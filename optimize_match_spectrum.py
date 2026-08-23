"""
Spectrum Matching Optimization for LWFA

Everything that used to be hardcoded here (model, target spectrum, bounds,
sampler settings, seeds and the optimization approaches themselves) now lives in
a config file - see configs/match_spectrum.yaml. Command line flags override
the config; a config only needs to list the keys it changes.

USAGE:
------
# Run with the default config (configs/match_spectrum.yaml)
python optimize_match_spectrum.py

# Run with a custom config
python optimize_match_spectrum.py --config configs/my_experiment.yaml

# Override config values from the command line
python optimize_match_spectrum.py --mode multi --approach bayes_adam --seeds 351

# Extend an existing comparison with new approaches
python optimize_match_spectrum.py --mode extend --output comparison_20260114_110450 --approaches adam_lbfgs bayes_adam_lbfgs

MODES:
------
comparison - run every configured approach across `run.comparison_seeds`
multi      - run a single approach across `run.n_seeds` random seeds
extend     - add approaches to an existing comparison directory

APPROACHES:
-----------
Defined in the config under `approaches` as a list of stages. The first stage
starts from random parameters, each later stage starts from the best parameters
so far, and the best stage overall is reported. Defaults ship with:

1) bayesian_only    - Bayesian optimization (100 calls)
2) adam_only        - Adam from random start (100 steps)
3) lbfgs_only       - LBFGS from random start (100 steps)
4) bayes_adam       - Bayesian (100) + Adam (50)
5) bayes_lbfgs      - Bayesian (100) + LBFGS (50)
6) adam_lbfgs       - Adam (50) + LBFGS (50)
7) bayes_adam_lbfgs - Bayesian (100) + Adam (50) + LBFGS (50)
8) bayes_lbfgs_adam - Bayesian (100) + LBFGS (50) + Adam (50)
9) bayes_sgd        - Bayesian (100) + SGD (50)
10) bayes_lbfgs_langevin - Bayesian (100) + LBFGS (50) + Langevin/SGLD (50)
11) bayes_langevin_lbfgs - Bayesian (100) + Langevin/SGLD (50) + LBFGS (50)
12) bayes_prior_only     - Bayesian (100) with the surrogate loss cube as GP prior mean
13) bayes_prior_lbfgs_adam - Bayes+prior (100) + LBFGS (50) + RAdam (50)
14) bayes_prior_adam_lbfgs - Bayes+prior (100) + RAdam (50) + LBFGS (50)

Stage methods: bayesian, adam (RAdam), lbfgs, sgd, langevin (SGLD; kwargs
noise_scale = sqrt(temperature), decay_power for the annealed step size
eps_t = lr / (1 + step)**decay_power), bayesian_prior (GP prior mean from a
precomputed 3D loss cube; kwargs prior_npz - null gives the matched flat-mean
baseline).
"""

import os
import copy
import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from skopt import gp_minimize
from skopt.space import Real
import pandas as pd
import json
import yaml
from datetime import datetime
import logging
from schedulefree import RAdamScheduleFree

from src.modules_1d import EDMPrecond
from src.diffusion import DifferentiableEdmSampler
from src.utils import deflection_biexp_calc

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG_PATH = "configs/match_spectrum.yaml"

# Built-in defaults. A config file is deep-merged on top of this, so it only has
# to specify the keys it wants to change. Keep in sync with the shipped YAML.
DEFAULT_CONFIG = {
    'model_path': "models/edm_4kepochs/ema_ckpt_final.pt",
    'target_spectrum_csv': "avg_spectrum_45_25_20.csv",
    'device': "cuda:1",
    'run': {
        'mode': 'comparison',
        'output': None,
        'approach': 'bayes_adam',
        'approaches': None,
        'n_seeds': 10,
        'multi_seed_base': 42,
        'comparison_seeds': [67, 156, 236, 391, 429, 504, 742, 782, 823, 918],
        'real_vs_generated': False,
    },
    'optimizer': {
        'laser_energy_bounds': [5.0, 50.0],
        'pressure_bounds': [1.0, 50.0],
        'acquisition_time_bounds': [5.0, 100.0],
        'batch_size': 16,
        'spectrum_length': 256,
        'features': ["E", "P", "ms"],
        'num_sampling_steps': 18,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'rho': 7,
        'cfg_scale': 3.0,
        'smooth_output': True,
        'smooth_kernel_size': 9,
        'smooth_sigma': 2.0,
        'normalize_spectrum': False,
    },
    'approaches': {
        'bayesian_only': {'label': 'Bayesian Only', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10}]},
        'adam_only': {'label': 'Adam Only', 'stages': [
            {'method': 'adam', 'n_steps': 100, 'lr': 2.0}]},
        'lbfgs_only': {'label': 'LBFGS Only', 'stages': [
            {'method': 'lbfgs', 'n_steps': 100, 'lr': 2.0}]},
        'bayes_adam': {'label': 'Bayes + Adam', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_lbfgs': {'label': 'Bayes + LBFGS', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0}]},
        'adam_lbfgs': {'label': 'Adam + LBFGS', 'stages': [
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_adam_lbfgs': {'label': 'Bayes + Adam + LBFGS', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_lbfgs_adam': {'label': 'Bayes + LBFGS + Adam', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0},
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_sgd': {'label': 'Bayes + SGD', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'sgd', 'n_steps': 50, 'lr': 2.0, 'momentum': 0.9}]},
        'bayes_lbfgs_langevin': {'label': 'Bayes + LBFGS + Langevin', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0},
            {'method': 'langevin', 'n_steps': 50, 'lr': 2.0, 'noise_scale': 1.0, 'decay_power': 0.5}]},
        'bayes_langevin_lbfgs': {'label': 'Bayes + Langevin + LBFGS', 'stages': [
            {'method': 'bayesian', 'n_calls': 100, 'n_initial': 10},
            {'method': 'langevin', 'n_steps': 50, 'lr': 2.0, 'noise_scale': 1.0, 'decay_power': 0.5},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_prior_only': {'label': 'Bayes (surrogate prior)', 'stages': [
            {'method': 'bayesian_prior', 'n_calls': 100, 'n_initial': 10,
             'prior_npz': 'loss_landscape_bounds.npz'}]},
        'bayes_prior_lbfgs_adam': {'label': 'Bayes(prior) + LBFGS + Adam', 'stages': [
            {'method': 'bayesian_prior', 'n_calls': 100, 'n_initial': 10,
             'prior_npz': 'loss_landscape_bounds.npz'},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0},
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0}]},
        'bayes_prior_adam_lbfgs': {'label': 'Bayes(prior) + Adam + LBFGS', 'stages': [
            {'method': 'bayesian_prior', 'n_calls': 100, 'n_initial': 10,
             'prior_npz': 'loss_landscape_bounds.npz'},
            {'method': 'adam', 'n_steps': 50, 'lr': 2.0},
            {'method': 'lbfgs', 'n_steps': 50, 'lr': 2.0}]},
    },
}


def deep_merge(base, override):
    """Recursively merge `override` into `base`. Lists and scalars replace."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    """Load a YAML/JSON config merged on top of DEFAULT_CONFIG."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if not os.path.exists(path):
        if os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {path}")
        print(f"⚠️  No config at {path}, using built-in defaults")
        return config

    with open(path, 'r') as f:
        user_config = json.load(f) if path.endswith('.json') else yaml.safe_load(f)

    print(f"📄 Config: {path}")
    return deep_merge(config, user_config or {})


def resolve_approaches(config, names):
    """Validate approach names against the config, defaulting to all of them."""
    known = config['approaches']
    names = names or list(known.keys())
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown approach(es) {unknown}. Available: {list(known.keys())}")
    return names


def approach_label(config, name):
    """Human readable label for an approach, falling back to its name."""
    return config['approaches'].get(name, {}).get('label', name)


def build_opt_params(config):
    """Assemble the SpectrumMatchingOptimizer kwargs from the config."""
    return {
        'model_path': config['model_path'],
        'target_spectrum_csv': config['target_spectrum_csv'],
        'device': config['device'] if torch.cuda.is_available() else "cpu",
        **config['optimizer'],
    }


def save_run_config(output_dir, config, opt_params, **extra):
    """Persist the resolved config next to the results."""
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({**extra,
                   'opt_params': {k: list(v) if isinstance(v, tuple) else v for k, v in opt_params.items()},
                   'config': config}, f, indent=2)

    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    """Set random seed for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def setup_logging(output_dir, name, seed):
    """Set up logging for optimization runs."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    logger = logging.getLogger(f"{name}_seed_{seed}")
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(os.path.join(logs_dir, f"{name}_seed_{seed}.log"), mode='w')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def close_logger(logger):
    """Close all logger handlers."""
    if logger:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def create_energy_axis(length=256, electron_pointing_pixel=62):
    """Create energy axis using biexponential deflection calculation."""
    deflection_MeV, _ = deflection_biexp_calc(batch_size=1, hor_image_size=max(length * 2, 512), 
                                               electron_pointing_pixel=electron_pointing_pixel)
    valid_energies = np.sort(deflection_MeV[0].cpu().numpy()[deflection_MeV[0].cpu().numpy() > 0])[::-1]
    return valid_energies[:length] if len(valid_energies) >= length else np.pad(valid_energies, (length - len(valid_energies), 0))


# =============================================================================
# OPTIMIZER CLASS
# =============================================================================

class SpectrumMatchingOptimizer:
    """Flexible optimizer for spectrum matching with Bayesian and/or gradient-based methods."""
    
    DEFAULT_PARAMS = DEFAULT_CONFIG['optimizer']

    def __init__(self, model_path, target_spectrum_csv, device="cuda", seed=None, logger=None, **kwargs):
        # Merge defaults with provided kwargs
        params = {**self.DEFAULT_PARAMS, **kwargs}
        for key, value in params.items():
            setattr(self, key, value)
        
        self.device = device
        self.seed = seed
        self.logger = logger
        self.target_spectrum_csv = target_spectrum_csv
        
        if seed is not None:
            set_seed(seed)
        
        # Load target spectrum
        self._load_target_spectrum(target_spectrum_csv)
        
        # Random starting parameters
        self.start_params = [
            np.random.uniform(*self.laser_energy_bounds),
            np.random.uniform(*self.pressure_bounds),
            np.random.uniform(*self.acquisition_time_bounds)
        ]
        
        # Initialize model and sampler
        self._init_model(model_path)
        
        # Optimization space for Bayesian
        self.dimensions = [
            Real(*self.laser_energy_bounds, name='laser_energy'),
            Real(*self.pressure_bounds, name='pressure'),
            Real(*self.acquisition_time_bounds, name='acquisition_time')
        ]
        
        # History tracking
        self.bayesian_history = []
        self.gradient_history = []

    def _init_model(self, model_path):
        """Initialize EDM model and sampler."""
        self.model = EDMPrecond(
            resolution=self.spectrum_length, settings_dim=len(self.features),
            sigma_min=0, sigma_max=float('inf'), sigma_data=0.112,
            model_type='UNet_conditional', device=self.device
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.sampler = DifferentiableEdmSampler(
            net=self.model, num_steps=self.num_sampling_steps,
            sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
        )
        self.sampler.initialize_latents(n_samples=self.batch_size, resolution=self.spectrum_length, device=self.device)

    def _load_target_spectrum(self, csv_path):
        """Load target spectrum from CSV file."""
        df = pd.read_csv(csv_path)
        self.target_energy_axis = torch.tensor(df['energy_MeV'].values, dtype=torch.float32, device=self.device)
        target_raw = torch.tensor(df['intensity'].values, dtype=torch.float32, device=self.device)
        
        if self.normalize_spectrum:
            self.target_spectrum = (target_raw - target_raw.min()) / (target_raw.max() - target_raw.min())
        else:
            self.target_spectrum = target_raw

    def _sample_spectrum(self, settings):
        """Generate spectrum from settings."""
        x = self.sampler.sample_differentiable(
            resolution=self.spectrum_length, device=self.device, settings=settings,
            n_samples=self.batch_size, cfg_scale=self.cfg_scale, settings_dim=len(self.features),
            smooth_output=self.smooth_output, smooth_kernel_size=self.smooth_kernel_size, smooth_sigma=self.smooth_sigma
        )
        return x.squeeze(1).mean(dim=0)

    def _compute_mse(self, spectrum):
        """Compute MSE between generated and target spectrum."""
        if self.normalize_spectrum:
            spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
        return torch.mean((spectrum - self.target_spectrum) ** 2)

    def _log_step(self, phase, step, params, mse):
        """Log optimization step."""
        if self.logger:
            self.logger.info(f"{phase} Step {step}: params={params}, MSE={mse:.6f}")

    # -------------------------------------------------------------------------
    # Optimization Methods
    # -------------------------------------------------------------------------
    
    def run_bayesian(self, n_calls=100, n_initial=10):
        """Run Bayesian optimization."""
        print(f"  Running Bayesian ({n_calls} calls)")
        
        def objective(params):
            settings = torch.tensor(params, device=self.device).unsqueeze(0)
            with torch.no_grad():
                spectrum = self._sample_spectrum(settings)
            mse = self._compute_mse(spectrum).item()
            
            self.bayesian_history.append({
                'laser_energy': params[0], 'pressure': params[1], 'acquisition_time': params[2],
                'objective': mse, 'spectrum': spectrum.cpu().numpy()
            })
            self._log_step("Bayesian", len(self.bayesian_history), params, mse)
            
            if len(self.bayesian_history) % 10 == 0:
                print(f"    Eval {len(self.bayesian_history)}: [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}] -> MSE={mse:.6f}")
            return mse
        
        result = gp_minimize(objective, self.dimensions, n_calls=n_calls, n_initial_points=n_initial,
                            x0=[self.start_params], acq_func='gp_hedge', random_state=self.seed or 42)
        
        best = min(self.bayesian_history, key=lambda x: x['objective'])
        print(f"  Bayesian Best: MSE={best['objective']:.6f}")
        return {'best_params': result.x, 'best_mse': result.fun}

    def run_bayesian_prior(self, n_calls=100, n_initial=10, prior_npz=None,
                           xi=0.01, n_candidates=4096):
        """Bayesian optimization with the surrogate loss landscape as the GP prior mean.

        A GP with prior mean m(x) has posterior mean m(x) + k(x,X)K^-1(y - m(X)),
        so it is implemented by fitting the GP to residuals y - m(x) and adding m
        back inside the acquisition. m(x) trilinearly interpolates a 3D loss cube
        precomputed by plot_loss_landscape.py (prior_npz), so the search starts
        from the surrogate's belief about the landscape instead of a flat prior.
        With prior_npz=None the identical loop runs with m = 0, giving an exactly
        matched no-prior (zeroth-order) baseline.

        Implemented as a self-contained EI loop (sklearn GP on the unit cube,
        candidate sampling) because skopt's gp_minimize has no hook for a
        non-constant prior mean.
        """
        from scipy.stats import norm
        from scipy.interpolate import RegularGridInterpolator
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

        bounds = np.array([self.laser_energy_bounds, self.pressure_bounds,
                           self.acquisition_time_bounds], dtype=float)
        span = bounds[:, 1] - bounds[:, 0]

        if prior_npz:
            if not os.path.exists(prior_npz):
                raise FileNotFoundError(
                    f"Prior cube {prior_npz} not found - generate it with e.g.\n"
                    f"  python plot_loss_landscape.py --e-min 5 --e-max 50 --p-min 1 "
                    f"--p-max 50 --t-min 5 --t-max 100 --output "
                    f"{os.path.splitext(prior_npz)[0]}.png")
            data = np.load(prior_npz)
            if 'Z' not in data.files or data['Z'].ndim != 3:
                raise ValueError(f"{prior_npz} holds no 3D loss cube - run "
                                 f"plot_loss_landscape.py in 3d mode first")
            if 'target' in data.files and str(data['target']) != self.target_spectrum_csv:
                print(f"  WARNING: prior cube target ({data['target']}) differs from "
                      f"this run's target ({self.target_spectrum_csv})")
            axes = [np.asarray(data[k], dtype=float) for k in ('E', 'P', 't_open')]
            cube = RegularGridInterpolator(axes, np.asarray(data['Z'], dtype=float))
            lo = np.array([k[0] for k in axes])
            hi = np.array([k[-1] for k in axes])
            if np.any(lo > bounds[:, 0]) or np.any(hi < bounds[:, 1]):
                print(f"  WARNING: cube covers E[{lo[0]:g},{hi[0]:g}] "
                      f"P[{lo[1]:g},{hi[1]:g}] t[{lo[2]:g},{hi[2]:g}], smaller than the "
                      f"search bounds; prior mean is clamped to the cube edge outside")
            mean_fn = lambda X: cube(np.clip(X, lo, hi))
            print(f"  Running Bayesian+prior ({n_calls} calls, prior={prior_npz})")
        else:
            mean_fn = lambda X: np.zeros(len(X))
            print(f"  Running Bayesian+prior ({n_calls} calls, flat prior)")

        rng = np.random.default_rng(self.seed if self.seed is not None else 42)
        # Mirrors skopt's GP recipe: ARD Matern 5/2 + white noise on the unit cube.
        # The upper length-scale bound is deliberately loose (skopt uses 100 too):
        # this objective is a broad plateau with one narrow basin, so the MLE length
        # scale is legitimately large and a tight ceiling censors it (sklearn then
        # emits a ConvergenceWarning about hitting the bound).
        kernel = (ConstantKernel(1.0, (0.01, 1000.0))
                  * Matern(length_scale=[0.3] * 3, length_scale_bounds=(0.01, 100.0), nu=2.5)
                  + WhiteKernel(1e-8, (1e-12, 1e-2)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-10,
                                      n_restarts_optimizer=2,
                                      random_state=int(self.seed or 42))

        def objective(x):
            settings = torch.tensor([float(v) for v in x], device=self.device).unsqueeze(0)
            with torch.no_grad():
                spectrum = self._sample_spectrum(settings)
            mse = self._compute_mse(spectrum).item()
            self.bayesian_history.append({
                'laser_energy': float(x[0]), 'pressure': float(x[1]),
                'acquisition_time': float(x[2]), 'objective': mse,
                'spectrum': spectrum.cpu().numpy()
            })
            self._log_step("Bayesian", len(self.bayesian_history), [float(v) for v in x], mse)
            if len(self.bayesian_history) % 10 == 0:
                print(f"    Eval {len(self.bayesian_history)}: "
                      f"[{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}] -> MSE={mse:.6f}")
            return mse

        # Initial design: the run's random start point, then uniform random points
        # (same role as x0 + n_initial_points in run_bayesian).
        init = [np.asarray(self.start_params, dtype=float)]
        init += [bounds[:, 0] + rng.random(3) * span for _ in range(max(n_initial - 1, 0))]

        X_obs, y_obs = [], []
        for it in range(n_calls):
            if it < len(init):
                x = init[it]
            else:
                X = np.asarray(X_obs)
                fitted = gp.fit((X - bounds[:, 0]) / span,
                                np.asarray(y_obs) - mean_fn(X))
                cand = bounds[:, 0] + rng.random((n_candidates, 3)) * span
                x_best = X_obs[int(np.argmin(y_obs))]
                local = np.clip(x_best + rng.normal(0.0, 0.02, (256, 3)) * span,
                                bounds[:, 0], bounds[:, 1])
                cand = np.vstack([cand, local])
                mu_r, sd = fitted.predict((cand - bounds[:, 0]) / span, return_std=True)
                mu = mu_r + mean_fn(cand)
                imp = min(y_obs) - mu - xi
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = np.where(sd > 0, imp / sd, 0.0)
                    ei = np.where(sd > 0, imp * norm.cdf(z) + sd * norm.pdf(z), 0.0)
                x = cand[int(np.argmax(ei))]
            y = objective(x)
            X_obs.append(np.asarray(x, dtype=float))
            y_obs.append(y)

        best = int(np.argmin(y_obs))
        print(f"  Bayesian+prior Best: MSE={y_obs[best]:.6f}")
        return {'best_params': [float(v) for v in X_obs[best]], 'best_mse': float(y_obs[best])}

    def run_adam(self, initial_params, n_steps=100, lr=2.0):
        """Run Adam optimization."""
        print(f"  Running Adam ({n_steps} steps)")
        
        params = [torch.tensor(p, device=self.device, requires_grad=True) for p in initial_params]
        optimizer = RAdamScheduleFree(params, lr=lr)
        optimizer.train()
        
        best_mse, best_params = float('inf'), initial_params
        
        for step in range(n_steps):
            optimizer.zero_grad()
            settings = torch.stack(params).unsqueeze(0)
            spectrum = self._sample_spectrum(settings)
            loss = self._compute_mse(spectrum)
            loss.backward()
            optimizer.step()
            
            mse = loss.item()
            if mse < best_mse:
                best_mse = mse
                best_params = [p.item() for p in params]
            
            self.gradient_history.append({
                'iteration': step, 'laser_energy': params[0].item(), 'pressure': params[1].item(),
                'acquisition_time': params[2].item(), 'objective': mse
            })
            self._log_step("Adam", step, [p.item() for p in params], mse)
            
            if step % 20 == 0 or step == n_steps - 1:
                print(f"    Step {step}: MSE={mse:.6f}")
        
        optimizer.eval()
        print(f"  Adam Best: MSE={best_mse:.6f}")
        return {'best_params': best_params, 'best_mse': best_mse}

    def run_lbfgs(self, initial_params, n_steps=100, lr=2.0):
        """Run LBFGS optimization."""
        print(f"  Running LBFGS ({n_steps} max iter)")
        
        params = [torch.tensor(p, device=self.device, requires_grad=True) for p in initial_params]
        optimizer = optim.LBFGS(params, lr=lr, max_iter=n_steps, line_search_fn='strong_wolfe')
        
        best_mse, best_params = float('inf'), initial_params
        step_count = [0]
        
        def closure():
            optimizer.zero_grad()
            settings = torch.stack(params).unsqueeze(0)
            spectrum = self._sample_spectrum(settings)
            loss = self._compute_mse(spectrum)
            loss.backward()
            
            mse = loss.item()
            nonlocal best_mse, best_params
            if mse < best_mse:
                best_mse = mse
                best_params = [p.item() for p in params]
            
            self.gradient_history.append({
                'iteration': step_count[0], 'laser_energy': params[0].item(), 'pressure': params[1].item(),
                'acquisition_time': params[2].item(), 'objective': mse
            })
            self._log_step("LBFGS", step_count[0], [p.item() for p in params], mse)
            
            if step_count[0] % 20 == 0:
                print(f"    Iter {step_count[0]}: MSE={mse:.6f}")
            step_count[0] += 1
            return loss
        
        optimizer.step(closure)
        print(f"  LBFGS Best: MSE={best_mse:.6f}")
        return {'best_params': best_params, 'best_mse': best_mse}

    def run_sgd(self, initial_params, n_steps=100, lr=2.0, momentum=0.9):
        """Run SGD optimization."""
        print(f"  Running SGD ({n_steps} steps)")
        
        params = [torch.tensor(p, device=self.device, requires_grad=True) for p in initial_params]
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
        
        best_mse, best_params = float('inf'), initial_params
        
        for step in range(n_steps):
            optimizer.zero_grad()
            settings = torch.stack(params).unsqueeze(0)
            spectrum = self._sample_spectrum(settings)
            loss = self._compute_mse(spectrum)
            loss.backward()
            optimizer.step()
            
            mse = loss.item()
            if mse < best_mse:
                best_mse = mse
                best_params = [p.item() for p in params]
            
            self.gradient_history.append({
                'iteration': step, 'laser_energy': params[0].item(), 'pressure': params[1].item(),
                'acquisition_time': params[2].item(), 'objective': mse
            })
            self._log_step("SGD", step, [p.item() for p in params], mse)
            
            if step % 20 == 0 or step == n_steps - 1:
                print(f"    Step {step}: MSE={mse:.6f}")
        
        print(f"  SGD Best: MSE={best_mse:.6f}")
        return {'best_params': best_params, 'best_mse': best_mse}

    def run_langevin(self, initial_params, n_steps=50, lr=2.0, noise_scale=1.0, decay_power=0.5):
        """Run Stochastic Gradient Langevin Dynamics (SGLD).

        Per step, with an annealed step size eps_t = lr / (1 + step)**decay_power:

            theta <- theta - eps_t * grad + sqrt(2*eps_t) * noise_scale * N(0, 1)

        The gradient term matches plain SGD (no momentum) so `lr` is comparable to the
        other stages. `noise_scale` = sqrt(temperature): 1.0 is canonical SGLD, 0.0 is
        plain SGD, small values give a mild perturbation. `decay_power=0` recovers a
        constant step size. Best-seen parameters are tracked and returned.
        """
        print(f"  Running Langevin/SGLD ({n_steps} steps)")

        params = [torch.tensor(p, device=self.device, requires_grad=True) for p in initial_params]
        best_mse, best_params = float('inf'), initial_params

        for step in range(n_steps):
            for p in params:
                p.grad = None
            settings = torch.stack(params).unsqueeze(0)
            spectrum = self._sample_spectrum(settings)
            loss = self._compute_mse(spectrum)

            # Snapshot the params that produced this loss before perturbing them.
            mse = loss.item()
            current = [p.item() for p in params]
            if mse < best_mse:
                best_mse = mse
                best_params = current

            loss.backward()
            eps = lr / (1.0 + step) ** decay_power
            with torch.no_grad():
                for p in params:
                    noise = torch.randn_like(p) * ((2.0 * eps) ** 0.5) * noise_scale
                    p.add_(-eps * p.grad + noise)

            self.gradient_history.append({
                'iteration': step, 'laser_energy': current[0], 'pressure': current[1],
                'acquisition_time': current[2], 'objective': mse
            })
            self._log_step("Langevin", step, current, mse)

            if step % 20 == 0 or step == n_steps - 1:
                print(f"    Step {step}: MSE={mse:.6f}")

        print(f"  Langevin Best: MSE={best_mse:.6f}")
        return {'best_params': best_params, 'best_mse': best_mse}

    def get_history(self):
        """Get all optimization history."""
        return {'bayesian': self.bayesian_history, 'gradient': self.gradient_history}


# =============================================================================
# APPROACH RUNNERS
# =============================================================================

def run_stage(optimizer, method, initial_params, **kwargs):
    """Run a single optimization stage. Extra kwargs go to the run_* method."""
    if method == 'bayesian':
        return optimizer.run_bayesian(**kwargs)
    elif method == 'adam':
        return optimizer.run_adam(initial_params, **kwargs)
    elif method == 'lbfgs':
        return optimizer.run_lbfgs(initial_params, **kwargs)
    elif method == 'sgd':
        return optimizer.run_sgd(initial_params, **kwargs)
    elif method == 'bayesian_prior':
        return optimizer.run_bayesian_prior(**kwargs)
    elif method == 'langevin':
        return optimizer.run_langevin(initial_params, **kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def run_approach(approach, opt_params, seed, output_dir, approaches_config):
    """Run a single optimization approach (a chain of stages) with given seed."""
    set_seed(seed)
    logger = setup_logging(output_dir, approach, seed)

    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=seed, logger=logger)

    stages = approaches_config[approach].get('stages')
    if not stages:
        raise ValueError(f"Approach '{approach}' defines no stages")

    # Each stage starts from the best parameters found so far; the best stage wins.
    best = None
    for stage in stages:
        stage_kwargs = dict(stage)
        method = stage_kwargs.pop('method')
        initial_params = best['best_params'] if best else optimizer.start_params

        result = run_stage(optimizer, method, initial_params, **stage_kwargs)
        if best is None or result['best_mse'] < best['best_mse']:
            best = result

    close_logger(logger)
    return {'best_params': best['best_params'], 'best_mse': best['best_mse'], 'history': optimizer.get_history()}


# =============================================================================
# COMPARISON TEST
# =============================================================================

def run_comparison_test(config, output_dir):
    """Run comparison test: approaches × seeds."""
    opt_params = build_opt_params(config)
    seeds = config['run']['comparison_seeds']
    approaches = resolve_approaches(config, config['run']['approaches'])
    approach_labels = [approach_label(config, a) for a in approaches]

    print("="*80)
    print("OPTIMIZER COMPARISON TEST")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Approaches: {approaches}")
    print(f"Output: {output_dir}")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)
    save_run_config(output_dir, config, opt_params, seeds=seeds, approaches=approaches)

    results = {approach: {} for approach in approaches}
    total = len(approaches) * len(seeds)
    run_num = 0
    
    for seed in seeds:
        for approach in approaches:
            run_num += 1
            print(f"\n{'='*20} Run {run_num}/{total}: {approach} (seed={seed}) {'='*20}")
            
            run_dir = os.path.join(output_dir, f"seed_{seed}", approach)
            os.makedirs(run_dir, exist_ok=True)

            result = run_approach(approach, opt_params, seed, run_dir, config['approaches'])
            results[approach][seed] = result

    # Save results summary
    summary = {approach: {seed: {'best_mse': r['best_mse'], 'best_params': r['best_params']} 
               for seed, r in seed_results.items()} for approach, seed_results in results.items()}
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot comparison
    plot_comparison(results, seeds, approaches, approach_labels, output_dir)

    # Optional: real experimental spectrum vs optimized generated spectrum
    if config['run'].get('real_vs_generated'):
        for approach in approaches:
            plot_real_vs_generated(opt_params, results[approach], seeds, output_dir, approach)

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Approach':<20} {'Mean MSE':>12} {'Std MSE':>12} {'Min MSE':>12}")
    print("-"*60)
    
    best_approach, best_mean = None, float('inf')
    for approach, label in zip(approaches, approach_labels):
        mses = [results[approach][s]['best_mse'] for s in seeds]
        mean_mse = np.mean(mses)
        if mean_mse < best_mean:
            best_mean, best_approach = mean_mse, label
        print(f"{label:<20} {mean_mse:>12.6f} {np.std(mses):>12.6f} {np.min(mses):>12.6f}")
    
    print("-"*60)
    print(f"🏆 Best: {best_approach} (Mean MSE = {best_mean:.6f})")
    
    return results


def plot_comparison(results, seeds, approaches, labels, output_dir):
    """Create comparison plots."""
    # Generate enough colors for all approaches
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(approaches))]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot
    ax = axes[0, 0]
    mse_data = [[results[a][s]['best_mse'] for s in seeds] for a in approaches]
    # Set tick labels explicitly: boxplot's `labels` kwarg was renamed
    # `tick_labels` in matplotlib 3.9, so avoid the parameter entirely.
    bp = ax.boxplot(mse_data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('MSE')
    ax.set_title('MSE Distribution by Approach')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3)
    
    # 2. Grouped bar chart
    ax = axes[0, 1]
    x = np.arange(len(seeds))
    width = 0.8 / len(approaches)
    for i, (approach, label, color) in enumerate(zip(approaches, labels, colors)):
        mses = [results[approach][s]['best_mse'] for s in seeds]
        offset = (i - len(approaches)/2 + 0.5) * width
        ax.bar(x + offset, mses, width, label=label, color=color, alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('MSE')
    ax.set_title('MSE by Seed')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=45)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Mean ± std bar chart
    ax = axes[1, 0]
    means = [np.mean([results[a][s]['best_mse'] for s in seeds]) for a in approaches]
    stds = [np.std([results[a][s]['best_mse'] for s in seeds]) for a in approaches]
    bars = ax.bar(range(len(approaches)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Mean MSE')
    ax.set_title('Mean MSE ± Std')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, f'{mean:.4f}', ha='center', fontsize=8)
    
    # 4. Convergence curves
    ax = axes[1, 1]
    for approach, label, color in zip(approaches, labels, colors):
        all_curves = []
        max_len = 0
        for seed in seeds:
            history = results[approach][seed]['history']
            objs = [h['objective'] for h in history['bayesian']] + [h['objective'] for h in history['gradient']]
            if objs:
                best_so_far = np.minimum.accumulate(objs)
                all_curves.append(best_so_far)
                max_len = max(max_len, len(best_so_far))
        
        if all_curves:
            padded = [np.pad(c, (0, max_len - len(c)), constant_values=c[-1]) for c in all_curves]
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            ax.plot(mean_curve, color=color, linewidth=2, label=label)
            ax.fill_between(range(len(mean_curve)), mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best MSE')
    ax.set_title('Convergence (Mean ± Std)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: {os.path.join(output_dir, 'comparison.png')}")
    plt.close()


def plot_real_vs_generated(opt_params, seed_results, seeds, output_dir, approach):
    """Overlay the real target spectrum (mean ± std) with the optimized generated
    spectrum (mean ± std across seeds).

    Regenerates each seed's spectrum from its best parameters using the same sampler
    settings the optimizer scored with. The model is loaded once and latents are
    re-seeded per seed so each spectrum matches its run.
    """
    print(f"\n  Building real-vs-generated figure for {approach}...")

    # Real target: energy_MeV, intensity (mean), optional intensity_std
    df = pd.read_csv(opt_params['target_spectrum_csv'])
    energy = df['energy_MeV'].values
    real_mean = df['intensity'].values
    real_std = df['intensity_std'].values if 'intensity_std' in df.columns else np.zeros_like(real_mean)

    # Load the model once via a single optimizer instance
    optimizer = SpectrumMatchingOptimizer(**opt_params)
    device = optimizer.device

    generated = []
    for seed in seeds:
        result = seed_results.get(seed)
        if result is None:
            continue
        set_seed(seed)
        optimizer.sampler.initialize_latents(
            n_samples=optimizer.batch_size, resolution=optimizer.spectrum_length, device=device)
        settings = torch.tensor(result['best_params'], device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            spectrum = optimizer._sample_spectrum(settings)
        generated.append(spectrum.cpu().numpy())

    if not generated:
        print("    No seed results to plot; skipping.")
        return

    gen = np.array(generated)
    gen_mean, gen_std = gen.mean(axis=0), gen.std(axis=0)

    # Two panels: raw and min-max normalized (for shape)
    def minmax(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, normalize in zip(axes, (False, True)):
        if normalize:
            rm, gm = minmax(real_mean), minmax(gen_mean)
            rs = real_std / (real_mean.max() - real_mean.min() + 1e-8)
            gs = gen_std / (gen_mean.max() - gen_mean.min() + 1e-8)
            title = "Min-max normalized (shape)"
        else:
            rm, gm, rs, gs = real_mean, gen_mean, real_std, gen_std
            title = "Raw intensity"

        ax.plot(energy, rm, color='black', lw=2, label='Real (mean)')
        ax.fill_between(energy, rm - rs, rm + rs, color='black', alpha=0.2, label='Real ±std (shots)')
        ax.plot(energy, gm, color='#d62728', lw=2, label='Generated (mean)')
        ax.fill_between(energy, gm - gs, gm + gs, color='#d62728', alpha=0.2, label='Generated ±std (seeds)')
        ax.set_xlabel('Energy [MeV]')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{approach}: real experimental target vs optimized generated spectrum "
                 f"({len(generated)} seeds)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"real_vs_generated_{approach}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close()


# =============================================================================
# EXTEND EXISTING COMPARISON
# =============================================================================

def extend_comparison(config, output_dir):
    """Add new approaches to an existing comparison folder."""
    opt_params = build_opt_params(config)
    seeds = config['run']['comparison_seeds']
    new_approaches = resolve_approaches(config, config['run']['approaches'])

    print("="*80)
    print("EXTENDING COMPARISON TEST")
    print("="*80)
    print(f"Output: {output_dir}")
    print(f"New approaches: {new_approaches}")
    print(f"Seeds: {seeds}")
    print("="*80)
    
    # Load existing results if available
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}
    
    # Run new approaches
    total = len(new_approaches) * len(seeds)
    run_num = 0
    
    new_results = {approach: {} for approach in new_approaches}
    
    for seed in seeds:
        for approach in new_approaches:
            run_num += 1
            print(f"\n{'='*20} Run {run_num}/{total}: {approach} (seed={seed}) {'='*20}")
            
            run_dir = os.path.join(output_dir, f"seed_{seed}", approach)
            os.makedirs(run_dir, exist_ok=True)
            
            result = run_approach(approach, opt_params, seed, run_dir, config['approaches'])
            new_results[approach][seed] = result
    
    # Merge with existing results
    for approach, seed_results in new_results.items():
        existing_results[approach] = {str(seed): {'best_mse': r['best_mse'], 'best_params': r['best_params']} 
                                       for seed, r in seed_results.items()}
    
    # Save merged results
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Load all results for plotting (need history too)
    all_approaches = list(existing_results.keys())
    all_labels = [approach_label(config, a) for a in all_approaches]

    # For new approaches, we have history; for old ones, create dummy history
    full_results = {}
    for approach in all_approaches:
        full_results[approach] = {}
        for seed in seeds:
            seed_key = str(seed) if str(seed) in existing_results[approach] else seed
            if approach in new_results and seed in new_results[approach]:
                full_results[approach][seed] = new_results[approach][seed]
            else:
                # Old approach - just use saved MSE/params, empty history for plotting
                saved = existing_results[approach].get(str(seed), existing_results[approach].get(seed, {}))
                full_results[approach][seed] = {
                    'best_mse': saved.get('best_mse', float('inf')),
                    'best_params': saved.get('best_params', [0, 0, 0]),
                    'history': {'bayesian': [], 'gradient': []}
                }
    
    # Plot updated comparison
    plot_comparison(full_results, seeds, all_approaches, all_labels, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (ALL APPROACHES)")
    print("="*80)
    print(f"{'Approach':<25} {'Mean MSE':>12} {'Std MSE':>12} {'Min MSE':>12}")
    print("-"*65)
    
    best_approach, best_mean = None, float('inf')
    for approach, label in zip(all_approaches, all_labels):
        mses = [full_results[approach][s]['best_mse'] for s in seeds]
        mean_mse = np.mean(mses)
        if mean_mse < best_mean:
            best_mean, best_approach = mean_mse, label
        print(f"{label:<25} {mean_mse:>12.6f} {np.std(mses):>12.6f} {np.min(mses):>12.6f}")
    
    print("-"*65)
    print(f"🏆 Best: {best_approach} (Mean MSE = {best_mean:.6f})")
    
    return full_results


# =============================================================================
# MULTI-SEED RUN (ORIGINAL FUNCTIONALITY)
# =============================================================================

def run_multi_seed(config, output_dir):
    """Run multi-seed optimization with a single approach."""
    opt_params = build_opt_params(config)
    approach = resolve_approaches(config, [config['run']['approach']])[0]
    n_seeds = config['run']['n_seeds']

    np.random.seed(config['run']['multi_seed_base'])
    seeds = np.random.randint(0, 100000, size=1000).tolist()[:n_seeds]

    print("="*80)
    print(f"MULTI-SEED OPTIMIZATION: {approach.upper()}")
    print("="*80)
    print(f"Seeds: {n_seeds}, Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    save_run_config(output_dir, config, opt_params, seeds=seeds, approach=approach)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} Run {i+1}/{n_seeds} (seed={seed}) {'='*20}")

        run_dir = os.path.join(output_dir, f"seed_{seed}")
        result = run_approach(approach, opt_params, seed, run_dir, config['approaches'])
        result['seed'] = seed
        results.append(result)
        
        print(f"  Result: MSE={result['best_mse']:.6f}, Params={[f'{p:.2f}' for p in result['best_params']]}")
    
    # Save results
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump([{'seed': r['seed'], 'best_mse': r['best_mse'], 'best_params': r['best_params']} for r in results], f, indent=2)
    
    # Summary
    mses = [r['best_mse'] for r in results]
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mean MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"Best: {np.min(mses):.6f} (seed={seeds[np.argmin(mses)]})")
    print(f"Worst: {np.max(mses):.6f} (seed={seeds[np.argmax(mses)]})")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spectrum Matching Optimization")
    parser.add_argument('--config', default=DEFAULT_CONFIG_PATH, help='Path to YAML/JSON config file')
    # The flags below override the corresponding config values when given.
    parser.add_argument('--mode', choices=['comparison', 'multi', 'extend'], default=None, help='Run mode')
    parser.add_argument('--approach', default=None, help='Approach for multi-seed mode')
    parser.add_argument('--approaches', nargs='+', default=None, help='Approaches for comparison/extend mode')
    parser.add_argument('--seeds', type=int, default=None, help='Number of seeds for multi-seed mode')
    parser.add_argument('--output', default=None, help='Output directory (required for extend mode)')
    parser.add_argument('--device', default=None, help='Device to use')
    parser.add_argument('--target', default=None, help='Target spectrum CSV')
    parser.add_argument('--model', default=None, help='Model checkpoint path')
    args = parser.parse_args()

    config = load_config(args.config)

    # Command line overrides
    overrides = {'mode': args.mode, 'approach': args.approach, 'approaches': args.approaches,
                 'n_seeds': args.seeds, 'output': args.output}
    config['run'].update({k: v for k, v in overrides.items() if v is not None})
    for key, value in [('device', args.device), ('target_spectrum_csv', args.target), ('model_path', args.model)]:
        if value is not None:
            config[key] = value

    run_config = config['run']
    mode = run_config['mode']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode == 'comparison':
        output_dir = run_config['output'] or f"comparison_{timestamp}"
        run_comparison_test(config, output_dir)
    elif mode == 'extend':
        if not run_config['output']:
            raise ValueError("--output (or run.output) is required for extend mode")
        if not run_config['approaches']:
            raise ValueError("--approaches (or run.approaches) is required for extend mode")
        extend_comparison(config, run_config['output'])
    else:
        output_dir = run_config['output'] or f"multi_{run_config['approach']}_{timestamp}"
        run_multi_seed(config, output_dir)


if __name__ == "__main__":
    main()
