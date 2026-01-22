"""
Spectrum Matching Optimization for LWFA

Supports multiple optimization modes:
- Multi-seed optimization with a single approach
- Comparison test across 7 approaches with 10 seeds each
- Extend existing comparison with additional approaches

USAGE:
------
# Run comparison test (7 approaches × 10 seeds)
python optimize_match_spectrum.py --mode comparison

# Extend existing comparison with new approaches
python optimize_match_spectrum.py --mode extend --output comparison_20260114_110450 --approaches adam_lbfgs bayes_adam_lbfgs

# Run multi-seed with specific approach
python optimize_match_spectrum.py --mode multi --approach bayes_adam --seeds 351

APPROACHES:
-----------
1) bayesian_only   - Bayesian optimization (100 steps)
2) adam_only       - Adam from random start (100 steps)
3) lbfgs_only      - LBFGS from random start (100 steps)
4) bayes_adam      - Bayesian (100) + Adam (50)
5) bayes_lbfgs     - Bayesian (100) + LBFGS (50)
6) adam_lbfgs      - Adam (50) + LBFGS (50)
7) bayes_adam_lbfgs - Bayesian (100) + Adam (50) + LBFGS (50)
8) bayes_lbfgs_adam - Bayesian (100) + LBFGS (50) + Adam (50)
9) bayes_sgd       - Bayesian (100) + SGD (50)
"""

import os
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
from datetime import datetime
import logging
from schedulefree import RAdamScheduleFree

from src.modules_1d import EDMPrecond
from src.diffusion import DifferentiableEdmSampler
from src.utils import deflection_biexp_calc

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
    
    DEFAULT_PARAMS = {
        'pressure_bounds': (1.0, 50.0),
        'laser_energy_bounds': (5.0, 50.0),
        'acquisition_time_bounds': (5.0, 100.0),
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
    }
    
    def __init__(self, model_path, target_spectrum_csv, device="cuda", seed=None, logger=None, **kwargs):
        # Merge defaults with provided kwargs
        params = {**self.DEFAULT_PARAMS, **kwargs}
        for key, value in params.items():
            setattr(self, key, value)
        
        self.device = device
        self.seed = seed
        self.logger = logger
        
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

    def get_history(self):
        """Get all optimization history."""
        return {'bayesian': self.bayesian_history, 'gradient': self.gradient_history}


# =============================================================================
# APPROACH RUNNERS
# =============================================================================

def run_approach(approach, opt_params, seed, output_dir):
    """Run a single optimization approach with given seed."""
    set_seed(seed)
    logger = setup_logging(output_dir, approach, seed)
    
    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=seed, logger=logger)
    
    if approach == 'bayesian_only':
        result = optimizer.run_bayesian(n_calls=100)
    
    elif approach == 'adam_only':
        result = optimizer.run_adam(optimizer.start_params, n_steps=100)
    
    elif approach == 'lbfgs_only':
        result = optimizer.run_lbfgs(optimizer.start_params, n_steps=100)
    
    elif approach == 'bayes_adam':
        bayes_result = optimizer.run_bayesian(n_calls=100)
        adam_result = optimizer.run_adam(bayes_result['best_params'], n_steps=50)
        result = adam_result if adam_result['best_mse'] < bayes_result['best_mse'] else bayes_result
    
    elif approach == 'bayes_lbfgs':
        bayes_result = optimizer.run_bayesian(n_calls=100)
        lbfgs_result = optimizer.run_lbfgs(bayes_result['best_params'], n_steps=50)
        result = lbfgs_result if lbfgs_result['best_mse'] < bayes_result['best_mse'] else bayes_result
    
    elif approach == 'adam_lbfgs':
        adam_result = optimizer.run_adam(optimizer.start_params, n_steps=50)
        lbfgs_result = optimizer.run_lbfgs(adam_result['best_params'], n_steps=50)
        result = lbfgs_result if lbfgs_result['best_mse'] < adam_result['best_mse'] else adam_result
    
    elif approach == 'bayes_adam_lbfgs':
        bayes_result = optimizer.run_bayesian(n_calls=100)
        adam_result = optimizer.run_adam(bayes_result['best_params'], n_steps=50)
        best_after_adam = adam_result if adam_result['best_mse'] < bayes_result['best_mse'] else bayes_result
        lbfgs_result = optimizer.run_lbfgs(best_after_adam['best_params'], n_steps=50)
        result = lbfgs_result if lbfgs_result['best_mse'] < best_after_adam['best_mse'] else best_after_adam
    
    elif approach == 'bayes_lbfgs_adam':
        bayes_result = optimizer.run_bayesian(n_calls=100)
        lbfgs_result = optimizer.run_lbfgs(bayes_result['best_params'], n_steps=50)
        best_after_lbfgs = lbfgs_result if lbfgs_result['best_mse'] < bayes_result['best_mse'] else bayes_result
        adam_result = optimizer.run_adam(best_after_lbfgs['best_params'], n_steps=50)
        result = adam_result if adam_result['best_mse'] < best_after_lbfgs['best_mse'] else best_after_lbfgs
    
    elif approach == 'bayes_sgd':
        bayes_result = optimizer.run_bayesian(n_calls=100)
        sgd_result = optimizer.run_sgd(bayes_result['best_params'], n_steps=50)
        result = sgd_result if sgd_result['best_mse'] < bayes_result['best_mse'] else bayes_result
    
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    close_logger(logger)
    return {'best_params': result['best_params'], 'best_mse': result['best_mse'], 'history': optimizer.get_history()}


# =============================================================================
# COMPARISON TEST
# =============================================================================

ALL_APPROACHES = ['bayesian_only', 'adam_only', 'lbfgs_only', 'bayes_adam', 'bayes_lbfgs', 'adam_lbfgs', 'bayes_adam_lbfgs', 'bayes_lbfgs_adam', 'bayes_sgd']
ALL_LABELS = ['Bayesian Only', 'Adam Only', 'LBFGS Only', 'Bayes + Adam', 'Bayes + LBFGS', 'Adam + LBFGS', 'Bayes + Adam + LBFGS', 'Bayes + LBFGS + Adam', 'Bayes + SGD']
COMPARISON_SEEDS = [67, 156, 236, 391, 429, 504, 742, 782, 823, 918]


def run_comparison_test(opt_params, output_dir, approaches=None):
    """Run comparison test: approaches × 10 seeds."""
    seeds = COMPARISON_SEEDS
    approaches = approaches or ALL_APPROACHES
    approach_labels = [ALL_LABELS[ALL_APPROACHES.index(a)] for a in approaches]
    
    print("="*80)
    print("OPTIMIZER COMPARISON TEST")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Approaches: {approaches}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({'seeds': seeds, 'approaches': approaches, 'opt_params': {k: list(v) if isinstance(v, tuple) else v for k, v in opt_params.items()}}, f, indent=2)
    
    results = {approach: {} for approach in approaches}
    total = len(approaches) * len(seeds)
    run_num = 0
    
    for seed in seeds:
        for approach in approaches:
            run_num += 1
            print(f"\n{'='*20} Run {run_num}/{total}: {approach} (seed={seed}) {'='*20}")
            
            run_dir = os.path.join(output_dir, f"seed_{seed}", approach)
            os.makedirs(run_dir, exist_ok=True)
            
            result = run_approach(approach, opt_params, seed, run_dir)
            results[approach][seed] = result
    
    # Save results summary
    summary = {approach: {seed: {'best_mse': r['best_mse'], 'best_params': r['best_params']} 
               for seed, r in seed_results.items()} for approach, seed_results in results.items()}
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot comparison
    plot_comparison(results, seeds, approaches, approach_labels, output_dir)
    
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
    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(approaches))]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot
    ax = axes[0, 0]
    mse_data = [[results[a][s]['best_mse'] for s in seeds] for a in approaches]
    bp = ax.boxplot(mse_data, labels=labels, patch_artist=True)
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


# =============================================================================
# EXTEND EXISTING COMPARISON
# =============================================================================

def extend_comparison(opt_params, output_dir, new_approaches):
    """Add new approaches to an existing comparison folder."""
    seeds = COMPARISON_SEEDS
    
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
            
            result = run_approach(approach, opt_params, seed, run_dir)
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
    all_labels = [ALL_LABELS[ALL_APPROACHES.index(a)] if a in ALL_APPROACHES else a for a in all_approaches]
    
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

def run_multi_seed(opt_params, approach, n_seeds, output_dir):
    """Run multi-seed optimization with a single approach."""
    np.random.seed(42)
    seeds = np.random.randint(0, 100000, size=1000).tolist()[:n_seeds]
    
    print("="*80)
    print(f"MULTI-SEED OPTIMIZATION: {approach.upper()}")
    print("="*80)
    print(f"Seeds: {n_seeds}, Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump({'seeds': seeds, 'approach': approach, 'opt_params': {k: list(v) if isinstance(v, tuple) else v for k, v in opt_params.items()}}, f, indent=2)
    
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} Run {i+1}/{n_seeds} (seed={seed}) {'='*20}")
        
        run_dir = os.path.join(output_dir, f"seed_{seed}")
        result = run_approach(approach, opt_params, seed, run_dir)
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
    parser.add_argument('--mode', choices=['comparison', 'multi', 'extend'], default='comparison', help='Run mode')
    parser.add_argument('--approach', default='bayes_adam', help='Approach for multi-seed mode')
    parser.add_argument('--approaches', nargs='+', default=None, help='Approaches for extend mode')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds for multi-seed mode')
    parser.add_argument('--output', default=None, help='Output directory (required for extend mode)')
    parser.add_argument('--device', default='cuda:1', help='Device to use')
    parser.add_argument('--target', default='avg_spectrum_45_25_20.csv', help='Target spectrum CSV')
    args = parser.parse_args()
    
    # Common parameters
    opt_params = {
        'model_path': "models/edm_4kepochs/ema_ckpt_final.pt",
        'device': args.device if torch.cuda.is_available() else "cpu",
        'target_spectrum_csv': args.target,
        'pressure_bounds': (1.0, 50.0),
        'laser_energy_bounds': (5.0, 50.0),
        'acquisition_time_bounds': (5.0, 100.0),
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.mode == 'comparison':
        output_dir = args.output or f"comparison_{timestamp}"
        run_comparison_test(opt_params, output_dir)
    elif args.mode == 'extend':
        if not args.output:
            raise ValueError("--output is required for extend mode")
        if not args.approaches:
            raise ValueError("--approaches is required for extend mode")
        extend_comparison(opt_params, args.output, args.approaches)
    else:
        output_dir = args.output or f"multi_{args.approach}_{timestamp}"
        run_multi_seed(opt_params, args.approach, args.seeds, output_dir)


if __name__ == "__main__":
    main()
