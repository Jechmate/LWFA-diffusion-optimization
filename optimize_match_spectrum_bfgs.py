"""
Multi-Seed Spectrum Matching Optimization for LWFA

Simplified script that runs hybrid Bayesian + gradient-based optimization to match a target spectrum
across multiple random seeds for variance analysis.

USAGE:
------
python optimize_match_spectrum.py

This will run optimization with 10 different seeds to match the spectrum from avg_spectrum.csv

FEATURES:
---------
✓ Match generated spectrum to target spectrum from CSV
✓ Hybrid Bayesian + gradient-based optimization (bfgs or Adam)
✓ Multi-seed variance analysis (10 seeds)
✓ Comprehensive visualization comparing convergence across seeds
✓ Option to skip Bayesian phase (bayesian_n_calls=0) for pure gradient-based optimization

OPTIMIZERS:
-----------
✓ bfgs: Second-order optimizer with line search (default)
✓ Adam: First-order adaptive learning rate optimizer

SEEDS USED: [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
"""

import os
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

# Import necessary modules from the project
from src.modules_1d import EDMPrecond
from src.diffusion import EdmSampler, transform_vector, gaussian_smooth_1d
from src.utils import deflection_biexp_calc

def set_seed(seed=42):
    """Set random seed for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def setup_logging(output_dir, seed):
    """Set up detailed logging for optimization runs."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    logger_name = f"spectrum_matching_seed_{seed}"
    log_filename = f"optimization_seed_{seed}.log"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    log_path = os.path.join(logs_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Starting Spectrum Matching Optimization - Seed: {seed}")
    
    return logger

def create_energy_axis(length=256, electron_pointing_pixel=62):
    """Create energy axis for plotting using biexponential deflection calculation."""
    temp_size = max(length * 2, 512)  
    deflection_MeV, _ = deflection_biexp_calc(
        batch_size=1, 
        hor_image_size=temp_size, 
        electron_pointing_pixel=electron_pointing_pixel
    )
    deflection_array = deflection_MeV[0].cpu().numpy()
    valid_energies = deflection_array[deflection_array > 0]
    valid_energies_sorted = np.sort(valid_energies)[::-1]
    
    if len(valid_energies_sorted) >= length:
        energy_axis = valid_energies_sorted[:length]
    else:
        energy_axis = np.concatenate([
            np.zeros(length - len(valid_energies_sorted)),
            valid_energies_sorted[::-1]
        ])
    return energy_axis

class DifferentiableEdmSampler(EdmSampler):
    """Differentiable version of EDM sampler that allows gradients to flow through the sampling process."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_latents = None
        self.latents_device = None
        self.latents_shape = None
    
    def initialize_latents(self, n_samples, resolution, device):
        """Initialize latents once for deterministic sampling."""
        self.latents_shape = (n_samples, 1, resolution)
        self.latents_device = device
        self.stored_latents = self.randn_like(torch.empty(self.latents_shape, device=device))
        print(f"Initialized deterministic latents with shape {self.latents_shape} on {device}")
    
    def sample_differentiable(self, resolution, device, settings=None, n_samples=1, cfg_scale=3, 
                             settings_dim=0, smooth_output=False, smooth_kernel_size=5, smooth_sigma=1.0):
        """Differentiable version of the sample method that preserves gradients."""
        # Use stored deterministic latents if available
        if (self.stored_latents is not None and 
            self.latents_shape == (n_samples, 1, resolution) and 
            self.latents_device == device):
            latents = self.stored_latents
        else:
            latents = self.randn_like(torch.empty((n_samples, 1, resolution), device=device))
            print(f"Warning: Creating new latents - algorithm may not be fully deterministic")

        sigma_min = self.sigma_min
        sigma_max = self.sigma_max

        # Time step discretization
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop
        x_next = latents.to(torch.float32) * t_steps[0]
       
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Increase noise temporarily
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)

            # Euler step
            if cfg_scale == -1:
                denoised = self.net(x_hat, t_hat, settings).to(torch.float32)
            elif settings_dim != 0: 
                denoised_uncond = self.net(x_hat, t_hat, None).to(torch.float32)
                denoised_cond = self.net(x_hat, t_hat, settings).to(torch.float32)
                denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond) 
            else:
                denoised_uncond = self.net(x_hat, t_hat, None).to(torch.float32)
                denoised = denoised_uncond

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
        
            # Apply 2nd order correction
            if i < self.num_steps - 1:
                if cfg_scale == -1:
                    denoised = self.net(x_next, t_next, settings).to(torch.float32)
                elif settings_dim != 0: 
                    denoised_uncond = self.net(x_next, t_next, None).to(torch.float32)
                    denoised_cond = self.net(x_next, t_next, settings).to(torch.float32)
                    denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
                else:
                    denoised_uncond = self.net(x_next, t_next, None).to(torch.float32)
                    denoised = denoised_uncond
                    
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    
        x_next = transform_vector(x_next)
        if smooth_output:
            x_next = gaussian_smooth_1d(x_next, kernel_size=smooth_kernel_size, sigma=smooth_sigma)
        return x_next

class SpectrumMatchingOptimizer:
    """
    Hybrid Bayesian + gradient-based optimizer for matching a target spectrum.
    Minimizes MSE between generated and target spectra.
    
    Supports two gradient-based optimizers for fine-tuning:
    - bfgs: Second-order optimizer with line search
    - Adam: First-order adaptive learning rate optimizer
    """
    def __init__(
        self,
        model_path,
        target_spectrum_csv='avg_spectrum.csv',
        device="cuda",
        pressure_bounds=(1.0, 50.0),
        laser_energy_bounds=(5.0, 50.0),
        acquisition_time_bounds=(5.0, 100.0),
        bayesian_n_calls=100,
        bayesian_n_initial_points=10,
        finetune_max_iter=50,
        finetune_lr=2.0,
        finetune_optimizer='bfgs',
        batch_size=16,
        output_dir="spectrum_matching_results",
        spectrum_length=256,
        features=["E", "P", "ms"],
        num_sampling_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        cfg_scale=3.0,
        smooth_output=True,
        smooth_kernel_size=9,
        smooth_sigma=2.0,
        normalize_spectrum=False,
        seed=None,
        logger=None
    ):
        self.device = device
        self.pressure_bounds = pressure_bounds
        self.laser_energy_bounds = laser_energy_bounds
        self.acquisition_time_bounds = acquisition_time_bounds
        
        self.bayesian_n_calls = bayesian_n_calls
        self.bayesian_n_initial_points = bayesian_n_initial_points
        self.finetune_max_iter = finetune_max_iter
        self.finetune_lr = finetune_lr
        self.finetune_optimizer = finetune_optimizer.lower()
        
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.spectrum_length = spectrum_length
        self.features = features
        
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.cfg_scale = cfg_scale
        
        self.smooth_output = smooth_output
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        self.normalize_spectrum = normalize_spectrum
        
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        
        # Set up logger
        self.logger = logger
        
        # Load target spectrum
        self._load_target_spectrum(target_spectrum_csv)
        
        # Random starting parameters
        self.laser_energy = np.random.uniform(laser_energy_bounds[0], laser_energy_bounds[1])
        self.pressure = np.random.uniform(pressure_bounds[0], pressure_bounds[1])
        self.acquisition_time_ms = np.random.uniform(acquisition_time_bounds[0], acquisition_time_bounds[1])
        
        print(f"Spectrum Matching Optimization Setup:")
        if bayesian_n_calls == 0:
            print(f"  Mode: {self.finetune_optimizer.upper()}-only (bayesian_n_calls=0)")
            if self.finetune_optimizer == 'adam':
                print(f"  Adam: {finetune_max_iter} epochs, lr={finetune_lr}")
            else:
                print(f"  bfgs: max {finetune_max_iter} iter, lr={finetune_lr}")
        else:
            print(f"  Phase 1 - Bayesian: {bayesian_n_calls} calls, {bayesian_n_initial_points} initial")
            if self.finetune_optimizer == 'adam':
                print(f"  Phase 2 - Adam: {finetune_max_iter} epochs, lr={finetune_lr}")
            else:
                print(f"  Phase 2 - bfgs: max {finetune_max_iter} iter, lr={finetune_lr}")
        print(f"  Normalization: {'Enabled' if normalize_spectrum else 'Disabled (using raw intensities)'}")
        print(f"  Starting: Laser={self.laser_energy:.2f}, Pressure={self.pressure:.2f}, Time={self.acquisition_time_ms:.2f}ms")
        
        # Create directories
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize EDM model
        self.model = EDMPrecond(
            resolution=spectrum_length,
            settings_dim=len(features),
            sigma_min=0,
            sigma_max=float('inf'),
            sigma_data=0.112,
            model_type='UNet_conditional',
            device=device
        ).to(device)
        self.load_model(model_path)
        
        # Initialize differentiable sampler
        self.sampler = DifferentiableEdmSampler(
            net=self.model, 
            num_steps=self.num_sampling_steps, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            rho=self.rho
        )
        
        # Initialize latents once for deterministic sampling
        self.sampler.initialize_latents(
            n_samples=self.batch_size,
            resolution=self.spectrum_length,
            device=self.device
        )
        
        # Define optimization space for Bayesian phase
        self.dimensions = [
            Real(laser_energy_bounds[0], laser_energy_bounds[1], name='laser_energy'),
            Real(pressure_bounds[0], pressure_bounds[1], name='pressure'),
            Real(acquisition_time_bounds[0], acquisition_time_bounds[1], name='acquisition_time')
        ]
        
        # Initialize history
        self.bayesian_history = []
        self.bfgs_history = []

    def load_model(self, model_path):
        """Load the pre-trained EDM model"""
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"EDM model loaded from {model_path}")

    def _load_target_spectrum(self, csv_path):
        """Load target spectrum from CSV file."""
        print(f"Loading target spectrum from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'energy_MeV' not in df.columns or 'intensity' not in df.columns:
            raise ValueError(f"CSV must contain 'energy_MeV' and 'intensity' columns")
        
        # Store as tensors on device
        self.target_energy_axis = torch.tensor(df['energy_MeV'].values, dtype=torch.float32, device=self.device)
        target_spectrum_raw = torch.tensor(df['intensity'].values, dtype=torch.float32, device=self.device)
        
        if self.normalize_spectrum:
            # Normalize target spectrum to [0, 1] range
            target_min = target_spectrum_raw.min()
            target_max = target_spectrum_raw.max()
            
            if target_max > target_min:
                self.target_spectrum = (target_spectrum_raw - target_min) / (target_max - target_min)
                print(f"  Loaded and normalized target spectrum: {len(self.target_spectrum)} points")
            else:
                raise ValueError("Target spectrum has no variation (min == max)")
        else:
            # Use raw intensity values
            self.target_spectrum = target_spectrum_raw
            print(f"  Loaded target spectrum (no normalization): {len(self.target_spectrum)} points")
        
        print(f"  Energy range: [{self.target_energy_axis.min():.2f}, {self.target_energy_axis.max():.2f}] MeV")

    def calculate_spectrum_distance(self, spectrum):
        """Calculate MSE distance between generated spectrum and target spectrum."""
        if len(spectrum) != len(self.target_spectrum):
            raise ValueError(f"Spectrum length mismatch: generated={len(spectrum)}, target={len(self.target_spectrum)}")
        
        if self.normalize_spectrum:
            # Normalize generated spectrum to [0, 1] range
            spectrum_min = torch.min(spectrum)
            spectrum_max = torch.max(spectrum)
            eps = 1e-8
            spectrum_normalized = (spectrum - spectrum_min) / (spectrum_max - spectrum_min + eps)
            
            # Calculate MSE with normalized spectra
            mse = torch.mean((spectrum_normalized - self.target_spectrum) ** 2)
        else:
            # Calculate MSE with raw intensity values
            mse = torch.mean((spectrum - self.target_spectrum) ** 2)
        
        return mse

    def bayesian_objective_function(self, params):
        """Objective function for Bayesian optimization phase."""
        laser_energy, pressure, acquisition_time = params
        
        try:
            settings = torch.tensor([laser_energy, pressure, acquisition_time], device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                x = self.sampler.sample_differentiable(
                    resolution=self.spectrum_length,
                    device=self.device,
                    settings=settings,
                    n_samples=self.batch_size,
                    cfg_scale=self.cfg_scale,
                    settings_dim=len(self.features),
                    smooth_output=self.smooth_output,
                    smooth_kernel_size=self.smooth_kernel_size,
                    smooth_sigma=self.smooth_sigma
                )
            
            deflection_MeV_np = create_energy_axis(256, 62)[::-1]
            deflection_MeV = torch.tensor(deflection_MeV_np.copy(), device=self.device)
            spectrum_intensity = x.squeeze(1).mean(dim=0)
            
            spectrum_distance = self.calculate_spectrum_distance(spectrum_intensity)
            objective = spectrum_distance.item()
            
            # Store history
            self.bayesian_history.append({
                'laser_energy': laser_energy,
                'pressure': pressure,
                'acquisition_time': acquisition_time,
                'spectrum_distance': objective,
                'objective': objective,
                'spectrum': spectrum_intensity.detach().cpu().numpy(),
                'energy_values': deflection_MeV.detach().cpu().numpy()
            })
            
            # Log to file
            if self.logger:
                self.logger.info(f"Bayesian Step {len(self.bayesian_history)}: "
                               f"laser_energy={laser_energy:.6f}, pressure={pressure:.6f}, "
                               f"acquisition_time={acquisition_time:.6f}, MSE={objective:.6f}")
            
            print(f"Bayesian Eval {len(self.bayesian_history)}: "
                  f"[{laser_energy:.2f}, {pressure:.2f}, {acquisition_time:.2f}] -> MSE={objective:.6f}")
            
            return objective
            
        except Exception as e:
            print(f"Error in Bayesian objective: {e}")
            return 1e6

    def finetune_objective_and_grad(self, params_tensor):
        """Objective function with gradients for bfgs fine-tuning phase."""
        laser_energy, pressure, acquisition_time = params_tensor
        settings = torch.stack([laser_energy, pressure, acquisition_time]).unsqueeze(0)
        
        x = self.sampler.sample_differentiable(
            resolution=self.spectrum_length,
            device=self.device,
            settings=settings,
            n_samples=self.batch_size,
            cfg_scale=self.cfg_scale,
            settings_dim=len(self.features),
            smooth_output=self.smooth_output,
            smooth_kernel_size=self.smooth_kernel_size,
            smooth_sigma=self.smooth_sigma
        )
        
        spectrum_intensity = x.squeeze(1).mean(dim=0)
        spectrum_distance = self.calculate_spectrum_distance(spectrum_intensity)
        
        return spectrum_distance

    def run_bayesian_phase(self):
        """Run the Bayesian optimization exploration phase."""
        print(f"Phase 1: Bayesian Optimization ({self.bayesian_n_calls} evaluations)")
        
        x0 = [self.laser_energy, self.pressure, self.acquisition_time_ms]
        
        result = gp_minimize(
            func=self.bayesian_objective_function,
            dimensions=self.dimensions,
            n_calls=self.bayesian_n_calls,
            n_initial_points=self.bayesian_n_initial_points,
            x0=[x0],
            acq_func='gp_hedge',
            random_state=self.seed if self.seed else 42
        )
        
        best_objective = result.fun
        best_params = result.x
        best_call = min(self.bayesian_history, key=lambda x: x['objective'])
        best_spectrum_dist = best_call['spectrum_distance']
        
        print(f"  Best: [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}] -> MSE={best_spectrum_dist:.6f}")
        
        return {
            'best_params': best_params, 
            'best_objective': best_objective,
            'best_spectrum_dist': best_spectrum_dist,
            'result': result
        }
    
    def run_bfgs_phase(self, initial_params):
        """Run BFGS with parameter transformation for bounds."""
        from scipy.optimize import minimize
        
        print(f"Phase 2: BFGS Fine-tuning (max {self.finetune_max_iter} iterations)")
        
        # Transform parameters to unconstrained space
        def to_unconstrained(params, bounds):
            """Map bounded parameters to unbounded space using sigmoid-like transform."""
            result = []
            for val, (low, high) in zip(params, bounds):
                # Map [low, high] -> (-inf, +inf) using logit
                normalized = (val - low) / (high - low)
                normalized = np.clip(normalized, 1e-7, 1 - 1e-7)  # Avoid log(0)
                result.append(np.log(normalized / (1 - normalized)))
            return np.array(result)
        
        def to_constrained(unconstrained_params, bounds):
            """Map unbounded parameters back to bounded space using sigmoid."""
            result = []
            for val, (low, high) in zip(unconstrained_params, bounds):
                # Map (-inf, +inf) -> [low, high] using sigmoid
                sigmoid = 1 / (1 + np.exp(-val))
                result.append(low + sigmoid * (high - low))
            return np.array(result)
        
        bounds = [
            (self.laser_energy_bounds[0], self.laser_energy_bounds[1]),
            (self.pressure_bounds[0], self.pressure_bounds[1]),
            (self.acquisition_time_bounds[0], self.acquisition_time_bounds[1])
        ]
        
        best_objective = float('inf')
        best_spectrum_dist = None
        iteration_count = 0
        
        def objective_and_grad_numpy(unconstrained_params):
            """Objective function in unconstrained space."""
            nonlocal iteration_count, best_objective, best_spectrum_dist
            
            # Transform back to constrained space
            params_numpy = to_constrained(unconstrained_params, bounds)
            
            # Convert to torch tensors with gradients
            params_tensor = torch.tensor(params_numpy, device=self.device, dtype=torch.float32, requires_grad=True)
            
            # Compute loss
            loss = self.finetune_objective_and_grad(params_tensor)
            loss.backward()
            
            # Extract values
            objective_value = loss.item()
            grad_constrained = params_tensor.grad.cpu().numpy()
            
            # Transform gradients using chain rule
            grad_unconstrained = np.zeros_like(unconstrained_params)
            for i, (unc_val, (low, high)) in enumerate(zip(unconstrained_params, bounds)):
                sigmoid = 1 / (1 + np.exp(-unc_val))
                # d(constrained)/d(unconstrained) = sigmoid * (1 - sigmoid) * (high - low)
                grad_unconstrained[i] = grad_constrained[i] * sigmoid * (1 - sigmoid) * (high - low)
            
            # Store history
            with torch.no_grad():
                laser_energy, pressure, acquisition_time = params_tensor
                settings = torch.stack([laser_energy, pressure, acquisition_time]).unsqueeze(0)
                x = self.sampler.sample_differentiable(
                    resolution=self.spectrum_length,
                    device=self.device,
                    settings=settings,
                    n_samples=self.batch_size,
                    cfg_scale=self.cfg_scale,
                    settings_dim=len(self.features),
                    smooth_output=self.smooth_output,
                    smooth_kernel_size=self.smooth_kernel_size,
                    smooth_sigma=self.smooth_sigma
                )
                deflection_MeV_np = create_energy_axis(256, 62)[::-1]
                deflection_MeV = torch.tensor(deflection_MeV_np.copy(), device=self.device)
                spectrum_intensity = x.squeeze(1).mean(dim=0)
                spectrum_dist = objective_value
            
            if objective_value < best_objective:
                best_objective = objective_value
                best_spectrum_dist = spectrum_dist
            
            self.bfgs_history.append({
                'iteration': iteration_count,
                'laser_energy': params_numpy[0],
                'pressure': params_numpy[1],
                'acquisition_time': params_numpy[2],
                'spectrum_distance': spectrum_dist,
                'objective': objective_value,
                'spectrum': spectrum_intensity.detach().cpu().numpy(),
                'energy_values': deflection_MeV.detach().cpu().numpy()
            })
            
            if self.logger:
                self.logger.info(f"BFGS Step {iteration_count}: "
                            f"laser_energy={params_numpy[0]:.6f}, pressure={params_numpy[1]:.6f}, "
                            f"acquisition_time={params_numpy[2]:.6f}, MSE={spectrum_dist:.6f}")
            
            if iteration_count % 5 == 0:
                print(f"  BFGS Iter {iteration_count}: [{params_numpy[0]:.3f}, {params_numpy[1]:.3f}, {params_numpy[2]:.3f}] -> MSE={spectrum_dist:.6f}")
            
            iteration_count += 1
            
            return objective_value, grad_unconstrained
        
        # Transform initial parameters to unconstrained space
        x0_unconstrained = to_unconstrained(initial_params, bounds)
        
        # Run BFGS optimization
        result = minimize(
            fun=objective_and_grad_numpy,
            x0=x0_unconstrained,
            method='BFGS',
            jac=True,
            options={'maxiter': self.finetune_max_iter}
        )
        
        # Transform final parameters back to constrained space
        final_params = to_constrained(result.x, bounds).tolist()
        print(f"  Final: [{final_params[0]:.3f}, {final_params[1]:.3f}, {final_params[2]:.3f}] -> MSE={best_spectrum_dist:.6f}")
        
        return {
            'best_params': final_params,
            'best_objective': best_objective,
            'best_spectrum_dist': best_spectrum_dist,
            'iterations': iteration_count,
            'scipy_result': result
        }

    def run_finetune_phase(self, initial_params):
        """Run the fine-tuning phase with the selected optimizer."""
        return self.run_bfgs_phase(initial_params)

    def optimize(self):
        """Run the complete hybrid optimization."""
        
        # Check if we should skip Bayesian phase
        if self.bayesian_n_calls == 0:
            optimizer_name = self.finetune_optimizer.upper()
            print(f"Starting {optimizer_name}-only Optimization (Spectrum Matching)")
            print(f"  Skipping Bayesian phase (bayesian_n_calls=0)")
            print(f"  Starting from random parameters (seed={self.seed})")
            
            # Use random starting parameters generated in __init__
            random_params = [self.laser_energy, self.pressure, self.acquisition_time_ms]
            print(f"  Random start: [{random_params[0]:.3f}, {random_params[1]:.3f}, {random_params[2]:.3f}]")
            
            # Run optimizer directly from random starting point
            finetune_result = self.run_finetune_phase(random_params)
            
            overall_best_params = finetune_result['best_params']
            overall_best_objective = finetune_result['best_objective']
            overall_best_spectrum_dist = finetune_result['best_spectrum_dist']
            best_phase = optimizer_name
            
            print(f"\nOptimization Complete!")
            print(f"  Best result from {best_phase} phase")
            print(f"  Final params: [{overall_best_params[0]:.3f}, {overall_best_params[1]:.3f}, {overall_best_params[2]:.3f}]")
            print(f"  Best spectrum distance (MSE): {overall_best_spectrum_dist:.6f}")
            
            result = {
                'best_params': overall_best_params,
                'best_objective': overall_best_objective,
                'best_spectrum_dist': overall_best_spectrum_dist,
                'best_phase': best_phase,
                'bayesian_phase': None,
                'finetune_phase': finetune_result,
                'bayesian_history': self.bayesian_history,
                'bfgs_history': self.bfgs_history
            }
            
            return result
        
        # Original hybrid optimization path
        optimizer_name = self.finetune_optimizer.upper()
        print(f"Starting Hybrid Bayesian-{optimizer_name} Optimization (Spectrum Matching)")
        
        # Phase 1: Bayesian exploration
        bayesian_result = self.run_bayesian_phase()
        
        # Phase 2: Fine-tuning
        finetune_result = self.run_finetune_phase(bayesian_result['best_params'])
        
        # Determine overall best
        if finetune_result['best_objective'] < bayesian_result['best_objective']:
            overall_best_params = finetune_result['best_params']
            overall_best_objective = finetune_result['best_objective']
            overall_best_spectrum_dist = finetune_result['best_spectrum_dist']
            best_phase = 'bfgs'
        else:
            overall_best_params = bayesian_result['best_params']
            overall_best_objective = bayesian_result['best_objective']
            overall_best_spectrum_dist = bayesian_result['best_spectrum_dist']
            best_phase = 'Bayesian'
        
        print(f"\nOptimization Complete!")
        print(f"  Best result from {best_phase} phase")
        print(f"  Final params: [{overall_best_params[0]:.3f}, {overall_best_params[1]:.3f}, {overall_best_params[2]:.3f}]")
        print(f"  Best spectrum distance (MSE): {overall_best_spectrum_dist:.6f}")
        
        # Calculate improvement
        if finetune_result['best_objective'] < bayesian_result['best_objective']:
            bayesian_best_dist = bayesian_result['best_spectrum_dist']
            if bayesian_best_dist > 0:
                improvement = ((bayesian_best_dist - overall_best_spectrum_dist) / bayesian_best_dist) * 100
                print(f"  bfgs improved by: {improvement:.2f}%")
        
        result = {
            'best_params': overall_best_params,
            'best_objective': overall_best_objective,
            'best_spectrum_dist': overall_best_spectrum_dist,
            'best_phase': best_phase,
            'bayesian_phase': bayesian_result,
            'finetune_phase': finetune_result,
            'bayesian_history': self.bayesian_history,
            'bfgs_history': self.bfgs_history
        }
        
        return result

    def plot_results(self):
        """Create visualization of the optimization results."""
        if not self.bayesian_history and not self.bfgs_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Extract data
        bayesian_objectives = [call['objective'] for call in self.bayesian_history]
        bfgs_objectives = [call['objective'] for call in self.bfgs_history] if self.bfgs_history else []
        
        # Plot 1: Convergence
        ax = axes[0, 0]
        if bayesian_objectives:
            bayesian_best_obj = np.minimum.accumulate(bayesian_objectives)
            ax.plot(range(len(bayesian_objectives)), bayesian_best_obj, 'b-', linewidth=2, label='Bayesian')
        
        if bfgs_objectives:
            starting_best_obj = bayesian_best_obj[-1] if bayesian_objectives else float('inf')
            bfgs_best_so_far = []
            current_best = starting_best_obj
            for obj in bfgs_objectives:
                current_best = min(current_best, obj)
                bfgs_best_so_far.append(current_best)
            
            offset = len(bayesian_objectives)
            combined_evals = [offset + i for i in range(len(bfgs_objectives))]
            ax.plot(combined_evals, bfgs_best_so_far, 'r-', linewidth=2, label='bfgs')
        
        ax.set_title('Spectrum Matching Convergence')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Spectrum Distance (MSE)')
        ax.legend()
        ax.grid(True)
        
        # Plot 2-4: Parameter evolution
        param_names = ['laser_energy', 'pressure', 'acquisition_time']
        param_labels = ['Laser Energy', 'Pressure', 'Acquisition Time (ms)']
        
        for idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            ax = axes[0, idx + 1] if idx < 2 else axes[1, 0]
            
            if bayesian_objectives:
                values = [call[param_name] for call in self.bayesian_history]
                ax.plot(range(len(values)), values, 'b.-', alpha=0.7, label='Bayesian')
            
            if bfgs_objectives:
                values_bfgs = [call[param_name] for call in self.bfgs_history]
                iters = [call['iteration'] for call in self.bfgs_history]
                offset = len(bayesian_objectives) if bayesian_objectives else 0
                offset_iters = [offset + iter_num for iter_num in iters]
                ax.plot(offset_iters, values_bfgs, 'r.-', alpha=0.7, label='bfgs')
            
            ax.set_title(f'{param_label} Evolution')
            ax.set_xlabel('Evaluation/Iteration')
            ax.set_ylabel(param_label)
            ax.legend()
            ax.grid(True)
        
        # Plot 5: Best spectrum vs target
        ax = axes[1, 1]
        all_evaluations = self.bayesian_history + self.bfgs_history
        if all_evaluations:
            best_eval = min(all_evaluations, key=lambda x: x['objective'])
            
            if 'spectrum' in best_eval and 'energy_values' in best_eval:
                generated_spectrum = best_eval['spectrum'].copy()
                energy_values = best_eval['energy_values']
                target_spectrum = self.target_spectrum.cpu().numpy()
                
                if self.normalize_spectrum:
                    gen_min = generated_spectrum.min()
                    gen_max = generated_spectrum.max()
                    if gen_max > gen_min:
                        generated_spectrum_plot = (generated_spectrum - gen_min) / (gen_max - gen_min)
                    else:
                        generated_spectrum_plot = generated_spectrum
                    
                    # Also normalize target spectrum the same way
                    target_min = target_spectrum.min()
                    target_max = target_spectrum.max()
                    if target_max > target_min:
                        target_spectrum_plot = (target_spectrum - target_min) / (target_max - target_min)
                    else:
                        target_spectrum_plot = target_spectrum
                else:
                    generated_spectrum_plot = generated_spectrum
                    target_spectrum_plot = target_spectrum
                
                # Plot both normalized spectra
                ax.plot(self.target_energy_axis.cpu().numpy(), generated_spectrum_plot, 'g-', linewidth=2.5, label='Generated (best)', alpha=0.8)
                ax.plot(self.target_energy_axis.cpu().numpy(), target_spectrum_plot, 
                    'r--', linewidth=2.5, label='Target', alpha=0.8)
                
                spectrum_dist = best_eval['spectrum_distance']
                ax.set_title(f"Best Spectrum Match (MSE={spectrum_dist:.6f})", fontweight='bold')
                ax.set_xlabel('Energy (MeV)', fontsize=10)
                ax.set_ylabel('Normalized Intensity', fontsize=10)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Set reasonable axis limits
                ax.set_xlim([energy_values.min(), energy_values.max()])
                # ax.set_ylim([-0.05, 1.1])
            else:
                ax.text(0.5, 0.5, 'No spectrum data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Best Spectrum Match')
        else:
            ax.text(0.5, 0.5, 'No evaluation data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Best Spectrum Match')
        
        # Plot 6: Parameter space exploration
        ax = axes[1, 2]
        if bayesian_objectives:
            laser_energies = [call['laser_energy'] for call in self.bayesian_history]
            pressures = [call['pressure'] for call in self.bayesian_history]
            acquisition_times = [call['acquisition_time'] for call in self.bayesian_history]
            scatter = ax.scatter(laser_energies, pressures, c=acquisition_times, 
                               cmap='Blues', alpha=0.7, s=50, label='Bayesian')
            
        if bfgs_objectives:
            laser_energies_bfgs = [call['laser_energy'] for call in self.bfgs_history]
            pressures_bfgs = [call['pressure'] for call in self.bfgs_history]
            acquisition_times_bfgs = [call['acquisition_time'] for call in self.bfgs_history]
            scatter2 = ax.scatter(laser_energies_bfgs, pressures_bfgs, c=acquisition_times_bfgs,
                                cmap='Reds', alpha=0.7, s=50, marker='^', label='bfgs')
        
        ax.set_title('Parameter Space Exploration')
        ax.set_xlabel('Laser Energy')
        ax.set_ylabel('Pressure')
        ax.legend()
        ax.grid(True)
        
        if bayesian_objectives or bfgs_objectives:
            cbar = plt.colorbar(scatter if bayesian_objectives else scatter2, ax=ax)
            cbar.set_label('Acquisition Time (ms)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'spectrum_matching_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.close()

def run_multi_seed_optimization():
    """Run spectrum matching optimization with multiple seeds and compare results."""
    model_path = "models/edm_4kepochs/ema_ckpt_final.pt"
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    
    seeds = [67, 156, 235, 391, 429, 504, 742, 782, 823, 918]
    
    # Create datetime-based folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"match_spectrum_{timestamp}"
    
    print("="*80)
    print("MULTI-SEED SPECTRUM MATCHING OPTIMIZATION")
    print("="*80)
    print(f"Running optimization with {len(seeds)} different seeds")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {base_output_dir}")
    print("="*80)
    
    # Common optimization parameters
    opt_params = {
        'model_path': model_path,
        'device': device,
        'target_spectrum_csv': 'avg_spectrum_45_25_20.csv',
        'pressure_bounds': (1.0, 50.0),
        'laser_energy_bounds': (5.0, 50.0),
        'acquisition_time_bounds': (5.0, 100.0),
        'bayesian_n_calls': 100,
        'bayesian_n_initial_points': 10,
        'finetune_max_iter': 50,
        'finetune_lr': 2.0,
        'finetune_optimizer': 'bfgs',  # Options: 'bfgs' or 'adam'
        'batch_size': 16,
        'num_sampling_steps': 18,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'rho': 7,
        'cfg_scale': 3.0,
        'smooth_output': True,
        'smooth_kernel_size': 9,
        'smooth_sigma': 2.0,
        'normalize_spectrum': False,
        'seed': None
    }
    
    # Save opt_params to JSON file in base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    opt_params_file = os.path.join(base_output_dir, "optimization_parameters.json")
    with open(opt_params_file, 'w') as f:
        # Convert to serializable format
        opt_params_serializable = {k: v if not isinstance(v, tuple) else list(v) 
                                   for k, v in opt_params.items()}
        json.dump(opt_params_serializable, f, indent=4)
    print(f"Optimization parameters saved to: {opt_params_file}")
    
    all_results = []
    all_optimizers = []
    
    # Run optimization for each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} RUN {i+1}/{len(seeds)} (Seed: {seed}) {'='*20}")
        
        set_seed(seed)
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        opt_params['output_dir'] = seed_output_dir
        opt_params['seed'] = seed
        
        # Set up logging for this seed
        logger = setup_logging(seed_output_dir, seed)
        opt_params['logger'] = logger
        
        optimizer = SpectrumMatchingOptimizer(**opt_params)
        results = optimizer.optimize()
        optimizer.plot_results()
        
        results['seed'] = seed
        results['run_id'] = i
        all_results.append(results)
        all_optimizers.append(optimizer)
        
        spectrum_dist = results['best_spectrum_dist']
        print(f"Run {i+1} complete - Best spectrum distance (MSE): {spectrum_dist:.6f}")
        print(f"Best params: [{results['best_params'][0]:.3f}, {results['best_params'][1]:.3f}, {results['best_params'][2]:.3f}]")
        
        # Close logger handlers to avoid file lock issues
        if logger:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    # Analyze results
    print("\n" + "="*80)
    print("MULTI-SEED ANALYSIS RESULTS")
    print("="*80)
    
    spectrum_dists = [r['best_spectrum_dist'] for r in all_results]
    laser_energies = [r['best_params'][0] for r in all_results]
    pressures = [r['best_params'][1] for r in all_results]
    acquisition_times = [r['best_params'][2] for r in all_results]
    
    print(f"🎯 SPECTRUM MATCHING STATISTICS:")
    print(f"  Mean spectrum distance (MSE): {np.mean(spectrum_dists):.6f}")
    print(f"  Std spectrum distance:        {np.std(spectrum_dists):.6f}")
    print(f"  Best match (lowest MSE):      {np.min(spectrum_dists):.6f} (Seed: {seeds[np.argmin(spectrum_dists)]})")
    print(f"  Worst match (highest MSE):    {np.max(spectrum_dists):.6f} (Seed: {seeds[np.argmax(spectrum_dists)]})")
    
    print(f"\n🎯 PARAMETER CONVERGENCE STATISTICS:")
    print(f"  Laser Energy:     {np.mean(laser_energies):.3f} ± {np.std(laser_energies):.3f}")
    print(f"  Pressure:         {np.mean(pressures):.3f} ± {np.std(pressures):.3f}")
    print(f"  Acquisition Time: {np.mean(acquisition_times):.3f} ± {np.std(acquisition_times):.3f}")
    
    # Create comprehensive plots
    plot_multi_seed_comparison(all_results, all_optimizers, seeds, base_output_dir)
    
    return all_results, all_optimizers

def plot_multi_seed_comparison(all_results, all_optimizers, seeds, base_output_dir):
    """Create comprehensive plots comparing multi-seed optimization results."""
    
    fig = plt.figure(figsize=(18, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Plot 1: Convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    for i, (result, optimizer) in enumerate(zip(all_results, all_optimizers)):
        seed = seeds[i]
        color = colors[i]
        
        if optimizer.bayesian_history:
            bayesian_objectives = [call['objective'] for call in optimizer.bayesian_history]
            bayesian_best_obj = np.minimum.accumulate(bayesian_objectives)
            ax1.plot(range(len(bayesian_objectives)), bayesian_best_obj, 
                    color=color, alpha=0.7, linewidth=1.5, label=f'Seed {seed}')
        
        if optimizer.bfgs_history:
            bfgs_objectives = [call['objective'] for call in optimizer.bfgs_history]
            offset = len(bayesian_objectives) if optimizer.bayesian_history else 0
            combined_evals = [offset + i for i in range(len(bfgs_objectives))]
            ax1.plot(combined_evals, bfgs_objectives, '--', color=color, alpha=0.7, linewidth=1.5)
    
    ax1.set_title('Convergence Comparison (Spectrum Matching)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Evaluation/Iteration')
    ax1.set_ylabel('Spectrum Distance (MSE)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Final spectrum distance by seed
    ax2 = plt.subplot(2, 3, 2)
    spectrum_dists = [r['best_spectrum_dist'] for r in all_results]
    bars = ax2.bar(range(len(seeds)), spectrum_dists, color=colors, alpha=0.7)
    ax2.set_title('Spectrum Distance (MSE) by Seed', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Run (Seed)')
    ax2.set_ylabel('Spectrum Distance (MSE)')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, spectrum_dists)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spectrum_dists)*0.01,
                f'{val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3-5: Parameter evolution comparison
    param_names = ['laser_energy', 'pressure', 'acquisition_time']
    param_labels = ['Laser Energy', 'Pressure', 'Acquisition Time (ms)']
    plot_positions = [(0, 2), (1, 0), (1, 1)]
    
    for param_name, param_label, (row, col) in zip(param_names, param_labels, plot_positions):
        ax = plt.subplot(2, 3, row * 3 + col + 1)
        
        for i, optimizer in enumerate(all_optimizers):
            color = colors[i]
            
            if optimizer.bayesian_history:
                values_bayes = [call[param_name] for call in optimizer.bayesian_history]
                ax.plot(range(len(values_bayes)), values_bayes, color=color, alpha=0.6, linewidth=1)
            
            if optimizer.bfgs_history:
                values_bfgs = [call[param_name] for call in optimizer.bfgs_history]
                iters = [call['iteration'] for call in optimizer.bfgs_history]
                offset = len(values_bayes) if optimizer.bayesian_history else 0
                offset_iters = [offset + iter_num for iter_num in iters]
                ax.plot(offset_iters, values_bfgs, '--', color=color, alpha=0.6, linewidth=1)
        
        ax.set_title(f'{param_label} Evolution Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel(param_label)
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Spectrum distance variance
    ax6 = plt.subplot(2, 3, 6)
    bars = ax6.bar(range(len(seeds)), spectrum_dists, color=colors, alpha=0.7)
    ax6.set_title('Spectrum Distance (MSE) by Seed', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Run (Seed)')
    ax6.set_ylabel('Spectrum Distance (MSE)')
    ax6.set_xticks(range(len(seeds)))
    ax6.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    for bar, dist in zip(bars, spectrum_dists):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spectrum_dists)*0.01,
                f'{dist:.6f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    plots_dir = os.path.join(base_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "multi_seed_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Multi-seed comparison plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Run multi-seed spectrum matching optimization
    results, optimizers = run_multi_seed_optimization()

