"""
Hybrid Bayesian-LBFGS Optimization for LWFA Parameter Optimization

This module provides deterministic optimization with multi-seed analysis capabilities.

USAGE:
------
Single-seed run (maximize):
    python optimize_conditional.py
    # Or: main('single', mode='maximize')

Single-seed run (target):
    main('single', mode='target', target=150.0)

Single-seed run (match spectrum):
    main('single', mode='match_spectrum')

Multi-seed analysis (10 seeds, maximize):
    main('multi', mode='maximize')
    
Multi-seed analysis (10 seeds, target):
    main('multi', mode='target', target=150.0)

Multi-seed analysis (10 seeds, match spectrum):
    main('multi', mode='match_spectrum')

FEATURES:
---------
✓ Three optimization modes: 'maximize', 'target', and 'match_spectrum'
✓ Fully deterministic latent reuse for reproducible results
✓ Hybrid Bayesian + gradient-based optimization  
✓ Multi-seed variance analysis with comprehensive statistics
✓ 6-panel visualization comparing convergence across seeds

OPTIMIZATION MODES:
-------------------
- 'maximize': Maximize the weighted sum (default)
- 'target': Achieve a specific target weighted sum value
- 'match_spectrum': Match generated spectrum to target spectrum from avg_spectrum.csv

SEEDS USED: [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
from skopt import gp_minimize
from skopt.space import Real
import logging
import json
from datetime import datetime
import pandas as pd

# Import necessary modules from the project
from src.modules_1d import EDMPrecond
from src.diffusion import DifferentiableEdmSampler, transform_vector, gaussian_smooth_1d
from src.utils import deflection_biexp_calc, calc_spec

def setup_logging(output_dir, method_name, trial_num=None):
    """Set up detailed logging for optimization runs."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    if trial_num is not None:
        logger_name = f"{method_name}_trial_{trial_num}"
        log_filename = f"{method_name}_trial_{trial_num}.log"
    else:
        logger_name = method_name
        log_filename = f"{method_name}.log"
    
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
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Starting optimization: {method_name}")
    if trial_num is not None:
        logger.info(f"Trial number: {trial_num}")
    
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

def set_seed(seed=42):
    """Set random seed for reproducible results."""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN
    # torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # For multiprocessing
    # torch.use_deterministic_algorithms(True)


class HybridBayesianLBFGSOptimizer:
    """
    Hybrid optimization approach that combines:
    1. Bayesian optimization for efficient exploration of parameter space
    2. LBFGS gradient-based fine-tuning starting from the best Bayesian result
    
    Supports three optimization modes:
    - 'maximize': Maximize the weighted sum (default)
    - 'target': Achieve a specific target weighted sum value
    - 'match_spectrum': Match generated spectrum to target spectrum from CSV
    """
    def __init__(
        self,
        model_path,
        device="cuda",
        pressure_bounds=(5.0, 30.0),
        laser_energy_bounds=(10.0, 50.0),
        acquisition_time_bounds=(5.0, 50.0),
        # Optimization mode
        optimization_mode='maximize',  # 'maximize', 'target', or 'match_spectrum'
        target_weighted_sum=None,  # Required when mode='target'
        target_spectrum_csv='avg_spectrum.csv',  # CSV file for mode='match_spectrum'
        # Bayesian phase parameters
        bayesian_n_calls=50,
        bayesian_n_initial_points=10,
        # LBFGS fine-tuning phase parameters  
        finetune_max_iter=50,
        finetune_lr=0.1,
        # LBFGS-specific parameters
        lbfgs_history_size=10,
        lbfgs_line_search_fn='strong_wolfe',
        # Common parameters
        batch_size=4,
        output_dir="hybrid_optimization_results",
        spectrum_length=256,
        features=["E", "P", "ms"],
        # Sampler parameters
        num_sampling_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        cfg_scale=3.0,
        # Smoothing parameters
        smooth_output=True,
        smooth_kernel_size=5,
        smooth_sigma=2.0,
        # Post-processing smoothing
        post_smooth=True,
        post_smooth_window=3,
        seed=None
    ):
        self.device = device
        self.pressure_bounds = pressure_bounds
        self.laser_energy_bounds = laser_energy_bounds
        self.acquisition_time_bounds = acquisition_time_bounds
        
        # Optimization mode
        self.optimization_mode = optimization_mode
        if optimization_mode not in ['maximize', 'target', 'match_spectrum']:
            raise ValueError(f"optimization_mode must be 'maximize', 'target', or 'match_spectrum', got '{optimization_mode}'")
        if optimization_mode == 'target' and target_weighted_sum is None:
            raise ValueError("target_weighted_sum must be provided when optimization_mode='target'")
        self.target_weighted_sum = target_weighted_sum
        
        # Load target spectrum for match_spectrum mode
        self.target_spectrum = None
        self.target_energy_axis = None
        if optimization_mode == 'match_spectrum':
            self._load_target_spectrum(target_spectrum_csv)
        
        # Optimization parameters
        self.bayesian_n_calls = bayesian_n_calls
        self.bayesian_n_initial_points = bayesian_n_initial_points
        self.finetune_max_iter = finetune_max_iter
        self.finetune_lr = finetune_lr
        
        # LBFGS parameters
        self.lbfgs_history_size = lbfgs_history_size
        self.lbfgs_line_search_fn = lbfgs_line_search_fn
        
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.spectrum_length = spectrum_length
        self.features = features
        
        # Sampler parameters
        self.num_sampling_steps = num_sampling_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.cfg_scale = cfg_scale
        
        # Smoothing parameters
        self.smooth_output = smooth_output
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        self.post_smooth = post_smooth
        self.post_smooth_window = post_smooth_window
        
        self.seed = seed
        
        if seed is not None:
            set_seed(seed)
        
        # Random starting parameters
        self.laser_energy = np.random.uniform(laser_energy_bounds[0], laser_energy_bounds[1])
        self.pressure = np.random.uniform(pressure_bounds[0], pressure_bounds[1])
        self.acquisition_time_ms = np.random.uniform(acquisition_time_bounds[0], acquisition_time_bounds[1])
        
        print(f"Hybrid Bayesian-LBFGS Optimization Setup:")
        mode_info = f"  Mode: {optimization_mode}"
        if optimization_mode == 'target':
            mode_info += f" (target={target_weighted_sum})"
        elif optimization_mode == 'match_spectrum':
            mode_info += f" (matching spectrum from CSV)"
        print(mode_info)
        print(f"  Phase 1 - Bayesian: {bayesian_n_calls} calls, {bayesian_n_initial_points} initial")
        print(f"  Phase 2 - LBFGS: max {finetune_max_iter} iter, lr={finetune_lr}")
        print(f"  Sampler: {self.num_sampling_steps} steps, σ∈[{self.sigma_min}, {self.sigma_max}], ρ={self.rho}")
        print(f"  Smoothing: output={self.smooth_output} (k={self.smooth_kernel_size}, σ={self.smooth_sigma}), post={self.post_smooth} (w={self.post_smooth_window})")
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
        
        # Initialize differentiable sampler (used for both phases)
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
        self.lbfgs_history = []
        self.logger = None

    def load_model(self, model_path):
        """Load the pre-trained EDM model"""
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"EDM model loaded from {model_path}")

    def _load_target_spectrum(self, csv_path):
        """Load target spectrum from CSV file for match_spectrum mode."""
        print(f"Loading target spectrum from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Expected columns: energy_MeV, intensity
        if 'energy_MeV' not in df.columns or 'intensity' not in df.columns:
            raise ValueError(f"CSV must contain 'energy_MeV' and 'intensity' columns")
        
        # Store as tensors on device
        self.target_energy_axis = torch.tensor(df['energy_MeV'].values, dtype=torch.float32, device=self.device)
        target_spectrum_raw = torch.tensor(df['intensity'].values, dtype=torch.float32, device=self.device)
        
        # Normalize target spectrum to [0, 1] range (min-max normalization)
        target_min = target_spectrum_raw.min()
        target_max = target_spectrum_raw.max()
        
        if target_max > target_min:
            self.target_spectrum = (target_spectrum_raw - target_min) / (target_max - target_min)
            print(f"  Loaded and normalized target spectrum: {len(self.target_spectrum)} points")
        else:
            raise ValueError("Target spectrum has no variation (min == max)")
        
        print(f"  Energy range: [{self.target_energy_axis.min():.2f}, {self.target_energy_axis.max():.2f}] MeV")
        print(f"  Original intensity range: [{target_spectrum_raw.min():.6f}, {target_spectrum_raw.max():.6f}]")
        print(f"  Normalized intensity range: [{self.target_spectrum.min():.6f}, {self.target_spectrum.max():.6f}]")

    def calculate_weighted_sum(self, spectrum, energy_values):
        """Calculate the weighted sum of intensities multiplied by MeV values."""
        return torch.sum(spectrum.to(self.device) * energy_values.to(self.device))
    
    def calculate_spectrum_distance(self, spectrum):
        """Calculate distance between generated spectrum and target spectrum.
        
        Uses Mean Squared Error (MSE) as the distance metric.
        Both spectra are normalized to [0, 1] range before comparison.
        """
        if self.target_spectrum is None:
            raise ValueError("Target spectrum not loaded")
        
        # Ensure both spectra have the same length
        if len(spectrum) != len(self.target_spectrum):
            raise ValueError(f"Spectrum length mismatch: generated={len(spectrum)}, target={len(self.target_spectrum)}")
        
        # Normalize generated spectrum to [0, 1] range (min-max normalization)
        # Use operations that preserve gradients
        spectrum_min = torch.min(spectrum)
        spectrum_max = torch.max(spectrum)
        
        # Add small epsilon to avoid division by zero and maintain gradients
        eps = 1e-8
        spectrum_normalized = (spectrum - spectrum_min) / (spectrum_max - spectrum_min + eps)
        
        # Calculate MSE between normalized spectra
        mse = torch.mean((spectrum_normalized - self.target_spectrum) ** 2)
        return mse
    
    def apply_post_smoothing(self, spectrum):
        """Apply Gaussian moving window averaging for post-processing smoothing."""
        if not self.post_smooth or self.post_smooth_window <= 1:
            return spectrum
        
        # Create Gaussian kernel for moving average
        window = self.post_smooth_window
        sigma = window / 3.0  # Rule of thumb: 3-sigma covers ~99.7% of distribution
        
        # Create 1D Gaussian kernel
        x = torch.arange(window, dtype=torch.float32, device=spectrum.device) - window // 2
        gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # Apply convolution (padding to maintain size)
        spectrum_padded = torch.nn.functional.pad(spectrum.unsqueeze(0).unsqueeze(0), 
                                                 (window//2, window//2), mode='reflect')
        smoothed = torch.nn.functional.conv1d(spectrum_padded, 
                                            gaussian_kernel.view(1, 1, -1), 
                                            padding=0)
        
        return smoothed.squeeze(0).squeeze(0)

    def compute_objective_value(self, weighted_sum_value=None, spectrum_distance=None):
        """Compute objective value based on optimization mode.
        
        Args:
            weighted_sum_value: The calculated weighted sum (used for 'maximize' and 'target' modes)
            spectrum_distance: The spectrum distance (used for 'match_spectrum' mode)
            
        Returns:
            Objective value to minimize (for both Bayesian and LBFGS phases)
        """
        if self.optimization_mode == 'maximize':
            # Maximize weighted sum = minimize negative weighted sum
            if weighted_sum_value is None:
                raise ValueError("weighted_sum_value required for 'maximize' mode")
            return -weighted_sum_value
        elif self.optimization_mode == 'target':
            # Achieve target = minimize absolute difference
            if weighted_sum_value is None:
                raise ValueError("weighted_sum_value required for 'target' mode")
            return abs(weighted_sum_value - self.target_weighted_sum)
        elif self.optimization_mode == 'match_spectrum':
            # Match spectrum = minimize distance
            if spectrum_distance is None:
                raise ValueError("spectrum_distance required for 'match_spectrum' mode")
            return spectrum_distance
        else:
            raise ValueError(f"Unknown optimization mode: {self.optimization_mode}")

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
            
            # Apply post-processing smoothing
            spectrum_intensity = self.apply_post_smoothing(spectrum_intensity)
            
            # Calculate metrics based on mode
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            
            if self.optimization_mode == 'match_spectrum':
                spectrum_distance = self.calculate_spectrum_distance(spectrum_intensity)
                objective = self.compute_objective_value(spectrum_distance=spectrum_distance.item())
                
                # Store history
                self.bayesian_history.append({
                    'laser_energy': laser_energy,
                    'pressure': pressure,
                    'acquisition_time': acquisition_time,
                    'weighted_sum': weighted_sum.item(),
                    'spectrum_distance': spectrum_distance.item(),
                    'objective': objective,
                    'spectrum': spectrum_intensity.detach().cpu().numpy(),
                    'energy_values': deflection_MeV.detach().cpu().numpy()
                })
                
                mode_str = f"spectrum_dist={spectrum_distance.item():.6f}"
            else:
                objective = self.compute_objective_value(weighted_sum_value=weighted_sum.item())
                
                # Store history
                self.bayesian_history.append({
                    'laser_energy': laser_energy,
                    'pressure': pressure,
                    'acquisition_time': acquisition_time,
                    'weighted_sum': weighted_sum.item(),
                    'objective': objective,
                    'spectrum': spectrum_intensity.detach().cpu().numpy(),
                    'energy_values': deflection_MeV.detach().cpu().numpy()
                })
                
                mode_str = f"target_diff={abs(weighted_sum.item() - self.target_weighted_sum):.4f}" if self.optimization_mode == 'target' else f"weighted_sum={weighted_sum.item():.4f}"
            
            print(f"Bayesian Eval {len(self.bayesian_history)}: "
                  f"[{laser_energy:.2f}, {pressure:.2f}, {acquisition_time:.2f}] -> {mode_str}")
            
            return objective
            
        except Exception as e:
            print(f"Error in Bayesian objective: {e}")
            return 1e6

    def finetune_objective_and_grad(self, params_tensor):
        """Objective function with gradients for fine-tuning optimization phase."""
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
        
        # Apply post-processing smoothing
        spectrum_intensity = self.apply_post_smoothing(spectrum_intensity)
        
        # Compute objective based on mode
        if self.optimization_mode == 'maximize':
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            objective = -weighted_sum  # Maximize = minimize negative
        elif self.optimization_mode == 'target':
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            # Minimize squared difference (differentiable version of absolute difference)
            objective = (weighted_sum - self.target_weighted_sum) ** 2
        elif self.optimization_mode == 'match_spectrum':
            # Minimize MSE between spectra
            spectrum_distance = self.calculate_spectrum_distance(spectrum_intensity)
            objective = spectrum_distance
        
        return objective

    def run_bayesian_phase(self):
        """Run the Bayesian optimization exploration phase."""
        if self.optimization_mode == 'target':
            mode_str = f"target={self.target_weighted_sum}"
        elif self.optimization_mode == 'match_spectrum':
            mode_str = "match_spectrum"
        else:
            mode_str = "maximize"
        print(f"Phase 1: Bayesian Optimization ({self.bayesian_n_calls} evaluations, mode={mode_str})")
        
        x0 = [self.laser_energy, self.pressure, self.acquisition_time_ms]
        
        result = gp_minimize(
            func=self.bayesian_objective_function,
            dimensions=self.dimensions,
            n_calls=self.bayesian_n_calls,
            n_initial_points=self.bayesian_n_initial_points,
            x0=[x0],
            acq_func='EI',
            random_state=self.seed if self.seed else 42
        )
        
        best_objective = result.fun
        best_params = result.x
        
        # Get best result from history
        best_call = min(self.bayesian_history, key=lambda x: x['objective'])
        best_weighted_sum = best_call['weighted_sum']
        
        if self.optimization_mode == 'maximize':
            print(f"  Best: [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}] -> weighted_sum={best_weighted_sum:.4f}")
        elif self.optimization_mode == 'target':
            target_diff = abs(best_weighted_sum - self.target_weighted_sum)
            print(f"  Best: [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}] -> weighted_sum={best_weighted_sum:.4f} (target_diff={target_diff:.4f})")
        elif self.optimization_mode == 'match_spectrum':
            best_spectrum_dist = best_call.get('spectrum_distance', 0.0)
            print(f"  Best: [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}] -> spectrum_dist={best_spectrum_dist:.6f}")
        
        return {
            'best_params': best_params, 
            'best_objective': best_objective,
            'best_weighted_sum': best_weighted_sum,
            'result': result
        }

    def run_finetune_phase(self, initial_params):
        """Run the LBFGS fine-tuning phase."""
        if self.optimization_mode == 'target':
            mode_str = f"target={self.target_weighted_sum}"
        elif self.optimization_mode == 'match_spectrum':
            mode_str = "match_spectrum"
        else:
            mode_str = "maximize"
        print(f"Phase 2: LBFGS Fine-tuning ({self.finetune_max_iter} max iterations, mode={mode_str})")
        
        # Initialize parameters with gradients
        laser_energy = torch.tensor(initial_params[0], device=self.device, requires_grad=True)
        pressure = torch.tensor(initial_params[1], device=self.device, requires_grad=True)
        acquisition_time = torch.tensor(initial_params[2], device=self.device, requires_grad=True)
        
        # Create LBFGS optimizer
        optimizer = optim.LBFGS(
            [laser_energy, pressure, acquisition_time], 
            lr=self.finetune_lr, 
            max_iter=self.finetune_max_iter,
            history_size=self.lbfgs_history_size,
            line_search_fn=self.lbfgs_line_search_fn
        )
        
        best_objective = float('inf')
        best_weighted_sum = None
        best_spectrum_dist = None
        iteration_count = 0
        
        # LBFGS requires closure function
        def closure():
            nonlocal iteration_count, best_objective, best_weighted_sum, best_spectrum_dist
            
            optimizer.zero_grad()
            
            loss = self.finetune_objective_and_grad(torch.stack([laser_energy, pressure, acquisition_time]))
            loss.backward()
            
            # Calculate metrics for logging
            with torch.no_grad():
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
                spectrum_intensity = self.apply_post_smoothing(spectrum_intensity)
                weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV).item()
                
                if self.optimization_mode == 'match_spectrum':
                    spectrum_dist = self.calculate_spectrum_distance(spectrum_intensity).item()
                else:
                    spectrum_dist = None
            
            current_objective = loss.item()
            if current_objective < best_objective:
                best_objective = current_objective
                best_weighted_sum = weighted_sum
                if self.optimization_mode == 'match_spectrum':
                    best_spectrum_dist = spectrum_dist
            
            # Store history
            history_entry = {
                'iteration': iteration_count,
                'laser_energy': laser_energy.item(),
                'pressure': pressure.item(),
                'acquisition_time': acquisition_time.item(),
                'weighted_sum': weighted_sum,
                'objective': current_objective
            }
            if self.optimization_mode == 'match_spectrum':
                history_entry['spectrum_distance'] = spectrum_dist
            self.lbfgs_history.append(history_entry)
            
            if iteration_count % 5 == 0:
                if self.optimization_mode == 'maximize':
                    print(f"  LBFGS Iter {iteration_count}: [{laser_energy.item():.3f}, {pressure.item():.3f}, {acquisition_time.item():.3f}] -> weighted_sum={weighted_sum:.4f}")
                elif self.optimization_mode == 'target':
                    target_diff = abs(weighted_sum - self.target_weighted_sum)
                    print(f"  LBFGS Iter {iteration_count}: [{laser_energy.item():.3f}, {pressure.item():.3f}, {acquisition_time.item():.3f}] -> weighted_sum={weighted_sum:.4f} (target_diff={target_diff:.4f})")
                elif self.optimization_mode == 'match_spectrum':
                    print(f"  LBFGS Iter {iteration_count}: [{laser_energy.item():.3f}, {pressure.item():.3f}, {acquisition_time.item():.3f}] -> spectrum_dist={spectrum_dist:.6f}")
            
            iteration_count += 1
            return loss
        
        optimizer.step(closure)
        
        final_params = [laser_energy.item(), pressure.item(), acquisition_time.item()]
        if self.optimization_mode == 'maximize':
            print(f"  Final: [{final_params[0]:.3f}, {final_params[1]:.3f}, {final_params[2]:.3f}] -> weighted_sum={best_weighted_sum:.4f}")
        elif self.optimization_mode == 'target':
            target_diff = abs(best_weighted_sum - self.target_weighted_sum)
            print(f"  Final: [{final_params[0]:.3f}, {final_params[1]:.3f}, {final_params[2]:.3f}] -> weighted_sum={best_weighted_sum:.4f} (target_diff={target_diff:.4f})")
        elif self.optimization_mode == 'match_spectrum':
            print(f"  Final: [{final_params[0]:.3f}, {final_params[1]:.3f}, {final_params[2]:.3f}] -> spectrum_dist={best_spectrum_dist:.6f}")
        
        result = {
            'best_params': final_params, 
            'best_objective': best_objective,
            'best_weighted_sum': best_weighted_sum,
            'iterations': iteration_count
        }
        if self.optimization_mode == 'match_spectrum':
            result['best_spectrum_dist'] = best_spectrum_dist
        
        return result

    def optimize(self):
        """Run the complete hybrid optimization."""
        if self.logger is None:
            if self.optimization_mode == 'target':
                mode_suffix = f"_target_{self.target_weighted_sum}"
            elif self.optimization_mode == 'match_spectrum':
                mode_suffix = "_match_spectrum"
            else:
                mode_suffix = "_maximize"
            self.logger = setup_logging(self.output_dir, f"Hybrid_Bayesian_LBFGS{mode_suffix}")
        
        mode_str = f"mode={self.optimization_mode}"
        if self.optimization_mode == 'target':
            mode_str += f", target={self.target_weighted_sum}"
        print(f"Starting Hybrid Bayesian-LBFGS Optimization ({mode_str})")
        
        # Phase 1: Bayesian exploration
        bayesian_result = self.run_bayesian_phase()
        
        # Phase 2: Fine-tuning
        finetune_result = self.run_finetune_phase(bayesian_result['best_params'])
        
        # Determine overall best based on objective
        if finetune_result['best_objective'] < bayesian_result['best_objective']:
            overall_best_weighted_sum = finetune_result['best_weighted_sum']
            overall_best_params = finetune_result['best_params']
            overall_best_objective = finetune_result['best_objective']
            overall_best_spectrum_dist = finetune_result.get('best_spectrum_dist', None)
            best_phase = 'LBFGS'
        else:
            overall_best_weighted_sum = bayesian_result['best_weighted_sum']
            overall_best_params = bayesian_result['best_params']
            overall_best_objective = bayesian_result['best_objective']
            # Get spectrum_dist from history for match_spectrum mode
            if self.optimization_mode == 'match_spectrum':
                best_call = min(self.bayesian_history, key=lambda x: x['objective'])
                overall_best_spectrum_dist = best_call.get('spectrum_distance', None)
            else:
                overall_best_spectrum_dist = None
            best_phase = 'Bayesian'
        
        print(f"\nOptimization Complete!")
        print(f"  Best result from {best_phase} phase")
        print(f"  Final params: [{overall_best_params[0]:.3f}, {overall_best_params[1]:.3f}, {overall_best_params[2]:.3f}]")
        
        if self.optimization_mode == 'maximize':
            print(f"  Best weighted sum: {overall_best_weighted_sum:.6f}")
            # Calculate improvement
            if finetune_result['best_objective'] < bayesian_result['best_objective']:
                improvement = ((finetune_result['best_weighted_sum'] - bayesian_result['best_weighted_sum']) / bayesian_result['best_weighted_sum']) * 100
                print(f"  LBFGS improved by: {improvement:.2f}%")
        elif self.optimization_mode == 'target':
            target_diff = abs(overall_best_weighted_sum - self.target_weighted_sum)
            print(f"  Best weighted sum: {overall_best_weighted_sum:.6f}")
            print(f"  Target difference: {target_diff:.6f}")
            print(f"  Target achievement: {(1 - target_diff/self.target_weighted_sum)*100:.2f}%")
        elif self.optimization_mode == 'match_spectrum':
            print(f"  Best spectrum distance (MSE): {overall_best_spectrum_dist:.6f}")
            # Calculate improvement
            if finetune_result['best_objective'] < bayesian_result['best_objective']:
                bayesian_best_call = min(self.bayesian_history, key=lambda x: x['objective'])
                bayesian_best_dist = bayesian_best_call.get('spectrum_distance', float('inf'))
                if bayesian_best_dist > 0:
                    improvement = ((bayesian_best_dist - overall_best_spectrum_dist) / bayesian_best_dist) * 100
                    print(f"  LBFGS improved by: {improvement:.2f}%")
        
        result = {
            'best_weighted_sum': overall_best_weighted_sum,
            'best_params': overall_best_params,
            'best_objective': overall_best_objective,
            'best_phase': best_phase,
            'bayesian_phase': bayesian_result,
            'finetune_phase': finetune_result,
            'bayesian_history': self.bayesian_history,
            'lbfgs_history': self.lbfgs_history
        }
        if self.optimization_mode == 'match_spectrum':
            result['best_spectrum_dist'] = overall_best_spectrum_dist
        
        return result

    def _track_best_params(self, history, param_name, starting_best_objective=float('inf'), starting_best_params=None):
        """Helper to track best-so-far parameter values based on objective."""
        best_params_list = []
        best_objective = starting_best_objective
        best_params = starting_best_params
        
        for call in history:
            if call['objective'] < best_objective:
                best_objective = call['objective']
                best_params = call
            best_params_list.append(best_params[param_name] if best_params else call[param_name])
        
        return best_params_list
    
    def plot_results(self):
        """Create visualization of the hybrid optimization results."""
        if not self.bayesian_history and not self.lbfgs_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Determine what to plot based on mode
        if self.optimization_mode == 'maximize':
            ylabel_convergence = 'Weighted Sum'
            title_suffix = '(Maximization)'
        elif self.optimization_mode == 'target':
            ylabel_convergence = f'Distance to Target ({self.target_weighted_sum})'
            title_suffix = f'(Target Seeking)'
        elif self.optimization_mode == 'match_spectrum':
            ylabel_convergence = 'Spectrum Distance (MSE)'
            title_suffix = '(Spectrum Matching)'
        
        # Extract data
        bayesian_scores = [call['weighted_sum'] for call in self.bayesian_history]
        bayesian_objectives = [call['objective'] for call in self.bayesian_history]
        lbfgs_scores = [call['weighted_sum'] for call in self.lbfgs_history] if self.lbfgs_history else []
        lbfgs_objectives = [call['objective'] for call in self.lbfgs_history] if self.lbfgs_history else []
        
        # Plot 1: Convergence - Best So Far
        ax = axes[0, 0]
        if bayesian_objectives:
            bayesian_best_obj = np.minimum.accumulate(bayesian_objectives)
            if self.optimization_mode == 'maximize':
                # For maximize mode, show weighted sum (negative of objective)
                bayesian_plot_values = -np.array(bayesian_best_obj)
            elif self.optimization_mode == 'target':
                # For target mode, show distance to target (objective itself)
                bayesian_plot_values = bayesian_best_obj
            elif self.optimization_mode == 'match_spectrum':
                # For match_spectrum mode, show spectrum distance (objective itself)
                bayesian_plot_values = bayesian_best_obj
            
            ax.plot(range(len(bayesian_objectives)), bayesian_plot_values, 'b-', linewidth=2, label='Bayesian Phase (Best So Far)')
        
        if lbfgs_objectives:
            # Calculate best so far for LBFGS phase, continuing from Bayesian best
            starting_best_obj = bayesian_best_obj[-1] if bayesian_objectives else float('inf')
            lbfgs_best_so_far = []
            current_best = starting_best_obj
            for obj in lbfgs_objectives:
                current_best = min(current_best, obj)
                lbfgs_best_so_far.append(current_best)
            
            if self.optimization_mode == 'maximize':
                lbfgs_plot_values = -np.array(lbfgs_best_so_far)
            elif self.optimization_mode == 'target':
                lbfgs_plot_values = lbfgs_best_so_far
            elif self.optimization_mode == 'match_spectrum':
                lbfgs_plot_values = lbfgs_best_so_far
            
            offset = len(bayesian_objectives)
            combined_evals = [offset + i for i in range(len(lbfgs_objectives))]
            ax.plot(combined_evals, lbfgs_plot_values, 'r-', linewidth=2, label='LBFGS Phase (Best So Far)')
        
        ax.set_title(f'Hybrid Optimization Convergence {title_suffix}')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel(ylabel_convergence)
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Parameter evolution - Laser Energy (Best So Far)
        ax = axes[0, 1]
        if bayesian_objectives:
            best_laser_energies = self._track_best_params(self.bayesian_history, 'laser_energy')
            ax.plot(range(len(best_laser_energies)), best_laser_energies, 'b.-', alpha=0.7, label='Bayesian (Best So Far)')
        
        if lbfgs_objectives:
            starting_best_obj = bayesian_best_obj[-1] if bayesian_objectives else float('inf')
            starting_best_params = min(self.bayesian_history, key=lambda x: x['objective']) if bayesian_objectives else None
            best_laser_energies_lbfgs = self._track_best_params(self.lbfgs_history, 'laser_energy', starting_best_obj, starting_best_params)
            
            iters = [call['iteration'] for call in self.lbfgs_history]
            offset = len(bayesian_objectives) if bayesian_objectives else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, best_laser_energies_lbfgs, 'r.-', alpha=0.7, label='LBFGS (Best So Far)')
        
        ax.set_title('Laser Energy Evolution')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Laser Energy')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Parameter evolution - Pressure (Best So Far)
        ax = axes[0, 2]
        if bayesian_objectives:
            best_pressures = self._track_best_params(self.bayesian_history, 'pressure')
            ax.plot(range(len(best_pressures)), best_pressures, 'b.-', alpha=0.7, label='Bayesian (Best So Far)')
        
        if lbfgs_objectives:
            starting_best_obj = bayesian_best_obj[-1] if bayesian_objectives else float('inf')
            starting_best_params = min(self.bayesian_history, key=lambda x: x['objective']) if bayesian_objectives else None
            best_pressures_lbfgs = self._track_best_params(self.lbfgs_history, 'pressure', starting_best_obj, starting_best_params)
            
            iters = [call['iteration'] for call in self.lbfgs_history]
            offset = len(bayesian_objectives) if bayesian_objectives else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, best_pressures_lbfgs, 'r.-', alpha=0.7, label='LBFGS (Best So Far)')
        
        ax.set_title('Pressure Evolution')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Pressure')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Parameter evolution - Acquisition Time
        ax = axes[1, 0]
        if bayesian_objectives:
            acquisition_times = [call['acquisition_time'] for call in self.bayesian_history]
            ax.plot(range(len(acquisition_times)), acquisition_times, 'b.-', alpha=0.7, label='Bayesian')
        
        if lbfgs_objectives:
            acquisition_times_lbfgs = [call['acquisition_time'] for call in self.lbfgs_history]
            iters = [call['iteration'] for call in self.lbfgs_history]
            offset = len(bayesian_objectives) if bayesian_objectives else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, acquisition_times_lbfgs, 'r.-', alpha=0.7, label='LBFGS')
        
        ax.set_title('Acquisition Time Evolution')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Acquisition Time (ms)')
        ax.legend()
        ax.grid(True)
        
        # Plot 5: Best spectrum
        ax = axes[1, 1]
        all_evaluations = self.bayesian_history + self.lbfgs_history
        if all_evaluations:
            best_eval = min(all_evaluations, key=lambda x: x['objective'])
            
            if 'spectrum' in best_eval and 'energy_values' in best_eval:
                generated_spectrum = best_eval['spectrum']
                
                # Normalize for display in match_spectrum mode to [0, 1] range
                if self.optimization_mode == 'match_spectrum':
                    spectrum_min = generated_spectrum.min()
                    spectrum_max = generated_spectrum.max()
                    if spectrum_max > spectrum_min:
                        generated_spectrum = (generated_spectrum - spectrum_min) / (spectrum_max - spectrum_min)
                
                ax.plot(best_eval['energy_values'], generated_spectrum, 'g-', linewidth=2, label='Generated')
                
                # Plot target spectrum if in match_spectrum mode
                if self.optimization_mode == 'match_spectrum' and self.target_spectrum is not None:
                    ax.plot(self.target_energy_axis.cpu().numpy(), self.target_spectrum.cpu().numpy(), 
                           'r--', linewidth=2, label='Target')
                    ax.legend()
                
                title_text = f"Best Spectrum"
                if self.optimization_mode == 'target':
                    title_text += f" (weighted_sum={best_eval['weighted_sum']:.2f}, target={self.target_weighted_sum:.2f})"
                elif self.optimization_mode == 'match_spectrum':
                    spectrum_dist = best_eval.get('spectrum_distance', 'N/A')
                    title_text += f" (Normalized MSE={spectrum_dist:.6f})" if isinstance(spectrum_dist, float) else f" (MSE={spectrum_dist})"
                else:
                    title_text += f" (weighted_sum={best_eval['weighted_sum']:.2f})"
                ax.set_title(title_text)
                ax.set_xlabel('Energy (MeV)')
                ylabel = 'Normalized Intensity' if self.optimization_mode == 'match_spectrum' else 'Intensity'
                ax.set_ylabel(ylabel)
                ax.grid(True)
        
        # Plot 6: Parameter trajectory in 2D parameter space  
        ax = axes[1, 2]
        if bayesian_objectives:
            laser_energies = [call['laser_energy'] for call in self.bayesian_history]
            pressures = [call['pressure'] for call in self.bayesian_history]
            acquisition_times = [call['acquisition_time'] for call in self.bayesian_history]
            scatter = ax.scatter(laser_energies, pressures, c=acquisition_times, 
                               cmap='Blues', alpha=0.7, s=50, label='Bayesian')
            
        if lbfgs_objectives:
            laser_energies_lbfgs = [call['laser_energy'] for call in self.lbfgs_history]
            pressures_lbfgs = [call['pressure'] for call in self.lbfgs_history]
            acquisition_times_lbfgs = [call['acquisition_time'] for call in self.lbfgs_history]
            scatter2 = ax.scatter(laser_energies_lbfgs, pressures_lbfgs, c=acquisition_times_lbfgs,
                                cmap='Reds', alpha=0.7, s=50, marker='^', label='LBFGS')
        
        ax.set_title('Parameter Space Exploration')
        ax.set_xlabel('Laser Energy')
        ax.set_ylabel('Pressure')
        ax.legend()
        ax.grid(True)
        
        # Add colorbar for acquisition time
        if bayesian_objectives or lbfgs_objectives:
            cbar = plt.colorbar(scatter if bayesian_objectives else scatter2, ax=ax)
            cbar.set_label('Acquisition Time (ms)')
        
        plt.tight_layout()
        if self.optimization_mode == 'target':
            mode_suffix = f"_target_{self.target_weighted_sum:.0f}"
        elif self.optimization_mode == 'match_spectrum':
            mode_suffix = "_match_spectrum"
        else:
            mode_suffix = "_maximize"
        plot_path = os.path.join(self.plots_dir, f'hybrid_optimization_results{mode_suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.show()

def run_multi_seed_optimization(optimization_mode='maximize', target_weighted_sum=None):
    """Run optimization with multiple seeds and compare results.
    
    Args:
        optimization_mode (str): 'maximize', 'target', or 'match_spectrum'
        target_weighted_sum (float): Target value when mode='target'
    
    This function runs the hybrid optimization with 10 different seeds:
    [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
    
    Returns:
        tuple: (all_results, all_optimizers)
            - all_results: List of optimization results for each seed
            - all_optimizers: List of optimizer objects for each seed
    
    Features:
        - Statistical analysis of weighted sum variance
        - Parameter convergence comparison across seeds  
        - 6-panel visualization comparing convergence
        - Individual seed plots saved to separate directories
    """
    model_path = "models/edm_4kepochs/ema_ckpt_final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seeds for multiple runs
    seeds = [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
    
    print("="*80)
    print("MULTI-SEED HYBRID OPTIMIZATION ANALYSIS")
    print("="*80)
    mode_str = f"mode={optimization_mode}"
    if optimization_mode == 'target':
        mode_str += f", target={target_weighted_sum}"
    print(f"Running optimization with {len(seeds)} different seeds ({mode_str})")
    print(f"Seeds: {seeds}")
    print("This will analyze convergence consistency and parameter variance")
    print("="*80)
    
    # Common optimization parameters
    opt_params = {
        'model_path': model_path,
        'device': device,
        'optimization_mode': optimization_mode,
        'target_weighted_sum': target_weighted_sum,
        'pressure_bounds': (1.0, 50.0),
        'laser_energy_bounds': (5.0, 50.0),
        'acquisition_time_bounds': (5.0, 100.0),
        'bayesian_n_calls': 100,
        'bayesian_n_initial_points': 10,
        'finetune_max_iter': 50,
        'finetune_lr': 2.0,
        'batch_size': 16,
        'num_sampling_steps': 18,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'rho': 7,
        'cfg_scale': 3.0,
        'smooth_output': True,
        'smooth_kernel_size': 9,
        'smooth_sigma': 2.0,
        'post_smooth': False,
        'post_smooth_window': 5,
        'seed': None  # Will be set for each run
    }
    
    # Store results from all runs
    all_results = []
    all_optimizers = []
    
    # Run optimization for each seed
    for i, seed in enumerate(seeds):
        print(f"\n{'='*20} RUN {i+1}/{len(seeds)} (Seed: {seed}) {'='*20}")
        
        # Set seed and create unique output directory
        set_seed(seed)
        if optimization_mode == 'target':
            mode_suffix = f"_target_{target_weighted_sum:.0f}"
        elif optimization_mode == 'match_spectrum':
            mode_suffix = "_match_spectrum"
        else:
            mode_suffix = "_maximize"
        opt_params['output_dir'] = f"multi_seed_demo{mode_suffix}/seed_{seed}"
        opt_params['seed'] = seed
        
        # Create and run optimizer
        optimizer = HybridBayesianLBFGSOptimizer(**opt_params)
        results = optimizer.optimize()
        
        # Generate plots for individual seed
        optimizer.plot_results()
        
        # Store results
        results['seed'] = seed
        results['run_id'] = i
        all_results.append(results)
        all_optimizers.append(optimizer)
        
        if optimization_mode == 'maximize':
            print(f"Run {i+1} complete - Best weighted sum: {results['best_weighted_sum']:.6f}")
        elif optimization_mode == 'target':
            target_diff = abs(results['best_weighted_sum'] - target_weighted_sum)
            print(f"Run {i+1} complete - Best weighted sum: {results['best_weighted_sum']:.6f} (target_diff={target_diff:.6f})")
        elif optimization_mode == 'match_spectrum':
            spectrum_dist = results.get('best_spectrum_dist', 'N/A')
            print(f"Run {i+1} complete - Best spectrum distance (MSE): {spectrum_dist:.6f}" if isinstance(spectrum_dist, float) else f"Run {i+1} complete - Best spectrum distance: {spectrum_dist}")
        print(f"Best params: [{results['best_params'][0]:.3f}, {results['best_params'][1]:.3f}, {results['best_params'][2]:.3f}]")
    
    # Analyze results
    print("\n" + "="*80)
    print("MULTI-SEED ANALYSIS RESULTS")
    print("="*80)
    
    # Extract key metrics
    weighted_sums = [r['best_weighted_sum'] for r in all_results]
    laser_energies = [r['best_params'][0] for r in all_results]
    pressures = [r['best_params'][1] for r in all_results]
    acquisition_times = [r['best_params'][2] for r in all_results]
    
    # Statistical analysis
    print(f"📊 WEIGHTED SUM STATISTICS:")
    print(f"  Mean: {np.mean(weighted_sums):.6f}")
    print(f"  Std:  {np.std(weighted_sums):.6f}")
    print(f"  Min:  {np.min(weighted_sums):.6f} (Seed: {seeds[np.argmin(weighted_sums)]})")
    print(f"  Max:  {np.max(weighted_sums):.6f} (Seed: {seeds[np.argmax(weighted_sums)]})")
    
    if optimization_mode == 'target':
        target_diffs = [abs(ws - target_weighted_sum) for ws in weighted_sums]
        print(f"\n🎯 TARGET ACHIEVEMENT STATISTICS:")
        print(f"  Mean distance to target: {np.mean(target_diffs):.6f}")
        print(f"  Std distance to target:  {np.std(target_diffs):.6f}")
        print(f"  Best achievement: {np.min(target_diffs):.6f} (Seed: {seeds[np.argmin(target_diffs)]})")
        print(f"  Worst achievement: {np.max(target_diffs):.6f} (Seed: {seeds[np.argmax(target_diffs)]})")
    elif optimization_mode == 'match_spectrum':
        spectrum_dists = [r.get('best_spectrum_dist', float('inf')) for r in all_results]
        print(f"\n🎯 SPECTRUM MATCHING STATISTICS:")
        print(f"  Mean spectrum distance (MSE): {np.mean(spectrum_dists):.6f}")
        print(f"  Std spectrum distance:        {np.std(spectrum_dists):.6f}")
        print(f"  Best match (lowest MSE):      {np.min(spectrum_dists):.6f} (Seed: {seeds[np.argmin(spectrum_dists)]})")
        print(f"  Worst match (highest MSE):    {np.max(spectrum_dists):.6f} (Seed: {seeds[np.argmax(spectrum_dists)]})")
    
    print(f"\n🎯 PARAMETER CONVERGENCE STATISTICS:")
    print(f"  Laser Energy:     {np.mean(laser_energies):.3f} ± {np.std(laser_energies):.3f}")
    print(f"  Pressure:         {np.mean(pressures):.3f} ± {np.std(pressures):.3f}")
    print(f"  Acquisition Time: {np.mean(acquisition_times):.3f} ± {np.std(acquisition_times):.3f}")
    
    # Create comprehensive plots
    plot_multi_seed_results(all_results, all_optimizers, seeds, optimization_mode, target_weighted_sum)
    
    return all_results, all_optimizers

def plot_multi_seed_results(all_results, all_optimizers, seeds, optimization_mode='maximize', target_weighted_sum=None):
    """Create comprehensive plots comparing multi-seed optimization results."""
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(18, 12))
    
    # Color palette for different seeds
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Determine what to plot based on mode
    if optimization_mode == 'maximize':
        ylabel_convergence = 'Weighted Sum'
        title_suffix = '(Maximization)'
    elif optimization_mode == 'target':
        ylabel_convergence = f'Distance to Target ({target_weighted_sum})'
        title_suffix = f'(Target={target_weighted_sum})'
    elif optimization_mode == 'match_spectrum':
        ylabel_convergence = 'Spectrum Distance (MSE)'
        title_suffix = '(Spectrum Matching)'
    
    # Plot 1: Convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    for i, (result, optimizer) in enumerate(zip(all_results, all_optimizers)):
        seed = seeds[i]
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            bayesian_objectives = [call['objective'] for call in optimizer.bayesian_history]
            bayesian_best_obj = np.minimum.accumulate(bayesian_objectives)
            
            if optimization_mode == 'maximize':
                bayesian_plot_values = -np.array(bayesian_best_obj)
            elif optimization_mode == 'target':
                bayesian_plot_values = bayesian_best_obj
            elif optimization_mode == 'match_spectrum':
                bayesian_plot_values = bayesian_best_obj
            
            ax1.plot(range(len(bayesian_objectives)), bayesian_plot_values, 
                    color=color, alpha=0.7, linewidth=1.5, label=f'Seed {seed}')
        
        # Fine-tuning phase continuation
        if optimizer.lbfgs_history:
            lbfgs_objectives = [call['objective'] for call in optimizer.lbfgs_history]
            offset = len(bayesian_objectives) if optimizer.bayesian_history else 0
            combined_evals = [offset + i for i in range(len(lbfgs_objectives))]
            
            if optimization_mode == 'maximize':
                lbfgs_plot_values = -np.array(lbfgs_objectives)
            elif optimization_mode == 'target':
                lbfgs_plot_values = lbfgs_objectives
            elif optimization_mode == 'match_spectrum':
                lbfgs_plot_values = lbfgs_objectives
            
            ax1.plot(combined_evals, lbfgs_plot_values, '--', color=color, alpha=0.7, linewidth=1.5)
    
    ax1.set_title(f'Convergence Comparison {title_suffix}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Evaluation/Iteration')
    ax1.set_ylabel(ylabel_convergence)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Final metric distribution (weighted sum or spectrum distance)
    ax2 = plt.subplot(2, 3, 2)
    
    if optimization_mode == 'match_spectrum':
        # Show spectrum distance for match_spectrum mode
        metric_values = [r.get('best_spectrum_dist', float('inf')) for r in all_results]
        title_text = 'Spectrum Distance (MSE) by Seed'
        ylabel_text = 'Spectrum Distance (MSE)'
    else:
        # Show weighted sum for maximize and target modes
        metric_values = [r['best_weighted_sum'] for r in all_results]
        title_text = 'Final Weighted Sum by Seed'
        ylabel_text = 'Best Weighted Sum'
        
    bars = ax2.bar(range(len(seeds)), metric_values, color=colors, alpha=0.7)
    
    if optimization_mode == 'target':
        title_text += f'\n(Target={target_weighted_sum})'
        # Add target line
        ax2.axhline(y=target_weighted_sum, color='red', linestyle='--', linewidth=2, label=f'Target={target_weighted_sum}')
        ax2.legend()
    
    ax2.set_title(title_text, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Run (Seed)')
    ax2.set_ylabel(ylabel_text)
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        format_str = f'{val:.6f}' if optimization_mode == 'match_spectrum' else f'{val:.2f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                format_str, ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Laser Energy evolution comparison
    ax3 = plt.subplot(2, 3, 3)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            laser_energies_bayes = [call['laser_energy'] for call in optimizer.bayesian_history]
            ax3.plot(range(len(laser_energies_bayes)), laser_energies_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            laser_energies_lbfgs = [call['laser_energy'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(laser_energies_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax3.plot(offset_iters, laser_energies_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax3.set_title('Laser Energy Evolution Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Evaluation/Iteration')
    ax3.set_ylabel('Laser Energy')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pressure evolution comparison
    ax4 = plt.subplot(2, 3, 4)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            pressures_bayes = [call['pressure'] for call in optimizer.bayesian_history]
            ax4.plot(range(len(pressures_bayes)), pressures_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            pressures_lbfgs = [call['pressure'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(pressures_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax4.plot(offset_iters, pressures_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax4.set_title('Pressure Evolution Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Evaluation/Iteration')
    ax4.set_ylabel('Pressure')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Acquisition Time evolution comparison
    ax5 = plt.subplot(2, 3, 5)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            acq_times_bayes = [call['acquisition_time'] for call in optimizer.bayesian_history]
            ax5.plot(range(len(acq_times_bayes)), acq_times_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            acq_times_lbfgs = [call['acquisition_time'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(acq_times_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax5.plot(offset_iters, acq_times_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax5.set_title('Acquisition Time Evolution Comparison', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Evaluation/Iteration')
    ax5.set_ylabel('Acquisition Time (ms)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Parameter variance analysis or target/spectrum achievement
    ax6 = plt.subplot(2, 3, 6)
    
    if optimization_mode == 'target':
        # Show distance to target for each seed
        weighted_sums = [r['best_weighted_sum'] for r in all_results]
        target_diffs = [abs(ws - target_weighted_sum) for ws in weighted_sums]
        
        bars = ax6.bar(range(len(seeds)), target_diffs, color=colors, alpha=0.7)
        ax6.set_title('Distance to Target by Seed', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Run (Seed)')
        ax6.set_ylabel('Distance to Target')
        ax6.set_xticks(range(len(seeds)))
        ax6.set_xticklabels([str(s) for s in seeds], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars, target_diffs):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(target_diffs)*0.01,
                    f'{diff:.3f}', ha='center', va='bottom', fontsize=8)
    elif optimization_mode == 'match_spectrum':
        # Show spectrum distance for each seed
        spectrum_dists = [r.get('best_spectrum_dist', float('inf')) for r in all_results]
        
        bars = ax6.bar(range(len(seeds)), spectrum_dists, color=colors, alpha=0.7)
        ax6.set_title('Spectrum Distance (MSE) by Seed', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Run (Seed)')
        ax6.set_ylabel('Spectrum Distance (MSE)')
        ax6.set_xticks(range(len(seeds)))
        ax6.set_xticklabels([str(s) for s in seeds], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, dist in zip(bars, spectrum_dists):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spectrum_dists)*0.01,
                    f'{dist:.6f}', ha='center', va='bottom', fontsize=8)
    else:
        # Show parameter variance
        laser_energies = [r['best_params'][0] for r in all_results]
        pressures = [r['best_params'][1] for r in all_results]
        acquisition_times = [r['best_params'][2] for r in all_results]
        
        param_names = ['Laser Energy', 'Pressure', 'Acq. Time']
        param_stds = [np.std(laser_energies), np.std(pressures), np.std(acquisition_times)]
        
        x_pos = np.arange(len(param_names))
        bars = ax6.bar(x_pos, param_stds, color=['red', 'green', 'blue'], alpha=0.7)
        ax6.set_title('Parameter Standard Deviation Across Seeds', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Standard Deviation')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(param_names)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, std in zip(bars, param_stds):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_stds)*0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if optimization_mode == 'target':
        mode_suffix = f"_target_{target_weighted_sum:.0f}"
    elif optimization_mode == 'match_spectrum':
        mode_suffix = "_match_spectrum"
    else:
        mode_suffix = "_maximize"
    os.makedirs(f"multi_seed_demo{mode_suffix}/plots", exist_ok=True)
    plot_path = f"multi_seed_demo{mode_suffix}/plots/multi_seed_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Multi-seed comparison plot saved to: {plot_path}")
    plt.show()

def main(run_mode='single', mode='maximize', target=None):
    """Run hybrid optimization demonstration.
    
    Args:
        run_mode (str): 'single' for single-seed run, 'multi' for multi-seed analysis
        mode (str): 'maximize', 'target', or 'match_spectrum' - optimization mode
        target (float): Target weighted sum value (required when mode='target')
    """
    
    if mode == 'target' and target is None:
        raise ValueError("target parameter must be provided when mode='target'")
    
    if run_mode == 'multi':
        # Run multi-seed analysis
        mode_str = f"🔬 Starting Multi-Seed Analysis (mode={mode}"
        if mode == 'target':
            mode_str += f", target={target}"
        mode_str += ")..."
        print(mode_str)
        all_results, all_optimizers = run_multi_seed_optimization(
            optimization_mode=mode,
            target_weighted_sum=target
        )
        return all_results, all_optimizers
    
    else:
        # Original single-seed run
        model_path = "models/edm_4kepochs/ema_ckpt_final.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("="*60)
        print(f"HYBRID BAYESIAN + GRADIENT OPTIMIZATION (mode={mode})")
        if mode == 'target':
            print(f"Target weighted sum: {target}")
        elif mode == 'match_spectrum':
            print(f"Matching spectrum from avg_spectrum.csv")
        print("="*60)
        seed = 69
        set_seed(seed)
        
        if mode == 'target':
            mode_suffix = f"_target_{target:.0f}"
        elif mode == 'match_spectrum':
            mode_suffix = "_match_spectrum"
        else:
            mode_suffix = "_maximize"
        
        optimizer = HybridBayesianLBFGSOptimizer(
            model_path=model_path,
            device=device,
            optimization_mode=mode,
            target_weighted_sum=target,
            pressure_bounds=(1.0, 50.0),
            laser_energy_bounds=(5.0, 50.0),
            acquisition_time_bounds=(5.0, 100.0),
            bayesian_n_calls=20,
            bayesian_n_initial_points=2,
            # LBFGS fine-tuning optimizer
            finetune_max_iter=20,
            finetune_lr=1.0,
            batch_size=16,
            output_dir=f"hybrid_optimization_demo{mode_suffix}",
            # Sampler parameters
            num_sampling_steps=30,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            cfg_scale=0.5,
            # Smoothing parameters
            smooth_output=True,
            smooth_kernel_size=5,
            smooth_sigma=2.0,
            # Post-processing smoothing
            post_smooth=True,
            post_smooth_window=5,
            seed=None
        )
        
        results = optimizer.optimize()
        optimizer.plot_results()
        
        print("="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        
        return results

if __name__ == "__main__":
    # Choose run mode and optimization mode
    # run_mode: 'single' or 'multi'
    # mode: 'maximize', 'target', or 'match_spectrum'
    # target: required when mode='target'
    
    # Example 1: Single-seed maximization
    # results = main('single', mode='maximize')
    
    # Example 2: Single-seed target seeking
    # results = main('single', mode='target', target=150.0)
    
    # Example 3: Single-seed spectrum matching
    # results = main('single', mode='match_spectrum')
    
    # Example 4: Multi-seed maximization
    # results = main('multi', mode='maximize')
    
    # Example 5: Multi-seed target seeking
    # results = main('multi', mode='target', target=900.0)
    
    # Example 6: Multi-seed spectrum matching
    results = main('multi', mode='match_spectrum')