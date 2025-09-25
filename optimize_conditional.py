"""
Hybrid Bayesian-LBFGS Optimization for LWFA Parameter Optimization

This module provides deterministic optimization with multi-seed analysis capabilities:

USAGE:
------
Single-seed run:
    python optimize_conditional.py
    # Or: main('single')

Multi-seed analysis (10 seeds):
    main('multi')  # In Python environment
    # Or change run_mode = 'multi' in __main__ section

FEATURES:
---------
✓ Fully deterministic latent reuse for reproducible results
✓ Hybrid Bayesian + gradient-based optimization  
✓ Multi-seed variance analysis with comprehensive statistics
✓ 9-panel visualization comparing convergence across seeds
✓ Parameter convergence analysis and coefficient of variation
✓ Best optimization phase analysis (Bayesian vs fine-tuning)

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

# Import necessary modules from the project
from src.modules_1d import EDMPrecond
from src.diffusion import EdmSampler, transform_vector, gaussian_smooth_1d
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
    
    def sample_differentiable(self, resolution, device, settings=None, n_samples=1, cfg_scale=3, settings_dim=0, smooth_output=False, smooth_kernel_size=5, smooth_sigma=1.0):
        """Differentiable version of the sample method that preserves gradients.
        
        Uses pre-initialized deterministic latents when available for fully reproducible sampling.
        """
        # Use stored deterministic latents if available and compatible, otherwise create new ones
        if (self.stored_latents is not None and 
            self.latents_shape == (n_samples, 1, resolution) and 
            self.latents_device == device):
            latents = self.stored_latents
            # print(f"Using deterministic latents for sampling") # Uncomment for debugging
        else:
            latents = self.randn_like(torch.empty((n_samples, 1, resolution), device=device))
            print(f"Warning: Creating new latents - algorithm may not be fully deterministic")

        sigma_min = self.sigma_min
        sigma_max = self.sigma_max

        # Time step discretization
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho)
                                                     - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop - no torch.no_grad() for differentiability
        x_next = latents.to(torch.float32) * t_steps[0]
       
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Increase noise temporarily
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)

            # Euler step - no torch.no_grad() for differentiability
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

class HybridBayesianLBFGSOptimizer:
    """
    Hybrid optimization approach that combines:
    1. Bayesian optimization for efficient exploration of parameter space
    2. LBFGS gradient-based fine-tuning starting from the best Bayesian result
    """
    def __init__(
        self,
        model_path,
        device="cuda",
        pressure_bounds=(5.0, 30.0),
        laser_energy_bounds=(10.0, 50.0),
        acquisition_time_bounds=(5.0, 50.0),
        # Bayesian phase parameters
        bayesian_n_calls=50,
        bayesian_n_initial_points=10,
        # Fine-tuning phase parameters  
        finetune_optimizer='lbfgs',  # 'lbfgs' or 'adam'
        finetune_max_iter=50,
        finetune_lr=0.1,
        # Adam-specific parameters (when using Adam)
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        # LBFGS-specific parameters (when using LBFGS)
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
        
        # Optimization parameters
        self.bayesian_n_calls = bayesian_n_calls
        self.bayesian_n_initial_points = bayesian_n_initial_points
        self.finetune_optimizer = finetune_optimizer.lower()
        self.finetune_max_iter = finetune_max_iter
        self.finetune_lr = finetune_lr
        
        # Adam parameters
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        
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
        
        print(f"Hybrid Bayesian-{self.finetune_optimizer.upper()} Optimization Setup:")
        print(f"  Phase 1 - Bayesian: {bayesian_n_calls} calls, {bayesian_n_initial_points} initial")
        print(f"  Phase 2 - {self.finetune_optimizer.upper()}: max {finetune_max_iter} iter, lr={finetune_lr}")
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

    def calculate_weighted_sum(self, spectrum, energy_values):
        """Calculate the weighted sum of intensities multiplied by MeV values."""
        return torch.sum(spectrum.to(self.device) * energy_values.to(self.device))
    
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
            
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            
            # Store history
            self.bayesian_history.append({
                'laser_energy': laser_energy,
                'pressure': pressure,
                'acquisition_time': acquisition_time,
                'weighted_sum': weighted_sum.item(),
                'spectrum': spectrum_intensity.detach().cpu().numpy(),
                'energy_values': deflection_MeV.detach().cpu().numpy()
            })
            
            print(f"Bayesian Eval {len(self.bayesian_history)}: "
                  f"[{laser_energy:.2f}, {pressure:.2f}, {acquisition_time:.2f}] -> {weighted_sum.item():.4f}")
            
            return -weighted_sum.item()  # Negative for minimization
            
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
        
        weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
        
        return -weighted_sum  # Negative for minimization

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
            acq_func='EI',
            random_state=self.seed if self.seed else 42
        )
        
        best_score = -result.fun
        best_params = result.x
        
        print(f"  Best: [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}] -> {best_score:.4f}")
        
        return {'best_params': best_params, 'best_score': best_score, 'result': result}

    def run_finetune_phase(self, initial_params):
        """Run the fine-tuning phase with selected optimizer (Adam or LBFGS)."""
        print(f"Phase 2: {self.finetune_optimizer.upper()} Fine-tuning ({self.finetune_max_iter} max iterations)")
        
        # Initialize parameters with gradients
        laser_energy = torch.tensor(initial_params[0], device=self.device, requires_grad=True)
        pressure = torch.tensor(initial_params[1], device=self.device, requires_grad=True)
        acquisition_time = torch.tensor(initial_params[2], device=self.device, requires_grad=True)
        
        # Create optimizer based on selection
        if self.finetune_optimizer == 'lbfgs':
            optimizer = optim.LBFGS(
                [laser_energy, pressure, acquisition_time], 
                lr=self.finetune_lr, 
                max_iter=self.finetune_max_iter,
                history_size=self.lbfgs_history_size,
                line_search_fn=self.lbfgs_line_search_fn
            )
        elif self.finetune_optimizer == 'adam':
            optimizer = optim.Adam(
                [laser_energy, pressure, acquisition_time],
                lr=self.finetune_lr,
                betas=self.adam_betas,
                eps=self.adam_eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.finetune_optimizer}. Choose 'adam' or 'lbfgs'.")
        
        best_score = -float('inf')
        iteration_count = 0
        
        if self.finetune_optimizer == 'lbfgs':
            # LBFGS requires closure function
            def closure():
                nonlocal iteration_count, best_score
                
                optimizer.zero_grad()
                
                loss = self.finetune_objective_and_grad(torch.stack([laser_energy, pressure, acquisition_time]))
                loss.backward()
                
                current_score = -loss.item()
                if current_score > best_score:
                    best_score = current_score
                
                # Store history
                self.lbfgs_history.append({
                    'iteration': iteration_count,
                    'laser_energy': laser_energy.item(),
                    'pressure': pressure.item(),
                    'acquisition_time': acquisition_time.item(),
                    'weighted_sum': current_score
                })
                
                if iteration_count % 5 == 0:
                    print(f"  {self.finetune_optimizer.upper()} Iter {iteration_count}: [{laser_energy.item():.3f}, {pressure.item():.3f}, {acquisition_time.item():.3f}] -> {current_score:.4f}")
                
                iteration_count += 1
                return loss
            
            optimizer.step(closure)
        
        else:  # Adam optimizer
            for iteration in range(self.finetune_max_iter):
                optimizer.zero_grad()
                
                # Clamp parameters to bounds (optional for Adam)
                with torch.no_grad():
                    laser_energy.clamp_(self.laser_energy_bounds[0], self.laser_energy_bounds[1])
                    pressure.clamp_(self.pressure_bounds[0], self.pressure_bounds[1])
                    acquisition_time.clamp_(self.acquisition_time_bounds[0], self.acquisition_time_bounds[1])
                
                loss = self.finetune_objective_and_grad(torch.stack([laser_energy, pressure, acquisition_time]))
                loss.backward()
                
                current_score = -loss.item()
                if current_score > best_score:
                    best_score = current_score
                
                # Store history
                self.lbfgs_history.append({  # Keep same history name for compatibility
                    'iteration': iteration,
                    'laser_energy': laser_energy.item(),
                    'pressure': pressure.item(),
                    'acquisition_time': acquisition_time.item(),
                    'weighted_sum': current_score
                })
                
                if iteration % 5 == 0:
                    print(f"  {self.finetune_optimizer.upper()} Iter {iteration}: [{laser_energy.item():.3f}, {pressure.item():.3f}, {acquisition_time.item():.3f}] -> {current_score:.4f}")
                
                optimizer.step()
                iteration_count = iteration + 1
        
        final_params = [laser_energy.item(), pressure.item(), acquisition_time.item()]
        print(f"  Final: [{final_params[0]:.3f}, {final_params[1]:.3f}, {final_params[2]:.3f}] -> {best_score:.4f}")
        
        return {'best_params': final_params, 'best_score': best_score, 'iterations': iteration_count}

    def optimize(self):
        """Run the complete hybrid optimization."""
        if self.logger is None:
            self.logger = setup_logging(self.output_dir, "Hybrid_Bayesian_LBFGS")
        
        print(f"Starting Hybrid Bayesian-{self.finetune_optimizer.upper()} Optimization")
        
        # Phase 1: Bayesian exploration
        bayesian_result = self.run_bayesian_phase()
        
        # Phase 2: Fine-tuning
        finetune_result = self.run_finetune_phase(bayesian_result['best_params'])
        
        # Determine overall best
        if finetune_result['best_score'] > bayesian_result['best_score']:
            overall_best_score = finetune_result['best_score']
            overall_best_params = finetune_result['best_params']
            best_phase = self.finetune_optimizer.upper()
        else:
            overall_best_score = bayesian_result['best_score']
            overall_best_params = bayesian_result['best_params']
            best_phase = 'Bayesian'
        
        print(f"\nOptimization Complete!")
        print(f"  Best result from {best_phase} phase")
        print(f"  Final params: [{overall_best_params[0]:.3f}, {overall_best_params[1]:.3f}, {overall_best_params[2]:.3f}]")
        print(f"  Best weighted sum: {overall_best_score:.6f}")
        
        # Calculate improvement
        if finetune_result['best_score'] > bayesian_result['best_score']:
            improvement = ((finetune_result['best_score'] - bayesian_result['best_score']) / bayesian_result['best_score']) * 100
            print(f"  {self.finetune_optimizer.upper()} improved by: {improvement:.2f}%")
        
        return {
            'best_weighted_sum': overall_best_score,
            'best_params': overall_best_params,
            'best_phase': best_phase,
            'bayesian_phase': bayesian_result,
            'finetune_phase': finetune_result,
            'bayesian_history': self.bayesian_history,
            'lbfgs_history': self.lbfgs_history  # Keep same name for plotting compatibility
        }

    def plot_results(self):
        """Create visualization of the hybrid optimization results."""
        if not self.bayesian_history and not self.lbfgs_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Extract data
        bayesian_scores = [call['weighted_sum'] for call in self.bayesian_history]
        lbfgs_scores = [call['weighted_sum'] for call in self.lbfgs_history] if self.lbfgs_history else []
        
        # Plot 1: Convergence
        ax = axes[0, 0]
        if bayesian_scores:
            bayesian_best = np.maximum.accumulate(bayesian_scores)
            ax.plot(range(len(bayesian_scores)), bayesian_best, 'b-', linewidth=2, label='Bayesian Phase')
        
        if lbfgs_scores:
            offset = len(bayesian_scores)
            combined_evals = [offset + i for i in range(len(lbfgs_scores))]
            ax.plot(combined_evals, lbfgs_scores, 'r-', linewidth=2, label=f'{self.finetune_optimizer.upper()} Phase')
        
        ax.set_title('Hybrid Optimization Convergence')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Weighted Sum')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Parameter evolution - Laser Energy
        ax = axes[0, 1]
        if bayesian_scores:
            laser_energies = [call['laser_energy'] for call in self.bayesian_history]
            ax.plot(range(len(laser_energies)), laser_energies, 'b.-', alpha=0.7, label='Bayesian')
        
        if lbfgs_scores:
            laser_energies_lbfgs = [call['laser_energy'] for call in self.lbfgs_history]
            iters = [call['iteration'] for call in self.lbfgs_history]
            # Offset iterations to continue from where Bayesian phase ended
            offset = len(bayesian_scores) if bayesian_scores else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, laser_energies_lbfgs, 'r.-', alpha=0.7, label=self.finetune_optimizer.upper())
        
        ax.set_title('Laser Energy Evolution')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Laser Energy')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Parameter evolution - Pressure
        ax = axes[0, 2]
        if bayesian_scores:
            pressures = [call['pressure'] for call in self.bayesian_history]
            ax.plot(range(len(pressures)), pressures, 'b.-', alpha=0.7, label='Bayesian')
        
        if lbfgs_scores:
            pressures_lbfgs = [call['pressure'] for call in self.lbfgs_history]
            iters = [call['iteration'] for call in self.lbfgs_history]
            # Offset iterations to continue from where Bayesian phase ended
            offset = len(bayesian_scores) if bayesian_scores else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, pressures_lbfgs, 'r.-', alpha=0.7, label=self.finetune_optimizer.upper())
        
        ax.set_title('Pressure Evolution')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Pressure')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Parameter evolution - Acquisition Time (Opening Time)
        ax = axes[1, 0]
        if bayesian_scores:
            acquisition_times = [call['acquisition_time'] for call in self.bayesian_history]
            ax.plot(range(len(acquisition_times)), acquisition_times, 'b.-', alpha=0.7, label='Bayesian')
        
        if lbfgs_scores:
            acquisition_times_lbfgs = [call['acquisition_time'] for call in self.lbfgs_history]
            iters = [call['iteration'] for call in self.lbfgs_history]
            # Offset iterations to continue from where Bayesian phase ended
            offset = len(bayesian_scores) if bayesian_scores else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax.plot(offset_iters, acquisition_times_lbfgs, 'r.-', alpha=0.7, label=self.finetune_optimizer.upper())
        
        ax.set_title('Acquisition Time Evolution (Opening Time)')
        ax.set_xlabel('Evaluation/Iteration')
        ax.set_ylabel('Acquisition Time (ms)')
        ax.legend()
        ax.grid(True)
        
        # Plot 5: Best spectrum
        ax = axes[1, 1]
        all_evaluations = self.bayesian_history + self.lbfgs_history
        if all_evaluations:
            best_eval = max(all_evaluations, key=lambda x: x['weighted_sum'])
            
            if 'spectrum' in best_eval and 'energy_values' in best_eval:
                ax.plot(best_eval['energy_values'], best_eval['spectrum'], 'g-', linewidth=2)
                ax.set_title('Best Spectrum')
                ax.set_xlabel('Energy (MeV)')
                ax.set_ylabel('Intensity')
                ax.grid(True)
        
        # Plot 6: Parameter trajectory in 3D parameter space  
        ax = axes[1, 2]
        if bayesian_scores:
            laser_energies = [call['laser_energy'] for call in self.bayesian_history]
            pressures = [call['pressure'] for call in self.bayesian_history]
            acquisition_times = [call['acquisition_time'] for call in self.bayesian_history]
            colors = plt.cm.Blues(np.linspace(0.3, 1, len(laser_energies)))
            scatter = ax.scatter(laser_energies, pressures, c=acquisition_times, 
                               cmap='Blues', alpha=0.7, s=50, label='Bayesian')
            
        if lbfgs_scores:
            laser_energies_lbfgs = [call['laser_energy'] for call in self.lbfgs_history]
            pressures_lbfgs = [call['pressure'] for call in self.lbfgs_history]
            acquisition_times_lbfgs = [call['acquisition_time'] for call in self.lbfgs_history]
            scatter2 = ax.scatter(laser_energies_lbfgs, pressures_lbfgs, c=acquisition_times_lbfgs,
                                cmap='Reds', alpha=0.7, s=50, marker='^', label=self.finetune_optimizer.upper())
        
        ax.set_title('Parameter Space Exploration')
        ax.set_xlabel('Laser Energy')
        ax.set_ylabel('Pressure')
        ax.legend()
        ax.grid(True)
        
        # Add colorbar for acquisition time
        if bayesian_scores or lbfgs_scores:
            cbar = plt.colorbar(scatter if bayesian_scores else scatter2, ax=ax)
            cbar.set_label('Acquisition Time (ms)')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'hybrid_optimization_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.show()

def run_multi_seed_optimization():
    """Run optimization with multiple seeds and compare results.
    
    This function runs the hybrid optimization with 10 different seeds:
    [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
    
    Returns:
        tuple: (all_results, all_optimizers)
            - all_results: List of optimization results for each seed
            - all_optimizers: List of optimizer objects for each seed
    
    Features:
        - Statistical analysis of weighted sum variance
        - Parameter convergence comparison across seeds  
        - Comprehensive 9-panel visualization
        - Analysis of which optimization phase performs best
    """
    model_path = "models/edm_4kepochs/ema_ckpt_final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seeds for multiple runs
    seeds = [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]
    
    print("="*80)
    print("MULTI-SEED HYBRID OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"Running optimization with {len(seeds)} different seeds: {seeds}")
    print("This will analyze convergence consistency and parameter variance")
    print("="*80)
    
    # Common optimization parameters
    opt_params = {
        'model_path': model_path,
        'device': device,
        'pressure_bounds': (1.0, 50.0),
        'laser_energy_bounds': (5.0, 50.0),
        'acquisition_time_bounds': (5.0, 100.0),
        'bayesian_n_calls': 20,
        'bayesian_n_initial_points': 2,
        'finetune_optimizer': 'lbfgs',
        'finetune_max_iter': 20,
        'finetune_lr': 1.0,
        'batch_size': 16,
        'num_sampling_steps': 30,
        'sigma_min': 0.002,
        'sigma_max': 80,
        'rho': 7,
        'cfg_scale': 0.5,
        'smooth_output': True,
        'smooth_kernel_size': 5,
        'smooth_sigma': 2.0,
        'post_smooth': True,
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
        opt_params['output_dir'] = f"multi_seed_demo/seed_{seed}"
        opt_params['seed'] = seed
        
        # Create and run optimizer
        optimizer = HybridBayesianLBFGSOptimizer(**opt_params)
        results = optimizer.optimize()
        
        # Store results
        results['seed'] = seed
        results['run_id'] = i
        all_results.append(results)
        all_optimizers.append(optimizer)
        
        print(f"Run {i+1} complete - Best weighted sum: {results['best_weighted_sum']:.6f}")
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
    best_phases = [r['best_phase'] for r in all_results]
    
    # Statistical analysis
    print(f"📊 WEIGHTED SUM STATISTICS:")
    print(f"  Mean: {np.mean(weighted_sums):.6f}")
    print(f"  Std:  {np.std(weighted_sums):.6f}")
    print(f"  Min:  {np.min(weighted_sums):.6f} (Seed: {seeds[np.argmin(weighted_sums)]})")
    print(f"  Max:  {np.max(weighted_sums):.6f} (Seed: {seeds[np.argmax(weighted_sums)]})")
    print(f"  CV:   {np.std(weighted_sums)/np.mean(weighted_sums)*100:.2f}%")
    
    print(f"\n🎯 PARAMETER CONVERGENCE STATISTICS:")
    print(f"  Laser Energy:     {np.mean(laser_energies):.3f} ± {np.std(laser_energies):.3f}")
    print(f"  Pressure:         {np.mean(pressures):.3f} ± {np.std(pressures):.3f}")
    print(f"  Acquisition Time: {np.mean(acquisition_times):.3f} ± {np.std(acquisition_times):.3f}")
    
    print(f"\n🏆 BEST PHASE ANALYSIS:")
    phase_counts = {}
    for phase in best_phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count}/{len(seeds)} runs ({count/len(seeds)*100:.1f}%)")
    
    # Create comprehensive plots
    plot_multi_seed_results(all_results, all_optimizers, seeds)
    
    return all_results, all_optimizers

def plot_multi_seed_results(all_results, all_optimizers, seeds):
    """Create comprehensive plots comparing multi-seed optimization results."""
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Color palette for different seeds
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Plot 1: Convergence comparison (top-left)
    ax1 = plt.subplot(3, 3, 1)
    for i, (result, optimizer) in enumerate(zip(all_results, all_optimizers)):
        seed = seeds[i]
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            bayesian_scores = [call['weighted_sum'] for call in optimizer.bayesian_history]
            bayesian_best = np.maximum.accumulate(bayesian_scores)
            ax1.plot(range(len(bayesian_scores)), bayesian_best, 
                    color=color, alpha=0.7, linewidth=1.5, label=f'Seed {seed}')
        
        # Fine-tuning phase continuation
        if optimizer.lbfgs_history:
            lbfgs_scores = [call['weighted_sum'] for call in optimizer.lbfgs_history]
            offset = len(bayesian_scores) if optimizer.bayesian_history else 0
            combined_evals = [offset + i for i in range(len(lbfgs_scores))]
            ax1.plot(combined_evals, lbfgs_scores, '--', color=color, alpha=0.7, linewidth=1.5)
    
    ax1.set_title('Convergence Comparison Across Seeds')
    ax1.set_xlabel('Evaluation/Iteration')
    ax1.set_ylabel('Weighted Sum')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Final weighted sum distribution (top-center)
    ax2 = plt.subplot(3, 3, 2)
    weighted_sums = [r['best_weighted_sum'] for r in all_results]
    bars = ax2.bar(range(len(seeds)), weighted_sums, color=colors, alpha=0.7)
    ax2.set_title('Final Weighted Sum by Seed')
    ax2.set_xlabel('Run (Seed)')
    ax2.set_ylabel('Best Weighted Sum')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, weighted_sums)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weighted_sums)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Parameter convergence scatter (top-right)
    ax3 = plt.subplot(3, 3, 3)
    laser_energies = [r['best_params'][0] for r in all_results]
    pressures = [r['best_params'][1] for r in all_results]
    acquisition_times = [r['best_params'][2] for r in all_results]
    
    scatter = ax3.scatter(laser_energies, pressures, c=acquisition_times, 
                         s=100, alpha=0.7, cmap='viridis', edgecolors='black')
    ax3.set_title('Parameter Convergence Points')
    ax3.set_xlabel('Laser Energy')
    ax3.set_ylabel('Pressure')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Acquisition Time (ms)')
    ax3.grid(True, alpha=0.3)
    
    # Add seed labels
    for i, (le, p, seed) in enumerate(zip(laser_energies, pressures, seeds)):
        ax3.annotate(str(seed), (le, p), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Plot 4: Laser Energy evolution comparison (middle-left)
    ax4 = plt.subplot(3, 3, 4)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            laser_energies_bayes = [call['laser_energy'] for call in optimizer.bayesian_history]
            ax4.plot(range(len(laser_energies_bayes)), laser_energies_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            laser_energies_lbfgs = [call['laser_energy'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(laser_energies_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax4.plot(offset_iters, laser_energies_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax4.set_title('Laser Energy Evolution Comparison')
    ax4.set_xlabel('Evaluation/Iteration')
    ax4.set_ylabel('Laser Energy')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Pressure evolution comparison (middle-center)
    ax5 = plt.subplot(3, 3, 5)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            pressures_bayes = [call['pressure'] for call in optimizer.bayesian_history]
            ax5.plot(range(len(pressures_bayes)), pressures_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            pressures_lbfgs = [call['pressure'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(pressures_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax5.plot(offset_iters, pressures_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax5.set_title('Pressure Evolution Comparison')
    ax5.set_xlabel('Evaluation/Iteration')
    ax5.set_ylabel('Pressure')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Acquisition Time evolution comparison (middle-right)
    ax6 = plt.subplot(3, 3, 6)
    for i, optimizer in enumerate(all_optimizers):
        color = colors[i]
        
        # Bayesian phase
        if optimizer.bayesian_history:
            acq_times_bayes = [call['acquisition_time'] for call in optimizer.bayesian_history]
            ax6.plot(range(len(acq_times_bayes)), acq_times_bayes, 
                    color=color, alpha=0.6, linewidth=1)
        
        # Fine-tuning phase
        if optimizer.lbfgs_history:
            acq_times_lbfgs = [call['acquisition_time'] for call in optimizer.lbfgs_history]
            iters = [call['iteration'] for call in optimizer.lbfgs_history]
            offset = len(acq_times_bayes) if optimizer.bayesian_history else 0
            offset_iters = [offset + iter_num for iter_num in iters]
            ax6.plot(offset_iters, acq_times_lbfgs, '--', color=color, alpha=0.6, linewidth=1)
    
    ax6.set_title('Acquisition Time Evolution Comparison')
    ax6.set_xlabel('Evaluation/Iteration')
    ax6.set_ylabel('Acquisition Time (ms)')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Parameter variance analysis (bottom-left)
    ax7 = plt.subplot(3, 3, 7)
    param_names = ['Laser Energy', 'Pressure', 'Acq. Time']
    param_means = [np.mean(laser_energies), np.mean(pressures), np.mean(acquisition_times)]
    param_stds = [np.std(laser_energies), np.std(pressures), np.std(acquisition_times)]
    
    x_pos = np.arange(len(param_names))
    bars = ax7.bar(x_pos, param_stds, color=['red', 'green', 'blue'], alpha=0.7)
    ax7.set_title('Parameter Standard Deviation Across Seeds')
    ax7.set_xlabel('Parameters')
    ax7.set_ylabel('Standard Deviation')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(param_names)
    ax7.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, std in zip(bars, param_stds):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_stds)*0.01,
                f'{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 8: Best phase distribution (bottom-center)
    ax8 = plt.subplot(3, 3, 8)
    best_phases = [r['best_phase'] for r in all_results]
    phase_counts = {}
    for phase in best_phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())
    colors_pie = ['lightblue', 'lightcoral', 'lightgreen'][:len(phases)]
    
    wedges, texts, autotexts = ax8.pie(counts, labels=phases, autopct='%1.1f%%', 
                                      colors=colors_pie, startangle=90)
    ax8.set_title('Best Phase Distribution')
    
    # Plot 9: Coefficient of variation (bottom-right)
    ax9 = plt.subplot(3, 3, 9)
    cv_weighted_sum = np.std(weighted_sums) / np.mean(weighted_sums) * 100
    cv_laser = np.std(laser_energies) / np.mean(laser_energies) * 100
    cv_pressure = np.std(pressures) / np.mean(pressures) * 100
    cv_acq_time = np.std(acquisition_times) / np.mean(acquisition_times) * 100
    
    cvs = [cv_weighted_sum, cv_laser, cv_pressure, cv_acq_time]
    labels = ['Weighted Sum', 'Laser Energy', 'Pressure', 'Acq. Time']
    
    bars = ax9.bar(range(len(labels)), cvs, color=['purple', 'red', 'green', 'blue'], alpha=0.7)
    ax9.set_title('Coefficient of Variation (%)')
    ax9.set_xlabel('Metrics')
    ax9.set_ylabel('CV (%)')
    ax9.set_xticks(range(len(labels)))
    ax9.set_xticklabels(labels, rotation=45)
    ax9.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv in zip(bars, cvs):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cvs)*0.01,
                f'{cv:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("multi_seed_demo/plots", exist_ok=True)
    plot_path = "multi_seed_demo/plots/multi_seed_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Multi-seed comparison plot saved to: {plot_path}")
    plt.show()

def main(run_mode='single'):
    """Run hybrid optimization demonstration.
    
    Args:
        run_mode (str): 'single' for single-seed run, 'multi' for multi-seed analysis
    """
    
    if run_mode == 'multi':
        # Run multi-seed analysis
        print("🔬 Starting Multi-Seed Analysis...")
        all_results, all_optimizers = run_multi_seed_optimization()
        return all_results, all_optimizers
    
    else:
        # Original single-seed run
        model_path = "models/edm_4kepochs/ema_ckpt_final.pt"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("="*60)
        print("HYBRID BAYESIAN + GRADIENT OPTIMIZATION")
        print("="*60)
        seed = 69
        set_seed(seed)
        
        optimizer = HybridBayesianLBFGSOptimizer(
            model_path=model_path,
            device=device,
            pressure_bounds=(1.0, 50.0),
            laser_energy_bounds=(5.0, 50.0),
            acquisition_time_bounds=(5.0, 100.0),
            bayesian_n_calls=20,
            bayesian_n_initial_points=2,
            # Select optimizer: 'lbfgs' or 'adam'
            finetune_optimizer='lbfgs',  # Try 'adam' for comparison
            finetune_max_iter=20,
            finetune_lr=1.0,
            batch_size=16,
            output_dir="hybrid_optimization_demo",
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
            post_smooth_window=5,  # Wider window for more smoothing
            seed=None
        )
        
        results = optimizer.optimize()
        optimizer.plot_results()
        
        print("="*60)
        print("NOTES ON OPTIMIZER SELECTION:")
        print("="*60)
        print("🎯 LBFGS (finetune_optimizer='lbfgs'):")
        print("  • Quasi-Newton method with limited memory")
        print("  • Uses second-order derivative approximation")
        print("  • Typically faster convergence for smooth problems")
        print("  • Good for problems with few parameters")
        print("  • Uses line search for step size")
        print()
        print("🚀 ADAM (finetune_optimizer='adam'):")
        print("  • Adaptive moment estimation")
        print("  • Uses first-order gradients only")
        print("  • More robust to noisy gradients")
        print("  • Better for larger parameter spaces")
        print("  • Easier to tune hyperparameters")
        print()
        print("💡 TO SWITCH OPTIMIZERS:")
        print("  Change 'finetune_optimizer' parameter:")
        print("  • 'lbfgs' for L-BFGS optimization")  
        print("  • 'adam' for Adam optimization")
        print()
        print("🔬 TO RUN MULTI-SEED ANALYSIS:")
        print("  Call main('multi') to run with 10 different seeds")
        print("  This will provide comprehensive variance analysis")
        print("="*60)
        
        return results

if __name__ == "__main__":
    # Choose run mode: 'single' or 'multi'
    # For quick testing, use 'single' 
    # For comprehensive analysis, use 'multi'
    
    run_mode = 'multi'  # Change to 'multi' for multi-seed analysis
    
    if run_mode == 'multi':
        print("🔬 MULTI-SEED OPTIMIZATION ANALYSIS")
        print("This will run 10 optimizations with different seeds:")
        print("Seeds: [42, 69, 100, 420, 1337, 1620, 1999, 2025, 2077, 3001]")
        print("Expected runtime: ~10-15 minutes (depending on hardware)")
        print()
        
        # Uncomment the next line to run multi-seed analysis
        results = main('multi')
        
    else:
        results = main('single')