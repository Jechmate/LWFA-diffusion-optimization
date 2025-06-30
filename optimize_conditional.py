import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random
import time
from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei
import pandas as pd
from scipy import stats
import gc  # Add garbage collection import
import logging
import json
from datetime import datetime

# Import necessary modules from the project - Updated for EDM 1D
from src.modules_1d import EDMPrecond
from src.diffusion import EdmSampler, transform_vector, gaussian_smooth_1d
from src.utils import deflection_biexp_calc, calc_spec

def setup_logging(output_dir, method_name, trial_num=None):
    """
    Set up detailed logging for optimization runs.
    
    Args:
        output_dir: Directory to save log files
        method_name: Name of the optimization method
        trial_num: Trial number (if applicable)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger name
    if trial_num is not None:
        logger_name = f"{method_name}_trial_{trial_num}"
        log_filename = f"{method_name}_trial_{trial_num}.log"
    else:
        logger_name = method_name
        log_filename = f"{method_name}.log"
    
    # Configure logger
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
    
    # Log start of optimization
    logger.info(f"Starting optimization: {method_name}")
    if trial_num is not None:
        logger.info(f"Trial number: {trial_num}")
    logger.info(f"Log file: {log_path}")
    
    return logger

def save_optimization_progress(output_dir, method_name, trial_num, progress_data):
    """
    Save detailed optimization progress to JSON file.
    
    Args:
        output_dir: Directory to save progress files
        method_name: Name of the optimization method
        trial_num: Trial number
        progress_data: Dictionary containing optimization progress
    """
    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    
    filename = f"{method_name}_trial_{trial_num}_progress.json"
    filepath = os.path.join(progress_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in progress_data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            serializable_data[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
        else:
            serializable_data[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Optimization progress saved to: {filepath}")

def cleanup_gpu_memory():
    """
    Explicitly clean up GPU memory and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"

def create_energy_axis(length=256, electron_pointing_pixel=62):
    """Create energy axis for plotting using biexponential deflection calculation."""
    # Use a larger size to ensure we get enough high-energy values
    temp_size = max(length * 2, 512)  
    
    # Calculate deflection using the biexponential model
    deflection_MeV, _ = deflection_biexp_calc(
        batch_size=1, 
        hor_image_size=temp_size, 
        electron_pointing_pixel=electron_pointing_pixel
    )
    
    # Convert to numpy and take first batch
    deflection_array = deflection_MeV[0].cpu().numpy()
    
    # Remove zeros and get valid energy values
    valid_energies = deflection_array[deflection_array > 0]
    
    # Sort in descending order to get highest energies first (top values)
    valid_energies_sorted = np.sort(valid_energies)[::-1]
    
    # Take the top 256 values (highest energies)
    if len(valid_energies_sorted) >= length:
        energy_axis = valid_energies_sorted[:length]
        # Reverse to have ascending order for plotting (lowest to highest)
        # energy_axis = energy_axis[::-1]
    else:
        # If we don't have enough values, pad with zeros
        energy_axis = np.concatenate([
            np.zeros(length - len(valid_energies_sorted)),
            valid_energies_sorted[::-1]
        ])
    return energy_axis

def set_seed(seed=42):
    """
    Set random seed for reproducible results across all random number generators.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

class DifferentiableEdmSampler(EdmSampler):
    """
    Differentiable version of EDM sampler that allows gradients to flow through
    the sampling process for optimization purposes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def sample_differentiable(self, resolution, device, settings=None, n_samples=1, cfg_scale=3, settings_dim=0, smooth_output=False, smooth_kernel_size=5, smooth_sigma=1.0):
        """
        Differentiable version of the sample method that preserves gradients.
        
        Args:
            resolution: Resolution of the output
            device: Device to run on
            settings: Conditioning settings
            n_samples: Number of samples to generate
            cfg_scale: Classifier-free guidance scale
            settings_dim: Dimension of settings
        """
        # Create initial latents
        latents = self.randn_like(torch.empty((n_samples, 1, resolution), device=device))

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

        # Main sampling loop - REMOVED torch.no_grad() for differentiability
        x_next = latents.to(torch.float32) * t_steps[0]
       
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            
            # Increase noise temporarily
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)

            # Euler step - REMOVED torch.no_grad() for differentiability
            denoised = None
            
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
                # REMOVED torch.no_grad() for differentiability
                denoised = None
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

class SpectrumWeightedSumOptimizer:
    def __init__(
        self,
        model_path,
        device="cuda",
        # Changed to use bounds instead of fixed values
        pressure_bounds=(10.0, 50.0),
        laser_energy_bounds=(10.0, 50.0),
        acquisition_time_bounds=(5.0, 50.0),
        # Optional: allow override of starting point
        pressure_start=None,
        laser_energy_start=None,
        acquisition_time_start=None,
        optimization_steps=100,
        lr=0.01,
        batch_size=4,
        save_images=True,
        output_dir="optimization_results",
        # EDM-specific parameters
        spectrum_length=256,
        features=["E", "P", "ms"],
        num_sampling_steps=18,
        seed=None,
        # Learning rate schedule parameters - Changed to regular cosine annealing
        use_cosine_annealing=True,
        T_max=None,  # Will default to optimization_steps if None
        eta_min=1e-6  # Minimum learning rate
    ):
        self.device = device
        self.pressure_bounds = pressure_bounds
        self.laser_energy_bounds = laser_energy_bounds
        self.acquisition_time_bounds = acquisition_time_bounds
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.save_images = save_images
        self.output_dir = output_dir
        self.spectrum_length = spectrum_length
        self.features = features
        self.num_sampling_steps = num_sampling_steps
        self.seed = seed
        
        # Learning rate schedule parameters - Updated for regular cosine annealing
        self.use_cosine_annealing = use_cosine_annealing
        self.T_max = T_max if T_max is not None else optimization_steps
        self.eta_min = eta_min
        
        # Set seed if provided
        if seed is not None:
            set_seed(seed)
        
        # Initialize starting points randomly from bounds if not provided
        if pressure_start is None:
            self.pressure = np.random.uniform(pressure_bounds[0], pressure_bounds[1])
        else:
            self.pressure = pressure_start
            
        if laser_energy_start is None:
            self.laser_energy = np.random.uniform(laser_energy_bounds[0], laser_energy_bounds[1])
        else:
            self.laser_energy = laser_energy_start
            
        if acquisition_time_start is None:
            self.acquisition_time_ms = np.random.uniform(acquisition_time_bounds[0], acquisition_time_bounds[1])
        else:
            self.acquisition_time_ms = acquisition_time_start
        
        print(f"Spectrum length: {spectrum_length}")
        print(f"Features: {features}")
        print(f"Random starting point:")
        print(f"  Laser energy: {self.laser_energy:.2f}")
        print(f"  Pressure: {self.pressure:.2f}")
        print(f"  Acquisition time: {self.acquisition_time_ms:.2f}ms")
        
        # Create organized directory structure
        if self.save_images or True:  # Always create dirs for plots
            self.images_dir = os.path.join(self.output_dir, "images")
            self.plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)
            print(f"Results will be saved to: {self.output_dir}")
            if self.save_images:
                print(f"  Images: {self.images_dir}")
            print(f"  Plots: {self.plots_dir}")
        
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
            num_steps=num_sampling_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
        )
        
        # Initialize optimization history
        self.weighted_sum_history = []
        self.spectrum_history = []
        # Add parameter tracking
        self.laser_energy_history = []
        self.pressure_history = []
        self.acquisition_time_history = []
        
        # Setup logging
        self.logger = None  # Will be set up when optimization starts
        self.trial_num = None  # Will be set when used in multi-trial runs

    def load_model(self, model_path):
        """Load the pre-trained EDM model"""
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.model.eval()  # Set to evaluation mode
            print(f"EDM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calculate_weighted_sum(self, spectrum, energy_values):
        """
        Calculate the weighted sum of intensities multiplied by MeV values.
        
        Args:
            spectrum (torch.Tensor): The intensity values of the spectrum
            energy_values (torch.Tensor): The corresponding energy values (MeV)
            
        Returns:
            torch.Tensor: Sum of intensities * MeV values
        """
        # Ensure inputs are on the correct device
        spectrum = spectrum.to(self.device)
        energy_values = energy_values.to(self.device)
        
        # Calculate weighted sum: sum(intensity * MeV)
        weighted_sum = torch.sum(spectrum * energy_values)
        return weighted_sum

    def save_spectra_batch(self, spectra, step, prefix="step"):
        """
        Save a batch of 1D spectra to disk.
        
        Args:
            spectra (torch.Tensor): Batch of 1D spectra to save
            step (int): Current optimization step
            prefix (str): Prefix for filename
        """
        if not self.save_images:
            return
            
        # Convert to numpy and handle batch dimension
        spectra_np = spectra.detach().cpu().numpy()
        
        for i, spectrum in enumerate(spectra_np):
            # Save as numpy array
            filename = f"{prefix}_{step:04d}_batch_{i:02d}.npy"
            filepath = os.path.join(self.images_dir, filename)
            np.save(filepath, spectrum)

    def create_lr_scheduler(self, optimizer):
        """Create cosine annealing scheduler"""
        if self.use_cosine_annealing:
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.T_max,
                eta_min=self.eta_min
            )
        else:
            return None

    def cleanup(self):
        """
        Clean up GPU memory used by the optimizer.
        Call this when done with optimization to free GPU memory.
        """
        # Delete model and sampler to free GPU memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'sampler'):
            del self.sampler
        
        # Clear GPU cache
        cleanup_gpu_memory()
        print(f"Optimizer cleanup completed. {get_gpu_memory_info()}")

    def optimize(self):
        """Run the optimization process to maximize the weighted sum"""
        return self._optimize_single_run()
    
    def _optimize_single_run(self):
        """Single optimization run with cosine annealing scheduler"""
        
        # Setup logging for this run
        if self.logger is None:
            method_name = "EDM_Cosine_Annealing"
            self.logger = setup_logging(self.output_dir, method_name, self.trial_num)
        
        # Initialize parameters as torch tensors with gradients
        laser_energy = torch.tensor(self.laser_energy, device=self.device, requires_grad=True)
        pressure = torch.tensor(self.pressure, device=self.device, requires_grad=True)
        acquisition_time = torch.tensor(self.acquisition_time_ms, device=self.device, requires_grad=True)
        
        # Log initial parameters
        self.logger.info("="*60)
        self.logger.info("OPTIMIZATION INITIALIZATION")
        self.logger.info("="*60)
        self.logger.info(f"Initial parameters:")
        self.logger.info(f"  Laser energy: {laser_energy.item():.4f}")
        self.logger.info(f"  Pressure: {pressure.item():.4f}")
        self.logger.info(f"  Acquisition time: {acquisition_time.item():.4f}ms")
        self.logger.info(f"Optimization steps: {self.optimization_steps}")
        self.logger.info(f"Learning rate: {self.lr}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Cosine annealing - T_max: {self.T_max}, eta_min: {self.eta_min}")
        
        print(f"Initial parameters:")
        print(f"  Laser energy: {laser_energy.item():.2f}")
        print(f"  Pressure: {pressure.item():.2f}")
        print(f"  Acquisition time: {acquisition_time.item():.2f}ms")

        # Create optimizer for parameters
        optimizer = optim.Adam([laser_energy, pressure, acquisition_time], lr=self.lr)
        
        # Create cosine annealing scheduler
        scheduler = self.create_lr_scheduler(optimizer)
        
        # Track best result
        best_weighted_sum = -float('inf')
        
        # Initialize progress tracking
        progress_data = {
            'method': 'EDM_Cosine_Annealing',
            'trial_num': self.trial_num,
            'steps': [],
            'weighted_sums': [],
            'laser_energies': [],
            'pressures': [],
            'acquisition_times': [],
            'learning_rates': [],
            'losses': [],
            'parameter_changes': [],
            'best_so_far': [],
            'timestamps': [],
            'initial_params': {
                'laser_energy': float(laser_energy.item()),
                'pressure': float(pressure.item()),
                'acquisition_time': float(acquisition_time.item())
            },
            'hyperparameters': {
                'lr': self.lr,
                'optimization_steps': self.optimization_steps,
                'batch_size': self.batch_size,
                'T_max': self.T_max,
                'eta_min': self.eta_min
            }
        }
        
        self.logger.info("="*60)
        self.logger.info("STARTING OPTIMIZATION LOOP")
        self.logger.info("="*60)
        
        # Store previous parameters for change tracking
        prev_laser = laser_energy.item()
        prev_pressure = pressure.item()
        prev_acquisition = acquisition_time.item()
        
        # Run optimization
        for step in tqdm(range(self.optimization_steps), desc="Optimizing weighted sum with cosine annealing"):
            step_start_time = time.time()
            optimizer.zero_grad()
            
            # Create settings tensor for the model (batch dimension added)
            settings = torch.stack([laser_energy, pressure, acquisition_time]).unsqueeze(0)
            
            # Generate 1D spectra using differentiable EDM sampling
            x = self.sampler.sample_differentiable(
                resolution=self.spectrum_length,
                device=self.device,
                settings=settings,
                n_samples=self.batch_size,
                cfg_scale=3.0,
                settings_dim=len(self.features),
                smooth_output=True,
                smooth_kernel_size=5,
                smooth_sigma=2.0
            )
            
            # Save spectra every 10 steps
            if step % 10 == 0:
                self.save_spectra_batch(x, step)
            
            # Get deflection values for the spectrum length
            deflection_MeV_np = create_energy_axis(256, 62)[::-1]
            deflection_MeV = torch.tensor(deflection_MeV_np.copy())
            deflection_MeV = deflection_MeV.to(self.device)
            
            # For 1D spectra, we can directly use the generated values
            x_spectr = x.squeeze(1)  # Remove channel dimension to get (batch_size, spectrum_length)
            spectrum_intensity = x_spectr.mean(dim=0)  # Shape: (spectrum_length,)
            
            # Calculate weighted sum (our objective to maximize)
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            
            # Since we want to maximize, use negative for minimization
            loss = -weighted_sum
            
            # Get current parameter values
            current_laser = laser_energy.item()
            current_pressure = pressure.item()
            current_acquisition = acquisition_time.item()
            current_lr = scheduler.get_last_lr()[0] if scheduler else self.lr
            
            # Calculate parameter changes
            laser_change = current_laser - prev_laser
            pressure_change = current_pressure - prev_pressure
            acquisition_change = current_acquisition - prev_acquisition
            
            # Update best result tracking
            if weighted_sum.item() > best_weighted_sum:
                best_weighted_sum = weighted_sum.item()
                
                # Save best spectra
                if self.save_images:
                    self.save_spectra_batch(x, step, prefix="best")
                
                self.logger.info(f"NEW BEST at step {step}: {best_weighted_sum:.6f}")
            
            # Store progress data
            progress_data['steps'].append(step)
            progress_data['weighted_sums'].append(float(weighted_sum.item()))
            progress_data['laser_energies'].append(float(current_laser))
            progress_data['pressures'].append(float(current_pressure))
            progress_data['acquisition_times'].append(float(current_acquisition))
            progress_data['learning_rates'].append(float(current_lr))
            progress_data['losses'].append(float(loss.item()))
            progress_data['parameter_changes'].append({
                'laser_change': float(laser_change),
                'pressure_change': float(pressure_change),
                'acquisition_change': float(acquisition_change)
            })
            progress_data['best_so_far'].append(float(best_weighted_sum))
            progress_data['timestamps'].append(time.time())
            
            # Detailed logging every step (but only print every 5 steps to avoid spam)
            log_message = (f"Step {step:3d}: "
                          f"WSum={weighted_sum.item():.6f}, "
                          f"Loss={loss.item():.6f}, "
                          f"LR={current_lr:.6f} | "
                          f"Laser={current_laser:.4f} (Δ{laser_change:+.4f}), "
                          f"Press={current_pressure:.4f} (Δ{pressure_change:+.4f}), "
                          f"Time={current_acquisition:.4f} (Δ{acquisition_change:+.4f})ms")
            
            if step % 5 == 0 or step < 10 or step >= self.optimization_steps - 5:
                self.logger.info(log_message)
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: Weighted Sum = {weighted_sum.item():.4f}, "
                      f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
                print(f"  Params: Laser={current_laser:.4f}, "
                      f"Pressure={current_pressure:.4f}, "
                      f"Time={current_acquisition:.4f}ms")
                print(f"  Changes: Laser={laser_change:+.4f}, "
                      f"Pressure={pressure_change:+.4f}, "
                      f"Time={acquisition_change:+.4f}ms")
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update learning rate scheduler (cosine annealing)
            if scheduler is not None:
                scheduler.step()
            
            # Store history for visualization
            self.weighted_sum_history.append(weighted_sum.item())
            
            # Track the current parameter values for next iteration
            self.laser_energy_history.append(current_laser)
            self.pressure_history.append(current_pressure)
            self.acquisition_time_history.append(current_acquisition)
            
            # Update previous values for next iteration change calculation
            prev_laser = current_laser
            prev_pressure = current_pressure
            prev_acquisition = current_acquisition
            
            # Save spectrum periodically
            if step % 10 == 0 or step == self.optimization_steps - 1:
                self.spectrum_history.append({
                    'weighted_sum': weighted_sum.item(),
                    'spectrum': spectrum_intensity.detach().cpu().numpy(),
                    'energy_values': deflection_MeV.detach().cpu().numpy(),
                    'laser_energy': current_laser,
                    'pressure': current_pressure,
                    'acquisition_time': current_acquisition
                })
                
        # Final logging
        self.logger.info("="*60)
        self.logger.info("OPTIMIZATION COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Best weighted sum achieved: {best_weighted_sum:.6f}")
        self.logger.info(f"Final parameters:")
        self.logger.info(f"  Laser energy: {laser_energy.item():.4f}")
        self.logger.info(f"  Pressure: {pressure.item():.4f}")
        self.logger.info(f"  Acquisition time: {acquisition_time.item():.4f}ms")
        
        # Calculate total parameter changes
        total_laser_change = laser_energy.item() - progress_data['initial_params']['laser_energy']
        total_pressure_change = pressure.item() - progress_data['initial_params']['pressure']
        total_acquisition_change = acquisition_time.item() - progress_data['initial_params']['acquisition_time']
        
        self.logger.info(f"Total parameter changes:")
        self.logger.info(f"  Laser energy: {total_laser_change:+.4f}")
        self.logger.info(f"  Pressure: {total_pressure_change:+.4f}")
        self.logger.info(f"  Acquisition time: {total_acquisition_change:+.4f}ms")
        
        print(f"Optimization completed. Best weighted sum: {best_weighted_sum:.4f}")
        
        # Print final parameters
        print(f"Final parameters:")
        print(f"  Laser energy: {laser_energy.item():.2f}")
        print(f"  Pressure: {pressure.item():.2f}")
        print(f"  Acquisition time: {acquisition_time.item():.2f}ms")
        
        # Save detailed progress data
        if self.trial_num is not None:
            save_optimization_progress(self.output_dir, "EDM_Cosine_Annealing", 
                                     self.trial_num, progress_data)
        
        # Return optimization results
        return {
            'best_weighted_sum': best_weighted_sum,
            'final_laser_energy': laser_energy.item(),
            'final_pressure': pressure.item(),
            'final_acquisition_time': acquisition_time.item(),
            'initial_laser_energy': self.laser_energy,
            'initial_pressure': self.pressure,
            'initial_acquisition_time': self.acquisition_time_ms,
            'seed': self.seed,
            'convergence_history': self.weighted_sum_history.copy(),
            'lr_scheduler': 'cosine_annealing',
            'progress_data': progress_data
        }
    
    def plot_results(self):
        """Visualize the optimization results with multiple plots"""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Weighted Sum history
        axs[0, 0].plot(self.weighted_sum_history)
        axs[0, 0].set_title('Weighted Sum (Intensity × MeV) vs. Iterations')
        axs[0, 0].set_xlabel('Iteration')
        axs[0, 0].set_ylabel('Weighted Sum')
        axs[0, 0].grid(True)
        
        # Plot 2: Laser Energy parameter change
        axs[0, 1].plot(self.laser_energy_history)
        axs[0, 1].set_title('Laser Energy Parameter vs. Iterations')
        axs[0, 1].set_xlabel('Iteration')
        axs[0, 1].set_ylabel('Laser Energy')
        axs[0, 1].axhline(y=self.laser_energy, color='r', linestyle='--', label=f'Initial: {self.laser_energy}')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Plot 3: Pressure parameter change
        axs[1, 0].plot(self.pressure_history)
        axs[1, 0].set_title('Pressure Parameter vs. Iterations')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Pressure')
        axs[1, 0].axhline(y=self.pressure, color='r', linestyle='--', label=f'Initial: {self.pressure}')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Plot 4: Acquisition Time parameter change
        axs[1, 1].plot(self.acquisition_time_history)
        axs[1, 1].set_title('Acquisition Time Parameter vs. Iterations')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Acquisition Time (ms)')
        axs[1, 1].axhline(y=self.acquisition_time_ms, color='r', linestyle='--', label=f'Initial: {self.acquisition_time_ms}')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'optimization_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Optimization results plot saved to: {plot_path}")
        plt.show()
        
        # Create a separate figure for the final spectrum
        if len(self.spectrum_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            final_spec = self.spectrum_history[-1]
            ax.plot(final_spec['energy_values'], final_spec['spectrum'], 'b-', linewidth=2, label='Final Spectrum')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Intensity')
            ax.grid(True)
            
            title = f'Final Spectrum\n'
            title += f'Weighted Sum = {final_spec["weighted_sum"]:.4f}\n'
            title += f'Params: Laser={final_spec["laser_energy"]:.2f}, '
            title += f'Pressure={final_spec["pressure"]:.2f}, '
            title += f'Time={final_spec["acquisition_time"]:.2f}ms'
            ax.set_title(title)
            ax.legend()
            
            plt.tight_layout()
            final_plot_path = os.path.join(self.plots_dir, 'final_spectrum.png')
            plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            print(f"Final spectrum plot saved to: {final_plot_path}")
            plt.show()

            # Plot evolution of spectra
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.linspace(0, len(self.spectrum_history)-1, min(5, len(self.spectrum_history))).astype(int)
            
            for idx in indices:
                spec_data = self.spectrum_history[idx]
                step_idx = idx * 10 if idx < len(self.spectrum_history) - 1 else self.optimization_steps - 1
                ax.plot(
                    spec_data['energy_values'], 
                    spec_data['spectrum'], 
                    label=f'Step {step_idx}, Sum={spec_data["weighted_sum"]:.2f}'
                )
            
            ax.set_title('Spectrum Evolution')
            ax.set_xlabel('Energy (MeV)')
            ax.set_ylabel('Intensity')
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            evolution_plot_path = os.path.join(self.plots_dir, 'spectrum_evolution.png')
            plt.savefig(evolution_plot_path, dpi=300, bbox_inches='tight')
            print(f"Spectrum evolution plot saved to: {evolution_plot_path}")
            plt.show()

class BayesianSpectrumOptimizer:
    """
    Bayesian optimization approach for maximizing spectrum weighted sum.
    Uses Gaussian Process surrogate model to efficiently explore parameter space.
    """
    def __init__(
        self,
        model_path,
        device="cuda",
        pressure_bounds=(5.0, 30.0),
        laser_energy_bounds=(10.0, 50.0),
        acquisition_time_bounds=(5.0, 50.0),
        n_calls=100,
        n_initial_points=1,  # Changed default to 1
        batch_size=4,
        save_images=True,
        output_dir="bayesian_optimization_results",
        # EDM-specific parameters
        spectrum_length=256,
        features=["E", "P", "ms"],
        num_sampling_steps=18,
        # Add initial parameter values
        pressure_start=None,
        laser_energy_start=None,
        acquisition_time_start=None
    ):
        self.device = device
        self.pressure_bounds = pressure_bounds
        self.laser_energy_bounds = laser_energy_bounds
        self.acquisition_time_bounds = acquisition_time_bounds
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.batch_size = batch_size
        self.save_images = save_images
        self.output_dir = output_dir
        self.spectrum_length = spectrum_length
        self.features = features
        self.num_sampling_steps = num_sampling_steps
        
        # Set initial parameters if provided
        self.pressure = pressure_start if pressure_start is not None else np.random.uniform(pressure_bounds[0], pressure_bounds[1])
        self.laser_energy = laser_energy_start if laser_energy_start is not None else np.random.uniform(laser_energy_bounds[0], laser_energy_bounds[1])
        self.acquisition_time_ms = acquisition_time_start if acquisition_time_start is not None else np.random.uniform(acquisition_time_bounds[0], acquisition_time_bounds[1])
        
        print(f"Bayesian Optimization Setup:")
        print(f"  Spectrum length: {spectrum_length}")
        print(f"  Features: {features}")
        print(f"  Parameter bounds:")
        print(f"    Laser energy: {laser_energy_bounds}")
        print(f"    Pressure: {pressure_bounds}")
        print(f"    Acquisition time: {acquisition_time_bounds}")
        print(f"  Initial parameters:")
        print(f"    Laser energy: {self.laser_energy:.2f}")
        print(f"    Pressure: {self.pressure:.2f}")
        print(f"    Acquisition time: {self.acquisition_time_ms:.2f}ms")
        
        # Create directory structure
        if self.save_images or True:
            self.images_dir = os.path.join(self.output_dir, "images")
            self.plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)
            print(f"Results will be saved to: {self.output_dir}")
        
        # Initialize EDM model (non-differentiable version for efficiency)
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
        
        # Initialize standard EDM sampler (no need for differentiable version)
        self.sampler = EdmSampler(
            net=self.model,
            num_steps=num_sampling_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7
        )
        
        # Define optimization space
        self.dimensions = [
            Real(laser_energy_bounds[0], laser_energy_bounds[1], name='laser_energy'),
            Real(pressure_bounds[0], pressure_bounds[1], name='pressure'),
            Real(acquisition_time_bounds[0], acquisition_time_bounds[1], name='acquisition_time')
        ]
        
        # Initialize optimization history
        self.call_history = []
        self.best_params_history = []
        self.best_values_history = []
        self.evaluation_times = []
        
        # Setup logging
        self.logger = None  # Will be set up when optimization starts
        self.trial_num = None  # Will be set when used in multi-trial runs

    def cleanup(self):
        """
        Clean up GPU memory used by the optimizer.
        Call this when done with optimization to free GPU memory.
        """
        # Delete model and sampler to free GPU memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'sampler'):
            del self.sampler
        
        # Clear GPU cache
        cleanup_gpu_memory()
        print(f"Bayesian optimizer cleanup completed. {get_gpu_memory_info()}")
        
    def load_model(self, model_path):
        """Load the pre-trained EDM model"""
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"EDM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calculate_weighted_sum(self, spectrum, energy_values):
        """Calculate the weighted sum of intensities multiplied by MeV values."""
        spectrum = spectrum.to(self.device)
        energy_values = energy_values.to(self.device)
        weighted_sum = torch.sum(spectrum * energy_values)
        return weighted_sum

    def objective_function(self, params):
        """
        Objective function for Bayesian optimization.
        Returns negative weighted sum (since we minimize in skopt but want to maximize weighted sum).
        
        Args:
            params (list): [laser_energy, pressure, acquisition_time]
        """
        laser_energy, pressure, acquisition_time = params
        start_time = time.time()
        
        try:
            # Create settings tensor
            settings = torch.tensor([laser_energy, pressure, acquisition_time], device=self.device).unsqueeze(0)
            
            # Generate spectra using EDM sampler (non-differentiable, more efficient)
            with torch.no_grad():
                x = self.sampler.sample(
                    resolution=self.spectrum_length,
                    device=self.device,
                    settings=settings,
                    n_samples=self.batch_size,
                    cfg_scale=3.0,
                    settings_dim=len(self.features),
                    smooth_output=True,
                    smooth_kernel_size=5,
                    smooth_sigma=2.0
                )
            
            # Get deflection values
            deflection_MeV_np = create_energy_axis(256, 62)[::-1]
            # print(deflection_MeV_np.shape)
            deflection_MeV = torch.tensor(deflection_MeV_np.copy())
            deflection_MeV = deflection_MeV.to(self.device)
            x_spectr = x.squeeze(1)  # Remove channel dimension to get (batch_size, spectrum_length)
            spectrum_intensity = x_spectr.mean(dim=0)  # Shape: (spectrum_length,)
            
            # Calculate weighted sum
            weighted_sum = self.calculate_weighted_sum(spectrum_intensity, deflection_MeV)
            
            eval_time = time.time() - start_time
            self.evaluation_times.append(eval_time)
            
            # Store call history
            call_info = {
                'laser_energy': laser_energy,
                'pressure': pressure,
                'acquisition_time': acquisition_time,
                'weighted_sum': weighted_sum.item(),
                'spectrum': spectrum_intensity.detach().cpu().numpy(),
                'energy_values': deflection_MeV.detach().cpu().numpy(),
                'evaluation_time': eval_time
            }
            self.call_history.append(call_info)
            
            # Update best so far for logging
            best_so_far = max([call['weighted_sum'] for call in self.call_history])
            
            # Log evaluation details
            if self.logger:
                self.logger.info(f"Eval {len(self.call_history):3d}: "
                               f"Params=[L:{laser_energy:.4f}, P:{pressure:.4f}, T:{acquisition_time:.4f}], "
                               f"WSum={weighted_sum.item():.6f}, "
                               f"Best={best_so_far:.6f}, "
                               f"Time={eval_time:.2f}s")
            
            print(f"Eval {len(self.call_history)}: "
                  f"Params=[{laser_energy:.2f}, {pressure:.2f}, {acquisition_time:.2f}], "
                  f"Weighted_sum={weighted_sum.item():.4f}, "
                  f"Time={eval_time:.2f}s")
            
            # Return negative for minimization
            return -weighted_sum.item()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in objective function: {e}")
            print(f"Error in objective function: {e}")
            return 1e6  # Return large positive value (bad for minimization)

    def optimize(self):
        """Run Bayesian optimization"""
        
        # Setup logging for this run
        if self.logger is None:
            method_name = "Bayesian_Optimization"
            self.logger = setup_logging(self.output_dir, method_name, self.trial_num)
        
        self.logger.info("="*60)
        self.logger.info("BAYESIAN OPTIMIZATION INITIALIZATION")
        self.logger.info("="*60)
        self.logger.info(f"Function evaluations: {self.n_calls}")
        self.logger.info(f"Initial evaluations: {self.n_initial_points} (using specified starting point)")
        self.logger.info(f"Initial point: [L:{self.laser_energy:.4f}, P:{self.pressure:.4f}, T:{self.acquisition_time_ms:.4f}]")
        self.logger.info(f"Parameter bounds:")
        self.logger.info(f"  Laser energy: {self.laser_energy_bounds}")
        self.logger.info(f"  Pressure: {self.pressure_bounds}")
        self.logger.info(f"  Acquisition time: {self.acquisition_time_bounds}")
        self.logger.info(f"Batch size: {self.batch_size}")
        
        print(f"Starting Bayesian optimization with {self.n_calls} function evaluations...")
        print(f"Initial evaluation: 1 (using specified starting point)")
        print(f"Starting point: Laser={self.laser_energy:.2f}, Pressure={self.pressure:.2f}, Time={self.acquisition_time_ms:.2f}ms")
        
        # Update dimensions to match the bounds used in objective function
        dimensions = [
            Real(self.laser_energy_bounds[0], self.laser_energy_bounds[1], name='laser_energy'),
            Real(self.pressure_bounds[0], self.pressure_bounds[1], name='pressure'),
            Real(self.acquisition_time_bounds[0], self.acquisition_time_bounds[1], name='acquisition_time')
        ]
        
        start_time = time.time()
        
        self.logger.info("="*60)
        self.logger.info("STARTING BAYESIAN OPTIMIZATION")
        self.logger.info("="*60)
        
        # Create initial point from the specified starting parameters
        x0 = [self.laser_energy, self.pressure, self.acquisition_time_ms]
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            x0=[x0],  # Use our specific initial point
            acq_func='EI',  # Expected Improvement
            random_state=42
        )
        
        total_time = time.time() - start_time
        
        # Final logging
        self.logger.info("="*60)
        self.logger.info("BAYESIAN OPTIMIZATION COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Total optimization time: {total_time:.2f}s")
        self.logger.info(f"Best parameters found:")
        self.logger.info(f"  Laser energy: {result.x[0]:.4f}")
        self.logger.info(f"  Pressure: {result.x[1]:.4f}")
        self.logger.info(f"  Acquisition time: {result.x[2]:.4f}")
        self.logger.info(f"Best weighted sum: {-result.fun:.6f}")
        
        # Create progress data for consistency with EDM
        if self.call_history:
            weighted_sums = [call['weighted_sum'] for call in self.call_history]
            best_so_far = np.maximum.accumulate(weighted_sums)
            
            progress_data = {
                'method': 'Bayesian_Optimization',
                'trial_num': self.trial_num,
                'evaluations': list(range(len(self.call_history))),
                'weighted_sums': weighted_sums,
                'laser_energies': [call['laser_energy'] for call in self.call_history],
                'pressures': [call['pressure'] for call in self.call_history],
                'acquisition_times': [call['acquisition_time'] for call in self.call_history],
                'evaluation_times': [call['evaluation_time'] for call in self.call_history],
                'best_so_far': best_so_far.tolist(),
                'hyperparameters': {
                    'n_calls': self.n_calls,
                    'n_initial_points': self.n_initial_points,
                    'batch_size': self.batch_size,
                    'acquisition_function': 'EI'
                }
            }
            
            # Save detailed progress data
            if self.trial_num is not None:
                save_optimization_progress(self.output_dir, "Bayesian_Optimization", 
                                         self.trial_num, progress_data)
        
        print(f"Bayesian optimization completed in {total_time:.2f}s")
        print(f"Best parameters found:")
        print(f"  Laser energy: {result.x[0]:.2f}")
        print(f"  Pressure: {result.x[1]:.2f}")
        print(f"  Acquisition time: {result.x[2]:.2f}")
        print(f"Best weighted sum: {-result.fun:.4f}")
        
        self.result = result
        return result

    def plot_results(self):
        """Visualize Bayesian optimization results"""
        if not self.call_history:
            print("No optimization history to plot")
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Convergence
        weighted_sums = [call['weighted_sum'] for call in self.call_history]
        best_so_far = np.maximum.accumulate(weighted_sums)
        
        axs[0, 0].plot(weighted_sums, 'b.', alpha=0.6, label='Individual evaluations')
        axs[0, 0].plot(best_so_far, 'r-', linewidth=2, label='Best so far')
        axs[0, 0].set_title('Bayesian Optimization Convergence')
        axs[0, 0].set_xlabel('Function Evaluation')
        axs[0, 0].set_ylabel('Weighted Sum')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot 2: Parameter exploration - Laser Energy
        laser_energies = [call['laser_energy'] for call in self.call_history]
        axs[0, 1].plot(laser_energies, 'g.-')
        axs[0, 1].set_title('Laser Energy Exploration')
        axs[0, 1].set_xlabel('Function Evaluation')
        axs[0, 1].set_ylabel('Laser Energy')
        axs[0, 1].grid(True)
        
        # Plot 3: Parameter exploration - Pressure
        pressures = [call['pressure'] for call in self.call_history]
        axs[1, 0].plot(pressures, 'm.-')
        axs[1, 0].set_title('Pressure Exploration')
        axs[1, 0].set_xlabel('Function Evaluation')
        axs[1, 0].set_ylabel('Pressure')
        axs[1, 0].grid(True)
        
        # Plot 4: Parameter exploration - Acquisition Time
        acq_times = [call['acquisition_time'] for call in self.call_history]
        axs[1, 1].plot(acq_times, 'c.-')
        axs[1, 1].set_title('Acquisition Time Exploration')
        axs[1, 1].set_xlabel('Function Evaluation')
        axs[1, 1].set_ylabel('Acquisition Time (ms)')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'bayesian_optimization_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Bayesian optimization results saved to: {plot_path}")
        plt.show()
        
        # Plot best spectrum
        best_idx = np.argmax(weighted_sums)
        best_call = self.call_history[best_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(best_call['energy_values'], best_call['spectrum'], 'b-', linewidth=2)
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Intensity')
        ax.grid(True)
        
        title = f'Best Spectrum (Bayesian Optimization)\n'
        title += f'Weighted Sum = {best_call["weighted_sum"]:.4f}\n'
        title += f'Params: Laser={best_call["laser_energy"]:.2f}, '
        title += f'Pressure={best_call["pressure"]:.2f}, '
        title += f'Time={best_call["acquisition_time"]:.2f}ms'
        ax.set_title(title)
        
        plt.tight_layout()
        best_plot_path = os.path.join(self.plots_dir, 'best_spectrum_bayesian.png')
        plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
        print(f"Best spectrum plot saved to: {best_plot_path}")
        plt.show()

def compare_optimization_methods(edm_optimizer, bayesian_optimizer, output_dir="comparison_results"):
    """
    Compare the results of EDM gradient-based optimization vs Bayesian optimization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get results
    edm_weighted_sums = edm_optimizer.weighted_sum_history
    bayesian_weighted_sums = [call['weighted_sum'] for call in bayesian_optimizer.call_history]
    bayesian_best_so_far = np.maximum.accumulate(bayesian_weighted_sums)
    
    # Create comparison plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Convergence comparison
    axs[0, 0].plot(edm_weighted_sums, 'b-', linewidth=2, label='EDM Gradient-based', alpha=0.8)
    axs[0, 0].plot(bayesian_best_so_far, 'r-', linewidth=2, label='Bayesian Optimization', alpha=0.8)
    axs[0, 0].set_title('Optimization Method Comparison')
    axs[0, 0].set_xlabel('Iteration/Evaluation')
    axs[0, 0].set_ylabel('Best Weighted Sum So Far')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot 2: Final parameter comparison - Laser Energy
    edm_final_laser = edm_optimizer.laser_energy_history[-1] if edm_optimizer.laser_energy_history else 0
    bayesian_best_idx = np.argmax(bayesian_weighted_sums)
    bayesian_final_laser = bayesian_optimizer.call_history[bayesian_best_idx]['laser_energy']
    
    axs[0, 1].bar(['EDM', 'Bayesian'], [edm_final_laser, bayesian_final_laser], 
                  color=['blue', 'red'], alpha=0.7)
    axs[0, 1].set_title('Final Laser Energy Parameters')
    axs[0, 1].set_ylabel('Laser Energy')
    axs[0, 1].grid(True, axis='y')
    
    # Plot 3: Final parameter comparison - Pressure  
    edm_final_pressure = edm_optimizer.pressure_history[-1] if edm_optimizer.pressure_history else 0
    bayesian_final_pressure = bayesian_optimizer.call_history[bayesian_best_idx]['pressure']
    
    axs[1, 0].bar(['EDM', 'Bayesian'], [edm_final_pressure, bayesian_final_pressure],
                  color=['blue', 'red'], alpha=0.7)
    axs[1, 0].set_title('Final Pressure Parameters')
    axs[1, 0].set_ylabel('Pressure')
    axs[1, 0].grid(True, axis='y')
    
    # Plot 4: Final parameter comparison - Acquisition Time
    edm_final_time = edm_optimizer.acquisition_time_history[-1] if edm_optimizer.acquisition_time_history else 0
    bayesian_final_time = bayesian_optimizer.call_history[bayesian_best_idx]['acquisition_time']
    
    axs[1, 1].bar(['EDM', 'Bayesian'], [edm_final_time, bayesian_final_time],
                  color=['blue', 'red'], alpha=0.7)
    axs[1, 1].set_title('Final Acquisition Time Parameters')
    axs[1, 1].set_ylabel('Acquisition Time (ms)')
    axs[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'optimization_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Optimization comparison plot saved to: {comparison_path}")
    plt.show()
    
    # Print comparison summary
    print("\n" + "="*50)
    print("OPTIMIZATION METHODS COMPARISON")
    print("="*50)
    
    edm_best = max(edm_weighted_sums) if edm_weighted_sums else 0
    bayesian_best = max(bayesian_weighted_sums) if bayesian_weighted_sums else 0
    
    print(f"EDM Gradient-based Optimization Results ({len(edm_weighted_sums)} trials):")
    print(f"  Mean: {edm_best:.4f}")
    print(f"  Final parameters: Laser={edm_final_laser:.2f}, Pressure={edm_final_pressure:.2f}, Time={edm_final_time:.2f}ms")
    print(f"  Number of evaluations: {len(edm_weighted_sums)}")
    
    print(f"\nBayesian Optimization Results ({len(bayesian_weighted_sums)} trials):")
    print(f"  Mean: {bayesian_best:.4f}")
    print(f"  Final parameters: Laser={bayesian_final_laser:.2f}, Pressure={bayesian_final_pressure:.2f}, Time={bayesian_final_time:.2f}ms")
    print(f"  Number of evaluations: {len(bayesian_weighted_sums)}")
    
    print(f"\nComparison:")
    if bayesian_best > edm_best:
        improvement = ((bayesian_best - edm_best) / edm_best) * 100
        print(f"  Bayesian optimization achieved {improvement:.2f}% better result")
    elif edm_best > bayesian_best:
        improvement = ((edm_best - bayesian_best) / bayesian_best) * 100
        print(f"  EDM gradient-based achieved {improvement:.2f}% better result")
    else:
        print(f"  Both methods achieved similar results")
    
    print("="*50)

def run_multi_trial_comparison(
    model_path,
    device="cuda",
    n_trials=10,
    seeds=None,
    # EDM parameters
    edm_optimization_steps=50,
    edm_lr=0.4,
    # Bayesian parameters
    bayesian_n_calls=100,
    bayesian_n_initial_points=10,
    # Common parameters
    pressure_bounds=(15.0, 30.0),
    laser_energy_bounds=(15.0, 30.0),
    acquisition_time_bounds=(10.0, 30.0),
    spectrum_length=256,
    features=["E", "P", "ms"],
    num_sampling_steps=30,
    batch_size=16,
    output_dir="multi_trial_results"
):
    """
    Run multiple trials of both optimization methods and compare results.
    Each trial uses different random starting parameters within the specified bounds.
    
    Args:
        n_trials: Number of trials to run for each method
        seeds: List of seeds to use (if None, will generate random seeds)
        other args: Parameters for the optimization methods
    
    Returns:
        dict: Results summary with statistics
    """
    if seeds is None:
        seeds = [42 + i for i in range(n_trials)]
    elif len(seeds) < n_trials:
        # Extend seeds if not enough provided
        seeds.extend([max(seeds) + i + 1 for i in range(n_trials - len(seeds))])
    
    print(f"Running {n_trials} trials for each optimization method...")
    print(f"Seeds: {seeds[:n_trials]}")
    print("Each trial will use different random starting parameters within the bounds.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for results
    edm_results = []
    bayesian_results = []
    
    # Run EDM optimization trials
    print("\n" + "="*50)
    print("RUNNING EDM GRADIENT-BASED OPTIMIZATION TRIALS")
    print("="*50)
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials} (seed: {seeds[trial]})")
        print(f"Before trial: {get_gpu_memory_info()}")
        
        trial_output_dir = os.path.join(output_dir, f"edm_trial_{trial+1}")
        
        edm_optimizer = SpectrumWeightedSumOptimizer(
            model_path=model_path,
            device=device,
            pressure_bounds=pressure_bounds,
            laser_energy_bounds=laser_energy_bounds,
            acquisition_time_bounds=acquisition_time_bounds,
            optimization_steps=edm_optimization_steps,
            lr=edm_lr,
            batch_size=batch_size,
            save_images=False,  # Don't save images for all trials
            output_dir=trial_output_dir,
            spectrum_length=spectrum_length,
            features=features,
            num_sampling_steps=num_sampling_steps,
            seed=seeds[trial]
        )
        
        # Set trial number for logging
        edm_optimizer.trial_num = trial + 1
        
        start_time = time.time()
        result = edm_optimizer.optimize()
        end_time = time.time()
        
        result['trial'] = trial + 1
        result['execution_time'] = end_time - start_time
        result['method'] = 'EDM'
        edm_results.append(result)
        
        print(f"Trial {trial + 1} completed: Best weighted sum = {result['best_weighted_sum']:.4f}")
        
        # Explicit cleanup after each trial
        edm_optimizer.cleanup()
        del edm_optimizer
        cleanup_gpu_memory()
        print(f"After cleanup: {get_gpu_memory_info()}")
    
    # Run Bayesian optimization trials
    print("\n" + "="*50)
    print("RUNNING BAYESIAN OPTIMIZATION TRIALS")
    print("="*50)
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials} (seed: {seeds[trial]})")
        print(f"Before trial: {get_gpu_memory_info()}")
        
        # Set seed for reproducibility
        set_seed(seeds[trial])
        
        trial_output_dir = os.path.join(output_dir, f"bayesian_trial_{trial+1}")
        
        bayesian_optimizer = BayesianSpectrumOptimizer(
            model_path=model_path,
            device=device,
            pressure_bounds=pressure_bounds,
            laser_energy_bounds=laser_energy_bounds,
            acquisition_time_bounds=acquisition_time_bounds,
            # Let each trial generate its own random starting parameters
            pressure_start=None,
            laser_energy_start=None,
            acquisition_time_start=None,
            n_calls=bayesian_n_calls,
            n_initial_points=1,  # Use only one initial point
            batch_size=batch_size,
            save_images=False,
            output_dir=trial_output_dir,
            spectrum_length=spectrum_length,
            features=features,
            num_sampling_steps=num_sampling_steps
        )
        
        # Set trial number for logging
        bayesian_optimizer.trial_num = trial + 1
        
        start_time = time.time()
        skopt_result = bayesian_optimizer.optimize()
        end_time = time.time()
        
        # Get best result from Bayesian optimization
        best_weighted_sum = -skopt_result.fun
        best_params = skopt_result.x
        
        # Find the best call in history
        weighted_sums = [call['weighted_sum'] for call in bayesian_optimizer.call_history]
        best_idx = np.argmax(weighted_sums)
        best_call = bayesian_optimizer.call_history[best_idx]
        
        result = {
            'best_weighted_sum': best_weighted_sum,
            'final_laser_energy': best_params[0],
            'final_pressure': best_params[1],
            'final_acquisition_time': best_params[2],
            'initial_laser_energy': bayesian_optimizer.call_history[0]['laser_energy'],
            'initial_pressure': bayesian_optimizer.call_history[0]['pressure'],
            'initial_acquisition_time': bayesian_optimizer.call_history[0]['acquisition_time'],
            'trial': trial + 1,
            'execution_time': end_time - start_time,
            'method': 'Bayesian',
            'seed': seeds[trial],
            'convergence_history': weighted_sums.copy(),
            'n_evaluations': len(weighted_sums)
        }
        
        bayesian_results.append(result)
        
        print(f"Trial {trial + 1} completed: Best weighted sum = {result['best_weighted_sum']:.4f}")
        
        # Explicit cleanup after each trial
        bayesian_optimizer.cleanup()
        del bayesian_optimizer
        cleanup_gpu_memory()
        print(f"After cleanup: {get_gpu_memory_info()}")
    
    # Analyze and summarize results
    print("\n" + "="*50)
    print("ANALYZING RESULTS")
    print("="*50)
    
    summary = analyze_multi_trial_results(edm_results, bayesian_results, output_dir)
    
    return {
        'edm_results': edm_results,
        'bayesian_results': bayesian_results,
        'summary': summary
    }

def analyze_multi_trial_results(edm_results, bayesian_results, output_dir):
    """
    Analyze and summarize the results from multiple trials.
    """
    # Extract performance metrics
    edm_best_scores = [r['best_weighted_sum'] for r in edm_results]
    bayesian_best_scores = [r['best_weighted_sum'] for r in bayesian_results]
    
    edm_times = [r['execution_time'] for r in edm_results]
    bayesian_times = [r['execution_time'] for r in bayesian_results]
    
    # Calculate statistics
    edm_stats = {
        'mean': np.mean(edm_best_scores),
        'std': np.std(edm_best_scores),
        'median': np.median(edm_best_scores),
        'min': np.min(edm_best_scores),
        'max': np.max(edm_best_scores),
        'mean_time': np.mean(edm_times),
        'std_time': np.std(edm_times)
    }
    
    bayesian_stats = {
        'mean': np.mean(bayesian_best_scores),
        'std': np.std(bayesian_best_scores),
        'median': np.median(bayesian_best_scores),
        'min': np.min(bayesian_best_scores),
        'max': np.max(bayesian_best_scores),
        'mean_time': np.mean(bayesian_times),
        'std_time': np.std(bayesian_times)
    }
    
    # Statistical significance test
    t_stat, p_value = stats.ttest_ind(edm_best_scores, bayesian_best_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(edm_best_scores) - 1) * edm_stats['std']**2 + 
                         (len(bayesian_best_scores) - 1) * bayesian_stats['std']**2) / 
                        (len(edm_best_scores) + len(bayesian_best_scores) - 2))
    cohens_d = (edm_stats['mean'] - bayesian_stats['mean']) / pooled_std
    
    # Print detailed results
    print(f"EDM Gradient-based Optimization Results ({len(edm_results)} trials):")
    print(f"  Mean: {edm_stats['mean']:.4f} ± {edm_stats['std']:.4f}")
    print(f"  Median: {edm_stats['median']:.4f}")
    print(f"  Range: [{edm_stats['min']:.4f}, {edm_stats['max']:.4f}]")
    print(f"  Mean execution time: {edm_stats['mean_time']:.2f} ± {edm_stats['std_time']:.2f} seconds")
    
    print(f"\nBayesian Optimization Results ({len(bayesian_results)} trials):")
    print(f"  Mean: {bayesian_stats['mean']:.4f} ± {bayesian_stats['std']:.4f}")
    print(f"  Median: {bayesian_stats['median']:.4f}")
    print(f"  Range: [{bayesian_stats['min']:.4f}, {bayesian_stats['max']:.4f}]")
    print(f"  Mean execution time: {bayesian_stats['mean_time']:.2f} ± {bayesian_stats['std_time']:.2f} seconds")
    
    print(f"\nStatistical Comparison:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")
    
    if p_value < 0.05:
        better_method = "EDM" if edm_stats['mean'] > bayesian_stats['mean'] else "Bayesian"
        print(f"  Result: {better_method} optimization is significantly better (p < 0.05)")
    else:
        print(f"  Result: No significant difference between methods (p >= 0.05)")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Effect size: {effect_size}")
    
    # Win rate
    edm_wins = sum(1 for i in range(len(edm_results)) if edm_best_scores[i] > bayesian_best_scores[i])
    bayesian_wins = sum(1 for i in range(len(bayesian_results)) if bayesian_best_scores[i] > edm_best_scores[i])
    ties = len(edm_results) - edm_wins - bayesian_wins
    
    print(f"\nHead-to-head comparison:")
    print(f"  EDM wins: {edm_wins}/{len(edm_results)} ({edm_wins/len(edm_results)*100:.1f}%)")
    print(f"  Bayesian wins: {bayesian_wins}/{len(bayesian_results)} ({bayesian_wins/len(bayesian_results)*100:.1f}%)")
    print(f"  Ties: {ties}")
    
    # Create summary plots
    plot_multi_trial_results(edm_results, bayesian_results, edm_stats, bayesian_stats, output_dir)
    
    # Save detailed results to CSV
    save_detailed_results(edm_results, bayesian_results, output_dir)
    
    summary = {
        'edm_stats': edm_stats,
        'bayesian_stats': bayesian_stats,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        'edm_wins': edm_wins,
        'bayesian_wins': bayesian_wins,
        'ties': ties,
        'total_trials': len(edm_results)
    }
    
    return summary

def plot_multi_trial_results(edm_results, bayesian_results, edm_stats, bayesian_stats, output_dir):
    """Create comprehensive plots for multi-trial results"""
    
    # Extract data
    edm_scores = [r['best_weighted_sum'] for r in edm_results]
    bayesian_scores = [r['best_weighted_sum'] for r in bayesian_results]
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Box plots comparison
    axs[0, 0].boxplot([edm_scores, bayesian_scores], labels=['EDM', 'Bayesian'])
    axs[0, 0].set_title('Performance Comparison (Box Plot)')
    axs[0, 0].set_ylabel('Best Weighted Sum')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram comparison
    axs[0, 1].hist(edm_scores, alpha=0.7, label='EDM', bins=10, color='blue')
    axs[0, 1].hist(bayesian_scores, alpha=0.7, label='Bayesian', bins=10, color='red')
    axs[0, 1].set_title('Performance Distribution')
    axs[0, 1].set_xlabel('Best Weighted Sum')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Trial-by-trial comparison
    trials = range(1, len(edm_results) + 1)
    axs[0, 2].plot(trials, edm_scores, 'bo-', label='EDM', alpha=0.7)
    axs[0, 2].plot(trials, bayesian_scores, 'ro-', label='Bayesian', alpha=0.7)
    axs[0, 2].set_title('Trial-by-Trial Performance')
    axs[0, 2].set_xlabel('Trial Number')
    axs[0, 2].set_ylabel('Best Weighted Sum')
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Execution time comparison
    edm_times = [r['execution_time'] for r in edm_results]
    bayesian_times = [r['execution_time'] for r in bayesian_results]
    axs[1, 0].boxplot([edm_times, bayesian_times], labels=['EDM', 'Bayesian'])
    axs[1, 0].set_title('Execution Time Comparison')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Convergence curves (sample)
    axs[1, 1].set_title('Sample Convergence Curves')
    
    # Plot convergence for first 3 trials of each method
    for i in range(min(3, len(edm_results))):
        if 'convergence_history' in edm_results[i]:
            axs[1, 1].plot(edm_results[i]['convergence_history'], 
                          color='blue', alpha=0.5, linewidth=1)
    
    for i in range(min(3, len(bayesian_results))):
        if 'convergence_history' in bayesian_results[i]:
            # For Bayesian, plot best-so-far
            convergence = bayesian_results[i]['convergence_history']
            best_so_far = np.maximum.accumulate(convergence)
            axs[1, 1].plot(best_so_far, color='red', alpha=0.5, linewidth=1)
    
    axs[1, 1].set_xlabel('Iteration/Evaluation')
    axs[1, 1].set_ylabel('Best Weighted Sum')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='EDM'),
                      Line2D([0], [0], color='red', lw=2, label='Bayesian')]
    axs[1, 1].legend(handles=legend_elements)
    
    # Plot 6: Summary statistics
    axs[1, 2].bar(['EDM Mean', 'Bayesian Mean'], 
                  [edm_stats['mean'], bayesian_stats['mean']], 
                  yerr=[edm_stats['std'], bayesian_stats['std']], 
                  color=['blue', 'red'], alpha=0.7, capsize=5)
    axs[1, 2].set_title('Mean Performance ± Std Dev')
    axs[1, 2].set_ylabel('Best Weighted Sum')
    axs[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'multi_trial_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Multi-trial comparison plot saved to: {plot_path}")
    plt.show()

def save_detailed_results(edm_results, bayesian_results, output_dir):
    """Save detailed results to CSV files"""
    
    # Convert results to DataFrames
    edm_df = pd.DataFrame(edm_results)
    bayesian_df = pd.DataFrame(bayesian_results)
    
    # Save to CSV
    edm_path = os.path.join(output_dir, 'edm_detailed_results.csv')
    bayesian_path = os.path.join(output_dir, 'bayesian_detailed_results.csv')
    
    edm_df.to_csv(edm_path, index=False)
    bayesian_df.to_csv(bayesian_path, index=False)
    
    print(f"Detailed EDM results saved to: {edm_path}")
    print(f"Detailed Bayesian results saved to: {bayesian_path}")
    
    # Create combined summary
    summary_data = []
    
    for result in edm_results:
        summary_data.append({
            'Method': 'EDM',
            'Trial': result['trial'],
            'Best_Weighted_Sum': result['best_weighted_sum'],
            'Final_Laser_Energy': result['final_laser_energy'],
            'Final_Pressure': result['final_pressure'],
            'Final_Acquisition_Time': result['final_acquisition_time'],
            'Execution_Time': result['execution_time'],
            'Seed': result['seed']
        })
    
    for result in bayesian_results:
        summary_data.append({
            'Method': 'Bayesian',
            'Trial': result['trial'],
            'Best_Weighted_Sum': result['best_weighted_sum'],
            'Final_Laser_Energy': result['final_laser_energy'],
            'Final_Pressure': result['final_pressure'],
            'Final_Acquisition_Time': result['final_acquisition_time'],
            'Execution_Time': result['execution_time'],
            'Seed': result['seed']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'optimization_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Combined summary saved to: {summary_path}")

def run_comprehensive_optimization_comparison(
    model_path,
    device="cuda",
    n_trials=5,
    seeds=None,
    # EDM parameters
    edm_optimization_steps=100,
    edm_lr=0.1,
    # Bayesian parameters
    bayesian_n_calls=100,
    bayesian_n_initial_points=10,
    # Common parameters
    pressure_bounds=(10.0, 50.0),
    laser_energy_bounds=(10.0, 50.0),
    acquisition_time_bounds=(5.0, 50.0),
    spectrum_length=256,
    features=["E", "P", "ms"],
    num_sampling_steps=30,
    batch_size=16,
    output_dir="comprehensive_comparison_results"
):
    """
    Run comprehensive comparison of all optimization methods:
    - EDM baseline (no exploration)
    - EDM with all exploration techniques
    - Bayesian optimization
    
    Each method is run with multiple seeds for statistical significance.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]  # Default 5 seeds
    elif len(seeds) < n_trials:
        seeds.extend([max(seeds) + i + 1 for i in range(n_trials - len(seeds))])
    
    print("="*80)
    print("COMPREHENSIVE OPTIMIZATION METHODS COMPARISON")  
    print("="*80)
    print(f"Running {n_trials} trials for each method with seeds: {seeds[:n_trials]}")
    print(f"Methods to compare:")
    print("  1. EDM with Cosine Annealing")
    print("  2. Bayesian Optimization")
    print("Each trial will use different random starting parameters within the bounds.")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define EDM configuration with cosine annealing
    edm_config = {
        "EDM_Cosine_Annealing": {
            "use_cosine_annealing": True,
            "T_max": 50,  # Match optimization_steps
            "eta_min": 1e-6,
            "description": "EDM optimization with cosine annealing scheduler"
        }
    }
    
    # Storage for all results
    all_results = {}
    
    # Generate different starting parameters for each trial
    trial_starting_params = []
    for trial in range(n_trials):
        # Set seed to ensure reproducible starting points for each trial
        np.random.seed(seeds[trial])
        
        trial_params = {
            'laser_energy': np.random.uniform(laser_energy_bounds[0], laser_energy_bounds[1]),
            'pressure': np.random.uniform(pressure_bounds[0], pressure_bounds[1]),
            'acquisition_time': np.random.uniform(acquisition_time_bounds[0], acquisition_time_bounds[1])
        }
        trial_starting_params.append(trial_params)
        
        print(f"Trial {trial + 1} starting parameters:")
        print(f"  Laser energy: {trial_params['laser_energy']:.2f}")
        print(f"  Pressure: {trial_params['pressure']:.2f}")
        print(f"  Acquisition time: {trial_params['acquisition_time']:.2f}ms")
    print()
    
    # Run EDM with cosine annealing
    print("RUNNING EDM WITH COSINE ANNEALING")
    print("="*50)
    
    config_name = "EDM_Cosine_Annealing"
    config = edm_config[config_name]
    
    print(f"\n{'-'*40}")
    print(f"Testing: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'-'*40}")
    
    method_results = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials} (seed: {seeds[trial]})")
        
        # Print memory status before trial
        print(f"    Before trial: {get_gpu_memory_info()}")
        
        trial_output_dir = os.path.join(output_dir, f"{config_name.lower()}_trial_{trial+1}")
        
        # Extract config parameters (excluding description)
        optimizer_params = {k: v for k, v in config.items() if k != "description"}
        
        # Get starting parameters for this trial
        start_params = trial_starting_params[trial]
        
        optimizer = SpectrumWeightedSumOptimizer(
            model_path=model_path,
            device=device,
            pressure_bounds=pressure_bounds,
            laser_energy_bounds=laser_energy_bounds,
            acquisition_time_bounds=acquisition_time_bounds,
            # Use different starting parameters for each trial
            pressure_start=start_params['pressure'],
            laser_energy_start=start_params['laser_energy'],
            acquisition_time_start=start_params['acquisition_time'],
            optimization_steps=edm_optimization_steps,
            lr=edm_lr,
            batch_size=batch_size,
            save_images=False,  # Don't save images for all trials
            output_dir=trial_output_dir,
            spectrum_length=spectrum_length,
            features=features,
            num_sampling_steps=num_sampling_steps,
            seed=seeds[trial],
            **optimizer_params
        )
        
        # Set trial number for logging
        optimizer.trial_num = trial + 1
        
        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()
        
        result['trial'] = trial + 1
        result['execution_time'] = end_time - start_time
        result['method'] = config_name
        result['description'] = config['description']
        method_results.append(result)
        
        print(f"    Result: {result['best_weighted_sum']:.4f} (time: {result['execution_time']:.1f}s)")
        
        # Explicit cleanup after each trial
        optimizer.cleanup()
        del optimizer
        
        print(f"    After cleanup: {get_gpu_memory_info()}")
    
    all_results[config_name] = method_results
    
    # Additional cleanup after method
    cleanup_gpu_memory()
    
    # Print summary for this method
    scores = [r['best_weighted_sum'] for r in method_results]
    print(f"  {config_name} Summary:")
    print(f"    Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"    Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"    Memory after method: {get_gpu_memory_info()}")
    
    # Run Bayesian optimization
    print(f"\n{'-'*40}")
    print("Testing: Bayesian Optimization")
    print("Description: Gaussian Process surrogate model with Expected Improvement")
    print(f"{'-'*40}")
    
    bayesian_results = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials} (seed: {seeds[trial]})")
        
        # Print memory status before trial
        print(f"    Before trial: {get_gpu_memory_info()}")
        
        # Set seed for reproducibility
        set_seed(seeds[trial])
        
        trial_output_dir = os.path.join(output_dir, f"bayesian_trial_{trial+1}")
        
        # Get starting parameters for this trial
        start_params = trial_starting_params[trial]
        
        bayesian_optimizer = BayesianSpectrumOptimizer(
            model_path=model_path,
            device=device,
            pressure_bounds=pressure_bounds,
            laser_energy_bounds=laser_energy_bounds,
            acquisition_time_bounds=acquisition_time_bounds,
            # Use different starting parameters for each trial
            pressure_start=start_params['pressure'],
            laser_energy_start=start_params['laser_energy'],
            acquisition_time_start=start_params['acquisition_time'],
            n_calls=bayesian_n_calls,
            n_initial_points=1,  # Use only one initial point
            batch_size=batch_size,
            save_images=False,
            output_dir=trial_output_dir,
            spectrum_length=spectrum_length,
            features=features,
            num_sampling_steps=num_sampling_steps
        )
        
        # Set trial number for logging
        bayesian_optimizer.trial_num = trial + 1
        
        start_time = time.time()
        skopt_result = bayesian_optimizer.optimize()
        end_time = time.time()
        
        # Get best result from Bayesian optimization
        best_weighted_sum = -skopt_result.fun
        best_params = skopt_result.x
        
        # Find the best call in history
        weighted_sums = [call['weighted_sum'] for call in bayesian_optimizer.call_history]
        best_idx = np.argmax(weighted_sums)
        best_call = bayesian_optimizer.call_history[best_idx]
        
        result = {
            'best_weighted_sum': best_weighted_sum,
            'final_laser_energy': best_params[0],
            'final_pressure': best_params[1],
            'final_acquisition_time': best_params[2],
            'initial_laser_energy': bayesian_optimizer.call_history[0]['laser_energy'],
            'initial_pressure': bayesian_optimizer.call_history[0]['pressure'],
            'initial_acquisition_time': bayesian_optimizer.call_history[0]['acquisition_time'],
            'trial': trial + 1,
            'execution_time': end_time - start_time,
            'method': 'Bayesian_Optimization',
            'description': 'Gaussian Process surrogate model with Expected Improvement',
            'seed': seeds[trial],
            'convergence_history': weighted_sums.copy(),
            'n_evaluations': len(weighted_sums)
        }
        
        bayesian_results.append(result)
        print(f"    Result: {result['best_weighted_sum']:.4f} (time: {result['execution_time']:.1f}s)")
        
        # Explicit cleanup after each trial
        bayesian_optimizer.cleanup()
        del bayesian_optimizer
        cleanup_gpu_memory()
        print(f"    After cleanup: {get_gpu_memory_info()}")
    
    all_results['Bayesian_Optimization'] = bayesian_results
    
    # Final cleanup after Bayesian optimization
    cleanup_gpu_memory()
    
    # Print summary for Bayesian
    bayesian_scores = [r['best_weighted_sum'] for r in bayesian_results]
    print(f"  Bayesian Optimization Summary:")
    print(f"    Mean: {np.mean(bayesian_scores):.4f} ± {np.std(bayesian_scores):.4f}")
    print(f"    Range: [{np.min(bayesian_scores):.4f}, {np.max(bayesian_scores):.4f}]")
    print(f"    Final memory: {get_gpu_memory_info()}")
    
    # Comprehensive analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    analyze_comprehensive_results(all_results, output_dir)
    
    return all_results

def analyze_comprehensive_results(all_results, output_dir):
    """
    Analyze and visualize results from all optimization methods.
    """
    # Extract summary statistics for each method
    method_stats = {}
    
    print(f"{'Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print("-" * 80)
    
    for method_name, results in all_results.items():
        scores = [r['best_weighted_sum'] for r in results]
        times = [r['execution_time'] for r in results]
        
        stat_dict = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'results': results
        }
        
        method_stats[method_name] = stat_dict
        
        print(f"{method_name:<25} {stat_dict['mean']:<10.4f} {stat_dict['std']:<10.4f} "
              f"{stat_dict['min']:<10.4f} {stat_dict['max']:<10.4f} {stat_dict['median']:<10.4f}")
    
    # Statistical significance testing (ANOVA + post-hoc)
    print(f"\nSTATISTICAL SIGNIFICANCE TESTING")
    print("-" * 50)
    
    # Prepare data for ANOVA
    all_scores = []
    method_labels = []
    
    for method_name, stat_dict in method_stats.items():
        all_scores.extend(stat_dict['scores'])
        method_labels.extend([method_name] * len(stat_dict['scores']))
    
    # Perform one-way ANOVA
    method_names = list(method_stats.keys())
    score_groups = [method_stats[name]['scores'] for name in method_names]
    
    f_stat, p_value = stats.f_oneway(*score_groups)
    print(f"One-way ANOVA:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"  Result: Significant differences between methods (p < 0.05)")
    else:
        print(f"  Result: No significant differences between methods (p >= 0.05)")
    
    # Find best performing method
    best_method = max(method_stats.items(), key=lambda x: x[1]['mean'])
    print(f"\nBest performing method: {best_method[0]}")
    print(f"  Mean score: {best_method[1]['mean']:.4f} ± {best_method[1]['std']:.4f}")
    
    # Pairwise comparisons with baseline
    baseline_scores = method_stats.get('EDM_Cosine_Annealing', {}).get('scores', [])
    if baseline_scores:
        print(f"\nComparisons with EDM Cosine Annealing:")
        print(f"{'Method':<25} {'Improvement':<12} {'p-value':<10} {'Significance':<12}")
        print("-" * 65)
        
        for method_name, method_stat in method_stats.items():
            if method_name == 'EDM_Cosine_Annealing':
                continue
                
            method_scores = method_stat['scores']
            
            # T-test comparison
            t_stat, p_val = stats.ttest_ind(method_scores, baseline_scores)
            
            # Calculate improvement
            baseline_mean = np.mean(baseline_scores)
            method_mean = method_stat['mean']
            improvement = ((method_mean - baseline_mean) / baseline_mean) * 100
            
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            print(f"{method_name:<25} {improvement:+8.1f}%   {p_val:<10.6f} {significance:<12}")
    
    # Create comprehensive visualization
    create_comprehensive_plots(method_stats, output_dir)
    
    # Save detailed results
    save_comprehensive_results(all_results, method_stats, output_dir)

def create_comprehensive_plots(method_stats, output_dir):
    """Create comprehensive visualization plots for all methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    method_names = list(method_stats.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    
    # Plot 1: Box plot comparison
    ax = axes[0, 0]
    score_data = [method_stats[name]['scores'] for name in method_names]
    bp = ax.boxplot(score_data, labels=[name.replace('_', '\n') for name in method_names], 
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Performance Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Weighted Sum')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean performance with error bars
    ax = axes[0, 1]
    means = [method_stats[name]['mean'] for name in method_names]
    stds = [method_stats[name]['std'] for name in method_names]
    
    bars = ax.bar(range(len(method_names)), means, yerr=stds, 
                  color=colors, alpha=0.7, capsize=5)
    ax.set_title('Mean Performance ± Standard Deviation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Weighted Sum')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Highlight best method
    best_idx = means.index(max(means))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2)
    
    # Plot 3: Execution time comparison
    ax = axes[0, 2]
    mean_times = [method_stats[name]['mean_time'] for name in method_names]
    std_times = [method_stats[name]['std_time'] for name in method_names]
    
    ax.bar(range(len(method_names)), mean_times, yerr=std_times,
           color=colors, alpha=0.7, capsize=5)
    ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance distribution (histogram)
    ax = axes[1, 0]
    for i, (name, stat_data) in enumerate(method_stats.items()):
        ax.hist(stat_data['scores'], alpha=0.6, label=name.replace('_', ' '), 
                color=colors[i], bins=8)
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Best Weighted Sum')
    ax.set_ylabel('Frequency')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Performance vs Time scatter
    ax = axes[1, 1]
    for i, (name, stat_data) in enumerate(method_stats.items()):
        scores = stat_data['scores']
        times = [r['execution_time'] for r in stat_data['results']]
        ax.scatter(times, scores, color=colors[i], label=name.replace('_', ' '), 
                  alpha=0.7, s=60)
    ax.set_title('Performance vs Execution Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Execution Time (seconds)')
    ax.set_ylabel('Best Weighted Sum')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Sample convergence curves
    ax = axes[1, 2]
    for i, (name, stat_data) in enumerate(method_stats.items()):
        # Plot convergence for first trial of each method
        first_result = stat_data['results'][0]
        if 'convergence_history' in first_result:
            convergence = first_result['convergence_history']
            if name == 'Bayesian_Optimization':
                # For Bayesian, plot best-so-far
                convergence = np.maximum.accumulate(convergence)
            ax.plot(convergence, color=colors[i], label=name.replace('_', ' '), 
                   alpha=0.8, linewidth=2)
    
    ax.set_title('Sample Convergence Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration/Evaluation')
    ax.set_ylabel('Best Weighted Sum')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive comparison plot saved to: {plot_path}")
    plt.show()

def save_comprehensive_results(all_results, method_stats, output_dir):
    """Save comprehensive results to files."""
    
    # Create summary DataFrame
    summary_data = []
    for method_name, stat_data in method_stats.items():
        summary_data.append({
            'Method': method_name,
            'Mean_Score': stat_data['mean'],
            'Std_Score': stat_data['std'],
            'Min_Score': stat_data['min'],
            'Max_Score': stat_data['max'],
            'Median_Score': stat_data['median'],
            'Mean_Time': stat_data['mean_time'],
            'Std_Time': stat_data['std_time']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'comprehensive_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Create detailed results DataFrame
    detailed_data = []
    for method_name, results in all_results.items():
        for result in results:
            detailed_data.append({
                'Method': method_name,
                'Trial': result['trial'],
                'Best_Weighted_Sum': result['best_weighted_sum'],
                'Final_Laser_Energy': result['final_laser_energy'],
                'Final_Pressure': result['final_pressure'],
                'Final_Acquisition_Time': result['final_acquisition_time'],
                'Execution_Time': result['execution_time'],
                'Seed': result['seed']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = os.path.join(output_dir, 'comprehensive_detailed_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")

def main():
    # Set parameters
    model_path = "models/edm_1d_spectrum_256pts_instancenorm_fixeddataset_10kEpochs/ema_ckpt_final.pt"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("EDM COSINE ANNEALING vs BAYESIAN OPTIMIZATION COMPARISON")
    print("="*80)
    print("This will run a complete comparison of:")
    print("  • EDM with Cosine Annealing")
    print("  • Bayesian Optimization")
    print()
    print("Each method will be run 5 times with different seeds for statistical significance.")
    print("Each trial will start from different randomly generated parameter values within the bounds.")
    print()
    print("Cosine Annealing Details:")
    print("  • T_max = 50 (total number of steps for annealing schedule)")
    print("  • eta_min = 1e-6 (minimum learning rate)")
    print("  • Gradually reduces learning rate following cosine curve")
    print()
    print("Bayesian Optimization Details:")
    print("  • Uses exactly 1 initial point (different random point for each trial)")
    print("  • Gaussian Process surrogate model with Expected Improvement")
    print()
    
    # Run comprehensive comparison
    results = run_comprehensive_optimization_comparison(
        model_path=model_path,
        device=device,
        n_trials=5,
        seeds=[42, 123, 456, 789, 999],
        edm_optimization_steps=50,
        edm_lr=2,
        # Bayesian parameters 
        bayesian_n_calls=50,
        bayesian_n_initial_points=1,  # Now uses 1 initial point (different for each trial)
        # Common parameters
        pressure_bounds=(10.0, 30.0),
        laser_energy_bounds=(10.0, 40.0),
        acquisition_time_bounds=(10.0, 40.0),
        spectrum_length=256,
        features=["E", "P", "ms"],
        num_sampling_steps=30,
        batch_size=16,
        output_dir="cosine_annealing_vs_bayesian_results"
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    
    # Print final recommendations
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Find best method overall
    method_means = {}
    for method_name, method_results in results.items():
        scores = [r['best_weighted_sum'] for r in method_results]
        method_means[method_name] = np.mean(scores)
    
    best_method = max(method_means.items(), key=lambda x: x[1])
    
    print(f"🏆 Best performing method: {best_method[0]}")
    print(f"   Average score: {best_method[1]:.4f}")
    
    edm_mean = method_means.get('EDM_Cosine_Annealing', 0)
    bayesian_mean = method_means.get('Bayesian_Optimization', 0)
    
    print(f"📊 EDM Cosine Annealing: {edm_mean:.4f}")
    print(f"📊 Bayesian Optimization: {bayesian_mean:.4f}")
    
    # Calculate difference
    if edm_mean > 0 and bayesian_mean > 0:
        if edm_mean >= bayesian_mean:
            improvement = ((edm_mean - bayesian_mean) / bayesian_mean) * 100
            print(f"📈 EDM Cosine Annealing is {improvement:.1f}% better")
        else:
            improvement = ((bayesian_mean - edm_mean) / edm_mean) * 100
            print(f"📈 Bayesian Optimization is {improvement:.1f}% better")
    
    print(f"\nAll results saved to: cosine_annealing_vs_bayesian_results/")
    print("Check the generated plots and CSV files for detailed analysis!")
    
    print("\n" + "="*80)
    print("ABOUT THE OPTIMIZATION METHODS:")
    print("="*80)
    print("EDM COSINE ANNEALING:")
    print("• Gradient-based optimization with differentiable EDM sampling")
    print("• Gradually reduces learning rate following a cosine curve")
    print("• T_max controls the total number of steps for the annealing schedule")
    print("• eta_min sets the minimum learning rate at the end of the schedule")
    print("• Provides smooth learning rate decay without periodic restarts")
    print("• More stable convergence compared to warm restarts")
    print()
    print("BAYESIAN OPTIMIZATION:")
    print("• Black-box optimization using Gaussian Process surrogate model")
    print("• Uses Expected Improvement acquisition function")
    print("• Starts with exactly 1 initial evaluation (same point as EDM)")
    print("• Explores parameter space efficiently with uncertainty quantification")
    print("• Good for expensive function evaluations")
    print("="*80)

if __name__ == "__main__":
    main()
