import torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from pathlib import Path
import os
import sys
from typing import Optional, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path if needed
# sys.path.append('.')

# Import your modules (adjust paths as needed)
from src.modules_1d import EDMPrecond
from src.diffusion import EdmSampler, transform_vector, gaussian_smooth_1d
from src.utils import deflection_biexp_calc

class SpectrumSamplingTool:
    """
    Interactive spectrum sampling tool for Jupyter notebooks.
    Allows users to select conditional parameters and visualize generated spectra.
    """
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        spectrum_length: int = 256,
        features: List[str] = ["E", "P", "ms"],
        num_sampling_steps: int = 30,
        sigma_data: float = 0.112
    ):
        """
        Initialize the spectrum sampling tool.
        
        Args:
            model_path: Path to the trained EDM model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            spectrum_length: Length of generated spectra (default: 256)
            features: List of feature names for conditioning (default: ["E", "P", "ms"])
            num_sampling_steps: Number of steps for sampling (default: 30)
            sigma_data: Sigma data parameter for the model (default: 0.112)
        """
        self.device = device
        self.spectrum_length = spectrum_length
        self.features = features
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        
        print(f"Initializing Spectrum Sampling Tool...")
        print(f"Device: {device}")
        print(f"Spectrum length: {spectrum_length}")
        print(f"Features: {features}")
        print(f"Sampling steps: {num_sampling_steps}")
        
        # Load the model
        self.model = self._load_model(model_path)
        
        # Initialize sampler
        self.sampler = EdmSampler(
            net=self.model,
            num_steps=num_sampling_steps,
            sigma_min=0.002,
            sigma_max=80,
            rho=7
        )
        
        # Create energy axis
        self.energy_axis = self._create_energy_axis()
        
        # Storage for generated samples
        self.current_samples = None
        self.sample_history = []
        
        # Initialize widgets
        self._create_widgets()
        
        print("✅ Initialization complete!")
    
    def _load_model(self, model_path: str) -> EDMPrecond:
        """Load the pre-trained EDM model."""
        try:
            model = EDMPrecond(
                resolution=self.spectrum_length,
                settings_dim=len(self.features),
                sigma_min=0,
                sigma_max=float('inf'),
                sigma_data=self.sigma_data,
                model_type='UNet_conditional',
                device=self.device
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            print(f"✅ Model loaded from: {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _create_energy_axis(self, electron_pointing_pixel: int = 62) -> np.ndarray:
        """Create energy axis for plotting using biexponential deflection calculation."""
        try:
            # Use a larger size to ensure we get enough high-energy values
            temp_size = max(self.spectrum_length * 2, 512)
            
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
            
            # Sort in descending order to get highest energies first
            valid_energies_sorted = np.sort(valid_energies)[::-1]
            
            # Take the top values (highest energies)
            if len(valid_energies_sorted) >= self.spectrum_length:
                energy_axis = valid_energies_sorted[:self.spectrum_length]
            else:
                # If we don't have enough values, pad with zeros
                energy_axis = np.concatenate([
                    np.zeros(self.spectrum_length - len(valid_energies_sorted)),
                    valid_energies_sorted[::-1]
                ])
            
            return energy_axis
            
        except Exception as e:
            print(f"⚠️ Could not create energy axis from deflection calculation: {e}")
            print("Using linear energy axis as fallback")
            return np.linspace(0, 100, self.spectrum_length)  # Fallback linear axis
    
    def _create_widgets(self):
        """Create interactive widgets for parameter control."""
        
        # Feature parameter sliders
        self.param_sliders = {}
        
        # Default parameter ranges (adjust as needed)
        param_ranges = {
            "E": (5.0, 50.0, 20.0),    # Energy: min, max, default
            "P": (5.0, 50.0, 15.0),    # Pressure: min, max, default  
            "ms": (5.0, 50.0, 30.0)    # Acquisition time: min, max, default
        }
        
        # Create sliders for each feature
        for feature in self.features:
            if feature in param_ranges:
                min_val, max_val, default_val = param_ranges[feature]
            else:
                min_val, max_val, default_val = 0.0, 100.0, 50.0
            
            slider = widgets.FloatSlider(
                value=default_val,
                min=min_val,
                max=max_val,
                step=0.1,
                description=f'{feature}:',
                style={'description_width': 'initial'},
                continuous_update=False
            )
            self.param_sliders[feature] = slider
        
        # Sampling parameters
        self.n_samples_slider = widgets.IntSlider(
            value=4,
            min=1,
            max=16,
            step=1,
            description='Number of Samples:',
            style={'description_width': 'initial'}
        )
        
        self.cfg_scale_slider = widgets.FloatSlider(
            value=3.0,
            min=1.0,
            max=10.0,
            step=0.1,
            description='CFG Scale:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        self.smooth_output_checkbox = widgets.Checkbox(
            value=True,
            description='Smooth Output',
            style={'description_width': 'initial'}
        )
        
        self.smooth_sigma_slider = widgets.FloatSlider(
            value=2.0,
            min=0.1,
            max=5.0,
            step=0.1,
            description='Smoothing Sigma:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # Control buttons
        self.generate_button = widgets.Button(
            description='Generate Spectra',
            button_style='primary',
            icon='play'
        )
        
        self.add_to_history_button = widgets.Button(
            description='Add to History',
            button_style='success',
            icon='plus'
        )
        
        self.clear_history_button = widgets.Button(
            description='Clear History',
            button_style='warning',
            icon='trash'
        )
        
        self.export_button = widgets.Button(
            description='Export Data',
            button_style='info',
            icon='download'
        )
        
        # Output widget for plots
        self.output_widget = widgets.Output()
        
        # Bind button callbacks
        self.generate_button.on_click(self._on_generate_click)
        self.add_to_history_button.on_click(self._on_add_to_history_click)
        self.clear_history_button.on_click(self._on_clear_history_click)
        self.export_button.on_click(self._on_export_click)
    
    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values from sliders."""
        return {feature: slider.value for feature, slider in self.param_sliders.items()}
    
    def _generate_spectra(
        self, 
        parameters: Dict[str, float], 
        n_samples: int, 
        cfg_scale: float,
        smooth_output: bool = True,
        smooth_sigma: float = 2.0
    ) -> np.ndarray:
        """Generate spectra with given parameters."""
        
        # Create settings tensor
        settings_values = [parameters[feature] for feature in self.features]
        settings_tensor = torch.tensor(settings_values, dtype=torch.float32).reshape(1, -1).to(self.device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.sampler.sample(
                resolution=self.spectrum_length,
                device=self.device,
                settings=settings_tensor,
                n_samples=n_samples,
                cfg_scale=cfg_scale,
                settings_dim=len(self.features),
                smooth_output=smooth_output,
                smooth_kernel_size=9,
                smooth_sigma=smooth_sigma
            )
            
            # Convert to numpy
            samples_np = samples.cpu().numpy()
            if samples_np.ndim == 3:  # (n_samples, channels, length)
                samples_np = samples_np[:, 0, :]  # Take first channel
            
            return samples_np
    
    def _plot_spectra(self, samples: np.ndarray, parameters: Dict[str, float], show_individual: bool = True):
        """Plot generated spectra."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Individual samples
        ax1 = axes[0]
        for i, spectrum in enumerate(samples):
            ax1.plot(self.energy_axis, spectrum, alpha=0.7, linewidth=1.5, label=f'Sample {i+1}')
        
        ax1.set_xlabel('Energy (MeV)')
        ax1.set_ylabel('Intensity')
        ax1.set_title(f'Individual Spectra (n={len(samples)})')
        ax1.grid(True, alpha=0.3)
        if len(samples) <= 8:  # Only show legend if not too many samples
            ax1.legend()
        
        # Plot 2: Average and statistics
        ax2 = axes[1]
        mean_spectrum = np.mean(samples, axis=0)
        std_spectrum = np.std(samples, axis=0)
        
        ax2.plot(self.energy_axis, mean_spectrum, 'b-', linewidth=2, label='Mean')
        ax2.fill_between(self.energy_axis, 
                        mean_spectrum - std_spectrum, 
                        mean_spectrum + std_spectrum, 
                        alpha=0.3, color='blue', label='±1 Std')
        
        ax2.set_xlabel('Energy (MeV)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Average Spectrum ± Standard Deviation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add parameter information as suptitle
        param_str = ', '.join([f'{k}={v:.2f}' for k, v in parameters.items()])
        fig.suptitle(f'Generated Spectra - Parameters: {param_str}', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"📊 Statistics:")
        print(f"  Mean intensity: {np.mean(mean_spectrum):.4f}")
        print(f"  Max intensity: {np.max(mean_spectrum):.4f}")
        print(f"  Total integrated intensity: {np.sum(mean_spectrum):.4f}")
        
        # Calculate weighted sum (intensity × energy)
        weighted_sum = np.sum(mean_spectrum * self.energy_axis)
        print(f"  Weighted sum (I × E): {weighted_sum:.4f}")
    
    def _plot_history_comparison(self):
        """Plot comparison of all samples in history."""
        if not self.sample_history:
            print("No samples in history to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All mean spectra
        ax1 = axes[0]
        for i, entry in enumerate(self.sample_history):
            mean_spectrum = np.mean(entry['samples'], axis=0)
            param_str = ', '.join([f'{k}={v:.1f}' for k, v in entry['parameters'].items()])
            ax1.plot(self.energy_axis, mean_spectrum, linewidth=2, label=f'{i+1}: {param_str}')
        
        ax1.set_xlabel('Energy (MeV)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Comparison of Mean Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Parameter space exploration
        ax2 = axes[1]
        if len(self.features) >= 2:
            feature1, feature2 = self.features[0], self.features[1]
            x_vals = [entry['parameters'][feature1] for entry in self.sample_history]
            y_vals = [entry['parameters'][feature2] for entry in self.sample_history]
            
            # Color by weighted sum
            weighted_sums = []
            for entry in self.sample_history:
                mean_spectrum = np.mean(entry['samples'], axis=0)
                weighted_sum = np.sum(mean_spectrum * self.energy_axis)
                weighted_sums.append(weighted_sum)
            
            scatter = ax2.scatter(x_vals, y_vals, c=weighted_sums, cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(scatter, ax=ax2, label='Weighted Sum (I × E)')
            
            ax2.set_xlabel(f'{feature1}')
            ax2.set_ylabel(f'{feature2}')
            ax2.set_title(f'Parameter Space Exploration\n({feature1} vs {feature2})')
            ax2.grid(True, alpha=0.3)
            
            # Add sample numbers
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                ax2.annotate(str(i+1), (x, y), xytext=(5, 5), textcoords='offset points')
        else:
            ax2.text(0.5, 0.5, 'Need at least 2 features\nfor parameter space plot', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Parameter Space')
        
        plt.tight_layout()
        plt.show()
    
    def _on_generate_click(self, button):
        """Handle generate button click."""
        with self.output_widget:
            clear_output(wait=True)
            
            try:
                # Get current parameters
                parameters = self._get_current_parameters()
                n_samples = self.n_samples_slider.value
                cfg_scale = self.cfg_scale_slider.value
                smooth_output = self.smooth_output_checkbox.value
                smooth_sigma = self.smooth_sigma_slider.value
                
                print(f"🔄 Generating {n_samples} spectra with parameters:")
                for feature, value in parameters.items():
                    print(f"  {feature}: {value:.2f}")
                print(f"  CFG Scale: {cfg_scale:.1f}")
                print(f"  Smooth Output: {smooth_output}")
                if smooth_output:
                    print(f"  Smooth Sigma: {smooth_sigma:.1f}")
                print()
                
                # Generate spectra
                samples = self._generate_spectra(
                    parameters, n_samples, cfg_scale, smooth_output, smooth_sigma
                )
                
                # Store current samples
                self.current_samples = {
                    'samples': samples,
                    'parameters': parameters.copy(),
                    'cfg_scale': cfg_scale,
                    'smooth_output': smooth_output,
                    'smooth_sigma': smooth_sigma
                }
                
                # Plot results
                self._plot_spectra(samples, parameters)
                
                print("✅ Generation complete!")
                
            except Exception as e:
                print(f"❌ Error generating spectra: {e}")
    
    def _on_add_to_history_click(self, button):
        """Handle add to history button click."""
        if self.current_samples is None:
            print("❌ No current samples to add. Generate spectra first.")
            return
        
        # Add current samples to history
        self.sample_history.append(self.current_samples.copy())
        print(f"✅ Added to history (now {len(self.sample_history)} entries)")
        
        with self.output_widget:
            # Update the comparison plot
            self._plot_history_comparison()
    
    def _on_clear_history_click(self, button):
        """Handle clear history button click."""
        self.sample_history = []
        print("🗑️ History cleared")
        
        with self.output_widget:
            clear_output(wait=True)
            print("History has been cleared.")
    
    def _on_export_click(self, button):
        """Handle export button click."""
        if not self.sample_history and self.current_samples is None:
            print("❌ No data to export.")
            return
        
        try:
            # Create export directory
            export_dir = Path("exported_spectra")
            export_dir.mkdir(exist_ok=True)
            
            # Export current samples if available
            if self.current_samples is not None:
                self._export_sample_set(self.current_samples, export_dir / "current_samples")
            
            # Export history
            for i, entry in enumerate(self.sample_history):
                self._export_sample_set(entry, export_dir / f"history_set_{i+1}")
            
            # Export summary
            self._export_summary(export_dir)
            
            print(f"✅ Data exported to: {export_dir.absolute()}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def _export_sample_set(self, sample_set: Dict[str, Any], output_dir: Path):
        """Export a single sample set."""
        output_dir.mkdir(exist_ok=True)
        
        samples = sample_set['samples']
        parameters = sample_set['parameters']
        
        # Save samples as numpy array
        np.save(output_dir / "samples.npy", samples)
        
        # Save energy axis
        np.save(output_dir / "energy_axis.npy", self.energy_axis)
        
        # Save individual spectra as CSV
        for i, spectrum in enumerate(samples):
            df = pd.DataFrame({
                'energy_MeV': self.energy_axis,
                'intensity': spectrum
            })
            df.to_csv(output_dir / f"spectrum_{i+1}.csv", index=False)
        
        # Save mean spectrum
        mean_spectrum = np.mean(samples, axis=0)
        df_mean = pd.DataFrame({
            'energy_MeV': self.energy_axis,
            'intensity': mean_spectrum,
            'std': np.std(samples, axis=0)
        })
        df_mean.to_csv(output_dir / "mean_spectrum.csv", index=False)
        
        # Save parameters
        param_df = pd.DataFrame([parameters])
        param_df.to_csv(output_dir / "parameters.csv", index=False)
        
        # Save metadata
        metadata = {
            'n_samples': len(samples),
            'spectrum_length': self.spectrum_length,
            'features': self.features,
            'cfg_scale': sample_set.get('cfg_scale', 'unknown'),
            'smooth_output': sample_set.get('smooth_output', 'unknown'),
            'smooth_sigma': sample_set.get('smooth_sigma', 'unknown')
        }
        
        with open(output_dir / "metadata.txt", 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
    
    def _export_summary(self, output_dir: Path):
        """Export summary of all samples."""
        if not self.sample_history:
            return
        
        # Create summary DataFrame
        summary_data = []
        for i, entry in enumerate(self.sample_history):
            samples = entry['samples']
            parameters = entry['parameters']
            
            mean_spectrum = np.mean(samples, axis=0)
            
            summary_row = {
                'set_number': i + 1,
                'n_samples': len(samples),
                'mean_intensity': np.mean(mean_spectrum),
                'max_intensity': np.max(mean_spectrum),
                'total_intensity': np.sum(mean_spectrum),
                'weighted_sum': np.sum(mean_spectrum * self.energy_axis),
                **parameters  # Add all parameters
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    def display_interface(self):
        """Display the interactive interface."""
        
        # Create layout
        param_box = widgets.VBox([
            widgets.HTML("<h3>📊 Conditional Parameters</h3>"),
            *list(self.param_sliders.values())
        ])
        
        sampling_box = widgets.VBox([
            widgets.HTML("<h3>⚙️ Sampling Parameters</h3>"),
            self.n_samples_slider,
            self.cfg_scale_slider,
            self.smooth_output_checkbox,
            self.smooth_sigma_slider
        ])
        
        control_box = widgets.VBox([
            widgets.HTML("<h3>🎮 Controls</h3>"),
            self.generate_button,
            widgets.HBox([
                self.add_to_history_button,
                self.clear_history_button
            ]),
            self.export_button
        ])
        
        # Left panel with controls
        left_panel = widgets.VBox([
            param_box,
            sampling_box,
            control_box
        ])
        
        # Right panel with output
        right_panel = widgets.VBox([
            widgets.HTML("<h3>📈 Generated Spectra</h3>"),
            self.output_widget
        ])
        
        # Main interface
        interface = widgets.HBox([
            left_panel,
            right_panel
        ])
        
        # Display title and interface
        title = widgets.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🔬 EDM Spectrum Sampling Tool</h1>
            <p>Interactive tool for generating and visualizing conditional electron spectra</p>
        </div>
        """)
        
        display(title)
        display(interface)
        
        # Initial message in output
        with self.output_widget:
            print("👋 Welcome to the EDM Spectrum Sampling Tool!")
            print()
            print("Instructions:")
            print("1. 🎛️ Adjust the conditional parameters using the sliders")
            print("2. ⚙️ Configure sampling parameters (number of samples, CFG scale, etc.)")
            print("3. 🎮 Click 'Generate Spectra' to create new samples")
            print("4. 📚 Use 'Add to History' to keep track of interesting parameter sets")
            print("5. 💾 Use 'Export Data' to save your results")
            print()
            print("Ready to generate spectra! 🚀")
    
    def generate_batch_comparison(
        self, 
        parameter_sets: List[Dict[str, float]], 
        n_samples: int = 4, 
        cfg_scale: float = 3.0
    ):
        """
        Generate and compare multiple parameter sets in batch mode.
        
        Args:
            parameter_sets: List of parameter dictionaries
            n_samples: Number of samples per parameter set
            cfg_scale: CFG scale for sampling
        """
        print(f"🔄 Generating batch comparison for {len(parameter_sets)} parameter sets...")
        
        results = []
        
        # Generate samples for each parameter set
        for i, parameters in enumerate(parameter_sets):
            print(f"  Generating set {i+1}/{len(parameter_sets)}: {parameters}")
            
            samples = self._generate_spectra(parameters, n_samples, cfg_scale)
            results.append({
                'samples': samples,
                'parameters': parameters,
                'mean_spectrum': np.mean(samples, axis=0)
            })
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: All mean spectra
        ax1 = axes[0, 0]
        for i, result in enumerate(results):
            param_str = ', '.join([f'{k}={v:.1f}' for k, v in result['parameters'].items()])
            ax1.plot(self.energy_axis, result['mean_spectrum'], linewidth=2, label=f'Set {i+1}')
        
        ax1.set_xlabel('Energy (MeV)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Comparison of Mean Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Parameter space (if 2+ features)
        ax2 = axes[0, 1]
        if len(self.features) >= 2:
            feature1, feature2 = self.features[0], self.features[1]
            x_vals = [r['parameters'][feature1] for r in results]
            y_vals = [r['parameters'][feature2] for r in results]
            
            # Color by weighted sum
            weighted_sums = [np.sum(r['mean_spectrum'] * self.energy_axis) for r in results]
            scatter = ax2.scatter(x_vals, y_vals, c=weighted_sums, cmap='viridis', s=100)
            plt.colorbar(scatter, ax=ax2, label='Weighted Sum')
            
            ax2.set_xlabel(feature1)
            ax2.set_ylabel(feature2)
            ax2.set_title(f'Parameter Space ({feature1} vs {feature2})')
            ax2.grid(True, alpha=0.3)
            
            # Add labels
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot 3: Intensity statistics
        ax3 = axes[1, 0]
        mean_intensities = [np.mean(r['mean_spectrum']) for r in results]
        max_intensities = [np.max(r['mean_spectrum']) for r in results]
        
        x_pos = np.arange(len(results))
        width = 0.35
        
        ax3.bar(x_pos - width/2, mean_intensities, width, label='Mean Intensity', alpha=0.7)
        ax3.bar(x_pos + width/2, max_intensities, width, label='Max Intensity', alpha=0.7)
        
        ax3.set_xlabel('Parameter Set')
        ax3.set_ylabel('Intensity')
        ax3.set_title('Intensity Statistics')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Set {i+1}' for i in range(len(results))])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Weighted sums
        ax4 = axes[1, 1]
        weighted_sums = [np.sum(r['mean_spectrum'] * self.energy_axis) for r in results]
        
        bars = ax4.bar(range(len(results)), weighted_sums, alpha=0.7, color='orange')
        ax4.set_xlabel('Parameter Set')
        ax4.set_ylabel('Weighted Sum (I × E)')
        ax4.set_title('Weighted Sum Comparison')
        ax4.set_xticks(range(len(results)))
        ax4.set_xticklabels([f'Set {i+1}' for i in range(len(results))])
        ax4.grid(True, alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(weighted_sums)
        bars[best_idx].set_color('red')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n📊 Batch Comparison Summary:")
        print("-" * 50)
        for i, result in enumerate(results):
            params = result['parameters']
            weighted_sum = weighted_sums[i]
            param_str = ', '.join([f'{k}={v:.2f}' for k, v in params.items()])
            print(f"Set {i+1}: {param_str} → Weighted Sum: {weighted_sum:.4f}")
        
        best_idx = np.argmax(weighted_sums)
        print(f"\n🏆 Best performing set: Set {best_idx+1}")
        
        return results

# Convenience functions for quick setup
def load_spectrum_tool(
    model_path: str,
    device: str = "auto",
    **kwargs
) -> SpectrumSamplingTool:
    """
    Convenience function to load and display the spectrum sampling tool.
    
    Args:
        model_path: Path to the trained model
        device: Device to use ("auto", "cuda", or "cpu")
        **kwargs: Additional arguments for SpectrumSamplingTool
    
    Returns:
        Initialized SpectrumSamplingTool instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tool = SpectrumSamplingTool(model_path, device=device, **kwargs)
    tool.display_interface()
    return tool

def quick_sample_and_plot(
    model_path: str,
    parameters: Dict[str, float],
    n_samples: int = 4,
    device: str = "auto",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to generate and plot spectra without the interactive interface.
    
    Args:
        model_path: Path to the trained model
        parameters: Dictionary of parameters (e.g., {"E": 20.0, "P": 15.0, "ms": 30.0})
        n_samples: Number of samples to generate
        device: Device to use ("auto", "cuda", or "cpu")
        **kwargs: Additional arguments for SpectrumSamplingTool
    
    Returns:
        Tuple of (samples, energy_axis)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create tool (without displaying interface)
    tool = SpectrumSamplingTool(model_path, device=device, **kwargs)
    
    # Generate samples
    samples = tool._generate_spectra(parameters, n_samples, cfg_scale=3.0)
    
    # Plot results
    tool._plot_spectra(samples, parameters)
    
    return samples, tool.energy_axis

# Example usage functions
def create_parameter_sweep(
    feature_ranges: Dict[str, Tuple[float, float, int]]
) -> List[Dict[str, float]]:
    """
    Create a parameter sweep for batch comparison.
    
    Args:
        feature_ranges: Dict mapping feature names to (min, max, num_points) tuples
        
    Example:
        ranges = {
            "E": (10.0, 30.0, 5),
            "P": (10.0, 20.0, 3),
            "ms": (20.0, 40.0, 3)
        }
        param_sets = create_parameter_sweep(ranges)
    
    Returns:
        List of parameter dictionaries
    """
    import itertools
    
    # Generate values for each feature
    feature_values = {}
    for feature, (min_val, max_val, num_points) in feature_ranges.items():
        if num_points == 1:
            feature_values[feature] = [(min_val + max_val) / 2]
        else:
            feature_values[feature] = np.linspace(min_val, max_val, num_points).tolist()
    
    # Create all combinations
    feature_names = list(feature_values.keys())
    value_combinations = list(itertools.product(*[feature_values[name] for name in feature_names]))
    
    # Convert to list of dictionaries
    parameter_sets = []
    for combination in value_combinations:
        param_dict = {name: value for name, value in zip(feature_names, combination)}
        parameter_sets.append(param_dict)
    
    return parameter_sets

# Example parameter sets for quick testing
EXAMPLE_PARAMETER_SETS = {
    "low_energy": {"E": 15.0, "P": 20.0, "ms": 25.0},
    "medium_energy": {"E": 25.0, "P": 15.0, "ms": 30.0},  
    "high_energy": {"E": 35.0, "P": 10.0, "ms": 20.0},
    "high_pressure": {"E": 20.0, "P": 40.0, "ms": 15.0},
    "long_acquisition": {"E": 30.0, "P": 25.0, "ms": 45.0}
} 