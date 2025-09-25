# 🔬 EDM Spectrum Sampling Tool

An interactive Jupyter notebook tool for generating and visualizing conditional electron spectra using trained EDM (Elucidating Diffusion Models).

## 🌟 Features

- **🎛️ Interactive Parameter Controls**: Real-time sliders for adjusting experimental conditions
- **📈 Live Visualization**: Instant spectrum generation and plotting
- **📚 Sample History**: Track and compare multiple parameter sets
- **🔍 Batch Analysis**: Systematic parameter sweep capabilities  
- **💾 Data Export**: Save results in multiple formats (CSV, NumPy, plots)
- **⚡ GPU Acceleration**: Automatic CUDA detection and utilization
- **🔧 Flexible Configuration**: Customizable for different model architectures

## 📦 Installation

### Prerequisites

```bash
# Required packages
pip install torch torchvision numpy matplotlib pandas ipywidgets jupyter
pip install scikit-optimize  # For advanced parameter sweeps

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension

# For JupyterLab users
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Project Setup

1. Place `spectrum_sampling_tool.py` in your project directory
2. Ensure your trained EDM model and `src/` modules are accessible
3. Update model paths in the example scripts

## 🚀 Quick Start

### Method 1: Interactive Jupyter Notebook

```python
from spectrum_sampling_tool import load_spectrum_tool

# Load and display interactive tool
tool = load_spectrum_tool(
    model_path="path/to/your/model.pt",
    device="auto",  # Automatically selects CUDA or CPU
    features=["E", "P", "ms"]  # Energy, Pressure, Acquisition time
)
```

### Method 2: Programmatic Usage

```python
from spectrum_sampling_tool import quick_sample_and_plot

# Generate spectra with specific parameters
parameters = {
    "E": 25.0,    # Laser Energy
    "P": 15.0,    # Gas Pressure 
    "ms": 30.0    # Acquisition time (ms)
}

samples, energy_axis = quick_sample_and_plot(
    model_path="path/to/your/model.pt",
    parameters=parameters,
    n_samples=6
)
```

### Method 3: Run Demo Script

```bash
# Update MODEL_PATH in the script first
python spectrum_sampling_demo.py
```

## 🎛️ Interactive Interface Guide

### Parameter Controls

**📊 Conditional Parameters:**
- **E (Energy)**: Laser energy parameter (typical range: 5-50)
- **P (Pressure)**: Gas pressure parameter (typical range: 5-50)  
- **ms (Time)**: Acquisition time in milliseconds (typical range: 5-50)

**⚙️ Sampling Parameters:**
- **Number of Samples**: How many spectra to generate (1-16)
- **CFG Scale**: Classifier-free guidance strength (1.0-10.0)
- **Smooth Output**: Apply Gaussian smoothing to results
- **Smooth Sigma**: Smoothing kernel width (0.1-5.0)

**🎮 Control Buttons:**
- **Generate Spectra**: Create new samples with current parameters
- **Add to History**: Save current results for comparison
- **Clear History**: Remove all saved results  
- **Export Data**: Save all data to files

### Visualization

The tool provides two main plots:

1. **Individual Spectra**: Shows all generated samples
2. **Statistics**: Mean spectrum with standard deviation bands

Additional information displayed:
- Parameter values used
- Intensity statistics (mean, max, total, weighted sum)
- Energy ranges and bandwidths

## 📊 Advanced Usage

### Batch Parameter Comparison

```python
from spectrum_sampling_tool import SpectrumSamplingTool

tool = SpectrumSamplingTool("path/to/model.pt")

# Compare different energy levels
parameter_sets = [
    {"E": 15.0, "P": 20.0, "ms": 25.0},  # Low energy
    {"E": 25.0, "P": 20.0, "ms": 25.0},  # Medium energy  
    {"E": 35.0, "P": 20.0, "ms": 25.0},  # High energy
]

results = tool.generate_batch_comparison(
    parameter_sets=parameter_sets,
    n_samples=4,
    cfg_scale=3.0
)
```

### Systematic Parameter Sweep

```python
from spectrum_sampling_tool import create_parameter_sweep

# Define parameter ranges
sweep_ranges = {
    "E": (10.0, 30.0, 5),    # Energy: 5 points from 10-30
    "P": (10.0, 20.0, 3),    # Pressure: 3 points from 10-20
    "ms": (20.0, 40.0, 3)    # Time: 3 points from 20-40
}

# Generate all combinations (5×3×3 = 45 parameter sets)
parameter_sets = create_parameter_sweep(sweep_ranges)

# Run systematic comparison
results = tool.generate_batch_comparison(parameter_sets)
```

### Example Parameter Sets

```python
from spectrum_sampling_tool import EXAMPLE_PARAMETER_SETS

# Available predefined parameter sets:
# - low_energy: {"E": 15.0, "P": 20.0, "ms": 25.0}
# - medium_energy: {"E": 25.0, "P": 15.0, "ms": 30.0}
# - high_energy: {"E": 35.0, "P": 10.0, "ms": 20.0}
# - high_pressure: {"E": 20.0, "P": 40.0, "ms": 15.0}
# - long_acquisition: {"E": 30.0, "P": 25.0, "ms": 45.0}

# Use example sets
results = tool.generate_batch_comparison(
    parameter_sets=list(EXAMPLE_PARAMETER_SETS.values())
)
```

## 💾 Data Export

The tool supports multiple export formats:

### Automatic Export (via button)
- `exported_spectra/current_samples/`: Latest generated spectra
- `exported_spectra/history_set_N/`: Each saved parameter set
- `exported_spectra/summary.csv`: Summary of all parameter sets

### File Formats
- `samples.npy`: Raw spectral data (NumPy array)
- `energy_axis.npy`: Energy values for x-axis
- `spectrum_N.csv`: Individual spectra in CSV format
- `mean_spectrum.csv`: Mean and standard deviation
- `parameters.csv`: Parameter values used
- `metadata.txt`: Generation settings and info

### Programmatic Access

```python
# Access raw data
samples = tool.current_samples['samples']  # Shape: (n_samples, spectrum_length)
energy_axis = tool.energy_axis  # Shape: (spectrum_length,)
parameters = tool.current_samples['parameters']  # Dict of parameter values

# Calculate statistics
mean_spectrum = np.mean(samples, axis=0)
std_spectrum = np.std(samples, axis=0)
total_intensity = np.sum(mean_spectrum)
weighted_sum = np.sum(mean_spectrum * energy_axis)
```

## 🔧 Customization

### Custom Parameter Ranges

```python
tool = SpectrumSamplingTool(
    model_path="path/to/model.pt",
    spectrum_length=256,
    features=["E", "P", "ms"],
    num_sampling_steps=30,
    device="cuda"
)

# Modify parameter ranges by updating widget properties after initialization
# (Advanced: requires modifying the widget creation code)
```

### Different Model Architectures

```python
tool = SpectrumSamplingTool(
    model_path="path/to/model.pt",
    spectrum_length=512,      # Different resolution
    features=["laser", "gas", "detector"],  # Different parameter names
    sigma_data=0.2           # Different normalization
)
```

### Energy Axis Customization

The tool automatically creates energy axes using deflection calculations. For custom energy ranges:

```python
# The energy axis is created in _create_energy_axis()
# Modify this method or override energy_axis after initialization
tool.energy_axis = np.linspace(0, 100, 256)  # Custom linear axis
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading:**
```python
# Check model file exists
from pathlib import Path
print(Path("your_model.pt").exists())

# Verify model architecture matches
tool = SpectrumSamplingTool(model_path, spectrum_length=256, features=["E", "P", "ms"])
```

**CUDA/GPU Issues:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Force CPU if needed
tool = SpectrumSamplingTool(model_path, device="cpu")
```

**Widget Display Issues:**
```bash
# Enable extensions
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Restart kernel
# Kernel → Restart & Clear Output
```

**Import Errors:**
```python
import sys
sys.path.append('.')  # Add project root to path

# Check dependencies
import torch, numpy, matplotlib, ipywidgets, pandas
```

### Debug Information

```python
from spectrum_sampling_tool import SpectrumSamplingTool
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
```

## 📖 Usage Examples

### Jupyter Notebook Cell Examples

```python
# Cell 1: Setup
from spectrum_sampling_tool import load_spectrum_tool
tool = load_spectrum_tool("path/to/model.pt")

# Cell 2: Quick generation
from spectrum_sampling_tool import quick_sample_and_plot
samples, energy = quick_sample_and_plot(
    "path/to/model.pt", 
    {"E": 20, "P": 15, "ms": 30}
)

# Cell 3: Batch comparison  
parameter_sets = [
    {"E": 15, "P": 15, "ms": 30},
    {"E": 25, "P": 15, "ms": 30}, 
    {"E": 35, "P": 15, "ms": 30}
]
results = tool.generate_batch_comparison(parameter_sets)

# Cell 4: Export data
# Use the Export button in the interface or:
# tool._on_export_click(None)
```

### Analysis Workflow

```python
# 1. Generate multiple parameter sets
tool = SpectrumSamplingTool("model.pt")

# 2. Explore parameter space interactively
# (Use the interface)

# 3. Add interesting results to history
# (Use "Add to History" button)

# 4. Run systematic comparison
param_sweep = create_parameter_sweep({
    "E": (15, 35, 5), "P": (10, 30, 4), "ms": (20, 40, 3)
})
results = tool.generate_batch_comparison(param_sweep[:10])

# 5. Export and analyze
# (Use "Export Data" button)
# Load exported CSV files for further analysis
```

## 📚 API Reference

### Main Classes

**`SpectrumSamplingTool`**: Core sampling and visualization tool
- `__init__(model_path, device, spectrum_length, features, ...)`
- `display_interface()`: Show interactive widgets
- `generate_batch_comparison(parameter_sets, ...)`: Compare multiple parameter sets
- `_generate_spectra(parameters, n_samples, ...)`: Generate spectra programmatically

### Utility Functions

**`load_spectrum_tool(model_path, **kwargs)`**: Quick interactive setup
**`quick_sample_and_plot(model_path, parameters, **kwargs)`**: Fast non-interactive sampling
**`create_parameter_sweep(feature_ranges)`**: Generate parameter combinations

### Constants

**`EXAMPLE_PARAMETER_SETS`**: Predefined interesting parameter combinations

## 🤝 Contributing

To extend the tool:

1. **Add new features**: Modify the `features` parameter and update parameter ranges
2. **Custom visualizations**: Override `_plot_spectra()` method
3. **Different models**: Adapt `_load_model()` for your architecture
4. **Export formats**: Extend `_export_sample_set()` method

## 📄 License

This tool is designed to work with the existing EDM diffusion model codebase. Please ensure compliance with your project's license terms.

## 🆘 Support

For issues related to:
- **Model loading**: Check path and model compatibility
- **Performance**: Use fewer samples, reduce resolution, or switch to CPU
- **Visualization**: Verify Jupyter widgets installation
- **Customization**: See the customization examples above

---

**Happy spectrum generation!** 🚀🔬