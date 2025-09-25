"""
EDM Spectrum Sampling Tool - Demo Script

This script demonstrates how to use the interactive spectrum sampling tool
for generating and visualizing conditional electron spectra.

Usage:
    python spectrum_sampling_demo.py

For Jupyter notebook usage, see the README.md file.
"""

from spectrum_sampling_tool import (
    SpectrumSamplingTool, 
    load_spectrum_tool, 
    quick_sample_and_plot,
    create_parameter_sweep,
    EXAMPLE_PARAMETER_SETS
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Update this path to your trained model
    MODEL_PATH = "models/edm_1d_spectrum_256pts_instancenorm_fixeddataset_10kEpochs/ema_ckpt_final.pt"
    
    print("🔬 EDM Spectrum Sampling Tool - Demo")
    print("="*50)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model.")
        return
    
    print(f"✅ Model found: {MODEL_PATH}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Demo 1: Quick sample generation
    print("📊 Demo 1: Quick Sample Generation")
    print("-" * 30)
    
    parameters = {
        "E": 25.0,    # Laser Energy
        "P": 15.0,    # Pressure 
        "ms": 30.0    # Acquisition time (ms)
    }
    
    print(f"Generating spectra with parameters: {parameters}")
    
    try:
        samples, energy_axis = quick_sample_and_plot(
            model_path=MODEL_PATH,
            parameters=parameters,
            n_samples=6,
            device="auto"
        )
        
        print(f"✅ Generated {len(samples)} spectra with shape {samples.shape}")
        print(f"   Energy axis: {len(energy_axis)} points, {energy_axis.min():.2f}-{energy_axis.max():.2f} MeV")
        
    except Exception as e:
        print(f"❌ Error in quick sampling: {e}")
        print("   This might be due to missing dependencies or incorrect model path.")
        return
    
    print("\n" + "="*50)
    
    # Demo 2: Batch comparison
    print("📈 Demo 2: Batch Parameter Comparison")
    print("-" * 30)
    
    try:
        # Initialize tool for batch comparison
        tool = SpectrumSamplingTool(
            model_path=MODEL_PATH,
            device="auto",
            num_sampling_steps=20  # Faster for demo
        )
        
        # Test different energy levels
        energy_comparison = [
            {"E": 15.0, "P": 20.0, "ms": 25.0},  # Low energy
            {"E": 25.0, "P": 20.0, "ms": 25.0},  # Medium energy
            {"E": 35.0, "P": 20.0, "ms": 25.0},  # High energy
        ]
        
        print("Comparing different energy levels...")
        results = tool.generate_batch_comparison(
            parameter_sets=energy_comparison,
            n_samples=3,  # Fewer samples for speed
            cfg_scale=3.0
        )
        
        print(f"✅ Batch comparison completed for {len(results)} parameter sets")
        
    except Exception as e:
        print(f"❌ Error in batch comparison: {e}")
    
    print("\n" + "="*50)
    
    # Demo 3: Parameter sweep
    print("🔍 Demo 3: Parameter Sweep")
    print("-" * 30)
    
    try:
        # Create a small parameter sweep for demo
        sweep_ranges = {
            "E": (20.0, 30.0, 3),    # Energy: 20, 25, 30
            "P": (15.0, 25.0, 2),    # Pressure: 15, 25
            "ms": (25.0, 35.0, 2)    # Time: 25, 35
        }
        
        parameter_sets = create_parameter_sweep(sweep_ranges)
        print(f"Created {len(parameter_sets)} parameter combinations:")
        for i, params in enumerate(parameter_sets):
            print(f"  {i+1}: {params}")
        
        print("\nRunning parameter sweep (subset for demo)...")
        subset_params = parameter_sets[:4]  # Use first 4 combinations
        
        sweep_results = tool.generate_batch_comparison(
            parameter_sets=subset_params,
            n_samples=2,  # Fewer samples for speed
            cfg_scale=3.0
        )
        
        print(f"✅ Parameter sweep completed for {len(sweep_results)} sets")
        
    except Exception as e:
        print(f"❌ Error in parameter sweep: {e}")
    
    print("\n" + "="*50)
    
    # Demo 4: Example parameter sets
    print("🎯 Demo 4: Example Parameter Sets")
    print("-" * 30)
    
    print("Available example parameter sets:")
    for name, params in EXAMPLE_PARAMETER_SETS.items():
        print(f"  {name}: {params}")
    
    try:
        # Test a few example sets
        example_subset = {
            k: v for i, (k, v) in enumerate(EXAMPLE_PARAMETER_SETS.items()) 
            if i < 3  # First 3 examples
        }
        
        print(f"\nTesting {len(example_subset)} example sets...")
        example_comparison = list(example_subset.values())
        
        example_results = tool.generate_batch_comparison(
            parameter_sets=example_comparison,
            n_samples=3,
            cfg_scale=3.0
        )
        
        print(f"✅ Example comparison completed")
        
        # Analyze results
        print("\n📊 Analysis:")
        for i, (result, (name, params)) in enumerate(zip(example_results, example_subset.items())):
            mean_spectrum = result['mean_spectrum']
            total_intensity = np.sum(mean_spectrum)
            peak_intensity = np.max(mean_spectrum)
            weighted_sum = np.sum(mean_spectrum * tool.energy_axis)
            
            print(f"  {name}:")
            print(f"    Total intensity: {total_intensity:.4f}")
            print(f"    Peak intensity: {peak_intensity:.4f}")
            print(f"    Weighted sum (I×E): {weighted_sum:.4f}")
        
    except Exception as e:
        print(f"❌ Error in example comparison: {e}")
    
    print("\n" + "="*50)
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. 🚀 Launch interactive tool: Use load_spectrum_tool() in Jupyter")
    print("2. 🔬 Explore parameters: Try different combinations")
    print("3. 📊 Analyze results: Use the batch comparison features")
    print("4. 💾 Export data: Save interesting results for further analysis")
    print("\nFor interactive usage, see the Jupyter notebook examples in the README.")

def interactive_demo():
    """
    Start the interactive tool directly.
    This function can be called from a Jupyter notebook.
    """
    MODEL_PATH = "models/edm_1d_spectrum_256pts_instancenorm_fixeddataset_10kEpochs/ema_ckpt_final.pt"
    
    print("🚀 Starting Interactive Spectrum Sampling Tool...")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the function to point to your trained model.")
        return None
    
    # Load and display the interactive tool
    tool = load_spectrum_tool(
        model_path=MODEL_PATH,
        device="auto",
        spectrum_length=256,
        features=["E", "P", "ms"],
        num_sampling_steps=30
    )
    
    return tool

if __name__ == "__main__":
    main() 