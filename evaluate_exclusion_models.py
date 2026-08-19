import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

# Configuration
CONFIG = {
    'model_dir': 'models',
    'data_dir': 'spectra',
    'params_file': 'params.csv',
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'resolution': 256,
    'features': ['E', 'P', 'ms'],
    'settings_dim': 3,
    'num_steps': 18,
    'cfg_scale': 3.0,
    'results_dir': 'exclusion_evaluation_results'
}

print(f"Using device: {CONFIG['device']}")

def load_exclusion_model(model_path, config, sigma_data=0.112):
    """Load a trained diffusion model from a path.

    sigma_data is NOT stored in the checkpoints (it is a plain attribute of
    EDMPrecond, not a registered buffer), so the caller must supply the value
    the model was trained with - see compute_norm_stats().
    """
    from src.modules_1d import EDMPrecond
    
    # Initialize model
    model = EDMPrecond(
        resolution=config['resolution'],
        settings_dim=config['settings_dim'],
        sigma_min=0,
        sigma_max=float('inf'),
        sigma_data=sigma_data,
        model_type='UNet_conditional',
        device=config['device']
    ).to(config['device'])
    
    # Prefer the final EMA checkpoint; fall back to the latest epoch file
    checkpoint_files = [
        'ema_ckpt_final.pt',
        'ema_ckpt.pt',
    ]
    
    if os.path.exists(model_path):
        epoch_files = [f for f in os.listdir(model_path) if f.startswith('ema_ckpt_epoch_') and f.endswith('.pt')]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
            checkpoint_files.append(epoch_files[-1])
    
    checkpoint_path = None
    for checkpoint_file in checkpoint_files:
        potential_path = os.path.join(model_path, checkpoint_file)
        if os.path.exists(potential_path):
            checkpoint_path = potential_path
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def _folder_intensity_stats(data_dir, resolution=256):
    """Per-experiment sufficient statistics of the raw intensities.

    For each spectra/<N> folder: (full-file max, and count / sum / sum-of-squares
    over the first `resolution` bins). One pass over the data; per-exclusion
    normalization constants are then combined from these without re-reading.
    """
    stats = {}
    for folder in sorted(Path(data_dir).iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        fmax, n, s, q = -np.inf, 0, 0.0, 0.0
        for csv_file in folder.glob('*.csv'):
            vals = pd.read_csv(csv_file)['intensity'].values.astype(np.float64)
            fmax = max(fmax, float(vals.max()))
            t = vals[:resolution]
            n += t.size
            s += float(t.sum())
            q += float((t ** 2).sum())
        stats[int(folder.name)] = (fmax, n, s, q)
    return stats


def compute_norm_stats(folder_stats, excluded_exp):
    """Reproduce the training normalization constants for one exclusion run.

    SpectrumDataset(normalize=True) divides by the max intensity over the FULL
    (untruncated) files of every non-excluded experiment, maps to [-1, 1] via
    (x/max)*2 - 1, then truncates to 256 bins; calculate_sigma_data() is the
    (unbiased) std of that tensor. std((2/M)*x - 1) = (2/M)*std(x), so both
    constants follow from the per-folder sufficient statistics.
    """
    included = [v for k, v in folder_stats.items() if k != excluded_exp]
    max_intensity = max(v[0] for v in included)
    n = sum(v[1] for v in included)
    s = sum(v[2] for v in included)
    q = sum(v[3] for v in included)
    var_raw = (q - s * s / n) / (n - 1)
    sigma_data = (2.0 / max_intensity) * float(np.sqrt(var_raw))
    return float(max_intensity), sigma_data


def sample_spectra_for_experiment(model, sampler, experiment_settings, n_samples, config, cfg_scale=3.0):
    """Generate spectra for a specific experiment."""
    model.eval()
    
    with torch.no_grad():
        settings_tensor = torch.tensor(
            experiment_settings, dtype=torch.float32
        ).reshape(1, -1).repeat(n_samples, 1).to(config['device'])
        
        # Generate samples
        samples = sampler.sample(
            resolution=config['resolution'],
            device=config['device'],
            settings=settings_tensor,
            n_samples=n_samples,
            cfg_scale=cfg_scale,
            settings_dim=config['settings_dim'],
            smooth_output=False,
            smooth_kernel_size=9,
            smooth_sigma=2.0
        )
        
        # Convert to numpy and return as 2D array (n_samples, resolution)
        samples_np = samples.cpu().numpy()
        if samples_np.ndim == 3:  # (n_samples, channels, length)
            samples_np = samples_np[:, 0, :]  # Take first channel
        
        return samples_np

def load_original_spectra(experiment_folder, resolution=256):
    """Load original spectra from CSV files in the experiment folder."""
    csv_files = sorted(Path(experiment_folder).glob("*.csv"))
    spectra = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'intensity' in df.columns:
            intensity = df['intensity'].values[:resolution]
            spectra.append(intensity)
        else:
            print(f"Warning: No 'intensity' column found in {csv_file}")
    
    return np.array(spectra) if spectra else None


def load_energy_axis(experiment_folder, resolution=256):
    csv_files = sorted(Path(experiment_folder).glob("*.csv"))
    if not csv_files:
        return None
    return pd.read_csv(csv_files[0])['energy'].values[:resolution]

def calculate_wasserstein_distance_1d_spectra(original_spectra, generated_spectra, min_avg_intensity=0.01):
    """
    Calculate Wasserstein distance between original and generated 1D spectra bin-by-bin.
    
    Parameters:
    - original_spectra: numpy array of shape (n_original_samples, resolution)
    - generated_spectra: numpy array of shape (n_generated_samples, resolution)  
    - min_avg_intensity: minimum average intensity for a bin to be included
    
    Returns:
    - average_distance: average Wasserstein distance across bins
    - detailed_results: detailed results per bin
    """
    if original_spectra is None or generated_spectra is None:
        return float('nan'), {}
    
    # Use model resolution (256) and truncate original spectra to match
    model_resolution = generated_spectra.shape[1]  # Should be 256
    
    # Truncate original spectra to first model_resolution points (highest energy end)
    if original_spectra.shape[1] > model_resolution:
        original_truncated = original_spectra[:, :model_resolution]
        print(f"Truncating original spectra from {original_spectra.shape[1]} to {model_resolution} points (highest energy end)")
    else:
        original_truncated = original_spectra
    
    resolution = model_resolution
    distances = []
    detailed_results = {}
    
    for bin_idx in range(resolution):
        original_bin = original_truncated[:, bin_idx]
        generated_bin = generated_spectra[:, bin_idx]
        
        # Calculate average intensities
        orig_avg = np.mean(original_bin)
        gen_avg = np.mean(generated_bin)
        
        # Skip bins with very low average intensity
        if orig_avg < min_avg_intensity and gen_avg < min_avg_intensity:
            detailed_results[bin_idx] = {
                'included': False,
                'reason': f'Low intensity ({orig_avg:.4f}, {gen_avg:.4f})',
                'orig_avg': orig_avg,
                'gen_avg': gen_avg
            }
            continue
        
        try:
            # Calculate Wasserstein distance for this bin
            wasserstein_dist = stats.wasserstein_distance(original_bin, generated_bin)
            
            detailed_results[bin_idx] = {
                'included': True,
                'wasserstein_distance': wasserstein_dist,
                'orig_avg': orig_avg,
                'gen_avg': gen_avg
            }
            
            distances.append(wasserstein_dist)
            
        except Exception as e:
            detailed_results[bin_idx] = {
                'included': False,
                'reason': f'Error: {str(e)}',
                'orig_avg': orig_avg,
                'gen_avg': gen_avg
            }
    
    average_distance = np.mean(distances) if distances else float('nan')
    return average_distance, detailed_results

def parse_exclusion_model_name(model_name):
    """Extract excluded experiment number from model name."""
    # e.g. edm_4kepochs_full_dataset_exclude_15
    match = re.search(r"exclude_(\d+)$", model_name)
    if match:
        return int(match.group(1))
    return None

def get_experiment_settings(experiment_num, params_file):
    """Get experimental settings for a given experiment number."""
    params_df = pd.read_csv(params_file)
    
    # Find the row for this experiment
    exp_row = params_df[params_df['experiment'] == experiment_num]
    
    if exp_row.empty:
        return None
    
    # Extract the features we need: E, P, ms
    settings = [
        float(exp_row.iloc[0]['E']),
        float(exp_row.iloc[0]['P']), 
        float(exp_row.iloc[0]['ms'])
    ]
    
    return settings

def save_generated_spectra(generated_spectra, experiment_num, model_name, results_dir):
    """Save generated spectra to files."""
    from optimize_conditional import create_energy_axis
    save_dir = Path(results_dir) / model_name / str(experiment_num)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(save_dir / "generated_spectra.npy", generated_spectra)
    
    # Also save individual CSV files to match original format
    energy_axis = create_energy_axis(generated_spectra.shape[1])
    
    for i, spectrum in enumerate(generated_spectra):
        df = pd.DataFrame({
            'energy': energy_axis,
            'intensity': spectrum
        })
        df.to_csv(save_dir / f"{i}.csv", index=False)
    
    return save_dir

def plot_comparison(original_spectra, generated_spectra, experiment_num, model_name, save_path=None):
    """Plot comparison between original and generated spectra."""
    from optimize_conditional import create_energy_axis
    # Use model resolution and truncate original spectra to match
    model_resolution = generated_spectra.shape[1]  # Should be 256
    
    # Truncate original spectra to first model_resolution points (highest energy end)
    if original_spectra.shape[1] > model_resolution:
        original_truncated = original_spectra[:, :model_resolution]
        print(f"Truncating original spectra for plotting: {original_spectra.shape[1]} -> {model_resolution} points")
    else:
        original_truncated = original_spectra
    
    # Create energy axis for the model resolution
    energy_axis = create_energy_axis(model_resolution)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average spectra comparison
    plt.subplot(2, 2, 1)
    orig_mean = np.mean(original_truncated, axis=0)
    gen_mean = np.mean(generated_spectra, axis=0)
    
    plt.plot(energy_axis, orig_mean, 'b-', label='Original (avg)', linewidth=2)
    plt.plot(energy_axis, gen_mean, 'r--', label='Generated (avg)', linewidth=2)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectra Comparison - Exp {experiment_num}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Individual sample comparison
    plt.subplot(2, 2, 2)
    n_show = min(5, len(original_truncated), len(generated_spectra))
    
    for i in range(n_show):
        plt.plot(energy_axis, original_truncated[i], 'b-', alpha=0.6, linewidth=1)
        plt.plot(energy_axis, generated_spectra[i], 'r--', alpha=0.6, linewidth=1)
    
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Intensity')
    plt.title(f'Sample Spectra (first {n_show}) - Exp {experiment_num}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Intensity distribution comparison
    plt.subplot(2, 2, 3)
    orig_flat = original_truncated.flatten()
    gen_flat = generated_spectra.flatten()
    
    plt.hist(orig_flat, bins=50, alpha=0.6, label='Original', density=True, color='blue')
    plt.hist(gen_flat, bins=50, alpha=0.6, label='Generated', density=True, color='red')
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.title('Intensity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation comparison
    plt.subplot(2, 2, 4)
    orig_std = np.std(original_truncated, axis=0)
    gen_std = np.std(generated_spectra, axis=0)
    
    plt.plot(energy_axis, orig_std, 'b-', label='Original std', linewidth=2)
    plt.plot(energy_axis, gen_std, 'r--', label='Generated std', linewidth=2)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Standard Deviation')
    plt.title('Spectral Variability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model: {model_name} - Experiment: {experiment_num}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _mean_std_along_energy(spectra, energy_axis):
    """Mean ± std with energy sorted increasing so fill_between is well-defined."""
    order = np.argsort(energy_axis)
    energy = energy_axis[order]
    sorted_spectra = spectra[:, order]
    mean = np.mean(sorted_spectra, axis=0)
    std = np.std(sorted_spectra, axis=0)
    return energy, mean, std


def plot_all_exclusions_overview(panels, save_path, cfg_scale, num_steps):
    """One panel per excluded experiment: real vs generated mean ± std."""
    panels = sorted(panels, key=lambda p: p['excluded_experiment'])
    n = len(panels)
    if n == 0:
        return

    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    energy_axis = None
    for panel in panels:
        if panel.get('energy_axis') is not None:
            energy_axis = np.asarray(panel['energy_axis'])
            break
    if energy_axis is None:
        from optimize_conditional import create_energy_axis
        energy_axis = create_energy_axis(CONFIG['resolution'])
    real_color = '#1f77b4'
    gen_color = '#d62728'

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, panel in zip(axes, panels):
        original = panel['original_spectra']
        generated = panel['generated_spectra']
        if original.shape[1] > generated.shape[1]:
            original = original[:, :generated.shape[1]]

        e_real, real_mean, real_std = _mean_std_along_energy(original, energy_axis)
        e_gen, gen_mean, gen_std = _mean_std_along_energy(generated, energy_axis)

        ax.fill_between(e_real, real_mean - real_std, real_mean + real_std,
                        color=real_color, alpha=0.25, linewidth=0)
        ax.plot(e_real, real_mean, color=real_color, lw=1.4, label='Real')
        ax.fill_between(e_gen, gen_mean - gen_std, gen_mean + gen_std,
                        color=gen_color, alpha=0.25, linewidth=0)
        ax.plot(e_gen, gen_mean, color=gen_color, lw=1.4, ls='--', label='Generated')

        settings = panel.get('experiment_settings')
        exp = panel['excluded_experiment']
        if settings is not None:
            ax.set_title(f'Exp {exp}  E={settings[0]:g}, P={settings[1]:g}, t={settings[2]:g}',
                         fontsize=9)
        else:
            ax.set_title(f'Exp {exp}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    for i, ax in enumerate(axes[:n]):
        if i // ncols == nrows - 1:
            ax.set_xlabel('Energy (MeV)', fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel('Intensity', fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f'Leave-one-out spectra  (CFG={cfg_scale:g}, steps={num_steps})',
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Overview plot saved to {save_path}")


def collect_saved_exclusion_panels(results_dir, data_dir, cfg_scale, num_steps):
    """Reload real/generated spectra from a previous evaluation run."""
    param_dir = Path(results_dir) / f"cfg{cfg_scale}_steps{num_steps}"
    if not param_dir.exists():
        return []

    panels = []
    for model_dir in sorted(param_dir.iterdir(), key=lambda p: parse_exclusion_model_name(p.name) or 0):
        if not model_dir.is_dir():
            continue
        excluded_exp = parse_exclusion_model_name(model_dir.name)
        if excluded_exp is None:
            continue

        exp_dir = model_dir / str(excluded_exp)
        npy_path = exp_dir / "generated_spectra.npy"
        if npy_path.exists():
            generated = np.load(npy_path)
        else:
            generated = load_original_spectra(exp_dir, CONFIG['resolution'])
        original = load_original_spectra(Path(data_dir) / str(excluded_exp), CONFIG['resolution'])
        if generated is None or original is None:
            print(f"Skipping exp {excluded_exp}: missing spectra")
            continue

        panels.append({
            'excluded_experiment': excluded_exp,
            'experiment_settings': get_experiment_settings(excluded_exp, CONFIG['params_file']),
            'original_spectra': original,
            'generated_spectra': generated,
            'energy_axis': load_energy_axis(Path(data_dir) / str(excluded_exp), CONFIG['resolution']),
        })
    return panels

def evaluate_all_exclusion_models(cfg_scale=3.0, num_steps=18):
    """Main evaluation function."""
    
    # Create results directory
    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Find all exclusion models (edm_4kepochs_full_dataset_exclude_*)
    model_dir = Path(CONFIG['model_dir'])
    exclusion_models = [
        d for d in model_dir.iterdir()
        if d.is_dir() and 'exclude_' in d.name
    ]
    exclusion_models.sort(key=lambda p: parse_exclusion_model_name(p.name) or 0)
    
    print(f"Found {len(exclusion_models)} exclusion models")
    print(f"Using CFG scale: {cfg_scale}, Sampling steps: {num_steps}")
    
    # Results storage
    all_results = []

    # Sufficient statistics for the per-exclusion training normalization
    print("Computing per-exclusion normalization statistics...")
    folder_stats = _folder_intensity_stats(CONFIG['data_dir'], CONFIG['resolution'])

    for model_path in tqdm(exclusion_models, desc="Evaluating models"):
        model_name = model_path.name
        
        # Extract excluded experiment number
        excluded_exp = parse_exclusion_model_name(model_name)
        if excluded_exp is None:
            print(f"Could not parse experiment number from {model_name}")
            continue
        
        print(f"\nEvaluating {model_name} (excluded experiment: {excluded_exp})")
        
        try:
            # Constants this model was trained with (not stored in checkpoints)
            max_intensity, sigma_data = compute_norm_stats(folder_stats, excluded_exp)
            print(f"Training normalization: max_intensity={max_intensity:.4f}, "
                  f"sigma_data={sigma_data:.4f}")

            # Load the model
            model = load_exclusion_model(str(model_path), CONFIG, sigma_data=sigma_data)
            from src.diffusion import EdmSampler
            sampler = EdmSampler(net=model, num_steps=num_steps)
            
            # Get experimental settings for the excluded experiment
            experiment_settings = get_experiment_settings(excluded_exp, CONFIG['params_file'])
            if experiment_settings is None:
                print(f"Could not find settings for experiment {excluded_exp}")
                continue
            
            print(f"Experiment settings: E={experiment_settings[0]}, P={experiment_settings[1]}, ms={experiment_settings[2]}")
            
            # Load original spectra for the excluded experiment
            original_data_path = Path(CONFIG['data_dir']) / str(excluded_exp)
            original_spectra = load_original_spectra(original_data_path, CONFIG['resolution'])
            
            if original_spectra is None:
                print(f"Could not load original spectra for experiment {excluded_exp}")
                continue
            
            n_original_samples = len(original_spectra)
            print(f"Loaded {n_original_samples} original spectra")
            
            # Generate the same number of samples
            print(f"Generating {n_original_samples} samples...")
            generated_spectra = sample_spectra_for_experiment(
                model, sampler, experiment_settings, n_original_samples, CONFIG, cfg_scale
            )
            
            print(f"Generated spectra shape: {generated_spectra.shape}")

            # transform_vector() in the sampler de-normalizes with a hardcoded 2.5,
            # but training normalized by max_intensity (the full-file max of this
            # model's reduced dataset). Rescale so generated spectra are in the
            # same physical units as the raw experimental spectra.
            generated_spectra = generated_spectra * (max_intensity / 2.5)
            
            # Create subdirectory for this parameter combination
            param_dir = f"cfg{cfg_scale}_steps{num_steps}"
            save_dir = save_generated_spectra(generated_spectra, excluded_exp, model_name, 
                                            str(results_dir / param_dir))
            print(f"Saved generated spectra to {save_dir}")
            
            # Calculate Wasserstein distance
            avg_wasserstein, detailed_results = calculate_wasserstein_distance_1d_spectra(
                original_spectra, generated_spectra, min_avg_intensity=0.01
            )
            
            print(f"Average Wasserstein distance: {avg_wasserstein:.6f}")
            
            # Create comparison plot
            plot_path = save_dir / "comparison_plot.png"
            plot_comparison(original_spectra, generated_spectra, excluded_exp, model_name, plot_path)
            
            # Store results
            result = {
                'model_name': model_name,
                'excluded_experiment': excluded_exp,
                'cfg_scale': cfg_scale,
                'num_steps': num_steps,
                'experiment_settings': experiment_settings,
                'n_samples': n_original_samples,
                'max_intensity': max_intensity,
                'sigma_data': sigma_data,
                'avg_wasserstein_distance': avg_wasserstein,
                'original_mean_intensity': np.mean(original_spectra),
                'generated_mean_intensity': np.mean(generated_spectra),
                'original_std_intensity': np.std(original_spectra),
                'generated_std_intensity': np.std(generated_spectra),
                'detailed_results': detailed_results,
                'original_spectra': original_spectra,
                'generated_spectra': generated_spectra,
                'energy_axis': load_energy_axis(original_data_path, CONFIG['resolution']),
            }
            
            all_results.append(result)
            
        except Exception as e:
            import traceback
            print(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()
            continue
        finally:
            if 'model' in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if all_results:
        param_dir = results_dir / f"cfg{cfg_scale}_steps{num_steps}"
        plot_all_exclusions_overview(
            all_results,
            param_dir / "all_exclusions_overview.png",
            cfg_scale,
            num_steps,
        )
    
    return all_results

def run_comprehensive_evaluation():
    """Run evaluation on all exclusion models at the configured sampler settings."""
    
    cfg_scales = [CONFIG['cfg_scale']]
    num_steps_list = [CONFIG['num_steps']]
    
    print("Starting exclusion model evaluation...")
    print(f"CFG scales: {cfg_scales}")
    print(f"Sampling steps: {num_steps_list}")
    
    # Collect all results
    all_comprehensive_results = []
    
    for cfg_scale in cfg_scales:
        for num_steps in num_steps_list:
            print(f"\n{'='*80}")
            print(f"EVALUATING: CFG={cfg_scale}, Steps={num_steps}")
            print(f"{'='*80}")
            
            # Run evaluation for this parameter combination
            results = evaluate_all_exclusion_models(cfg_scale=cfg_scale, num_steps=num_steps)
            all_comprehensive_results.extend(results)
    
    # Process results
    results_dir = Path(CONFIG['results_dir'])
    
    # Save detailed results
    summary_results = []
    for result in all_comprehensive_results:
        summary_results.append({
            'model_name': result['model_name'],
            'excluded_experiment': result['excluded_experiment'],
            'cfg_scale': result['cfg_scale'],
            'num_steps': result['num_steps'],
            'E': result['experiment_settings'][0],
            'P': result['experiment_settings'][1], 
            'ms': result['experiment_settings'][2],
            'n_samples': result['n_samples'],
            'max_intensity': result['max_intensity'],
            'sigma_data': result['sigma_data'],
            'avg_wasserstein_distance': result['avg_wasserstein_distance'],
            'original_mean_intensity': result['original_mean_intensity'],
            'generated_mean_intensity': result['generated_mean_intensity'],
            'original_std_intensity': result['original_std_intensity'],
            'generated_std_intensity': result['generated_std_intensity']
        })
    
    # Save comprehensive results
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        comprehensive_csv_path = results_dir / "comprehensive_evaluation_results.csv"
        summary_df.to_csv(comprehensive_csv_path, index=False)
        print(f"\nComprehensive results saved to {comprehensive_csv_path}")
        
        # Calculate averages across experiments for each parameter combination
        param_averages = []
        for cfg_scale in cfg_scales:
            for num_steps in num_steps_list:
                param_results = summary_df[
                    (summary_df['cfg_scale'] == cfg_scale) & 
                    (summary_df['num_steps'] == num_steps)
                ]
                
                if not param_results.empty:
                    avg_result = {
                        'cfg_scale': cfg_scale,
                        'num_steps': num_steps,
                        'n_experiments': len(param_results),
                        'avg_wasserstein_distance': param_results['avg_wasserstein_distance'].mean(),
                        'std_wasserstein_distance': param_results['avg_wasserstein_distance'].std(),
                        'min_wasserstein_distance': param_results['avg_wasserstein_distance'].min(),
                        'max_wasserstein_distance': param_results['avg_wasserstein_distance'].max(),
                        'avg_original_mean_intensity': param_results['original_mean_intensity'].mean(),
                        'avg_generated_mean_intensity': param_results['generated_mean_intensity'].mean(),
                    }
                    param_averages.append(avg_result)
        
        # Save parameter averages
        if param_averages:
            param_df = pd.DataFrame(param_averages)
            param_csv_path = results_dir / "parameter_averages.csv"
            param_df.to_csv(param_csv_path, index=False)
            print(f"Parameter averages saved to {param_csv_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("COMPREHENSIVE EVALUATION SUMMARY")
            print("="*80)
            print(f"Total parameter combinations: {len(param_averages)}")
            
            # Find best parameters
            best_params = param_df.loc[param_df['avg_wasserstein_distance'].idxmin()]
            print(f"\nBest parameters (lowest avg Wasserstein distance):")
            print(f"  CFG Scale: {best_params['cfg_scale']}")
            print(f"  Sampling Steps: {best_params['num_steps']}")
            print(f"  Average Wasserstein Distance: {best_params['avg_wasserstein_distance']:.6f}")
            print(f"  Evaluated on {best_params['n_experiments']} experiments")
            
            # Print top 5 parameter combinations
            print(f"\nTop 5 parameter combinations:")
            top_5 = param_df.nsmallest(5, 'avg_wasserstein_distance')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {i}. CFG={row['cfg_scale']}, Steps={row['num_steps']}: "
                      f"{row['avg_wasserstein_distance']:.6f} ± {row['std_wasserstein_distance']:.6f}")
    
    return all_comprehensive_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot-only', action='store_true',
        help='Rebuild the 22-experiment overview from saved spectra without resampling',
    )
    args = parser.parse_args()

    if args.plot_only:
        panels = collect_saved_exclusion_panels(
            CONFIG['results_dir'], CONFIG['data_dir'],
            CONFIG['cfg_scale'], CONFIG['num_steps'],
        )
        if not panels:
            raise SystemExit('No saved exclusion spectra found to plot.')
        out = Path(CONFIG['results_dir']) / f"cfg{CONFIG['cfg_scale']}_steps{CONFIG['num_steps']}" / "all_exclusions_overview.png"
        plot_all_exclusions_overview(panels, out, CONFIG['cfg_scale'], CONFIG['num_steps'])
    else:
        print("Starting exclusion model evaluation...")
        results = run_comprehensive_evaluation()
        print("Exclusion evaluation complete!") 