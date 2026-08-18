"""
2D loss landscape L(E, P) for the spectrum-matching objective, with t_open fixed.

Tests the degeneracy hypothesis: whether recovered optima lie on an elongated
low-loss valley rather than at a unique point. The loss is exactly the MSE the
optimizer minimizes (model-generated spectrum vs the target CSV), reusing
SpectrumMatchingOptimizer so the landscape matches the optimization run.

The sampler latents are initialized once and reused for every grid cell, so the
surface reflects parameter changes rather than per-evaluation sampling noise.

    python plot_loss_landscape.py                        # 40x40, default window
    python plot_loss_landscape.py --resolution 8 --output /tmp/ll_test.png
    python plot_loss_landscape.py --e-min 5 --e-max 50 --p-min 1 --p-max 50
    python plot_loss_landscape.py --results comparison_xxx/results.json
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from optimize_match_spectrum import load_config, build_opt_params, SpectrumMatchingOptimizer

# Reference regions (for context rectangles); see conversation/analysis.
OPT_BOUNDS = {'E': (5.0, 50.0), 'P': (1.0, 50.0)}
TRAIN_RANGE = {'E': (12.0, 26.0), 'P': (10.0, 38.0)}


def compute_landscape(optimizer, e_vals, p_vals, t_open, device):
    """Evaluate the MSE loss over the (E, P) grid at fixed t_open."""
    Z = np.zeros((len(p_vals), len(e_vals)), dtype=np.float64)
    total = Z.size
    with torch.no_grad():
        for j, P in enumerate(p_vals):
            for i, E in enumerate(e_vals):
                settings = torch.tensor([[E, P, t_open]], device=device, dtype=torch.float32)
                spectrum = optimizer._sample_spectrum(settings)
                Z[j, i] = optimizer._compute_mse(spectrum).item()
            done = (j + 1) * len(e_vals)
            print(f"  row {j+1}/{len(p_vals)} ({done}/{total} cells)", end='\r')
    print()
    return Z


def load_recovered_optima(results_path):
    """Extract (E, P) from best_params in a results.json.

    Handles both shapes the optimizer writes: comparison mode
    {approach: {seed: {best_params}}} and multi mode [{seed, best_params}].
    """
    with open(results_path) as f:
        results = json.load(f)

    def collect(entry, out):
        bp = entry.get('best_params') if isinstance(entry, dict) else None
        if bp and len(bp) >= 2:
            out.append((bp[0], bp[1]))  # best_params = [E, P, t_open]

    points = []
    if isinstance(results, list):                 # multi mode
        for entry in results:
            collect(entry, points)
    elif isinstance(results, dict):               # comparison mode
        for seed_map in results.values():
            for entry in seed_map.values():
                collect(entry, points)
    return points


def main():
    parser = argparse.ArgumentParser(description="Plot L(E, P) loss landscape at fixed t_open")
    parser.add_argument('--config', default='configs/comparison_langevin_avg_45_25_20.json',
                        help='Config defining model/target/sampler (loss definition)')
    parser.add_argument('--device', default=None, help='Override device from config')
    parser.add_argument('--e-min', type=float, default=10.0)
    parser.add_argument('--e-max', type=float, default=60.0)
    parser.add_argument('--p-min', type=float, default=1.0)
    parser.add_argument('--p-max', type=float, default=55.0)
    parser.add_argument('--resolution', type=int, default=40, help='Grid points per axis')
    parser.add_argument('--t-open', type=float, default=20.0, help='Fixed acquisition time (ms)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for latent initialization')
    parser.add_argument('--mark', default='45,25', help='"E,P" optimum marker, or "none"')
    parser.add_argument('--results', default=None, help='results.json to overlay recovered optima')
    parser.add_argument('--linear-color', action='store_true', help='Linear color scale (default log)')
    parser.add_argument('--output', default='loss_landscape_EP.png')
    args = parser.parse_args()

    # Reuse the exact loss setup from the optimization config
    config = load_config(args.config)
    opt_params = build_opt_params(config)
    if args.device is not None:
        opt_params['device'] = args.device if torch.cuda.is_available() else 'cpu'
    device = opt_params['device']

    print(f"Loss landscape over E[{args.e_min},{args.e_max}] x P[{args.p_min},{args.p_max}], "
          f"t_open={args.t_open}, {args.resolution}x{args.resolution}")
    print(f"Target: {opt_params['target_spectrum_csv']}  |  device: {device}")

    # Single optimizer instance: loads model, fixes latents (seeded)
    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=args.seed)

    e_vals = np.linspace(args.e_min, args.e_max, args.resolution)
    p_vals = np.linspace(args.p_min, args.p_max, args.resolution)
    Z = compute_landscape(optimizer, e_vals, p_vals, args.t_open, device)

    # Report the grid minimum
    jmin, imin = np.unravel_index(np.argmin(Z), Z.shape)
    print(f"Grid min: L={Z[jmin, imin]:.6g} at E={e_vals[imin]:.2f}, P={p_vals[jmin]:.2f}")

    # Save raw grid for reuse
    npz_path = os.path.splitext(args.output)[0] + '.npz'
    np.savez(npz_path, E=e_vals, P=p_vals, Z=Z, t_open=args.t_open)
    print(f"Saved grid: {npz_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    norm = None if args.linear_color else LogNorm(vmin=max(Z.min(), 1e-9), vmax=Z.max())
    mesh = ax.pcolormesh(e_vals, p_vals, Z, shading='auto', cmap='viridis', norm=norm)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Loss (MSE)')

    # Contour lines for valley shape
    levels = np.logspace(np.log10(max(Z.min(), 1e-9)), np.log10(Z.max()), 12) if not args.linear_color \
        else np.linspace(Z.min(), Z.max(), 12)
    cs = ax.contour(e_vals, p_vals, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.6)

    # True optimum marker
    if args.mark.lower() != 'none':
        me, mp = (float(v) for v in args.mark.split(','))
        ax.plot(me, mp, marker='*', markersize=18, color='red', markeredgecolor='black',
                label=f'Target optimum ({me:g}, {mp:g})')

    # Recovered optima overlay
    if args.results and os.path.exists(args.results):
        pts = load_recovered_optima(args.results)
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=30, color='orange', edgecolor='black', linewidths=0.5,
                       label=f'Recovered optima (n={len(pts)})', zorder=5)

    # Context rectangles: optimizer bounds and training range
    for region, style, name in [(OPT_BOUNDS, dict(edgecolor='white', linestyle='--', linewidth=1.2), 'opt bounds'),
                                (TRAIN_RANGE, dict(edgecolor='cyan', linestyle=':', linewidth=1.2), 'training range')]:
        (e0, e1), (p0, p1) = region['E'], region['P']
        ax.add_patch(plt.Rectangle((e0, p0), e1 - e0, p1 - p0, fill=False, **style))
        ax.text(e1, p1, f' {name}', color=style['edgecolor'], fontsize=7, va='bottom', ha='left')

    ax.set_xlabel('E — laser energy')
    ax.set_ylabel('P — pressure [bar]')
    ax.set_title(f'Loss landscape L(E, P) at t_open = {args.t_open:g} ms')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {args.output}")
    plt.close()


if __name__ == '__main__':
    main()
