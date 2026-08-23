"""
Loss landscape scan for the spectrum-matching objective.

Scans the merit function L over the conditioning parameters (E, P, t_open) and
plots it, to test whether recovered optima lie on a degenerate low-loss valley
rather than at a unique point.

Two modes:
  3d (default) - scan the full (E, P, t_open) cube, then plot the three parameter
                 pairs. Each pair is shown twice: as a SLICE through the optimum
                 (the third parameter fixed) and as a MIN-PROJECTION (best loss
                 over any value of the third parameter). The min-projection is the
                 stronger degeneracy test: it asks whether some other parameter can
                 compensate.
  2d           - scan a single (E, P) plane at fixed t_open.

The loss is exactly the MSE the optimizer minimizes, reusing
SpectrumMatchingOptimizer so the landscape matches the optimization runs.

Determinism: the sampler latents are initialized once and the SAME latents are
reused for every grid point, so the surface reflects parameter changes rather than
sampling noise. Grid points are evaluated in batches through a single sampler call;
with S_churn=0 and the model in eval() this is numerically identical to evaluating
them one at a time (verify with --verify-batching).

    python plot_loss_landscape.py                             # 3D cube, 25^3
    python plot_loss_landscape.py --resolution 30
    python plot_loss_landscape.py --mode 2d --resolution 40    # single (E,P) plane
    python plot_loss_landscape.py --from-npz loss_landscape_3d.npz   # replot only
    python plot_loss_landscape.py --self-test                  # axes check, no GPU
    python plot_loss_landscape.py --verify-batching 6          # batched == sequential
    python plot_loss_landscape.py --results comparison_xxx/results.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Axis metadata. Cube Z is indexed Z[i_E, j_P, k_T].
AXES = ['E', 'P', 't_open']
LABELS = {
    'E': 'E — laser energy',
    'P': 'P — pressure [bar]',
    't_open': 't_open — acquisition time [ms]',
}
# Reference regions for context rectangles (optimizer bounds / training data range).
OPT_BOUNDS = {'E': (5.0, 50.0), 'P': (1.0, 50.0), 't_open': (5.0, 100.0)}
TRAIN_RANGE = {'E': (12.0, 26.0), 'P': (10.0, 38.0), 't_open': (10.0, 40.0)}
PAIRS = [('E', 'P'), ('E', 't_open'), ('P', 't_open')]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_points(optimizer, points, batch_points=8, progress_every=25):
    """Evaluate the MSE loss at each row of `points` (N, 3) = [E, P, t_open].

    Points are evaluated in chunks of `batch_points` through one sampler call each.
    Every point is given the same batch of latents (tiled), so the result matches
    sequential evaluation exactly while running far fewer GPU calls.
    """
    import torch

    device = optimizer.device
    n_latents = optimizer.batch_size
    resolution = optimizer.spectrum_length

    sampler = optimizer.sampler
    base_latents = sampler.stored_latents            # (batch_size, 1, resolution)
    base_shape = sampler.latents_shape
    if base_latents is None:
        raise RuntimeError("Sampler latents are not initialized")

    losses = np.empty(len(points), dtype=np.float64)
    n_chunks = int(np.ceil(len(points) / batch_points))

    try:
        with torch.no_grad():
            for c, start in enumerate(range(0, len(points), batch_points)):
                chunk = points[start:start + batch_points]
                k = len(chunk)

                # Same latents for every point in the chunk: k blocks of batch_size.
                sampler.stored_latents = base_latents.repeat(k, 1, 1)
                sampler.latents_shape = (k * n_latents, 1, resolution)

                settings = torch.tensor(np.asarray(chunk), device=device, dtype=torch.float32)
                settings = settings.repeat_interleave(n_latents, dim=0)   # (k*batch_size, 3)

                x = sampler.sample_differentiable(
                    resolution=resolution, device=device, settings=settings,
                    n_samples=k * n_latents, cfg_scale=optimizer.cfg_scale,
                    settings_dim=len(optimizer.features),
                    smooth_output=optimizer.smooth_output,
                    smooth_kernel_size=optimizer.smooth_kernel_size,
                    smooth_sigma=optimizer.smooth_sigma,
                )
                # (k*batch_size, 1, res) -> average the latents within each point
                spectra = x.squeeze(1).view(k, n_latents, resolution).mean(dim=1)

                for r in range(k):
                    losses[start + r] = optimizer._compute_mse(spectra[r]).item()

                if (c + 1) % progress_every == 0 or c + 1 == n_chunks:
                    done = min(start + batch_points, len(points))
                    print(f"    {done}/{len(points)} points", end='\r')
    finally:
        sampler.stored_latents = base_latents
        sampler.latents_shape = base_shape
    print()
    return losses


def verify_batching(optimizer, n_points, rng, ranges):
    """Sanity check: batched evaluation must match one-at-a-time evaluation."""
    pts = np.column_stack([rng.uniform(*ranges[a], size=n_points) for a in AXES])
    batched = evaluate_points(optimizer, pts, batch_points=max(2, n_points))
    single = evaluate_points(optimizer, pts, batch_points=1)
    diff = np.abs(batched - single)
    print(f"  batched vs sequential: max |diff| = {diff.max():.3e}, "
          f"max relative = {(diff / np.maximum(single, 1e-12)).max():.3e}")
    return diff.max()


def scan_grid(optimizer, grids, fixed=None, batch_points=8):
    """Scan a full grid. `grids` maps axis name -> values; `fixed` maps the rest.

    Returns an array shaped by the scanned axes in AXES order.
    """
    scanned = [a for a in AXES if a in grids]
    mesh = np.meshgrid(*[grids[a] for a in scanned], indexing='ij')
    shape = mesh[0].shape

    points = np.zeros((mesh[0].size, 3))
    for col, axis in enumerate(AXES):
        if axis in grids:
            points[:, col] = mesh[scanned.index(axis)].ravel()
        else:
            points[:, col] = fixed[axis]

    print(f"  evaluating {len(points)} points in chunks of {batch_points}...")
    losses = evaluate_points(optimizer, points, batch_points=batch_points)
    return losses.reshape(shape)


# =============================================================================
# PLOTTING
# =============================================================================

def reduce_pair(Z, xa, ya, reduce, center_idx, maximize=False):
    """Collapse the cube Z[i_E, j_P, k_T] onto the (xa, ya) plane.

    Returns an array indexed [y, x], ready for pcolormesh(x_vals, y_vals, ...).
    The 'min' reduction means "best achievable over the third parameter", so it
    becomes a max-projection when the objective is maximized.
    """
    other = [a for a in AXES if a not in (xa, ya)][0]
    axis = AXES.index(other)
    if reduce == 'min':
        plane = Z.max(axis=axis) if maximize else Z.min(axis=axis)
    else:  # slice through the centre value
        plane = np.take(Z, center_idx[other], axis=axis)
    # `plane` is indexed by the two remaining axes in AXES order; orient as [y, x]
    remaining = [a for a in AXES if a != other]
    return plane if remaining == [ya, xa] else plane.T


def draw_panel(ax, x_vals, y_vals, plane, xa, ya, linear=False, mark=None,
               overlay=None, context=True, norm=None):
    """Draw one panel. Pass `norm` to share a colour scale across panels.

    Non-positive cells cannot be shown on a log scale; they are masked and drawn
    in the colormap's "bad" colour (grey), which for the beam objectives means
    the model produced a degenerate flat spectrum there - no beam at all.
    """
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#d9d9d9')
    if norm is None:
        vmin = max(plane[plane > 0].min() if np.any(plane > 0) else 1e-12, 1e-12)
        norm = None if linear else LogNorm(vmin=vmin, vmax=plane.max())
    shown = plane if linear else np.ma.masked_less_equal(plane, 0)
    mesh = ax.pcolormesh(x_vals, y_vals, shown, shading='auto', cmap=cmap, norm=norm)

    if linear:
        levels = np.linspace(plane.min(), plane.max(), 12)
    else:
        lo, hi = norm.vmin, norm.vmax
        levels = np.logspace(np.log10(lo), np.log10(hi), 10)
    ax.contour(x_vals, y_vals, shown, levels=levels, colors='white', alpha=0.3, linewidths=0.6)

    if mark is not None:
        ax.plot(mark[0], mark[1], marker='*', markersize=16, color='red',
                markeredgecolor='black', linestyle='none', label='Optimum / slice')

    if overlay:
        xs, ys = zip(*overlay)
        ax.scatter(xs, ys, s=26, color='orange', edgecolor='black', linewidths=0.5,
                   zorder=5, label=f'Recovered (n={len(overlay)})')

    if context:
        for region, colour, name in ((OPT_BOUNDS, 'white', 'bounds'),
                                     (TRAIN_RANGE, 'cyan', 'training')):
            x0, x1 = region[xa]
            y0, y1 = region[ya]
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=colour, linestyle='--', linewidth=1.0))
            ax.text(x1, y1, f' {name}', color=colour, fontsize=6, va='bottom', ha='left')

    ax.set_xlabel(LABELS[xa])
    ax.set_ylabel(LABELS[ya])
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(y_vals.min(), y_vals.max())
    return mesh


def plot_pairs(grids, Z, centre, output, linear=False, overlay_pts=None,
               reduces=('slice', 'min'), maximize=False, value_label='Loss (MSE)',
               dynamic_range=4):
    """Plot all three parameter pairs, one row per reduction."""
    center_idx = {a: int(np.argmin(np.abs(grids[a] - centre[a]))) for a in AXES}

    # Build every panel first so they can share one colour scale. Without this
    # each panel self-scales and the rows cannot be compared by eye.
    planes = {(r, c): reduce_pair(Z, xa, ya, reduce, center_idx, maximize=maximize)
              for r, reduce in enumerate(reduces)
              for c, (xa, ya) in enumerate(PAIRS)}
    allv = np.concatenate([p.ravel() for p in planes.values()])
    pos = allv[allv > 0]
    shared = None
    if not linear and pos.size:
        vmax = float(pos.max())
        # Clip the useless low tail: keep `dynamic_range` decades below the peak,
        # but never below the smallest positive value actually present.
        vmin = max(float(pos.min()), vmax / 10 ** dynamic_range)
        shared = LogNorm(vmin=vmin, vmax=vmax)
        n_bad = int((allv <= 0).sum())
        print(f"  colour scale: {vmin:.3g} .. {vmax:.3g} ({dynamic_range} decades, shared); "
              f"{n_bad}/{allv.size} cells non-positive -> grey")

    fig, axes = plt.subplots(len(reduces), 3, figsize=(19, 5.6 * len(reduces)), squeeze=False)
    for r, reduce in enumerate(reduces):
        for c, (xa, ya) in enumerate(PAIRS):
            ax = axes[r][c]
            plane = planes[(r, c)]
            other = [a for a in AXES if a not in (xa, ya)][0]
            overlay = ([(p[AXES.index(xa)], p[AXES.index(ya)]) for p in overlay_pts]
                       if overlay_pts else None)
            mesh = draw_panel(ax, grids[xa], grids[ya], plane, xa, ya, linear=linear,
                              mark=(centre[xa], centre[ya]), overlay=overlay, norm=shared)
            fig.colorbar(mesh, ax=ax).set_label(value_label)
            if reduce == 'min':
                ax.set_title(f'{"max" if maximize else "min"} over '
                             f'{LABELS[other].split(" — ")[0]}')
            else:
                ax.set_title(f'{LABELS[other].split(" — ")[0]} = {grids[other][center_idx[other]]:.3g} (slice)')
            if r == 0 and c == 0:
                ax.legend(loc='upper left', fontsize=7)

    proj = 'max' if maximize else 'min'
    fig.suptitle(f'{value_label} over parameter pairs '
                 f'(top: slice through optimum, bottom: {proj}-projection)'
                 if len(reduces) > 1 else f'{value_label} over parameter pairs', fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output}")
    plt.close()


def plot_single_plane(x_vals, y_vals, plane, xa, ya, third_name, third_val, output,
                      centre=None, linear=False, overlay_pts=None,
                      value_label='Loss (MSE)'):
    """Plot one (x, y) plane - the 2D mode."""
    fig, ax = plt.subplots(figsize=(9, 7))
    overlay = ([(p[AXES.index(xa)], p[AXES.index(ya)]) for p in overlay_pts]
               if overlay_pts else None)
    mark = (centre[xa], centre[ya]) if centre else None
    mesh = draw_panel(ax, x_vals, y_vals, plane, xa, ya, linear=linear,
                      mark=mark, overlay=overlay)
    fig.colorbar(mesh, ax=ax).set_label(value_label)
    ax.set_title(f'Loss landscape L({xa}, {ya}) at {third_name} = {third_val:g}')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output}")
    plt.close()


# =============================================================================
# HELPERS
# =============================================================================

def load_recovered_optima(results_path):
    """Extract [E, P, t_open] from best_params in a results.json.

    Handles both shapes the optimizer writes: comparison mode
    {approach: {seed: {best_params}}} and multi mode [{seed, best_params}].
    """
    with open(results_path) as f:
        results = json.load(f)

    points = []

    def collect(entry):
        bp = entry.get('best_params') if isinstance(entry, dict) else None
        if bp and len(bp) >= 3:
            points.append(list(bp[:3]))

    if isinstance(results, list):
        for entry in results:
            collect(entry)
    elif isinstance(results, dict):
        for seed_map in results.values():
            for entry in seed_map.values():
                collect(entry)
    return points


def resolve_centre(spec, grids, Z, maximize):
    """Marker / slice location: an explicit "E,P,t", or the grid optimum."""
    if spec not in (None, 'auto'):
        return dict(zip(AXES, [float(v) for v in spec.split(',')]))
    idx = np.unravel_index(np.argmax(Z) if maximize else np.argmin(Z), Z.shape)
    scanned = [a for a in AXES if a in grids]
    centre = {a: float(grids[a][i]) for a, i in zip(scanned, idx)}
    for a in AXES:                      # 2d mode leaves one axis unscanned
        centre.setdefault(a, float(np.mean(grids[a])) if a in grids else 0.0)
    print("  marker/slice set to the grid optimum: "
          + ", ".join(f"{a}={centre[a]:.2f}" for a in scanned))
    return centre


def report_minimum(grids, Z, maximize=False, value_label='L'):
    """Print where the scanned optimum sits."""
    idx = np.unravel_index(np.argmax(Z) if maximize else np.argmin(Z), Z.shape)
    scanned = [a for a in AXES if a in grids]
    loc = ", ".join(f"{a}={grids[a][i]:.2f}" for a, i in zip(scanned, idx))
    print(f"Grid {'max' if maximize else 'min'}: {value_label}={Z[idx]:.6g} at {loc}")


def self_test():
    """Check axis/orientation handling on a synthetic cube (no GPU, no model)."""
    print("Self-test: synthetic cube L = (E-45)^2 + 0.5*(P-25)^2 + 0.1*(t-20)^2")
    grids = {'E': np.linspace(10, 60, 21), 'P': np.linspace(1, 55, 23),
             't_open': np.linspace(5, 60, 25)}
    EE, PP, TT = np.meshgrid(grids['E'], grids['P'], grids['t_open'], indexing='ij')
    Z = (EE - 45) ** 2 + 0.5 * (PP - 25) ** 2 + 0.1 * (TT - 20) ** 2 + 1e-6
    centre = {'E': 45.0, 'P': 25.0, 't_open': 20.0}
    center_idx = {a: int(np.argmin(np.abs(grids[a] - centre[a]))) for a in AXES}

    ok = True
    for xa, ya in PAIRS:
        plane = reduce_pair(Z, xa, ya, 'min', center_idx)
        assert plane.shape == (len(grids[ya]), len(grids[xa])), \
            f"{xa},{ya}: shape {plane.shape} != {(len(grids[ya]), len(grids[xa]))}"
        # min-projection minimum must land on the centre in both plotted axes
        jy, ix = np.unravel_index(np.argmin(plane), plane.shape)
        got = (grids[xa][ix], grids[ya][jy])
        want = (grids[xa][center_idx[xa]], grids[ya][center_idx[ya]])
        match = np.allclose(got, want)
        ok &= match
        print(f"  pair ({xa:6s},{ya:6s}) shape={plane.shape} argmin={got} expected={want} {'OK' if match else 'MISMATCH'}")
    report_minimum(grids, Z)
    plot_pairs(grids, Z, centre, 'loss_landscape_selftest.png')
    print("Self-test", "passed" if ok else "FAILED")
    return ok


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scan and plot the loss landscape")
    parser.add_argument('--config', default='configs/comparison_langevin_avg_45_25_20.json',
                        help='Config defining model/target/sampler (the loss definition)')
    parser.add_argument('--mode', choices=['2d', '3d'], default='3d')
    parser.add_argument('--device', default=None, help='Override device from config')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Points per axis (default 25 for 3d, 40 for 2d)')
    parser.add_argument('--e-min', type=float, default=10.0)
    parser.add_argument('--e-max', type=float, default=60.0)
    parser.add_argument('--p-min', type=float, default=1.0)
    parser.add_argument('--p-max', type=float, default=55.0)
    parser.add_argument('--t-min', type=float, default=5.0)
    parser.add_argument('--t-max', type=float, default=60.0)
    parser.add_argument('--t-open', type=float, default=20.0, help='Fixed t_open in 2d mode')
    parser.add_argument('--centre', '--center', dest='centre', default=None,
                        help='"E,P,t" marker and slice location, or "auto" for the grid '
                             'optimum (default: 45,25,20 for the MSE objective, auto '
                             'for the maximized beam objectives)')
    parser.add_argument('--dynamic-range', type=float, default=4,
                        help='Decades below the peak to show on the shared log colour scale')
    parser.add_argument('--batch-points', type=int, default=8,
                        help='Grid points per sampler call (memory/speed knob)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for latent initialization')
    parser.add_argument('--results', default=None, help='results.json to overlay recovered optima')
    parser.add_argument('--linear-color', action='store_true', help='Linear color scale (default log)')
    parser.add_argument('--output', default=None, help='Output PNG (default depends on mode)')
    parser.add_argument('--from-npz', default=None, help='Replot from a saved scan, no recompute')
    parser.add_argument('--self-test', action='store_true', help='Check axis handling, no GPU needed')
    parser.add_argument('--verify-batching', type=int, default=0, metavar='N',
                        help='Evaluate N random points batched vs sequential and compare')
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)

    overlay_pts = (load_recovered_optima(args.results)
                   if args.results and os.path.exists(args.results) else None)
    if overlay_pts:
        print(f"Overlaying {len(overlay_pts)} recovered optima from {args.results}")

    # ---- Replot from a cached scan -----------------------------------------
    if args.from_npz:
        data = np.load(args.from_npz)
        Z = data['Z']
        if Z.ndim == 3:
            grids = {a: data[a] for a in AXES}
            mx = bool(data['maximize']) if 'maximize' in data.files else False
            lbl = str(data['objective']) if 'objective' in data.files else 'Loss (MSE)'
            report_minimum(grids, Z, maximize=mx, value_label=lbl)
            centre = resolve_centre(args.centre if args.centre else ('auto' if mx else '45,25,20'),
                                    grids, Z, mx)
            plot_pairs(grids, Z, centre, args.output or 'loss_landscape_pairs.png',
                       linear=args.linear_color, overlay_pts=overlay_pts,
                       maximize=mx, value_label=lbl, dynamic_range=args.dynamic_range)
        else:
            g = {'E': data['E'], 'P': data['P']}
            centre = resolve_centre(args.centre or '45,25,20', g, Z, False)
            plot_single_plane(data['E'], data['P'], Z, 'E', 'P', 't_open',
                              float(data['t_open']), args.output or 'loss_landscape_EP.png',
                              centre=centre, linear=args.linear_color, overlay_pts=overlay_pts)
        return

    # ---- Build the optimizer (loads model, fixes latents) -------------------
    import torch
    from optimize_match_spectrum import load_config, build_opt_params, SpectrumMatchingOptimizer

    config = load_config(args.config)
    opt_params = build_opt_params(config)
    if args.device is not None:
        opt_params['device'] = args.device if torch.cuda.is_available() else 'cpu'

    print(f"Target: {opt_params['target_spectrum_csv']}  |  device: {opt_params['device']}")
    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=args.seed)

    # Maximized objectives are returned negated (the optimizers minimize); store
    # and plot the physical quantity instead, and flip the projection sense.
    maximize = getattr(optimizer, 'maximize', False)
    objective = getattr(optimizer, 'objective', 'mse')
    VALUE_LABELS = {'mse': 'Loss (MSE)',
                    'beam_energy': r'$\int I(E)\,E^p\,dE$  (charge x energy)',
                    'mean_energy': r'$\int I E^p dE / \int I dE$  (mean energy)'}
    value_label = VALUE_LABELS.get(objective, objective)
    if maximize:
        print(f"Objective '{objective}' is MAXIMIZED; plotting the figure of merit")

    if args.verify_batching:
        ranges = {'E': (args.e_min, args.e_max), 'P': (args.p_min, args.p_max),
                  't_open': (args.t_min, args.t_max)}
        verify_batching(optimizer, args.verify_batching, np.random.default_rng(0), ranges)
        return

    resolution = args.resolution or (25 if args.mode == '3d' else 40)

    if args.mode == '3d':
        grids = {
            'E': np.linspace(args.e_min, args.e_max, resolution),
            'P': np.linspace(args.p_min, args.p_max, resolution),
            't_open': np.linspace(args.t_min, args.t_max, resolution),
        }
        print(f"3D scan {resolution}^3 = {resolution**3} points over "
              f"E[{args.e_min},{args.e_max}] P[{args.p_min},{args.p_max}] t[{args.t_min},{args.t_max}]")
        Z = scan_grid(optimizer, grids, batch_points=args.batch_points)
        if maximize:
            Z = -Z
        report_minimum(grids, Z, maximize=maximize, value_label=value_label.split()[0])

        output = args.output or 'loss_landscape_pairs.png'
        npz = os.path.splitext(output)[0] + '.npz'
        np.savez(npz, Z=Z, target=opt_params['target_spectrum_csv'],
                 objective=objective, maximize=maximize, **grids)
        print(f"Saved grid: {npz}")
        centre = resolve_centre(args.centre if args.centre else ('auto' if maximize else '45,25,20'),
                                grids, Z, maximize)
        plot_pairs(grids, Z, centre, output, linear=args.linear_color,
                   overlay_pts=overlay_pts, maximize=maximize, value_label=value_label,
                   dynamic_range=args.dynamic_range)
    else:
        grids = {
            'E': np.linspace(args.e_min, args.e_max, resolution),
            'P': np.linspace(args.p_min, args.p_max, resolution),
        }
        print(f"2D scan {resolution}x{resolution} = {resolution**2} points at t_open={args.t_open}")
        Z = scan_grid(optimizer, grids, fixed={'t_open': args.t_open},
                      batch_points=args.batch_points)   # indexed [E, P]
        if maximize:
            Z = -Z
        report_minimum(grids, Z, maximize=maximize, value_label=value_label.split()[0])

        output = args.output or 'loss_landscape_EP.png'
        npz = os.path.splitext(output)[0] + '.npz'
        np.savez(npz, Z=Z, E=grids['E'], P=grids['P'], t_open=args.t_open,
                 target=opt_params['target_spectrum_csv'])
        print(f"Saved grid: {npz}")
        centre = resolve_centre(args.centre if args.centre else ('auto' if maximize else '45,25,20'),
                                grids, Z, maximize)
        plot_single_plane(grids['E'], grids['P'], Z.T, 'E', 'P', 't_open', args.t_open,
                          output, centre=centre, linear=args.linear_color,
                          overlay_pts=overlay_pts, value_label=value_label)


if __name__ == '__main__':
    main()
