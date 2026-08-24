"""
Does RAdam-last actually find flatter minima?

The appendix argues that the final stochastic stage relaxes the solution away from
sharp, bin-aligned minima toward wide ones, via the free-energy expression
F = L + (T/2) log det H_perp. That is a mechanism, not a measurement: the MSE
table is equally consistent with the duller explanation that a large-step method
simply keeps descending where L-BFGS stalled. This script tests the claim
directly by measuring curvature and perturbation sensitivity at the endpoints of

    BO -> L-BFGS            (no stochastic stage)
    BO -> L-BFGS -> RAdam   (stochastic stage last)

For a given seed both pipelines share an identical BO -> L-BFGS prefix, so the
second is exactly the first plus 50 RAdam steps. The comparison is therefore
PAIRED within seed, and is analysed with an exact Wilcoxon signed-rank test.

Measured at each endpoint (a 3D problem, so this is cheap):
  * finite-difference Hessian in NORMALISED coordinates -> eigenvalues, trace,
    max eigenvalue, and log det over the non-degenerate subspace (the quantity
    the appendix's free energy actually contains);
  * mean/max excess loss over a sphere of perturbations at one or more radii,
    which is the operationally meaningful quantity: how much the spectrum
    degrades if the machine cannot hold the setting exactly.

Both absolute and loss-relative excesses are reported, because the two arms do
not stop at the same loss and a raw comparison would confound flatness with
depth.

    python flatness_analysis.py --run-dir comparison_20260114_110450
    python flatness_analysis.py --run-dir <dir> --endpoint final   # theory-faithful
    python flatness_analysis.py --self-test                        # no GPU needed
"""

import os
import re
import json
import glob
import argparse
import itertools

import numpy as np


AXES = ['E', 'P', 't_open']
STEP_RE = re.compile(
    r'(Bayesian|LBFGS|Adam|SGD|Langevin) Step (\d+): params=\['
    r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?),\s*'
    r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?),\s*'
    r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\]')


# =============================================================================
# ENDPOINTS
# =============================================================================

def endpoints_from_results(run_dir, approaches):
    """Best-seen endpoint per (approach, seed), as reported in results.json."""
    with open(os.path.join(run_dir, 'results.json')) as f:
        results = json.load(f)
    out = {}
    for approach in approaches:
        if approach not in results:
            raise SystemExit(f"approach {approach!r} not in {run_dir}/results.json; "
                             f"available: {sorted(results)}")
        out[approach] = {int(seed): np.asarray(r['best_params'], dtype=float)
                         for seed, r in results[approach].items()}
    return out


def endpoints_from_logs(run_dir, approaches):
    """Final iterate per (approach, seed): the last logged step of the last stage.

    This is what the free-energy argument is actually about. The reported
    best_params is instead an argmin over the whole trajectory, a selection rule
    that favours sharp minima and so understates the effect being tested.
    """
    out = {a: {} for a in approaches}
    for seed_dir in sorted(glob.glob(os.path.join(run_dir, 'seed_*'))):
        sn = os.path.basename(seed_dir)
        try:
            seed = int(sn.split('_')[1])
        except (IndexError, ValueError):
            continue
        for approach in approaches:
            lf = os.path.join(seed_dir, approach, 'logs', f'{approach}_{sn}.log')
            if not os.path.exists(lf):
                continue
            last = None
            with open(lf) as f:
                for line in f:
                    m = STEP_RE.search(line)
                    if m:
                        last = m
            if last:
                out[approach][seed] = np.array(
                    [float(last.group(i)) for i in (3, 4, 5)], dtype=float)
    return out


# =============================================================================
# PROBE GEOMETRY
# =============================================================================

def sphere_directions(n, rng):
    """n roughly uniform unit directions in 3D (normalised Gaussians)."""
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def build_probes(u0, h, radii, directions):
    """Probe offsets around u0 in normalised coordinates.

    Returns (points, index) where index maps a semantic key to row numbers.
    """
    pts = [u0.copy()]
    index = {'center': 0}

    for i in range(3):                                   # Hessian diagonal
        for sign in (+1, -1):
            e = np.zeros(3); e[i] = sign * h
            index[('diag', i, sign)] = len(pts)
            pts.append(u0 + e)

    for i, j in itertools.combinations(range(3), 2):     # Hessian off-diagonal
        for si in (+1, -1):
            for sj in (+1, -1):
                e = np.zeros(3); e[i] = si * h; e[j] = sj * h
                index[('off', i, j, si, sj)] = len(pts)
                pts.append(u0 + e)

    for r in radii:                                      # perturbation spheres
        rows = []
        for d in directions:
            rows.append(len(pts))
            pts.append(u0 + r * d)
        index[('ball', r)] = rows

    return np.asarray(pts), index


def hessian_from_probes(L, index, h):
    """Symmetric 3x3 finite-difference Hessian in normalised coordinates."""
    H = np.zeros((3, 3))
    L0 = L[index['center']]
    for i in range(3):
        H[i, i] = (L[index[('diag', i, +1)]] - 2 * L0
                   + L[index[('diag', i, -1)]]) / h ** 2
    for i, j in itertools.combinations(range(3), 2):
        val = (L[index[('off', i, j, +1, +1)]] - L[index[('off', i, j, +1, -1)]]
               - L[index[('off', i, j, -1, +1)]] + L[index[('off', i, j, -1, -1)]]) / (4 * h ** 2)
        H[i, j] = H[j, i] = val
    return H


def point_metrics(L, index, h, radii, eig_floor_rel=1e-3):
    """Curvature and sensitivity summaries at one endpoint."""
    L0 = float(L[index['center']])
    H = hessian_from_probes(L, index, h)
    eig = np.linalg.eigvalsh(H)

    pos = eig[eig > eig_floor_rel * max(abs(eig).max(), 1e-30)]
    m = {
        'L_star': L0,
        'trace_H': float(eig.sum()),
        'max_eig': float(eig.max()),
        'min_eig': float(eig.min()),
        'n_eig_nondegenerate': int(pos.size),
        # the free-energy term: log det restricted to the non-degenerate subspace
        'logdet_H_perp': float(np.log(pos).sum()) if pos.size else float('nan'),
    }
    for r in radii:
        vals = L[index[('ball', r)]]
        excess = vals - L0
        m[f'ball{r}_mean_excess'] = float(excess.mean())
        m[f'ball{r}_max_excess'] = float(excess.max())
        m[f'ball{r}_mean_rel'] = float(excess.mean() / L0) if L0 > 0 else float('nan')
    return m


# =============================================================================
# PAIRED STATISTICS
# =============================================================================

def wilcoxon_exact(diffs):
    """Two-sided exact Wilcoxon signed-rank test. Zeros dropped, ties get
    average ranks. Exact null by enumerating all 2^n sign assignments."""
    d = np.asarray([x for x in diffs if x != 0], dtype=float)
    n = d.size
    if n == 0:
        return float('nan'), float('nan'), 0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    absd = np.abs(d)
    for v in np.unique(absd):                       # average ranks within ties
        tie = absd == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()

    W = float(ranks[d > 0].sum())
    total = ranks.sum()
    obs = abs(W - total / 2)
    count = 0
    for signs in itertools.product((0, 1), repeat=n):
        stat = ranks[np.asarray(signs, dtype=bool)].sum()
        if abs(stat - total / 2) >= obs - 1e-12:
            count += 1
    return W, count / 2 ** n, n


def sign_test(diffs):
    pos = int(sum(1 for x in diffs if x > 0))
    neg = int(sum(1 for x in diffs if x < 0))
    return pos, neg


# =============================================================================
# MAIN
# =============================================================================

def self_test():
    """Validate the Hessian and the Wilcoxon test on cases with known answers."""
    ok = True

    # analytic quadratic: L = 0.5 u^T A u  -> Hessian is exactly A
    A = np.array([[3.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 0.4]])
    u0 = np.zeros(3)
    h = 0.01
    dirs = sphere_directions(8, np.random.default_rng(0))
    pts, index = build_probes(u0, h, (0.02,), dirs)
    L = np.array([0.5 * p @ A @ p for p in pts])
    H = hessian_from_probes(L, index, h)
    err = np.abs(H - A).max()
    print(f"  Hessian vs analytic A: max abs error = {err:.2e}", end='  ')
    ok &= err < 1e-6
    print("OK" if err < 1e-6 else "FAIL")

    # anisotropic quadratic: sphere excess should track the mean eigenvalue
    r = 0.02
    vals = L[index[('ball', r)]]
    predicted = 0.5 * r ** 2 * np.trace(A) / 3
    print(f"  sphere mean excess {vals.mean():.3e} vs 0.5 r^2 tr(A)/3 = {predicted:.3e}",
          end='  ')
    close = abs(vals.mean() - predicted) / predicted < 0.5
    ok &= close
    print("OK" if close else "FAIL")

    # Wilcoxon against a hand-checkable case: all differences same sign -> p = 2/2^n
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    W, p, n = wilcoxon_exact(d)
    expect = 2 / 2 ** 5
    print(f"  Wilcoxon all-positive n=5: W={W}, p={p:.4f} (expect {expect:.4f})", end='  ')
    ok &= abs(p - expect) < 1e-12
    print("OK" if abs(p - expect) < 1e-12 else "FAIL")

    # symmetric differences -> p = 1
    W, p, n = wilcoxon_exact([1.0, -1.0, 2.0, -2.0])
    print(f"  Wilcoxon symmetric: p={p:.3f} (expect 1.000)", end='  ')
    ok &= abs(p - 1.0) < 1e-12
    print("OK" if abs(p - 1.0) < 1e-12 else "FAIL")

    print("\nself-test:", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Test whether RAdam-last finds flatter minima")
    ap.add_argument('--run-dir', help='Comparison run directory containing both approaches')
    ap.add_argument('--config', default='configs/comparison_langevin_avg_45_25_20.json',
                    help='Config defining the objective (must match the run)')
    ap.add_argument('--approaches', nargs=2, default=['bayes_lbfgs', 'bayes_lbfgs_adam'],
                    metavar=('BASELINE', 'STOCHASTIC'),
                    help='Baseline arm and the arm with the stochastic stage last')
    ap.add_argument('--endpoint', choices=['best', 'final', 'both'], default='best',
                    help="'best' = reported best_params (what the paper tabulates); "
                         "'final' = last iterate from the logs (what the theory predicts)")
    ap.add_argument('--device', default=None)
    ap.add_argument('--h', type=float, default=0.01,
                    help='Finite-difference step, fraction of each search range')
    ap.add_argument('--radii', type=float, nargs='+', default=[0.01, 0.02],
                    help='Perturbation sphere radii, fraction of each search range')
    ap.add_argument('--n-directions', type=int, default=32)
    ap.add_argument('--batch-points', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for probe directions')
    ap.add_argument('--csv', default='flatness_analysis.csv')
    ap.add_argument('--plot', default='flatness_analysis.png')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.run_dir:
        raise SystemExit("--run-dir is required (or use --self-test)")

    import torch
    from optimize_match_spectrum import (load_config, build_opt_params,
                                         SpectrumMatchingOptimizer, set_seed)
    from plot_loss_landscape import evaluate_points

    config = load_config(args.config)
    opt_params = build_opt_params(config)
    if args.device:
        opt_params['device'] = args.device if torch.cuda.is_available() else 'cpu'

    baseline, stochastic = args.approaches
    modes = ['best', 'final'] if args.endpoint == 'both' else [args.endpoint]

    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=0)
    lo = np.array([optimizer.laser_energy_bounds[0], optimizer.pressure_bounds[0],
                   optimizer.acquisition_time_bounds[0]], dtype=float)
    hi = np.array([optimizer.laser_energy_bounds[1], optimizer.pressure_bounds[1],
                   optimizer.acquisition_time_bounds[1]], dtype=float)
    span = hi - lo
    to_phys = lambda u: lo + u * span
    to_norm = lambda c: (np.asarray(c, dtype=float) - lo) / span

    dirs = sphere_directions(args.n_directions, np.random.default_rng(args.seed))
    rows = []

    for mode in modes:
        ends = (endpoints_from_results(args.run_dir, args.approaches) if mode == 'best'
                else endpoints_from_logs(args.run_dir, args.approaches))
        seeds = sorted(set(ends[baseline]) & set(ends[stochastic]))
        print(f"\n=== endpoint = {mode}: {len(seeds)} paired seeds ===")
        if not seeds:
            print("  no seeds common to both approaches; skipping")
            continue

        for s in seeds:
            # Reproduce this seed's objective exactly: same latents the run used.
            set_seed(s)
            optimizer.sampler.initialize_latents(
                n_samples=optimizer.batch_size,
                resolution=optimizer.spectrum_length, device=optimizer.device)

            for approach in (baseline, stochastic):
                u0 = to_norm(ends[approach][s])
                pts_u, index = build_probes(u0, args.h, tuple(args.radii), dirs)
                pts_c = np.array([to_phys(u) for u in pts_u])
                out = (pts_c < lo - 1e-9) | (pts_c > hi + 1e-9)
                if out.any():
                    print(f"    note: seed {s} {approach}: {out.any(axis=1).sum()} probe "
                          f"points fall outside the search bounds")
                L = evaluate_points(optimizer, pts_c, batch_points=args.batch_points,
                                    progress_every=10**9)
                m = point_metrics(L, index, args.h, tuple(args.radii))
                m.update({'endpoint': mode, 'approach': approach, 'seed': s,
                          'E': ends[approach][s][0], 'P': ends[approach][s][1],
                          't_open': ends[approach][s][2]})
                rows.append(m)
            print(f"  seed {s}: done")

    if not rows:
        raise SystemExit("nothing evaluated")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(f"\nSaved: {args.csv}")

    metrics = ['L_star', 'trace_H', 'max_eig', 'logdet_H_perp'] + \
              [f'ball{r}_mean_excess' for r in args.radii] + \
              [f'ball{r}_mean_rel' for r in args.radii]

    for mode in df['endpoint'].unique():
        sub = df[df.endpoint == mode]
        a = sub[sub.approach == baseline].set_index('seed')
        b = sub[sub.approach == stochastic].set_index('seed')
        common = sorted(set(a.index) & set(b.index))
        print("\n" + "=" * 92)
        print(f"PAIRED COMPARISON ({mode} endpoint, n={len(common)} seeds)")
        print(f"  baseline   = {baseline}")
        print(f"  stochastic = {stochastic}   (lower = flatter, except L_star)")
        print("=" * 92)
        print(f"{'metric':<26}{'baseline':>13}{'stochastic':>13}{'ratio':>9}"
              f"{'n_lower':>9}{'p (exact)':>11}")
        print("-" * 92)
        for k in metrics:
            va, vb = a.loc[common, k].values, b.loc[common, k].values
            good = np.isfinite(va) & np.isfinite(vb)
            if good.sum() < 2:
                continue
            d = vb[good] - va[good]                       # stochastic - baseline
            _, p, _ = wilcoxon_exact(list(d))
            pos, neg = sign_test(list(d))
            ratio = (np.median(vb[good]) / np.median(va[good])
                     if np.median(va[good]) != 0 else float('nan'))
            print(f"{k:<26}{np.median(va[good]):>13.3e}{np.median(vb[good]):>13.3e}"
                  f"{ratio:>9.2f}{neg:>6}/{good.sum():<3}{p:>11.4f}")
        print("-" * 92)
        print("medians shown; 'n_lower' counts seeds where the stochastic arm is LOWER;")
        print("p is a two-sided exact Wilcoxon signed-rank test on the paired differences.")

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mode0 = df['endpoint'].unique()[0]
    sub = df[df.endpoint == mode0]
    a = sub[sub.approach == baseline].set_index('seed')
    b = sub[sub.approach == stochastic].set_index('seed')
    common = sorted(set(a.index) & set(b.index))
    key = f'ball{args.radii[0]}_mean_excess'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, k, title in ((axes[0], key, f'Mean excess loss, r={args.radii[0]}'),
                         (axes[1], 'trace_H', 'Curvature: trace of Hessian'),
                         (axes[2], 'logdet_H_perp', r'Free-energy term: $\log\det H_\perp$')):
        for s in common:
            ax.plot([0, 1], [a.loc[s, k], b.loc[s, k]], '-o', color='#888888',
                    ms=4, lw=1, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['BO→L-BFGS', 'BO→L-BFGS→RAdam'], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if k != 'logdet_H_perp':
            ax.set_yscale('log')
    fig.suptitle(f'Is the stochastic stage selecting flatter minima?  '
                 f'({mode0} endpoint, n={len(common)} seeds, paired)', fontsize=12)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {args.plot}")


if __name__ == '__main__':
    main()
