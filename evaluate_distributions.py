"""
Distributional evaluation: are the learned spectrum distributions calibrated,
and are quantiles and tails reproduced?

Averaged per-bin distances say whether the mean is right; they say nothing about
whether the spread is right. This script treats the GENERATED ensemble as a
predictive distribution and the REAL shots as observations drawn from truth -
the correct framing when there are few real shots (31-109 per experiment) and
arbitrarily many generated ones.

Diagnostics
-----------
PIT / rank histogram  For every real shot and energy bin, the rank of the real
                      value within the generated ensemble. Uniform = calibrated.
                      U-shaped = generated too narrow (under-dispersed);
                      dome = too wide; sloped = biased.
CRPS                  Continuous Ranked Probability Score, a proper scoring rule
                      (the distributional generalisation of MAE), per bin and
                      aggregated, with a skill score against the real-climatology
                      reference.
Dispersion ratio      sigma_gen / sigma_real per energy bin - directly shows where
                      in the spectrum the width is wrong.
Scalar summaries      Per shot: total charge, mean energy, peak intensity and
                      high-energy tail charge. The real and generated
                      DISTRIBUTIONS of each are compared (quantiles + KS).

Reads the ensembles saved by evaluate_exclusion_models.py, so no GPU is needed.
Generate a large generated ensemble first:

    python evaluate_exclusion_models.py --n-generated 1000
    python evaluate_distributions.py
    python evaluate_distributions.py --cfg-scales 1.0 2.0 3.0 5.0   # calibration vs CFG

CAVEAT: with 31-109 real shots, quantiles beyond ~95% are dominated by sampling
noise. Statistics above --max-quantile are reported but flagged as unreliable.
"""

import os
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG = {
    'results_dir': 'exclusion_evaluation_results',
    'data_dir': 'spectra',
    'params_file': 'params.csv',
    'resolution': 256,
}
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# np.trapezoid only exists from NumPy 2.0; np.trapz is deprecated there but still
# present. Fall back so this runs on either.
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# =============================================================================
# LOADING
# =============================================================================

def load_real(experiment, data_dir, resolution):
    """Real shots for one experiment: (n_shots, resolution) plus the energy axis."""
    files = sorted(Path(data_dir, str(experiment)).glob('*.csv'))
    if not files:
        return None, None
    energy, rows = None, []
    for f in files:
        df = pd.read_csv(f)
        if energy is None:
            energy = df['energy'].values[:resolution]
        rows.append(df['intensity'].values[:resolution])
    return np.array(rows), energy


def parse_excluded(name):
    import re
    m = re.search(r'exclude_(\d+)$', name)
    return int(m.group(1)) if m else None


def collect_pairs(results_dir, data_dir, cfg_scale, num_steps, resolution):
    """(experiment, real, generated, energy) for every exclusion model in a run."""
    param_dir = Path(results_dir) / f"cfg{cfg_scale}_steps{num_steps}"
    if not param_dir.exists():
        return []
    out = []
    for model_dir in sorted(param_dir.iterdir()):
        exp = parse_excluded(model_dir.name) if model_dir.is_dir() else None
        if exp is None:
            continue
        npy = model_dir / str(exp) / 'generated_spectra.npy'
        if not npy.exists():
            continue
        gen = np.load(npy)
        real, energy = load_real(exp, data_dir, resolution)
        if real is None:
            continue
        n = min(real.shape[1], gen.shape[1])
        out.append({'experiment': exp, 'real': real[:, :n],
                    'generated': gen[:, :n], 'energy': energy[:n]})
    return out


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def pit_values(real, generated):
    """PIT / rank of each real value within the generated ensemble, per bin.

    Randomised ranks break ties uniformly so that a perfectly calibrated model
    gives exactly uniform PIT even with discrete ensembles.
    """
    rng = np.random.default_rng(0)
    below = (generated[None, :, :] < real[:, None, :]).sum(axis=1)
    equal = (generated[None, :, :] == real[:, None, :]).sum(axis=1)
    n = generated.shape[0]
    return (below + rng.random(below.shape) * (equal + 1)) / (n + 1)


def crps_ensemble(real, generated):
    """CRPS per (shot, bin) via the energy form; lower is better.

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated from the ensemble.
    """
    n = generated.shape[0]
    term1 = np.abs(generated[None, :, :] - real[:, None, :]).mean(axis=1)
    g = np.sort(generated, axis=0)
    # E|X - X'| via the sorted-sample identity, O(n log n) instead of O(n^2)
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    term2 = 2.0 * (g * w).sum(axis=0) / (n * n)
    return term1 - 0.5 * term2


def scalar_summaries(spectra, energy):
    """Per-shot physical summaries. Trapezoid over the true non-uniform axis."""
    order = np.argsort(energy)
    e = energy[order]
    y = spectra[:, order]
    charge = _trapz(y, e, axis=1)
    weighted = _trapz(y * e, e, axis=1)
    hi = e >= 20.0
    tail = _trapz(y[:, hi], e[hi], axis=1) if hi.sum() > 1 else np.zeros(len(y))
    return {
        'total_charge': charge,
        'mean_energy': np.divide(weighted, charge, out=np.zeros_like(charge),
                                 where=charge > 0),
        'peak_intensity': y.max(axis=1),
        'tail_charge_20MeV': tail,
    }


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / len(a)
    cdf_b = np.searchsorted(b, allv, side='right') / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# =============================================================================
# REPORT + PLOTS
# =============================================================================

def analyse(pairs, max_quantile, out_prefix, cfg_scale, num_steps):
    rows, all_pit, disp_curves, scalars = [], [], [], {}

    for p in pairs:
        real, gen, energy = p['real'], p['generated'], p['energy']
        pit = pit_values(real, gen)
        crps = crps_ensemble(real, gen)

        # restrict dispersion to bins with real signal
        active = real.mean(axis=0) > 0.01 * real.mean(axis=0).max()
        sd_real = real.std(axis=0)
        sd_gen = gen.std(axis=0)
        ratio = np.divide(sd_gen, sd_real, out=np.full_like(sd_gen, np.nan),
                          where=sd_real > 0)

        # CRPS skill vs using the real ensemble's own climatology
        clim = crps_ensemble(real, real).mean()
        skill = 1 - crps.mean() / clim if clim > 0 else np.nan

        rs, gs = scalar_summaries(real, energy), scalar_summaries(gen, energy)
        scalars[p['experiment']] = (rs, gs)

        rows.append({
            'experiment': p['experiment'], 'n_real': len(real), 'n_gen': len(gen),
            'pit_mean': pit.mean(), 'pit_frac_outside': float(
                ((pit < 1 / (len(gen) + 1)) | (pit > len(gen) / (len(gen) + 1))).mean()),
            'crps': crps.mean(), 'crps_skill': skill,
            'dispersion_ratio': float(np.nanmedian(ratio[active])),
            **{f'ks_{k}': ks_statistic(rs[k], gs[k]) for k in rs},
        })
        all_pit.append(pit.ravel())
        disp_curves.append((energy, ratio, active))

    df = pd.DataFrame(rows).sort_values('experiment')
    pit_all = np.concatenate(all_pit)

    print("\n" + "=" * 100)
    print(f"DISTRIBUTIONAL EVALUATION   CFG={cfg_scale}, steps={num_steps}, "
          f"{len(pairs)} experiments")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- overall verdict -----------------------------------------------------
    disp = df['dispersion_ratio'].median()
    # uniformity of the pooled PIT, as a chi-square-like deviation
    hist, _ = np.histogram(pit_all, bins=20, range=(0, 1))
    expected = len(pit_all) / 20
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    # U vs dome: compare edge bins to centre bins
    edges = hist[:2].sum() + hist[-2:].sum()
    centre = hist[9:11].sum() * 2
    print("\n" + "-" * 100)
    print(f"median dispersion ratio sigma_gen/sigma_real = {disp:.2f}")
    if disp < 0.8:
        print("  -> generated spectra are UNDER-dispersed: the model is too confident, "
              "tails too thin")
    elif disp > 1.25:
        print("  -> generated spectra are OVER-dispersed: too much shot-to-shot spread")
    else:
        print("  -> spread is broadly consistent with the real shots")
    print(f"pooled PIT uniformity chi2 = {chi2:.0f} over 20 bins "
          f"({len(pit_all):,} values; 0 = perfectly calibrated)")
    print(f"  edge/centre mass ratio = {edges / max(centre, 1):.2f} "
          f"({'U-shaped -> under-dispersed' if edges > centre * 1.2 else 'dome -> over-dispersed' if centre > edges * 1.2 else 'flat -> calibrated'})")
    print(f"mean CRPS = {df['crps'].mean():.4g}, median skill vs real climatology = "
          f"{df['crps_skill'].median():.3f} (1 = perfect, 0 = no better than climatology)")
    print(f"\nNOTE: with {df['n_real'].min()}-{df['n_real'].max()} real shots per "
          f"experiment, quantiles beyond {max_quantile:.0%} are sampling-noise dominated;")
    print("      the body of the distribution is measurable, the extreme tail is not.")

    # --- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.hist(pit_all, bins=20, range=(0, 1), color='#1f77b4', alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.axhline(len(pit_all) / 20, color='red', ls='--', lw=2, label='calibrated (uniform)')
    ax.set_xlabel('PIT (rank of real shot within generated ensemble)')
    ax.set_ylabel('count')
    ax.set_title('Calibration: PIT histogram\nU-shape = too narrow, dome = too wide')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for energy, ratio, active in disp_curves:
        ax.plot(energy[active], ratio[active], alpha=0.35, lw=1, color='#1f77b4')
    ax.axhline(1.0, color='red', ls='--', lw=2, label='perfect')
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel(r'$\sigma_{gen}/\sigma_{real}$')
    ax.set_yscale('log')
    ax.set_title('Dispersion ratio per energy bin (one line per experiment)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    keys = ['total_charge', 'mean_energy', 'peak_intensity', 'tail_charge_20MeV']
    x = np.arange(len(keys))
    ax.bar(x, [df[f'ks_{k}'].median() for k in keys], color='#d62728', alpha=0.8,
           edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('_', '\n') for k in keys], fontsize=8)
    ax.set_ylabel('median KS statistic (real vs generated)')
    ax.set_title('Scalar summary distributions\n0 = identical, 1 = disjoint')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    ex = sorted(scalars)[len(scalars) // 2]
    rs, gs = scalars[ex]
    qs = np.linspace(0.02, max_quantile, 40)
    for k, c in zip(keys, ('#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd')):
        r, g = np.quantile(rs[k], qs), np.quantile(gs[k], qs)
        scale = max(np.abs(r).max(), 1e-12)
        ax.plot(r / scale, g / scale, 'o-', ms=3, color=c, alpha=0.8, label=k)
    lim = ax.get_xlim()
    ax.plot(lim, lim, 'k--', lw=1, label='ideal')
    ax.set_xlabel('real quantile (normalised)')
    ax.set_ylabel('generated quantile (normalised)')
    ax.set_title(f'Q-Q of scalar summaries, experiment {ex}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Distributional evaluation - CFG={cfg_scale}, steps={num_steps}',
                 fontsize=14)
    plt.tight_layout()
    out = f"{out_prefix}_cfg{cfg_scale}_steps{num_steps}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out}")
    return df, {'dispersion': disp, 'pit_chi2': chi2,
                'edge_centre': edges / max(centre, 1),
                'crps': df['crps'].mean(), 'crps_skill': df['crps_skill'].median()}


def plot_cfg_trend(trend, out_prefix):
    """Calibration and accuracy against CFG scale."""
    cfgs = sorted(trend)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, label, ref in (
            (axes[0], 'dispersion', r'median $\sigma_{gen}/\sigma_{real}$', 1.0),
            (axes[1], 'edge_centre', 'PIT edge/centre mass', 1.0),
            (axes[2], 'crps', 'mean CRPS (lower better)', None)):
        ax.plot(cfgs, [trend[c][key] for c in cfgs], 'o-', lw=2, color='#1f77b4')
        if ref is not None:
            ax.axhline(ref, color='red', ls='--', lw=1.5, label='calibrated')
            ax.legend(fontsize=8)
        ax.set_xlabel('CFG scale')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Distributional calibration vs classifier-free guidance scale', fontsize=13)
    plt.tight_layout()
    out = f"{out_prefix}_cfg_trend.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Distributional evaluation of generated spectra")
    parser.add_argument('--results-dir', default=CONFIG['results_dir'])
    parser.add_argument('--data-dir', default=CONFIG['data_dir'])
    parser.add_argument('--cfg-scales', type=float, nargs='+', default=[3.0])
    parser.add_argument('--num-steps', type=int, default=18)
    parser.add_argument('--max-quantile', type=float, default=0.95,
                        help='Highest quantile treated as measurable')
    parser.add_argument('--output-prefix', default='distribution_eval')
    parser.add_argument('--csv', default='distribution_eval.csv')
    args = parser.parse_args()

    trend, frames = {}, []
    for cfg_scale in args.cfg_scales:
        pairs = collect_pairs(args.results_dir, args.data_dir, cfg_scale,
                              args.num_steps, CONFIG['resolution'])
        if not pairs:
            print(f"No saved ensembles for CFG={cfg_scale}, steps={args.num_steps} "
                  f"in {args.results_dir} - run evaluate_exclusion_models.py first.")
            continue
        df, summary = analyse(pairs, args.max_quantile, args.output_prefix,
                              cfg_scale, args.num_steps)
        df.insert(1, 'cfg_scale', cfg_scale)
        frames.append(df)
        trend[cfg_scale] = summary

    if not frames:
        raise SystemExit('Nothing to analyse.')
    pd.concat(frames).to_csv(args.csv, index=False)
    print(f"Saved: {args.csv}")

    if len(trend) > 1:
        plot_cfg_trend(trend, args.output_prefix)
        print("\nCalibration vs CFG:")
        print(f"  {'cfg':>5}{'dispersion':>13}{'edge/centre':>13}{'CRPS':>12}")
        for c in sorted(trend):
            t = trend[c]
            print(f"  {c:>5}{t['dispersion']:>13.2f}{t['edge_centre']:>13.2f}{t['crps']:>12.4g}")


if __name__ == '__main__':
    main()
