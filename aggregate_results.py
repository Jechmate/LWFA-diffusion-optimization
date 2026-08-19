"""
Aggregate optimization results from several runs into one tidy CSV + comparison.

Optimization runs each write their own results file, in one of three shapes:

  comparison mode   results.json       {approach: {seed: {best_mse, best_params}}}
  multi mode        results.json       [{seed, best_mse, best_params}]
  legacy            *_results.json     [{seed, optimizer_type, best_objective, ...}]

This script normalizes all of them into one row per (run, approach, seed) and
writes a single CSV that can be appended to as new runs finish - the aggregate
table the project was missing. Results from runs with different target spectra
are not comparable on raw MSE, so the target is recorded per row and mixing is
refused unless explicitly forced.

    # combine runs into the aggregate CSV and plot them
    python aggregate_results.py comparison_A comparison_B multi_bayesian_only_X

    # add a newly finished run to the existing table (idempotent)
    python aggregate_results.py comparison_C --append all_results.csv

    # nicer names in the plots
    python aggregate_results.py comparison_A --label comparison_A=Langevin

Only the plotting path needs numpy/matplotlib; the CSV path is pure stdlib.
"""

import os
import csv
import json
import glob
import argparse
import statistics
from collections import defaultdict

FIELDS = ['run', 'approach', 'seed', 'best_mse', 'E', 'P', 't_open',
          'target', 'best_phase', 'source']


# =============================================================================
# LOADING
# =============================================================================

def find_results_file(path):
    """Locate the results JSON for a run directory (or accept a .json directly)."""
    if os.path.isfile(path):
        return path
    preferred = os.path.join(path, 'results.json')
    if os.path.exists(preferred):
        return preferred
    candidates = sorted(glob.glob(os.path.join(path, '*_results.json')))
    return candidates[0] if candidates else None


def find_target(run_dir):
    """Recover which target spectrum a run optimized against."""
    # Current runs: config.json written by save_run_config
    cfg = os.path.join(run_dir, 'config.json')
    if os.path.exists(cfg):
        try:
            with open(cfg) as f:
                data = json.load(f)
            target = (data.get('opt_params', {}).get('target_spectrum_csv')
                      or data.get('config', {}).get('target_spectrum_csv'))
            if target:
                return target
        except (json.JSONDecodeError, AttributeError):
            pass
    # Legacy runs
    legacy = os.path.join(run_dir, 'optimization_parameters.json')
    if os.path.exists(legacy):
        try:
            with open(legacy) as f:
                return json.load(f).get('target_spectrum_csv', 'unknown')
        except json.JSONDecodeError:
            pass
    return 'unknown'


def approach_from_dirname(name):
    """Derive an approach name from a run directory name.

    'multi_bayesian_only_20260813_112907' -> 'bayesian_only'
    """
    base = os.path.basename(os.path.normpath(name))
    parts = base.split('_')
    # drop a trailing YYYYMMDD_HHMMSS timestamp
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    if parts and parts[0] in ('multi', 'comparison'):
        parts = parts[1:]
    return '_'.join(parts) if parts else base


def _as_seed(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _row(run, approach, seed, entry, target, source):
    """Build one tidy row from a result entry, tolerating both loss key names."""
    mse = entry.get('best_mse', entry.get('best_objective'))
    params = entry.get('best_params') or []
    params = list(params) + [None] * (3 - len(params))
    return {
        'run': run, 'approach': approach, 'seed': _as_seed(seed), 'best_mse': mse,
        'E': params[0], 'P': params[1], 't_open': params[2],
        'target': target, 'best_phase': entry.get('best_phase', ''), 'source': source,
    }


def load_run(path, label=None):
    """Load one run directory (or results file) into tidy rows."""
    results_file = find_results_file(path)
    if results_file is None:
        existing = ', '.join(sorted(os.listdir(path))[:6]) if os.path.isdir(path) else '?'
        print(f"  ! {path}: no results.json or *_results.json (contains: {existing}) - skipped")
        return []

    run_dir = path if os.path.isdir(path) else os.path.dirname(path)
    run = label or os.path.basename(os.path.normpath(run_dir))
    target = find_target(run_dir)

    with open(results_file) as f:
        data = json.load(f)

    rows = []
    if isinstance(data, dict):                      # comparison mode
        for approach, seed_map in data.items():
            if not isinstance(seed_map, dict):
                continue
            for seed, entry in seed_map.items():
                if isinstance(entry, dict):
                    rows.append(_row(run, approach, seed, entry, target, results_file))
    elif isinstance(data, list):                    # multi mode or legacy
        fallback = approach_from_dirname(run_dir)
        for entry in data:
            if not isinstance(entry, dict):
                continue
            approach = entry.get('optimizer_type') or fallback
            rows.append(_row(run, approach, entry.get('seed'), entry, target, results_file))

    n_approaches = len({r['approach'] for r in rows})
    print(f"  + {run}: {len(rows)} rows, {n_approaches} approach(es), "
          f"target={target}  [{os.path.basename(results_file)}]")
    return rows


# =============================================================================
# CSV
# =============================================================================

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDS})
    print(f"Wrote {len(rows)} rows -> {path}")


def merge_rows(existing, new):
    """Merge new rows over existing ones, de-duped on (run, approach, seed)."""
    def key(r):
        return (str(r.get('run')), str(r.get('approach')), str(r.get('seed')))

    merged = {key(r): r for r in existing}
    added = updated = 0
    for row in new:
        if key(row) in merged:
            updated += 1
        else:
            added += 1
        merged[key(row)] = row
    print(f"Merge: {added} new rows, {updated} replaced, {len(merged)} total")
    return list(merged.values())


# =============================================================================
# SUMMARY + PLOT
# =============================================================================

def group_values(rows, group_by):
    """Map group name -> list of (mse, E, P, t_open), skipping unusable rows."""
    groups = defaultdict(list)
    for row in rows:
        try:
            mse = float(row['best_mse'])
        except (TypeError, ValueError):
            continue

        def num(key):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None

        groups[row[group_by]].append((mse, num('E'), num('P'), num('t_open')))
    return groups


def print_summary(rows, group_by):
    groups = group_values(rows, group_by)
    print("\n" + "=" * 88)
    print(f"SUMMARY BY {group_by.upper()}")
    print("=" * 88)
    print(f"{group_by:<26} {'n':>5} {'mean MSE':>13} {'std':>13} {'min':>13} {'median':>13}")
    print("-" * 88)
    stats = []
    for name, vals in groups.items():
        mses = [v[0] for v in vals]
        stats.append((statistics.mean(mses), name, len(mses),
                      statistics.pstdev(mses) if len(mses) > 1 else 0.0,
                      min(mses), statistics.median(mses)))
    for mean, name, n, std, mn, med in sorted(stats):
        print(f"{name:<26} {n:>5} {mean:>13.3e} {std:>13.3e} {mn:>13.3e} {med:>13.3e}")
    print("-" * 88)
    if stats:
        best = sorted(stats)[0]
        print(f"Best mean MSE: {best[1]} ({best[0]:.3e}, n={best[2]})")


def plot_comparison(rows, output, group_by):
    """Box plot + mean/std bars + recovered (E, P) scatter. Lazily imports mpl."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = group_values(rows, group_by)
    if not groups:
        print("No plottable rows; skipping figure.")
        return

    names = [name for _, name in sorted((statistics.mean([v[0] for v in vals]), name)
                                        for name, vals in groups.items())]
    data = [[v[0] for v in groups[n]] for n in names]
    labels = [f"{n}\n(n={len(groups[n])})" for n in names]
    colors = [plt.cm.tab10(i % 10) for i in range(len(names))]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    # `labels` was renamed `tick_labels` in matplotlib 3.9; set ticks explicitly.
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_ylabel('Best MSE (log)')
    ax.set_title('MSE distribution')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    means = [statistics.mean(d) for d in data]
    stds = [statistics.pstdev(d) if len(d) > 1 else 0.0 for d in data]
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Mean MSE (log)')
    ax.set_title('Mean MSE ± std')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    for name, color in zip(names, colors):
        pts = [(v[1], v[2]) for v in groups[name] if v[1] is not None and v[2] is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=22, color=color, alpha=0.7, edgecolor='black',
                       linewidths=0.3, label=f"{name} (n={len(pts)})")
    ax.set_xlabel('E — laser energy')
    ax.set_ylabel('P — pressure [bar]')
    ax.set_title('Recovered optima')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {output}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate optimization results across runs")
    parser.add_argument('runs', nargs='+', help='Run directories (or results .json files)')
    parser.add_argument('--output', default='all_results.csv', help='Aggregate CSV to write')
    parser.add_argument('--append', default=None,
                        help='Merge into this existing CSV (de-duped on run/approach/seed)')
    parser.add_argument('--plot', default='comparison_all.png', help='Figure path ("none" to skip)')
    parser.add_argument('--group-by', choices=['approach', 'run'], default='approach')
    parser.add_argument('--label', action='append', default=[], metavar='DIR=NAME',
                        help='Friendly name for a run, repeatable')
    parser.add_argument('--force-mixed-targets', action='store_true',
                        help='Allow combining runs that used different target spectra')
    args = parser.parse_args()

    labels = dict(item.split('=', 1) for item in args.label if '=' in item)

    print(f"Loading {len(args.runs)} run(s):")
    rows = []
    for path in args.runs:
        key = os.path.basename(os.path.normpath(path))
        rows.extend(load_run(path, label=labels.get(path, labels.get(key))))

    if not rows:
        raise SystemExit("No results found in the given directories.")

    targets = {r['target'] for r in rows}
    if len(targets) > 1:
        msg = (f"Runs used different target spectra: {sorted(targets)}. Raw MSE is not "
               f"comparable across targets.")
        if not args.force_mixed_targets:
            raise SystemExit(f"ERROR: {msg} Use --force-mixed-targets to override.")
        print(f"WARNING: {msg}")

    if args.append:
        if os.path.exists(args.append):
            rows = merge_rows(read_csv(args.append), rows)
        else:
            print(f"{args.append} does not exist yet; creating it.")
        args.output = args.append

    write_csv(rows, args.output)
    print_summary(rows, args.group_by)

    if args.plot.lower() != 'none':
        plot_comparison(rows, args.plot, args.group_by)


if __name__ == '__main__':
    main()
