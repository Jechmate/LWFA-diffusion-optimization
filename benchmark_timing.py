"""
Timing benchmarks for the diffusion sampler.

Measures the two costs that set the price of an optimization run:

  1. one GRADIENT evaluation - forward through the sampler + backward, i.e. exactly
     what one Adam / L-BFGS / SGLD iteration costs. The gradient flows through every
     sampling step, so the per-step cost is reported as the MARGINAL cost (slope of
     a time-vs-num_steps sweep), which is more honest than dividing by num_steps:
     part of each call is fixed overhead that does not scale with the step count.
  2. generating a BATCH of spectra at inference (no autograd) - the cost paid by
     evaluate_exclusion_models.py and by every Bayesian objective evaluation.

Both use CUDA synchronisation around each timed region and discard warmup
iterations, without which the numbers measure kernel-launch latency and one-off
cuDNN autotuning rather than compute.

    python benchmark_timing.py
    python benchmark_timing.py --batch-size 32 --num-steps 18 --repeats 20
    python benchmark_timing.py --sweep-steps 1 4 8 18 30 --json timing.json
"""

import time
import json
import argparse
import statistics

import numpy as np
import torch

from optimize_match_spectrum import load_config, build_opt_params, SpectrumMatchingOptimizer


# =============================================================================
# TIMING HELPERS
# =============================================================================

def _sync(device):
    """CUDA kernels are async; without this we would time the launch, not the work."""
    if torch.cuda.is_available() and 'cuda' in str(device):
        torch.cuda.synchronize()


def timeit(fn, repeats, warmup, device):
    """Return per-call wall times in seconds, discarding warmup iterations."""
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def summarize(times):
    return {
        'median': statistics.median(times),
        'mean': statistics.mean(times),
        'std': statistics.pstdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
        'n': len(times),
    }


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.3f} s"


def peak_memory_mb(device):
    if torch.cuda.is_available() and 'cuda' in str(device):
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return float('nan')


def reset_memory(device):
    if torch.cuda.is_available() and 'cuda' in str(device):
        torch.cuda.reset_peak_memory_stats(device)


# =============================================================================
# BENCHMARKS
# =============================================================================

def make_gradient_fn(optimizer, start_params, backward=True):
    """One optimizer-style gradient evaluation: sample -> MSE -> backward."""
    device = optimizer.device

    def run():
        params = [torch.tensor(float(p), device=device, requires_grad=True)
                  for p in start_params]
        settings = torch.stack(params).unsqueeze(0)
        spectrum = optimizer._sample_spectrum(settings)
        loss = optimizer._compute_mse(spectrum)
        if backward:
            loss.backward()
    return run


def make_forward_nograd_fn(optimizer, start_params):
    """Forward only, no autograd graph - the Bayesian objective's cost."""
    device = optimizer.device

    def run():
        with torch.no_grad():
            settings = torch.tensor([float(p) for p in start_params],
                                    device=device).unsqueeze(0)
            spectrum = optimizer._sample_spectrum(settings)
            optimizer._compute_mse(spectrum)
    return run


def make_batch_sample_fn(model, device, settings_row, batch_size, num_steps,
                         cfg_scale, settings_dim, resolution):
    """Inference-path batch generation (fresh latents, no autograd)."""
    from src.diffusion import EdmSampler
    sampler = EdmSampler(net=model, num_steps=num_steps)
    settings = torch.tensor(settings_row, dtype=torch.float32,
                            device=device).reshape(1, -1).repeat(batch_size, 1)

    def run():
        with torch.no_grad():
            sampler.sample(
                resolution=resolution, device=device, settings=settings,
                n_samples=batch_size, cfg_scale=cfg_scale,
                settings_dim=settings_dim, smooth_output=False,
            )
    return run


def set_model_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


def linear_fit(xs, ys):
    """Least-squares slope/intercept: marginal per-step cost and fixed overhead."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Time sampler gradient and batch generation")
    parser.add_argument('--config', default='configs/match_spectrum.yaml',
                        help='Config supplying model / sampler settings')
    parser.add_argument('--device', default=None, help='Override device from config')
    parser.add_argument('--repeats', type=int, default=20, help='Timed iterations per benchmark')
    parser.add_argument('--warmup', type=int, default=3, help='Discarded warmup iterations')
    parser.add_argument('--batch-size', type=int, default=32, help='Spectra per batch-generation call')
    parser.add_argument('--num-steps', type=int, default=18, help='Sampler steps')
    parser.add_argument('--settings', default='45,25,20', help='"E,P,t" to condition on')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweep-steps', type=int, nargs='+', default=None, metavar='S',
                        help='Also sweep these step counts to get the marginal per-step cost')
    parser.add_argument('--json', default=None, help='Write results to this JSON file')
    args = parser.parse_args()

    settings_row = [float(v) for v in args.settings.split(',')]

    config = load_config(args.config)
    opt_params = build_opt_params(config)
    if args.device is not None:
        opt_params['device'] = args.device if torch.cuda.is_available() else 'cpu'
    opt_params['num_sampling_steps'] = args.num_steps
    device = opt_params['device']

    print("=" * 78)
    print("SAMPLER TIMING BENCHMARK")
    print("=" * 78)
    print(f"device={device}  model={opt_params['model_path']}")
    print(f"sampler: {args.num_steps} steps, cfg_scale={opt_params['cfg_scale']}, "
          f"resolution={opt_params['spectrum_length']}")
    print(f"gradient batch (latents per evaluation): {opt_params['batch_size']}")
    print(f"generation batch: {args.batch_size}")
    print(f"repeats={args.repeats} (+{args.warmup} warmup), settings={settings_row}")
    if torch.cuda.is_available() and 'cuda' in str(device):
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 78)

    optimizer = SpectrumMatchingOptimizer(**opt_params, seed=args.seed)
    model = optimizer.model
    results = {'config': {
        'device': str(device), 'num_steps': args.num_steps,
        'grad_batch': opt_params['batch_size'], 'gen_batch': args.batch_size,
        'cfg_scale': opt_params['cfg_scale'], 'repeats': args.repeats,
    }}

    rows = []

    def bench(name, fn, note=''):
        reset_memory(device)
        times = timeit(fn, args.repeats, args.warmup, device)
        s = summarize(times)
        s['peak_mem_mb'] = peak_memory_mb(device)
        s['note'] = note
        results[name] = s
        rows.append((name, s))
        return s

    # --- gradient path -------------------------------------------------------
    set_model_grad(model, True)
    grad = bench('gradient_eval',
                 make_gradient_fn(optimizer, settings_row, backward=True),
                 'sample + MSE + backward (as Adam/L-BFGS/SGLD do)')
    fwd_graph = bench('forward_with_graph',
                      make_gradient_fn(optimizer, settings_row, backward=False),
                      'forward building the autograd graph, no backward')
    fwd_nograd = bench('forward_no_grad',
                       make_forward_nograd_fn(optimizer, settings_row),
                       'forward under no_grad (Bayesian objective)')

    # Model parameters currently receive gradients too - nothing freezes them.
    set_model_grad(model, False)
    grad_frozen = bench('gradient_eval_frozen_model',
                        make_gradient_fn(optimizer, settings_row, backward=True),
                        'same, but model parameters requires_grad=False')
    set_model_grad(model, True)

    # --- inference batch generation -----------------------------------------
    gen = bench('batch_generation',
                make_batch_sample_fn(model, device, settings_row, args.batch_size,
                                     args.num_steps, opt_params['cfg_scale'],
                                     len(opt_params['features']),
                                     opt_params['spectrum_length']),
                f'{args.batch_size} spectra, {args.num_steps} steps, no_grad')

    # --- report --------------------------------------------------------------
    print(f"\n{'benchmark':<30}{'median':>11}{'mean':>11}{'std':>10}{'min':>11}{'peak MB':>10}")
    print("-" * 78)
    for name, s in rows:
        print(f"{name:<30}{fmt(s['median']):>11}{fmt(s['mean']):>11}"
              f"{fmt(s['std']):>10}{fmt(s['min']):>11}{s['peak_mem_mb']:>10.0f}")
    print("-" * 78)

    print("\nDerived:")
    bwd = grad['median'] - fwd_graph['median']
    print(f"  backward alone                  {fmt(bwd)} "
          f"({bwd / grad['median'] * 100:.0f}% of a gradient evaluation)")
    print(f"  autograd-graph overhead on fwd  {fmt(fwd_graph['median'] - fwd_nograd['median'])}")
    saving = grad['median'] - grad_frozen['median']
    print(f"  freezing model params saves     {fmt(saving)} "
          f"({saving / grad['median'] * 100:+.0f}%)")
    print(f"  gradient eval / naive per step  {fmt(grad['median'] / args.num_steps)} "
          f"(= total / {args.num_steps}; see the sweep for the true marginal cost)")
    print(f"  batch generation per spectrum   {fmt(gen['median'] / args.batch_size)}")
    results['derived'] = {
        'backward_only': bwd,
        'graph_overhead': fwd_graph['median'] - fwd_nograd['median'],
        'frozen_model_saving': saving,
        'gradient_per_step_naive': grad['median'] / args.num_steps,
        'generation_per_spectrum': gen['median'] / args.batch_size,
    }

    # --- optional step sweep -> marginal per-step cost ------------------------
    if args.sweep_steps:
        print(f"\n{'steps':>7}{'gradient':>13}{'generation':>14}")
        print("-" * 34)
        sweep = {'steps': [], 'gradient': [], 'generation': []}
        for steps in args.sweep_steps:
            optimizer.sampler.num_steps = steps
            g = summarize(timeit(make_gradient_fn(optimizer, settings_row),
                                 args.repeats, args.warmup, device))['median']
            b = summarize(timeit(
                make_batch_sample_fn(model, device, settings_row, args.batch_size,
                                     steps, opt_params['cfg_scale'],
                                     len(opt_params['features']),
                                     opt_params['spectrum_length']),
                args.repeats, args.warmup, device))['median']
            sweep['steps'].append(steps)
            sweep['gradient'].append(g)
            sweep['generation'].append(b)
            print(f"{steps:>7}{fmt(g):>13}{fmt(b):>14}")
        optimizer.sampler.num_steps = args.num_steps

        gs, gi = linear_fit(sweep['steps'], sweep['gradient'])
        bs, bi = linear_fit(sweep['steps'], sweep['generation'])
        print("-" * 34)
        print("Marginal cost per sampling step (least-squares slope):")
        print(f"  gradient    {fmt(gs)}/step   fixed overhead {fmt(gi)}")
        print(f"  generation  {fmt(bs)}/step   fixed overhead {fmt(bi)}")
        results['sweep'] = sweep
        results['sweep_fit'] = {'gradient_slope': gs, 'gradient_intercept': gi,
                                'generation_slope': bs, 'generation_intercept': bi}

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.json}")


if __name__ == '__main__':
    main()
