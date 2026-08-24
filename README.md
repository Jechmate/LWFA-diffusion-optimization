# Differentiable Control of Laser–Plasma Experiments with Conditional Diffusion Models

Code and data accompanying the paper. A conditional diffusion model (EDM) is
trained on measured electron energy spectra from the ALFA beamline. Because EDM
sampling integrates a probability-flow ODE, the map from experimental setting to
spectrum is differentiable end to end, so the setting that reproduces a target
spectrum can be recovered by backpropagating through the sampler with the network
weights held fixed.

## Data

- `spectra/<experiment>/*.csv` — 1279 measured spectra across 22 experimental
  settings. Each file is one recording: `energy` (MeV) and `intensity` (pC/MeV)
  over 450 bins, of which the first 256 (97.8 down to 4.3 MeV) are used.
- `params.csv` — the setting for each experiment: laser energy `E` (mJ), backing
  pressure `P` (bar), and valve opening time `ms`.
- `avg_spectrum_*.csv`, `target_real_*.csv` — optimisation targets. The first are
  model-generated at known settings; the second are shot-averaged measurements,
  regenerable with `build_real_target.py`.

## Model

`models/edm_4kepochs/ema_ckpt_final.pt` is the trained model used throughout.
Retraining from scratch takes about 1 h on a single A100:

```bash
python train_edm.py
```

## Reproducing the results

Optimisation strategies are declared as stage pipelines in
`configs/match_spectrum.yaml`; a config only needs to list what it changes.
Running with no arguments reproduces the strategy comparison from the paper:
the eight approaches listed below, each across ten seeds.

| approach | stages |
| --- | --- |
| `bayesian_only` | BO (100) |
| `adam_only` | RAdam (100) |
| `lbfgs_only` | L-BFGS (100) |
| `bayes_adam` | BO (100) + RAdam (50) |
| `bayes_lbfgs` | BO (100) + L-BFGS (50) |
| `adam_lbfgs` | RAdam (50) + L-BFGS (50) |
| `bayes_adam_lbfgs` | BO (100) + RAdam (50) + L-BFGS (50) |
| `bayes_lbfgs_adam` | BO (100) + L-BFGS (50) + RAdam (50) |

```bash
# Compare optimisation strategies across seeds (Table: strategy comparison)
python optimize_match_spectrum.py

# A subset, or any other defined approach, on demand
python optimize_match_spectrum.py --approaches bayes_lbfgs_adam bayes_lbfgs

# BO with the surrogate loss landscape as GP prior mean
python optimize_match_spectrum.py --config configs/comparison_prior_mean.json

# Stochastic-gradient Langevin variants
python optimize_match_spectrum.py --config configs/comparison_langevin_avg_45_25_20.json

# Objective landscape over the (E, P, t_open) cube (Figure: loss landscape)
python plot_loss_landscape.py --resolution 25

# Leave-one-setting-out evaluation, then the Wasserstein table
python evaluate_exclusion_models.py --n-generated 1000
python evaluate_distributions.py --export-wasserstein wasserstein_loo.csv

# Timing and memory per gradient evaluation (Table: hyperparameters)
python benchmark_timing.py --sweep-steps 1 4 8 18 30
```

`configs/match_spectrum.yaml` also defines approaches that are not in the
default set: the SGD and Langevin/SGLD variants and the surrogate-prior ones.
Select them with `--approaches` or by setting `run.approaches` in a config.

`plots.ipynb` builds the comparison figures and the LaTeX tables from finished
runs; `aggregate_results.py` merges results across runs into one CSV.

Several scripts have offline self-checks that need no GPU:

```bash
python plot_loss_landscape.py --self-test      # scan axis handling
python plot_loss_landscape.py --verify-batching 6   # batched == sequential
```

## Requirements

Python 3.10+, PyTorch (CUDA), NumPy, SciPy, scikit-learn, scikit-optimize,
pandas, matplotlib, PyYAML, schedulefree, tqdm.

## Note on run outputs

Optimisation runs, scan caches and analysis tables are reproducible from the
scripts above and are not tracked here; see `.gitignore`.
