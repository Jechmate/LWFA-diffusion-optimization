"""
Build a real experimental target spectrum for optimize_match_spectrum.py.

Averages the shots of one experiment folder in spectra/ and writes a CSV that the
optimizer can consume as its target (columns energy_MeV, intensity), plus a per-bin
std column used by the built-in real-vs-generated figure.

Default: experiment 14 (E=20, P=11, t=40 ms), the unique match for those settings.

    python build_real_target.py
    python build_real_target.py --experiment 14 --output target_real_exp14_E20_P11_t40.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

SPECTRUM_LENGTH = 256   # matches the model output length / training truncation


def build_target(experiment, spectra_dir="spectra", length=SPECTRUM_LENGTH):
    """Average all shots of `experiment` and return the first `length` bins."""
    files = sorted(glob.glob(os.path.join(spectra_dir, str(experiment), "*.csv")))
    if not files:
        raise FileNotFoundError(f"No shots found in {os.path.join(spectra_dir, str(experiment))}")

    energy = None
    intensities = []
    for fn in files:
        df = pd.read_csv(fn)
        if energy is None:
            energy = df["energy"].values
        intensities.append(df["intensity"].values)

    stack = np.array(intensities)                 # (n_shots, n_bins)
    energy = energy[:length]
    mean = stack.mean(axis=0)[:length]
    std = stack.std(axis=0)[:length]
    return energy, mean, std, len(files)


def main():
    parser = argparse.ArgumentParser(description="Build a real target spectrum CSV")
    parser.add_argument("--experiment", type=int, default=4, help="Experiment folder in spectra/")
    parser.add_argument("--spectra-dir", default="spectra")
    parser.add_argument("--output", default="target_real_exp4_E13_P15_t20.csv")
    parser.add_argument("--params", default="params.csv", help="params.csv for a settings printout")
    args = parser.parse_args()

    energy, mean, std, n_shots = build_target(args.experiment, args.spectra_dir)

    df = pd.DataFrame({"energy_MeV": energy, "intensity": mean, "intensity_std": std})
    df.to_csv(args.output, index=False)

    # Report
    settings = ""
    if os.path.exists(args.params):
        p = pd.read_csv(args.params).set_index("experiment")
        if args.experiment in p.index:
            r = p.loc[args.experiment]
            settings = f"  (E={r.E}, P={r.P} bar, t={r.ms} ms, gain={r.gain}, N={r.perc_N}%)"

    print(f"Experiment {args.experiment}: averaged {n_shots} shots{settings}")
    print(f"  bins={len(energy)}  energy {energy[0]:.2f} -> {energy[-1]:.2f} MeV")
    print(f"  intensity mean range {mean.min():.4f}..{mean.max():.4f}, peak std {std.max():.4f}")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
