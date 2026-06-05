"""
aggregate_results.py

Average per-seed metric files into a single mean/std summary.
Optionally remove per-seed files and extra checkpoints to save space.

Usage:
    python3 aggregate_results.py \
        --prefix results/rungB_20260605_1200/pg \
        --models "quantum qaoa node-quantum node-qaoa classical" \
        --seeds "0 1 2 3 4" \
        [--delete-seeds]
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def aggregate(
    prefix: str,
    models: list[str],
    seeds: list[str],
    metrics: list[str],
    delete_seeds: bool = False,
) -> None:
    for model in models:
        for metric in metrics:
            files = [f"{prefix}_{model}_s{s}_{metric}.txt" for s in seeds]
            existing = [f for f in files if os.path.exists(f)]
            if not existing:
                print(f"  skip  {model}/{metric}  (no files found)")
                continue

            arrs = [np.loadtxt(f) for f in existing]
            maxlen = max(len(a) for a in arrs)
            padded = np.array([
                np.pad(a, (0, maxlen - len(a)), constant_values=np.nan)
                for a in arrs
            ])
            avg = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)

            np.savetxt(f"{prefix}_{model}_avg_{metric}.txt", avg)
            np.savetxt(f"{prefix}_{model}_std_{metric}.txt", std)
            print(f"  saved {os.path.basename(prefix)}_{model}_avg_{metric}.txt"
                  f"  (n={len(existing)} seeds, len={maxlen})")

            if delete_seeds:
                for f in existing:
                    os.remove(f)

        # Keep only seed-0 checkpoint as the representative model; remove others.
        if delete_seeds:
            for s in seeds[1:]:
                pt = f"{prefix}_{model}_s{s}.pt"
                if os.path.exists(pt):
                    os.remove(pt)
            s0 = f"{prefix}_{model}_s0.pt"
            avg_pt = f"{prefix}_{model}_avg.pt"
            if os.path.exists(s0):
                os.rename(s0, avg_pt)
                print(f"  checkpoint: s0 → {os.path.basename(avg_pt)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Average per-seed RL result files")
    p.add_argument("--prefix",  required=True,
                   help="Path prefix shared by all files, e.g. results/rungB/pg")
    p.add_argument("--models",  required=True,
                   help="Space-separated model names")
    p.add_argument("--seeds",   required=True,
                   help="Space-separated seed integers used during training")
    p.add_argument("--metrics", default="rewards dists losses feas",
                   help="Space-separated metric names (default: rewards dists losses feas)")
    p.add_argument("--delete-seeds", action="store_true",
                   help="Remove per-seed .txt files and extra .pt files after aggregation")
    args = p.parse_args()

    models  = args.models.split()
    seeds   = args.seeds.split()
    metrics = args.metrics.split()

    print(f"[aggregate] prefix={args.prefix}")
    print(f"            models={models}  seeds={seeds}  delete={args.delete_seeds}")
    aggregate(args.prefix, models, seeds, metrics, args.delete_seeds)
    print("[aggregate] done")


if __name__ == "__main__":
    main()
