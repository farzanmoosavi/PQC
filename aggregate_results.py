"""
aggregate_results.py

Average per-seed metric files into mean/std curves, then write a one-row-per-model
summary CSV.  Optionally delete per-seed txt files and extra checkpoints.

Final output per rung (example for 5 models):
  summary.csv                        ← one row per model, final 100-ep stats
  dqn_quantum_avg_rewards.txt        ← full reward curve (mean over seeds)
  dqn_quantum_std_rewards.txt        ← std over seeds at each episode
  dqn_quantum_avg.pt                 ← checkpoint (seed-0 weights)
  ... (same pattern for every model × metric)

Usage:
    python3 aggregate_results.py \
        --prefix  results/rungB_20260605/pg \
        --models  "quantum qaoa node-quantum node-qaoa classical" \
        --seeds   "0 1 2 3 4 5 6" \
        --out-csv results/rungB_20260605/summary.csv \
        [--delete-seeds]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

METRICS = ("rewards", "dists", "losses", "complete")
TAIL = 100   # episodes used for "final" statistics in the summary CSV


def _load_curves(prefix: str, model: str, seeds: list[str], metric: str):
    files = [f"{prefix}_{model}_s{s}_{metric}.txt" for s in seeds]
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        return None, None, 0
    arrs = [np.loadtxt(f) for f in existing]
    maxlen = max(len(a) for a in arrs)
    padded = np.array([
        np.pad(a, (0, maxlen - len(a)), constant_values=np.nan)
        for a in arrs
    ])
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0), len(existing)


def aggregate(
    prefix: str,
    models: list[str],
    seeds: list[str],
    out_csv: str,
    delete_seeds: bool = False,
) -> None:
    rows = []

    for model in models:
        row = {"model": model}
        n_seeds_found = 0

        for metric in METRICS:
            avg, std, n = _load_curves(prefix, model, seeds, metric)
            if avg is None:
                print(f"  skip  {model}/{metric}  (no files)")
                row[f"{metric}_final"] = float("nan")
                row[f"{metric}_final_std"] = float("nan")
                continue

            n_seeds_found = max(n_seeds_found, n)
            np.savetxt(f"{prefix}_{model}_avg_{metric}.txt", avg)
            np.savetxt(f"{prefix}_{model}_std_{metric}.txt", std)
            print(f"  saved {os.path.basename(prefix)}_{model}_avg_{metric}.txt"
                  f"  (n={n} seeds)")

            tail = avg[-TAIL:]
            tail_std = std[-TAIL:]
            row[f"{metric}_final"]     = float(np.nanmean(tail))
            row[f"{metric}_final_std"] = float(np.nanmean(tail_std))

            if delete_seeds:
                for s in seeds:
                    f = f"{prefix}_{model}_s{s}_{metric}.txt"
                    if os.path.exists(f):
                        os.remove(f)

        row["n_seeds"] = n_seeds_found

        if delete_seeds:
            # Always keep seed "0" as the representative checkpoint regardless
            # of seed list order; delete all others.
            for s in seeds:
                if str(s) == "0":
                    continue
                pt = f"{prefix}_{model}_s{s}.pt"
                if os.path.exists(pt):
                    os.remove(pt)
            s0 = f"{prefix}_{model}_s0.pt"
            avg_pt = f"{prefix}_{model}_avg.pt"
            if os.path.exists(s0):
                os.rename(s0, avg_pt)
                print(f"  checkpoint → {os.path.basename(avg_pt)}")

        rows.append(row)

    if not rows:
        return

    # Write summary CSV
    fieldnames = [
        "model", "n_seeds",
        "rewards_final",   "rewards_final_std",
        "dists_final",     "dists_final_std",
        "complete_final",  "complete_final_std",
        "losses_final",    "losses_final_std",
    ]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  summary → {out_csv}")
    print(f"  {'model':<16} {'reward':>10} {'±':>8}  {'dist':>8}  {'complete':>8}")
    print(f"  {'-'*56}")
    for r in rows:
        print(f"  {r['model']:<16} {r['rewards_final']:>10.2f} "
              f"{r['rewards_final_std']:>8.2f}  "
              f"{r['dists_final']:>8.2f}  "
              f"{r['complete_final']:>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate per-seed RL result files")
    p.add_argument("--prefix",   required=True,
                   help="Path prefix, e.g. results/rungB_20260605/pg")
    p.add_argument("--models",   required=True,
                   help="Space-separated model names")
    p.add_argument("--seeds",    required=True,
                   help="Space-separated seed integers used during training")
    p.add_argument("--out-csv",  default=None,
                   help="Path for summary CSV (default: <prefix_dir>/summary.csv)")
    p.add_argument("--delete-seeds", action="store_true",
                   help="Remove per-seed .txt and extra .pt files after aggregation")
    args = p.parse_args()

    models = args.models.split()
    seeds  = args.seeds.split()
    out_csv = args.out_csv or os.path.join(os.path.dirname(args.prefix), "summary.csv")

    print(f"[aggregate] prefix={args.prefix}  seeds={seeds}  delete={args.delete_seeds}")
    aggregate(args.prefix, models, seeds, out_csv, args.delete_seeds)
    print("[aggregate] done")


if __name__ == "__main__":
    main()
