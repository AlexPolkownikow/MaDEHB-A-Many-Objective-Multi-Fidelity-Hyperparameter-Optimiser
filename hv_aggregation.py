
import json
import sys
from pathlib import Path

import numpy as np

# Order matters
ALGO_NAMES = [
    "MaDEHB",
    "SMAC-HB",
    "SMAC",
    "MaNSGA-II",
    "Optuna",
    "NSGA-III",
    "NSGA-II",
    "Random Search",
]


def load_hv_results(json_path: Path):
    """Returns hv_results (list of per-algorithm lists of HV values) or None."""
    try:
        with json_path.open("r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [skip] {json_path.name}: could not read/parse ({e})")
        return None

    if "hv_results" not in data:
        print(f"  [skip] {json_path.name}: no 'hv_results' key")
        return None

    hv_results = data["hv_results"]
    if len(hv_results) != len(ALGO_NAMES):
        print(
            f"  [skip] {json_path.name}: expected {len(ALGO_NAMES)} algorithms "
            f"in hv_results, found {len(hv_results)}"
        )
        return None

    return hv_results


def main(results_dir: str):
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {results_dir.resolve()}")
        return

    print(f"Found {len(json_files)} json file(s) in {results_dir.resolve()}:")
    pooled = {name: [] for name in ALGO_NAMES}

    for jf in json_files:
        hv_results = load_hv_results(jf)
        if hv_results is None:
            continue
        n_this_file = 0
        for name, values in zip(ALGO_NAMES, hv_results):
            values = np.asarray(values, dtype=float)
            n_nan = int(np.isnan(values).sum())
            if n_nan:
                print(f"  [warn] {jf.name}: {name} has {n_nan} NaN value(s), excluding them")
            values = values[~np.isnan(values)]
            pooled[name].extend(values.tolist())
            n_this_file += len(values)
        print(f"  [ok] {jf.name}: {n_this_file} total HV values read")

    # --- aggregate ---
    means, stds, counts = {}, {}, {}
    for name in ALGO_NAMES:
        vals = np.asarray(pooled[name], dtype=float)
        counts[name] = len(vals)
        if len(vals) == 0:
            means[name] = np.nan
            stds[name] = np.nan
        else:
            means[name] = float(np.mean(vals))
            stds[name] = float(np.std(vals))

    valid_means = [m for m in means.values() if not np.isnan(m)]
    if not valid_means:
        print("\nNo valid HV values found for any algorithm.")
        return

    lo, hi = min(valid_means), max(valid_means)
    spread = hi - lo

    normalized = {}
    for name in ALGO_NAMES:
        m = means[name]
        if np.isnan(m):
            normalized[name] = np.nan
        elif spread == 0:
            # every algorithm has the identical mean HV -- nothing to rank
            normalized[name] = 100.0
        else:
            normalized[name] = 100.0 * (m - lo) / spread

    # --- print table ---
    order = sorted(
        ALGO_NAMES,
        key=lambda n: (-normalized[n] if not np.isnan(normalized[n]) else float("inf")),
    )

    header = f"{'Algorithm':<14} {'N':>5} {'Mean HV':>12} {'Std HV':>10} {'Normalized (0-100)':>20}"
    print("\n" + header)
    print("-" * len(header))
    for name in order:
        if counts[name] == 0:
            print(f"{name:<14} {'--':>5} {'--':>12} {'--':>10} {'--':>20}")
            continue
        print(
            f"{name:<14} {counts[name]:>5} {means[name]:>12.5f} {stds[name]:>10.5f} "
            f"{normalized[name]:>20.2f}"
        )
    print()


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(results_dir)