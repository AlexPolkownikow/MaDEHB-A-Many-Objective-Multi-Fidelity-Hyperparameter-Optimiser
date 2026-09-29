import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
import hvwfg

# ---------------------------------------------------------
# HV Calculation
# ---------------------------------------------------------

def normalize_front_minspace(front: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = (maxs - mins).astype(float)
    denom[denom == 0.0] = 1.0 

    normalized = (front - mins) / denom

    return normalized


def compute_hv_minspace(front: np.ndarray, mins: np.ndarray, maxs: np.ndarray, objective_number: int) -> float:
    normalized = normalize_front_minspace(front, mins, maxs)
    ref = np.ones(objective_number) * (1.01)
    if objective_number == 1:
        # 1-D hypervolume: distance from the reference point to the best (smallest) value
        return float(max(0.0, ref[0] - normalized.min()))
    hv_value = hvwfg.wfg(normalized,ref)

    return hv_value



# ---------------------------------------------------------
# Load histories
# ---------------------------------------------------------
def load_madehb_history(path):
    df = pd.read_parquet(path)
    fitness_history = np.vstack(df["fitness"].values)
    costs = df["cost"].values  # fidelity_int
    return fitness_history, costs


def load_nsga_history(path):
    df = pd.DataFrame(pd.read_parquet(path))
    fitness_history = np.vstack(df["fitness"].values)
    return fitness_history


# ---------------------------------------------------------
# Build x-axes (weighted FEvals vs plain fevals)
# ---------------------------------------------------------
def build_madehb_x(costs, max_fidelity):
    weights = costs / float(max_fidelity)
    return np.cumsum(weights)


def build_nsga_x(n_evals):
    return np.arange(1, n_evals + 1, dtype=float)




# ---------------------------------------------------------
# Fitness array of a stored history item
# ---------------------------------------------------------
def _history_fitness(item):
    """(T, n_obj) fitness array of a history item, stored either as (fitness, costs) or as a bare fitness array."""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        try:
            candidate = np.asarray(item[0], dtype=float)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.ndim == 2:
            return candidate
    arr = np.asarray(item, dtype=float)
    return arr.reshape(len(arr), -1)


def _shuffle_within_generations(item, pop_size, rng):
    """Restore an unbiased evaluation order for NSGA-II / NSGA-III histories.

   """
    fitness = _history_fitness(item)
    if len(fitness) % pop_size != 0:
        raise ValueError(
            f"NSGA history has {len(fitness)} rows, not a multiple of pop_size={pop_size}; "
            "pass the correct --pop-size."
        )
    out = fitness.copy()
    for start in range(0, len(fitness), pop_size):
        out[start:start + pop_size] = fitness[start:start + pop_size][rng.permutation(pop_size)]
    return out


# ---------------------------------------------------------
# Incremental Pareto front
# ---------------------------------------------------------
class IncrementalParetoFront:
    def __init__(self, n_obj):
        self.points = np.empty((0, n_obj))

    def add(self, point):
        """Add a point (minimization). Returns True if the front changed."""
        point = np.asarray(point, dtype=float).reshape(1, -1)

        if self.points.shape[0] == 0:
            self.points = point.copy()
            return True

        # is the new point dominated by (or equal to) something already on the front?
        existing_dominates_new = np.any(
            np.all(self.points <= point, axis=1) & np.any(self.points < point, axis=1)
        )
        if existing_dominates_new:
            return False  # front unchanged, discard the new point

        # does the new point dominate any current front members? drop them
        new_dominates_existing = (
            np.all(point <= self.points, axis=1) & np.any(point < self.points, axis=1)
        )
        if np.any(new_dominates_existing):
            self.points = self.points[~new_dominates_existing]

        self.points = np.vstack([self.points, point])
        return True


# ---------------------------------------------------------
# HV curve from fitness history
# ---------------------------------------------------------
def compute_hv_curve_minspace(fitness_history, mins, maxs, objective_number, compute_hv_minspace):
    """
    fitness_history: (T, x) array in minimization space
    mins, maxs: global bounds (x,)
    compute_hv_minspace: function(front, mins, maxs) -> float (ideally in [0,1])

    Maintains the Pareto front incrementally (see IncrementalParetoFront)
    and only calls the (comparatively expensive) hypervolume routine when
    the front actually changed -- if the new evaluation didn't improve
    the front (very common later in a run, once an algorithm has mostly
    converged), the HV is unchanged too, so it's just reused instead of
    being recomputed from scratch.
    """
    hv_curve = np.empty(len(fitness_history))
    pf = IncrementalParetoFront(objective_number)
    current_hv = 0.0

    for t, point in enumerate(fitness_history):
        changed = pf.add(point)
        if changed:
            current_hv = compute_hv_minspace(pf.points, mins, maxs, objective_number)
        hv_curve[t] = current_hv

    return hv_curve


# ---------------------------------------------------------
# Parallel HV-curve computation across independent runs
# ---------------------------------------------------------
def _hv_curve_worker(args):
    fitness_history, mins, maxs, objective_number = args
    return compute_hv_curve_minspace(fitness_history, mins, maxs, objective_number, compute_hv_minspace)


def _default_n_jobs():
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1, int(val.split("(")[0]))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


# ---------------------------------------------------------
# Pad to max length 
# ---------------------------------------------------------

def pad_to_max_length(list_of_arrays):
    max_len = max(len(a) for a in list_of_arrays)
    padded = []
    for a in list_of_arrays:
        if len(a) < max_len:
            pad_width = max_len - len(a)
            # Forward-fill the last known value (hypervolume doesn't disappear when a run ends)
            last_val = a[-1] if len(a) > 0 else 0.0
            a = np.concatenate([a, np.full(pad_width, last_val)])
        padded.append(a)
    return np.vstack(padded)


# ---------------------------------------------------------
# Main visualization entry point
# ---------------------------------------------------------
def visualize_iaml_multi(
    madehb_histories,
    smachb_histories,
    smac_histories,
    optuna_histories,
    nsga2_histories,
    nsga3_histories,
    rand_s_histories,
    mins,
    maxs,
    max_fidelity,
    outfile="hv_progression_rbv2_xgboost.pdf",
    n_jobs=None,
):
    """
    Visualizes HV progression using mean +/- bootstrap CI.

    n_jobs: number of worker processes used to compute per-run HV curves
        in parallel (each run is independent). Defaults to the number of
        CPUs SLURM has allocated to the job (SLURM_CPUS_PER_TASK /
        SLURM_JOB_CPUS_PER_NODE) if set, else os.cpu_count(). Pass 1 to
        force single-process execution (e.g. for debugging).
    """

    hv_madehb_runs, x_madehb_runs = [], []
    hv_smachb_runs, x_smachb_runs = [], []
    hv_smac_runs,   x_smac_runs   = [], []
    hv_optuna_runs, x_optuna_runs = [], []
    hv_nsga2_runs,  x_nsga2_runs  = [], []
    hv_nsga3_runs,  x_nsga3_runs  = [], []
    hv_rand_s_runs, x_rand_s_runs = [], []

    objective_number = len(mins)

    if n_jobs is None:
        n_jobs = _default_n_jobs()

    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None

    def hv_curves_for(fitness_histories):
        fitness_histories = list(fitness_histories)
        args = [(fh, mins, maxs, objective_number) for fh in fitness_histories]
        if executor is None or len(args) <= 1:
            return [_hv_curve_worker(a) for a in args]
        return list(executor.map(_hv_curve_worker, args))

    try:
        # -----------------------------
        # MaDEHB (Multi-Fidelity)
        # -----------------------------
        madehb_fitness = [fh for fh, _ in madehb_histories]
        madehb_costs = [ch for _, ch in madehb_histories]
        for hv_curve, cost_history in zip(hv_curves_for(madehb_fitness), madehb_costs):
            cost_history = np.asarray(cost_history, dtype=float)
            x = np.cumsum(cost_history / float(max_fidelity))
            hv_madehb_runs.append(np.array(hv_curve))
            x_madehb_runs.append(np.array(x))

        # -----------------------------
        # SMAC-HB (Multi-Fidelity)
        # -----------------------------
        smachb_fitness = [fh for fh, _ in smachb_histories]
        smachb_costs = [ch for _, ch in smachb_histories]
        for hv_curve, cost_history in zip(hv_curves_for(smachb_fitness), smachb_costs):
            cost_history = np.asarray(cost_history, dtype=float)
            x = np.cumsum(cost_history / float(max_fidelity))
            hv_smachb_runs.append(np.array(hv_curve))
            x_smachb_runs.append(np.array(x))

        # -----------------------------
        # SMAC (Single-Fidelity)
        # -----------------------------
        smac_fitness = [fh for fh, _ in smac_histories]
        for fh, hv_curve in zip(smac_fitness, hv_curves_for(smac_fitness)):
            hv_smac_runs.append(np.array(hv_curve))
            x_smac_runs.append(np.arange(1, len(fh) + 1, dtype=float))

        # -----------------------------
        # Optuna (Single-Fidelity)
        # -----------------------------
        optuna_fitness = [fh for fh, _ in optuna_histories]
        for fh, hv_curve in zip(optuna_fitness, hv_curves_for(optuna_fitness)):
            hv_optuna_runs.append(np.array(hv_curve))
            x_optuna_runs.append(np.arange(1, len(fh) + 1, dtype=float))

        # -----------------------------
        # NSGA-II (Single-Fidelity)
        # -----------------------------
        nsga2_histories = list(nsga2_histories)
        for fh, hv_curve in zip(nsga2_histories, hv_curves_for(nsga2_histories)):
            hv_nsga2_runs.append(np.array(hv_curve))
            x_nsga2_runs.append(np.arange(1, len(fh) + 1, dtype=float))

        # -----------------------------
        # NSGA-III (Single-Fidelity)
        # -----------------------------
        nsga3_histories = list(nsga3_histories)
        for fh, hv_curve in zip(nsga3_histories, hv_curves_for(nsga3_histories)):
            hv_nsga3_runs.append(np.array(hv_curve))
            x_nsga3_runs.append(np.arange(1, len(fh) + 1, dtype=float))

        # -----------------------------
        # Random Search (Single-Fidelity)
        # -----------------------------
        rand_s_histories = list(rand_s_histories)
        for fh, hv_curve in zip(rand_s_histories, hv_curves_for(rand_s_histories)):
            hv_rand_s_runs.append(np.array(hv_curve))
            x_rand_s_runs.append(np.arange(1, len(fh) + 1, dtype=float))
    finally:
        if executor is not None:
            executor.shutdown()

    # -----------------------------
    # Stack HV curves
    # -----------------------------
    hv_madehb_stack = pad_to_max_length(hv_madehb_runs)
    hv_smachb_stack = pad_to_max_length(hv_smachb_runs)
    hv_smac_stack   = pad_to_max_length(hv_smac_runs)
    hv_optuna_stack = pad_to_max_length(hv_optuna_runs)
    hv_nsga2_stack  = pad_to_max_length(hv_nsga2_runs)
    hv_nsga3_stack  = pad_to_max_length(hv_nsga3_runs)
    hv_rand_s_stack = pad_to_max_length(hv_rand_s_runs)

    # -----------------------------
    # Mean + Bootstrap CI
    # -----------------------------
    madehb_mean, madehb_lower, madehb_upper = bootstrap_ci(hv_madehb_stack.T)
    smachb_mean, smachb_lower, smachb_upper = bootstrap_ci(hv_smachb_stack.T)
    smac_mean,   smac_lower,   smac_upper   = bootstrap_ci(hv_smac_stack.T)
    optuna_mean, optuna_lower, optuna_upper = bootstrap_ci(hv_optuna_stack.T)
    nsga2_mean,  nsga2_lower,  nsga2_upper  = bootstrap_ci(hv_nsga2_stack.T)
    nsga3_mean,  nsga3_lower,  nsga3_upper  = bootstrap_ci(hv_nsga3_stack.T)
    rand_s_mean, rand_s_lower, rand_s_upper = bootstrap_ci(hv_rand_s_stack.T)


    # X-axes: first run per algorithm
    x_madehb = x_madehb_runs[0]
    x_smachb = x_smachb_runs[0]
    x_smac   = x_smac_runs[0]
    x_optuna = x_optuna_runs[0]
    x_nsga2  = x_nsga2_runs[0]
    x_nsga3  = x_nsga3_runs[0]
    x_rand_s = x_rand_s_runs[0]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8, 5))

    # MaDEHB
    plt.plot(x_madehb, madehb_mean, label="MaDEHB", color="C0", linewidth=2)
    plt.fill_between(x_madehb, madehb_lower, madehb_upper, color="C0", alpha=0.2)

    # SMAC-HB
    plt.plot(x_smachb, smachb_mean, label="SMAC-HB(MeanAgg)", color="C1", linewidth=2)
    plt.fill_between(x_smachb, smachb_lower, smachb_upper, color="C1", alpha=0.2)

    # SMAC
    plt.plot(x_smac, smac_mean, label="SMAC(ParEGO)", color="C2", linewidth=2)
    plt.fill_between(x_smac, smac_lower, smac_upper, color="C2", alpha=0.2)

    # Optuna
    plt.plot(x_optuna, optuna_mean, label="Optuna", color="C3", linewidth=2)
    plt.fill_between(x_optuna, optuna_lower, optuna_upper, color="C3", alpha=0.2)

    # NSGA-II
    plt.plot(x_nsga2, nsga2_mean, label="NSGA-II", color="C4", linewidth=2)
    plt.fill_between(x_nsga2, nsga2_lower, nsga2_upper, color="C4", alpha=0.2)

    # NSGA-III
    plt.plot(x_nsga3, nsga3_mean, label="NSGA-III", color="C5", linewidth=2)
    plt.fill_between(x_nsga3, nsga3_lower, nsga3_upper, color="C5", alpha=0.2)

    # Random Search
    plt.plot(x_rand_s, rand_s_mean, label="Rand. Search", color="C6", linewidth=2)
    plt.fill_between(x_rand_s, rand_s_lower, rand_s_upper, color="C6", alpha=0.2)

    plt.ylim(0.6, 1.0)
    plt.xlabel("Weighted Function Evaluations")
    plt.ylabel("Mean Hypervolume")
    plt.title("RBV2 XGBOOST (Mean with Bootstrap Confidence Interval)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def bootstrap_ci(data, n_boot=2000, alpha=0.05, rng=None, t_chunk=300):
    """
    data: shape (T, n_runs)
    Returns mean, lower_ci, upper_ci for each time step.

    Statistically identical to the original procedure -- independent
    resampling (with replacement) of the n_runs values at each time step,
    n_boot times -- just vectorized with numpy instead of a Python loop
    that called np.random.choice T * n_boot times (which dominates runtime
    for anything but tiny T). Processed in chunks over T so memory stays
    bounded regardless of how long the histories are; lower t_chunk if you
    have many runs and hit memory pressure.
    """
    if rng is None:
        rng = np.random.default_rng()

    T, n_runs = data.shape
    lower = np.empty(T)
    upper = np.empty(T)
    lo_i = int(alpha / 2 * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot)

    for start in range(0, T, t_chunk):
        end = min(start + t_chunk, T)
        block = data[start:end]
        t_len = end - start

        idx = rng.integers(0, n_runs, size=(t_len, n_boot, n_runs))
        boot_samples = block[np.arange(t_len)[:, None, None], idx]
        boot_means = boot_samples.mean(axis=2)
        boot_means.sort(axis=1)

        lower[start:end] = boot_means[:, lo_i]
        upper[start:end] = boot_means[:, hi_i]

    means = data.mean(axis=1)
    return means, lower, upper

# ---------------------------------------------------------
# Main visualization entry point
# ---------------------------------------------------------
from scipy.stats import rankdata

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def rank_visualisation(
    madehb_histories,
    smachb_histories,
    smac_histories,
    mansga_histories,
    optuna_histories,
    nsga2_histories,
    nsga3_histories,
    rand_s_histories,
    mins,
    maxs,
    max_fidelity,
    outfile="rank_progression_rbv2_super.pdf",
    n_jobs=None,
    num_grid_points=500,
    title=None,
):
    objective_number = len(mins)
    if n_jobs is None:
        n_jobs = _default_n_jobs()

    executor = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None

    def hv_curves_for(fitness_histories):
        fitness_histories = list(fitness_histories)
        args = [(fh, mins, maxs, objective_number) for fh in fitness_histories]
        if executor is None or len(args) <= 1:
            return [_hv_curve_worker(a) for a in args]
        return list(executor.map(_hv_curve_worker, args))

    # Helper to unpack history item safely whether it's (fh, ch) or just fh
    def parse_run_item(item):
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], (tuple, list, np.ndarray))
        ):
            return item[0], item[1]
        return item, None

    alg_data = {
        "MaDEHB": {"x": [], "y": []},
        "SMAC-HB(MeanAgg)": {"x": [], "y": []},
        "SMAC(ParEGO)": {"x": [], "y": []},
        "MaNSGA-II": {"x": [], "y": []},
        "Optuna": {"x": [], "y": []},
        "NSGA-II": {"x": [], "y": []},
        "NSGA-III": {"x": [], "y": []},
        "Rand. Search": {"x": [], "y": []},
    }

    try:
        # Multi-Fidelity & Cost-Aware Algorithms (MaDEHB, SMAC-HB, MaNSGA-II)
        for alg_key, histories in [
            ("MaDEHB", madehb_histories),
            ("SMAC-HB(MeanAgg)", smachb_histories),
            ("MaNSGA-II", mansga_histories),
        ]:
            parsed = [parse_run_item(item) for item in histories]
            fhs = [p[0] for p in parsed]
            chs = [
                (
                    p[1]
                    if p[1] is not None
                    else np.ones(len(p[0]), dtype=float) * float(max_fidelity)
                )
                for p in parsed
            ]

            hv_curves = hv_curves_for(fhs)
            for hv_curve, cost_history in zip(hv_curves, chs):
                alg_data[alg_key]["x"].append(
                    np.cumsum(
                        np.asarray(cost_history, dtype=float)
                        / float(max_fidelity)
                    )
                )
                alg_data[alg_key]["y"].append(np.array(hv_curve))

        # Single-Fidelity Standard Algorithms (Evaluations = Iterations)
        for alg_key, histories in [
            ("SMAC(ParEGO)", smac_histories),
            ("Optuna", optuna_histories),
            ("NSGA-II", nsga2_histories),
            ("NSGA-III", nsga3_histories),
            ("Rand. Search", rand_s_histories),
        ]:
            fhs = [parse_run_item(item)[0] for item in histories]
            hv_curves = hv_curves_for(fhs)

            for fh, hv_curve in zip(fhs, hv_curves):
                alg_data[alg_key]["x"].append(
                    np.arange(1, len(fh) + 1, dtype=float)
                )
                alg_data[alg_key]["y"].append(np.array(hv_curve))
    finally:
        if executor is not None:
            executor.shutdown()

    max_x_val = max(
        (x[-1] for data in alg_data.values() for x in data["x"] if len(x) > 0),
        default=1.0,
    )
    x_common = np.linspace(0, max_x_val, num_grid_points)

    def interpolate_step(x, y, x_new):
        if len(x) == 0:
            return np.zeros_like(x_new)
        idx = np.clip(
            np.searchsorted(x, x_new, side="right") - 1, 0, len(y) - 1
        )
        y_interp = y[idx]
        y_interp[x_new < x[0]] = 0.0
        return y_interp

    alg_names = list(alg_data.keys())
    num_algs = len(alg_names)
    num_runs = min(len(alg_data[name]["x"]) for name in alg_names)

    hv_matrix = np.zeros((num_algs, num_grid_points, num_runs))
    for a_idx, name in enumerate(alg_names):
        for r_idx in range(num_runs):
            hv_matrix[a_idx, :, r_idx] = interpolate_step(
                alg_data[name]["x"][r_idx],
                alg_data[name]["y"][r_idx],
                x_common,
            )

    ranks = rankdata(-hv_matrix, axis=0)

    mean_ranks = np.zeros((num_algs, num_grid_points))
    lower_ranks = np.zeros((num_algs, num_grid_points))
    upper_ranks = np.zeros((num_algs, num_grid_points))
    for a_idx in range(num_algs):
        mean_ranks[a_idx], lower_ranks[a_idx], upper_ranks[a_idx] = (
            bootstrap_ci(ranks[a_idx])
        )

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    plt.figure(figsize=(8, 5))

    for a_idx, name in enumerate(alg_names):
        plt.plot(
            x_common,
            mean_ranks[a_idx],
            label=name,
            color=colors[a_idx],
            linewidth=2,
        )
        plt.fill_between(
            x_common,
            lower_ranks[a_idx],
            upper_ranks[a_idx],
            color=colors[a_idx],
            alpha=0.2,
        )

    plt.ylim(1, num_algs)
    plt.gca().invert_yaxis()
    plt.xlabel("Weighted Function Evaluations")
    plt.ylabel("Mean Rank")
    plt.title(
        title or "LCBench Rank Progression (Mean with Bootstrap Confidence Interval)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
 

import sys
import json

def average_rank_summary(
    madehb_histories,
    smachb_histories,
    smac_histories,
    mansga_histories,
    optuna_histories,
    nsga2_histories,
    nsga3_histories,
    rand_s_histories,
    mins,
    maxs,
    max_fidelity,
    n_jobs=None,
    *,
    budget_start=0.0,
    budget_end=None,
):
    mins, maxs = np.asarray(mins, dtype=float), np.asarray(maxs, dtype=float)
    if mins.ndim != 1 or mins.size == 0 or maxs.shape != mins.shape:
        raise ValueError("mins and maxs must be non-empty vectors of equal length.")
    if (not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs))
            or np.any(maxs < mins)):
        raise ValueError("Normalisation bounds must be finite, with maxs >= mins.")
    objective_number = len(mins)
    max_fidelity = float(max_fidelity)
    if not np.isfinite(max_fidelity) or max_fidelity <= 0:
        raise ValueError("max_fidelity must be finite and positive.")

    algorithms = [
        ("MaDEHB", list(madehb_histories), True),
        ("SMAC-HB(MeanAgg)", list(smachb_histories), True),
        ("SMAC(ParEGO)", list(smac_histories), False),
        ("MaNSGA-II", list(mansga_histories), True),
        ("Optuna", list(optuna_histories), False),
        ("NSGA-II", list(nsga2_histories), False),
        ("NSGA-III", list(nsga3_histories), False),
        ("Rand. Search", list(rand_s_histories), False),
    ]
    counts = {name: len(histories) for name, histories, _ in algorithms}
    if not all(counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(
            "All algorithms need the same non-zero number of runs; "
            f"received {counts}. No runs are silently discarded."
        )
    num_runs = next(iter(counts.values()))

    def parse_run(item):
        # Requiring a 2-D first element avoids misreading a bare history
        # containing exactly two objective vectors as (history, costs).
        if isinstance(item, (tuple, list)) and len(item) == 2:
            try:
                candidate = np.asarray(item[0], dtype=float)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate.ndim == 2:
                return candidate, item[1]
        return np.asarray(item, dtype=float), None

    runs_by_algorithm, all_runs, worker_args = [], [], []
    for name, histories, weighted in algorithms:
        runs = []
        for repetition, item in enumerate(histories):
            fitness, costs = parse_run(item)
            label = f"{name}, run {repetition + 1}"
            if (fitness.ndim != 2 or len(fitness) == 0
                    or fitness.shape[1] != objective_number
                    or not np.all(np.isfinite(fitness))):
                raise ValueError(
                    f"{label}: expected a non-empty, finite fitness history with "
                    f"{objective_number} objective columns."
                )
            if weighted and costs is not None:
                costs = np.asarray(costs, dtype=float)
                if (costs.shape != (len(fitness),)
                        or not np.all(np.isfinite(costs)) or np.any(costs < 0)):
                    raise ValueError(
                        f"{label}: supply one finite, non-negative cost "
                        "per fitness observation."
                    )
                x = np.cumsum(costs / max_fidelity)
            else:
                x = np.arange(1, len(fitness) + 1, dtype=float)
            if not np.all(np.isfinite(x)) or x[-1] <= 0:
                raise ValueError(f"{label}: total budget must be finite and positive.")
            run = {"x": x, "label": label}
            runs.append(run)
            all_runs.append(run)
            worker_args.append((fitness, mins, maxs, objective_number))
        runs_by_algorithm.append(runs)

    common_end = min(run["x"][-1] for run in all_runs)
    start = float(budget_start)
    end = common_end if budget_end is None else float(budget_end)
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end:
        raise ValueError("Require finite bounds with 0 <= budget_start < budget_end.")
    if end > common_end:
        if not np.isclose(end, common_end, rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"budget_end={end:g} exceeds the common recorded budget "
                f"{common_end:g}; use a smaller endpoint."
            )
        end = common_end  # Allow harmless floating-point accumulation error.
    if start >= end:
        raise ValueError("budget_start must be below the common recorded budget.")

    if n_jobs is None:
        n_jobs = _default_n_jobs()
    if (isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer or None.")

    def store_curves(curves):
        for run, curve in zip(all_runs, curves):
            y = np.asarray(curve, dtype=float)
            if y.shape != run["x"].shape or not np.all(np.isfinite(y)):
                raise ValueError(
                    f"{run['label']}: _hv_curve_worker must return one finite "
                    "hypervolume per fitness observation."
                )
            run["y"] = y

    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=min(int(n_jobs), len(all_runs))) as executor:
            store_curves(executor.map(_hv_curve_worker, worker_args))
    else:
        store_curves(map(_hv_curve_worker, worker_args))

    totals = np.zeros(len(algorithms), dtype=float)
    for repetition in range(num_runs):
        runs = [group[repetition] for group in runs_by_algorithm]
        # Ranks can change only when at least one run records a new result.
        changes = [np.array([start, end])]
        for run in runs:
            x = run["x"]
            changes.append(x[(x > start) & (x < end)])
        edges = np.unique(np.concatenate(changes))
        left_edges = edges[:-1]
        weights = np.diff(edges) / (end - start)

        hv = np.zeros((len(algorithms), len(left_edges)), dtype=float)
        for a_idx, run in enumerate(runs):
            indices = np.searchsorted(run["x"], left_edges, side="right") - 1
            observed = indices >= 0
            hv[a_idx, observed] = run["y"][indices[observed]]

        ranks = rankdata(-hv, axis=0, method="average")
        totals += ranks @ weights

    result = {
        name: float(totals[a_idx] / num_runs)
        for a_idx, (name, _, _) in enumerate(algorithms)
    }
    for name, value in result.items():
        print(f"{name}: {value:.6f}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot the rank progression, then print the average rank per algorithm.")
    parser.add_argument("json_dir", help="Directory containing the benchmark JSON files.")
    parser.add_argument("scenario", help="Scenario label used for the plot title and file name "
                                         "(rank_progression_<scenario>.pdf).")
    parser.add_argument("--budget-start", type=float, default=0.0,
                        help="First budget included in the rank summary (default: 0).")
    parser.add_argument("--budget-end", type=float, default=None,
                        help="Last budget included (default: the shortest recorded run).")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Worker processes for hypervolume calculations.")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size of NSGA-II / NSGA-III (rows per generation in their "
                             "stored histories). Default: the value stored in the JSON files; "
                             "35 if the JSON has none.")
    parser.add_argument("--max-fidelity", type=float, default=None,
                        help="Maximum fidelity of the scenario (costs are divided by it, so one "
                             "full-fidelity evaluation counts as 1 on the x-axis). Default: the value "
                             "stored in the JSON files; 52 (LCBench) if the JSON has none.")
    args = parser.parse_args()
    json_dir = args.json_dir

    # Storage for merged histories
    madehb_histories = []
    smachb_histories = []
    smac_histories = []
    mansga_histories = []
    optuna_histories = []
    nsga2_histories = []
    nsga3_histories = []
    rand_s_histories = []

    json_max_fidelities, json_pop_sizes = set(), set()
    summary_problems = []  # run-count problems only block the rank summary, not the plot

    # Iterate over all JSON files 
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue

        full_path = os.path.join(json_dir, fname)
        with open(full_path, "r") as f:
            data = json.load(f)

        # Extract histories (list of lists)
        histories = data["histories"]
        run_counts = [len(group) for group in histories]
        if len(run_counts) != 8 or len(set(run_counts)) != 1 or not all(run_counts):
            summary_problems.append(
                f"{fname}: rank summaries require eight algorithms with equal, "
                f"non-zero run counts in each file; received {run_counts}."
            )
        # histories structure:
        # [
        #   madehb_histories,
        #   smachb_histories,
        #   smac_histories,
        #   mansga_histories,
        #   optuna_histories,
        #   nsga3_histories,
        #   nsga2_histories,
        #   rand_s_histories
        # ]

        madehb_histories.extend(histories[0])
        smachb_histories.extend(histories[1])
        smac_histories.extend(histories[2])
        mansga_histories.extend(histories[3])
        optuna_histories.extend(histories[4])
        nsga3_histories.extend(histories[5])
        nsga2_histories.extend(histories[6])
        rand_s_histories.extend(histories[7])

        if "max_fidelity" in data:
            json_max_fidelities.add(float(data["max_fidelity"]))
        if "pop_size" in data:
            json_pop_sizes.add(int(data["pop_size"]))

    if len(json_pop_sizes) > 1:
        raise ValueError(f"JSON files disagree on pop_size: {sorted(json_pop_sizes)}")
    pop_size = args.pop_size if args.pop_size is not None else (next(iter(json_pop_sizes)) if json_pop_sizes else 35)
    shuffle_rng = np.random.default_rng(0)  # fixed seed -> reproducible plots
    nsga2_histories = [_shuffle_within_generations(h, pop_size, shuffle_rng) for h in nsga2_histories]
    nsga3_histories = [_shuffle_within_generations(h, pop_size, shuffle_rng) for h in nsga3_histories]

    all_histories = [madehb_histories, smachb_histories, smac_histories, mansga_histories,
                     optuna_histories, nsga2_histories, nsga3_histories, rand_s_histories]
    fitness_arrays = [_history_fitness(item) for group in all_histories for item in group]
    if not fitness_arrays:
        raise ValueError(f"No histories found in {json_dir}.")
    global_mins = np.min([f.min(axis=0) for f in fitness_arrays], axis=0)
    global_maxs = np.max([f.max(axis=0) for f in fitness_arrays], axis=0)

    if len(json_max_fidelities) > 1:
        raise ValueError(f"JSON files disagree on max_fidelity: {sorted(json_max_fidelities)}")
    if args.max_fidelity is not None:
        max_fidelity = args.max_fidelity
    elif json_max_fidelities:
        max_fidelity = next(iter(json_max_fidelities))
    else:
        max_fidelity = 52.0  # older LCBench JSONs without metadata
    scenario = args.scenario
    print(f"Scenario: {scenario} | max_fidelity: {max_fidelity:g} | pop_size: {pop_size} | objectives: {len(global_mins)}")

    comparison_args = dict(
        madehb_histories=madehb_histories,
        smachb_histories=smachb_histories,
        smac_histories=smac_histories,
        mansga_histories=mansga_histories,
        optuna_histories=optuna_histories,
        nsga2_histories=nsga2_histories,
        nsga3_histories=nsga3_histories,
        rand_s_histories=rand_s_histories,
        mins=global_mins,
        maxs=global_maxs,
        max_fidelity=max_fidelity,
        n_jobs=args.n_jobs,
    )
    # 1) rank progression plot
    outfile = f"fixed_rank_progression_{scenario.replace(' ', '_')}.pdf"
    rank_visualisation(
        **comparison_args,
        outfile=outfile,
        title=f"{scenario} Rank Progression (Mean with Bootstrap Confidence Interval)",
    )
    print(f"Visualization written to {outfile}")

    # 2) average-rank summary (printed after the plot)
    if summary_problems:
        raise ValueError("\n".join(summary_problems))
    print(f"\nAverage rank summary ({scenario}):")
    average_rank_summary(
        **comparison_args,
        budget_start=args.budget_start,
        budget_end=args.budget_end,
    )

if __name__ == "__main__":
    main()