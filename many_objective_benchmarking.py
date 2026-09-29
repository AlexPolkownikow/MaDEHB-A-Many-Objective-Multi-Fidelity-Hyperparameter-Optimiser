import time
import os
import numpy as np
import sys
import json


import onnxruntime as ort

_N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

_orig_init = ort.InferenceSession.__init__
def _patched_init(self, *args, sess_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = _N_THREADS
    sess_options.inter_op_num_threads = 1
    _orig_init(self, *args, sess_options=sess_options, **kwargs)
ort.InferenceSession.__init__ = _patched_init

from yahpo_gym import BenchmarkSet

os.environ["OMP_NUM_THREADS"] = "4"

from src.ma_dehb.optimizers.MaDEHB import MaDEHB
from src.ma_dehb.optimizers.adapted_MaNSGA_II import Ma_NSGA_II

from typing import Dict, Any

from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

import hvwfg
import optuna 
from ConfigSpace import Configuration

# SMAC related imports
import smac
print(smac.__file__)
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac.multi_objective.parego import ParEGO
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.intensifier.hyperband import Hyperband
from smac.callback import Callback
import logging
logging.getLogger("smac").setLevel(logging.ERROR)
logging.getLogger("smac").propagate = False

# ---------------------------------------------------------
# Custom Callbacks for Weighted Budgets
# ---------------------------------------------------------
# class WeightedBudgetTerminationCallback:
#     def __init__(self, target_weighted_fevals: float, max_fid: float):
#         self.target_weighted_fevals = target_weighted_fevals
#         self.max_fidelity = max_fid
#         self.accumulated_weighted_fevals = 0.0

#     def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
#         fidelity = trial.user_attrs.get("fidelity_reached", 0.0)
#         self.accumulated_weighted_fevals += float(fidelity) / float(self.max_fidelity)
#         if self.accumulated_weighted_fevals >= self.target_weighted_fevals:
#             study.stop()

class SMACWeightedBudgetCallback(Callback):
    def __init__(self, target_weighted_fevals: float, max_fid: float):
        self.target_weighted_fevals = target_weighted_fevals
        self.max_fidelity = max_fid
        self.accumulated_weighted_fevals = 0.0

    def on_tell_end(self, smac, info, value) -> bool | None:
        fidelity = info.budget if info.budget is not None else self.max_fidelity
        self.accumulated_weighted_fevals += float(fidelity) / float(self.max_fidelity)
        
        if self.accumulated_weighted_fevals >= self.target_weighted_fevals:
            return False 

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
    hv_value = hvwfg.wfg(normalized,ref)
    return hv_value

# ---------------------------------------------------------
# YAHPO-Gym problem wrapper for pymoo.
# ---------------------------------------------------------
class YAHPOProblem(Problem):
    def __init__(self, cs, objective, n_obj):
        self.cs = cs
        self.objective = objective
        self.hp_list = cs.get_hyperparameters()

        xl, xu = [], []
        self.is_log = []  

        for hp in self.hp_list:
            cname = hp.__class__.__name__

            if cname in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                log_scale = hp.log
                self.is_log.append(log_scale)
                if log_scale:
                    xl.append(np.log(hp.lower))
                    xu.append(np.log(hp.upper))
                else:
                    xl.append(hp.lower)
                    xu.append(hp.upper)

            elif cname == "CategoricalHyperparameter":
                self.is_log.append(False)
                xl.append(0); xu.append(len(hp.choices) - 1)

            elif cname == "OrdinalHyperparameter":
                self.is_log.append(False)
                xl.append(0); xu.append(len(hp.sequence) - 1)

            elif cname == "Constant":
                self.is_log.append(False)
                xl.append(0); xu.append(0)

            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp}")

        super().__init__(
            n_var=len(self.hp_list), n_obj=n_obj, n_constr=0,
            xl=np.array(xl, dtype=float), xu=np.array(xu, dtype=float),
        )

    def decode(self, x):
        #from ConfigSpace import Configuration

        values_dict = {}
        for val, hp, log_scale in zip(x, self.hp_list, self.is_log):
            cname = hp.__class__.__name__

            if cname == "UniformFloatHyperparameter":
                real_val = np.exp(val) if log_scale else val
                values_dict[hp.name] = float(np.clip(real_val, hp.lower, hp.upper))

            elif cname == "UniformIntegerHyperparameter":
                real_val = np.exp(val) if log_scale else val
                values_dict[hp.name] = int(round(np.clip(real_val, hp.lower, hp.upper)))

            elif cname == "CategoricalHyperparameter":
                values_dict[hp.name] = hp.choices[int(round(val))]

            elif cname == "OrdinalHyperparameter":
                values_dict[hp.name] = hp.sequence[int(round(val))]

            elif cname == "Constant":
                values_dict[hp.name] = hp.value

        raw_cfg = Configuration(self.cs, values=values_dict, allow_inactive_with_values=True)
        active_keys = self.cs.get_active_hyperparameters(raw_cfg)
        clean_values = {k: raw_cfg[k] for k in active_keys}
        return Configuration(self.cs, values=clean_values)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            config = self.decode(row)
            res = self.objective(config)
            F.append(res["fitness"])
        out["F"] = np.array(F)

# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
def benchmark_an_instance(instance_id: int) ->[]:
    scenario = "iaml_xgboost"
    targets = ["mmce", "rammodel", "timepredict", "nf", "rampredict"]

    bench = BenchmarkSet(scenario=scenario)
    bench.set_instance(bench.instances[instance_id])

    cs = bench.get_opt_space(drop_fidelity_params=True)
    dims = len(cs.get_hyperparameters())

    fidelity_name = bench.config.fidelity_params[0]
    cs_full = bench.get_opt_space(drop_fidelity_params=False)
    fid_hp = cs_full.get_hyperparameter(fidelity_name)

    min_fidelity = fid_hp.lower
    max_fidelity = fid_hp.upper

    print(f"Using fidelity '{fidelity_name}' in range [{min_fidelity}, {max_fidelity}]")

    secondary_fidelity_names = [
        fp for fp in bench.config.fidelity_params if fp != fidelity_name
    ]
    if secondary_fidelity_names:
        print(
            f"Fixing secondary fidelity/nuisance parameter(s) "
            f"{secondary_fidelity_names} at their maximum for every "
            f"evaluation (not tuned, not used as a budget axis)."
        )

    def _secondary_fidelity_values() -> Dict[str, Any]:
        vals = {}
        for fp_name in secondary_fidelity_names:
            fp_hp = cs_full.get_hyperparameter(fp_name)
            vals[fp_name] = fp_hp.sequence[-1] if hasattr(fp_hp, "sequence") else fp_hp.upper
        return vals

    #Multi-fidelity objective function wrapper
    def objective_mf(config, fidelity=None, **kwargs) -> Dict[str, Any]:
        if fidelity is None:
            fidelity = min_fidelity

        if not isinstance(config, dict):
            config = config.get_dictionary()
        config = dict(config) 

        config[fidelity_name] = float(fidelity)
        config.update(_secondary_fidelity_values())

        result = bench.objective_function(config)[0]

        mmce        = float(result["mmce"])
        rammodel    = float(result["rammodel"])
        timepredict = float(result["timepredict"])
        nf          = float(result["nf"])
        rampredict  = float(result["rampredict"])

        fitness = np.array([mmce, rammodel, timepredict, nf, rampredict], dtype=float)

        return {
            "fitness": fitness,
            "cost": float(fidelity),
        }

    def objective_full_fidelity(config, **kwargs) -> Dict[str, Any]:
        if not isinstance(config, dict):
            config = config.get_dictionary()
        config = dict(config) 

        fidelity_int = int(max_fidelity)
        config[fidelity_name] = float(fidelity_int)
        config.update(_secondary_fidelity_values())

        result = bench.objective_function(config)[0]

        mmce        = float(result["mmce"])
        rammodel    = float(result["rammodel"])
        timepredict = float(result["timepredict"])
        nf          = float(result["nf"])
        rampredict  = float(result["rampredict"])

        fitness = np.array([mmce, rammodel, timepredict, nf, rampredict], dtype=float)

        return {
            "fitness": fitness,
            "cost": float(fidelity_int),
        }


    def smac_objective(config, seed: int, budget: float | None = None):
        if not isinstance(config, dict):
            config = config.get_dictionary()
        config = dict(config) 

        fidelity_int = int(max_fidelity)
        config[fidelity_name] = float(fidelity_int)
        config.update(_secondary_fidelity_values())

        result = bench.objective_function(config)[0]

        mmce        = float(result["mmce"])
        rammodel    = float(result["rammodel"])
        timepredict = float(result["timepredict"])
        nf          = float(result["nf"])
        rampredict  = float(result["rampredict"])

        return [mmce, rammodel, timepredict, nf, rampredict]

    def smac_objective_mf(config, seed: int, budget: float | None = None):
        if budget is None:
            budget = max_fidelity

        if not isinstance(config, dict):
            config = config.get_dictionary()
        config = dict(config) 

        config[fidelity_name] = float(budget)
        config.update(_secondary_fidelity_values())

        result = bench.objective_function(config)[0]

        mmce        = float(result["mmce"])
        rammodel    = float(result["rammodel"])
        timepredict = float(result["timepredict"])
        nf          = float(result["nf"])
        rampredict  = float(result["rampredict"])

        return [mmce, rammodel, timepredict, nf, rampredict]


    def optuna_objective_full(trial):
        cfg_full = {}
        for hp in cs.get_hyperparameters():
            if hasattr(hp, "choices"):
                cfg_full[hp.name] = trial.suggest_categorical(hp.name, hp.choices)
            elif hasattr(hp, "lower") and hasattr(hp, "upper"):
                if isinstance(hp.lower, float):
                    cfg_full[hp.name] = trial.suggest_float(
                        hp.name, hp.lower, hp.upper, log=getattr(hp, "log", False)
                    )
                else:
                    cfg_full[hp.name] = trial.suggest_int(
                        hp.name, hp.lower, hp.upper, log=getattr(hp, "log", False)
                    )
            else:
                cfg_full[hp.name] = hp.value

        cfg = Configuration(cs, values=cfg_full, allow_inactive_with_values=True)
        active_names = cs.get_active_hyperparameters(cfg)
        cfg_active = {k: cfg[k] for k in active_names}

        base_dict = dict(cfg_active)
        base_dict[fidelity_name] = float(max_fidelity)
        base_dict.update(_secondary_fidelity_values())

        result = bench.objective_function(base_dict)[0]

        mmce        = float(result["mmce"])
        rammodel    = float(result["rammodel"])
        timepredict = float(result["timepredict"])
        nf          = float(result["nf"])
        rampredict  = float(result["rampredict"])

        return [mmce, rammodel, timepredict, nf, rampredict]

    n_obj = len(targets)
    ref_dirs_nsga3 = get_reference_directions("das-dennis", n_obj, n_partitions=3)
    pop_size = len(ref_dirs_nsga3)
    fevals = pop_size * 16
    number_of_runs = 20

    madehb_res_before_hv_calc = []
    smachb_res_before_hv_calc = []
    smac_res_before_hv_calc = []
    mansga_res_before_hv_calc = []
    optuna_res_before_hv_calc = []
    nsga3_res_before_hv_calc = []
    nsga2_res_before_hv_calc = []
    rand_s_res_before_hv_calc = []

    hv_madehb_final = []
    hv_smachb_final = []
    hv_smac_final = []
    hv_optuna_final = []
    hv_mansga_final = []
    hv_nsga3_final = []
    hv_nsga2_final = []
    hv_rand_s_final = []

    madehb_histories=[]
    smachb_histories=[]
    smac_histories=[]
    mansga_histories=[]
    optuna_histories=[]
    nsga2_histories=[]
    nsga3_histories=[]
    rand_s_histories = []

    for run_idx in range(number_of_runs):
        print(f"\n================ RUN {run_idx+1}/{number_of_runs} ================")

        # ----- Random Search (Single-Fidelity) -----
        print(f"\n[Random Search] Running for {fevals} function evaluations...")
        
        cs.seed(run_idx)
        rs_configs = cs.sample_configuration(size=fevals)
        if fevals == 1: 
            rs_configs = [rs_configs]
            
        rs_fitness_history = []
        for cfg in rs_configs:
            res = objective_full_fidelity(cfg)
            rs_fitness_history.append(res["fitness"])

        rs_fitness_history = np.array(rs_fitness_history)
        rand_s_histories.append(rs_fitness_history)

        nds_rs = NonDominatedSorting()
        fronts_rs = nds_rs.do(rs_fitness_history, return_rank=False)
        final_front_rs = rs_fitness_history[fronts_rs[0]]
        rand_s_res_before_hv_calc.append(final_front_rs)

        # ----- MaDEHB (multi-fidelity) -----
        dehb = MaDEHB(
            cs=cs_full,
            objective_number=len(targets),
            trade_off_param=0.01,
            f=objective_mf,
            dimensions=dims,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            seed=run_idx,
            n_workers=1,
        )

        print("\n[MaDEHB] Running for", fevals, "function evaluations...")
        result = dehb.run(fevals=fevals)

        history = np.array(result[3], dtype=object)
        fitness_history_madehb = np.array([entry[2] for entry in history])
        cost_history_madehb = np.array([entry[3] for entry in history])
        madehb_histories.append((fitness_history_madehb, cost_history_madehb))

        #The algorithm keeps track of the configs evaluated at full fidelity internally, and outputs them separately as well
        full_budget_configs, idx = np.unique(np.array(result[0][0]), axis=0, return_index=True)
        full_fitness = np.array(result[0][1])[idx]

        nds = NonDominatedSorting()
        fronts = nds.do(full_fitness, return_rank=False)
        final_front_madehb = full_fitness[fronts[0]]

        madehb_res_before_hv_calc.append(final_front_madehb)


        # ----- SMAC-HB (multi-fidelity) --------
        print("\n[SMAC(HB)] Running for", fevals, "weighted function evaluations...")

        scenario = Scenario(
            configspace=cs,
            objectives=targets,
            seed=run_idx,
            n_trials=10000, 
            n_workers=1,
            walltime_limit=float("inf"),
            min_budget=min_fidelity,  
            max_budget=max_fidelity,
        )

        initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)

        mo_algo = MeanAggregationStrategy(scenario)

        intensifier = Hyperband(
            scenario=scenario,
            eta=3,
            n_seeds=1,
        )

        smac_hb = HPOFacade(
            scenario=scenario,
            target_function=smac_objective_mf,
            initial_design=initial_design,
            multi_objective_algorithm=mo_algo,
            intensifier=intensifier,
            overwrite=True,
            callbacks=[SMACWeightedBudgetCallback(fevals, max_fidelity)]
        )

        incumbents_hb = smac_hb.optimize()

        fitness_history_smac_hb = []
        cost_history_smac_hb = []
        all_evals_for_history = []
        
        for trial_key, trial_value in smac_hb.runhistory._data.items():
            fidelity = float(trial_key.budget)
            cost_history_smac_hb.append(fidelity)
            
            obj_vals = np.array(trial_value.cost, dtype=float)
            all_evals_for_history.append(obj_vals)
            
            if fidelity == max_fidelity:
                fitness_history_smac_hb.append(obj_vals)

        if len(fitness_history_smac_hb) == 0:
             fitness_history_smac_hb = [all_evals_for_history[-1]]

        fitness_history_smac_hb = np.array(fitness_history_smac_hb)
        cost_history_smac_hb = np.array(cost_history_smac_hb)
        all_evals_for_history = np.array(all_evals_for_history)

        smachb_histories.append((all_evals_for_history, cost_history_smac_hb))

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_smac_hb, return_rank=False)
        final_front_smac_hb = fitness_history_smac_hb[fronts[0]]

        smachb_res_before_hv_calc.append(final_front_smac_hb)


        # ----- SMAC  (Single-Fidelity) --------
        print("\n[SMAC] Running for", fevals, "function evaluations...")

        scenario = Scenario(
            configspace=cs,
            objectives=targets,
            seed=run_idx,
            n_trials=fevals, 
            n_workers=1,
            walltime_limit=float("inf"), 
        )

        initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)

        mo_algo = ParEGO(scenario)

        intensifier = HPOFacade.get_intensifier(
            scenario,
            max_config_calls=1, 
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=smac_objective,
            initial_design=initial_design,
            multi_objective_algorithm=mo_algo,
            intensifier=intensifier,
            overwrite=True,
        )
        
        incumbents = smac.optimize()

        fitness_history_smac = []
        cost_history_smac = []

        for trial_key, trial_value in smac.runhistory._data.items():
            obj_vals = np.array(trial_value.cost, dtype=float)
            fitness_history_smac.append(obj_vals)
            cost_history_smac.append(float(max_fidelity))

        fitness_history_smac = np.array(fitness_history_smac)
        cost_history_smac = np.array(cost_history_smac)

        smac_histories.append((fitness_history_smac, cost_history_smac))

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_smac, return_rank=False)
        final_front_smac = fitness_history_smac[fronts[0]]

        smac_res_before_hv_calc.append(final_front_smac)

        # ----- MaNSGA-II (Single-Fidelity) --------
        print("\n[MaNSGA-II] Running for", fevals, "function evaluations...")
        mansga = Ma_NSGA_II(
            cs=cs,
            f=objective_full_fidelity,
            dimensions=dims,
            objective_number=len(targets),
            mutation_factor=0.5,
            crossover_prob=0.5,
            normalize_objective_space=False,
            pop_size=pop_size,
            seed=run_idx,
            trade_off_param=0.01,
        )

        traj, runtime, history, final_pop_fitness = mansga.run(
            generations=15,
            verbose=False,
        )

        fitness_history_mansga = np.array([entry[1] for entry in history], dtype=float)

        cost_history_mansga = np.array(runtime, dtype=float)

        mansga_histories.append((fitness_history_mansga, cost_history_mansga))

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_mansga, return_rank=False)
        final_front_mansga = fitness_history_mansga[fronts[0]]

        mansga_res_before_hv_calc.append(final_front_mansga)

        # ----- Optuna  (Single-Fidelity) --------
        print(f"\n[Optuna-MO] Running for {fevals} full-fidelity evaluations...")

        sampler = optuna.samplers.NSGAIISampler(seed=run_idx)
        study = optuna.create_study(
            directions=["minimize"] * n_obj,
            sampler=sampler
        )

        study.optimize(
            optuna_objective_full,
            n_trials=fevals
        )

        fitness_history_optuna = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                fitness_history_optuna.append(np.array(t.values, dtype=float))

        fitness_history_optuna = np.array(fitness_history_optuna)

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_optuna, return_rank=False)
        final_front_optuna = fitness_history_optuna[fronts[0]]

        optuna_res_before_hv_calc.append(final_front_optuna)
        optuna_histories.append((fitness_history_optuna, np.ones(len(fitness_history_optuna))*max_fidelity))


        # ----- NSGA-III  (Single-Fidelity) -----
        problem = YAHPOProblem(cs, objective_full_fidelity, n_obj=n_obj)

        nsga3 = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs_nsga3,
        )

        print("\n[NSGA-III] Running for", fevals, "function evaluations...")
        res_nsga3 = minimize(
            problem,
            nsga3,
            ("n_eval", fevals),
            seed=run_idx,
            verbose=False,
            save_history=True,
        )
        
        fitness_history_nsga3 = []
        for entry in res_nsga3.history:
            pop = entry.pop
            F = pop.get("F")
            for f in F:
                fitness_history_nsga3.append(f)

        fitness_history_nsga3 = np.array(fitness_history_nsga3)
        nsga3_histories.append(fitness_history_nsga3)

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_nsga3, return_rank=False)
        final_front_nsga3 = fitness_history_nsga3[fronts[0]]
        nsga3_res_before_hv_calc.append(final_front_nsga3)
        #nsga3_res_before_hv_calc.append(nsga3_F)


        # ----- NSGA-II  (Single-Fidelity) -----
        problem = YAHPOProblem(cs, objective_full_fidelity, n_obj=n_obj)
        nsga2 = NSGA2(
            pop_size=pop_size,
            eliminate_duplicates=True
        )

        print("\n[NSGA-II] Running for", fevals, "function evaluations...")
        res_nsga2 = minimize(
            problem,
            nsga2,
            ('n_eval', fevals),
            seed=run_idx,
            verbose=False,
            save_history=True,
        )
       
        fitness_history_nsga2 = []
        for entry in res_nsga2.history:
            pop = entry.pop
            F = pop.get("F")
            for f in F:
                fitness_history_nsga2.append(f)

        fitness_history_nsga2 = np.array(fitness_history_nsga2)
        nsga2_histories.append(fitness_history_nsga2)

        nds = NonDominatedSorting()
        fronts = nds.do(fitness_history_nsga2, return_rank=False)
        final_front_nsga2 = fitness_history_nsga2[fronts[0]]
        nsga2_res_before_hv_calc.append(final_front_nsga2)
        #nsga2_res_before_hv_calc.append(res_nsga2.F)

    # ----- HV Calculations -----
    all_fronts = np.vstack(
        madehb_res_before_hv_calc +
        smachb_res_before_hv_calc +
        smac_res_before_hv_calc +
        mansga_res_before_hv_calc +
        optuna_res_before_hv_calc +
        nsga3_res_before_hv_calc +
        nsga2_res_before_hv_calc +
        rand_s_res_before_hv_calc
    )

    mins = np.min(all_fronts, axis=0)
    maxs = np.max(all_fronts, axis=0)

    front_lists = [madehb_res_before_hv_calc, smachb_res_before_hv_calc, smac_res_before_hv_calc, mansga_res_before_hv_calc, optuna_res_before_hv_calc, nsga3_res_before_hv_calc, nsga2_res_before_hv_calc, rand_s_res_before_hv_calc]
    hv_lists = [hv_madehb_final, hv_smachb_final, hv_smac_final, hv_mansga_final, hv_optuna_final, hv_nsga3_final, hv_nsga2_final, hv_rand_s_final]

    for algo_idx, front_list in enumerate(front_lists):
        for front in front_list:
            hv = compute_hv_minspace(front, mins, maxs, len(targets))
            hv_lists[algo_idx].append(hv)

    print("\n======= SUMMARY OVER RUNS OF AN ISNTANCE ========")
    print("MaDEHB final HVs:", hv_madehb_final)
    print("SMAC(HB) final HVs:", hv_smachb_final)
    print("SMAC final HVs:", hv_smac_final)
    print("MaNSGA-II final HVs:", hv_mansga_final)
    print("Optuna final HVs:", hv_optuna_final)
    print("NSGA-III final HVs:", hv_nsga3_final)
    print("NSGA-II final HVs:", hv_nsga2_final)
    print("Rand. Search final HVs:", hv_rand_s_final)

    print("MaDEHB mean HV:", np.mean(hv_madehb_final))
    print("SMAC(HB) mean HV:", np.mean(hv_smachb_final))
    print("SMAC mean HV:", np.mean(hv_smac_final))
    print("MaNSGA-II mean HV:", np.mean(hv_mansga_final))
    print("Optuna mean HV:", np.mean(hv_optuna_final))
    print("NSGA-III mean HV:", np.mean(hv_nsga3_final))
    print("NSGA-II mean HV:", np.mean(hv_nsga2_final))
    print("Rand. Search mean HV:", np.mean(hv_rand_s_final))

    return (
            [np.array(hv_madehb_final), np.array(hv_smachb_final), np.array(hv_smac_final), np.array(hv_mansga_final), np.array(hv_optuna_final), np.array(hv_nsga3_final), np.array(hv_nsga2_final), np.array(hv_rand_s_final)],  
            [madehb_histories, smachb_histories, smac_histories, mansga_histories, optuna_histories, nsga3_histories, nsga2_histories, rand_s_histories], 
            mins, maxs
        )

def to_jsonable(obj):
    """Recursively convert numpy arrays and other non-JSON types into JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def main():
    # Parse instance ID from command line
    if len(sys.argv) < 2:
        raise ValueError("Usage: python benchmark.py <instance_id>")

    instance_id = int(sys.argv[1])

    # Run one instance
    res, histories, mins, maxs = benchmark_an_instance(instance_id)

    # Convert everything to JSON-safe format
    json_data = {
        "hv_results": to_jsonable(res),
        "histories": to_jsonable(histories),
        "mins": to_jsonable(mins),
        "maxs": to_jsonable(maxs),
    }

    # Save to a JSON file
    outfile = f"/home/uzq06230/hpo_benchmarking_slurm_output/iaml_xgboost/instance_{instance_id}.json"
    #outfile = f"/home/uzq06230/MaDEHB/job_scheduling_test/instance_{instance_id}.json"
    with open(outfile, "w") as f:
        json.dump(json_data, f)

    print(f"Saved results for instance {instance_id} to {outfile}")


if __name__ == "__main__":
    main()