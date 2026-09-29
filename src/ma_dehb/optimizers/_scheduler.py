import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Timer
from typing import List, Tuple, Union

import ConfigSpace
import numpy as np
import pandas as pd
from ..utils._compat import Client
from loguru import logger

from ..utils import ConfigRepository, SHBracketManager
from .de import AsyncDE
from .adapted_MaNSGA_II import Ma_NSGA_II, AsyncMa_NSGA_II

_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB",
}


class MaDEHBBase:
    def __init__(self, cs=None, objective_number=1, trade_off_param=None, f=None, dimensions=None, mutation_factor=None,
                 crossover_prob=None, strategy=None, min_fidelity=None,
                 max_fidelity=None, eta=None, min_clip=None, max_clip=None, seed=None,
                 boundary_fix_type="random", max_age=np.inf, resume=False,
                 normalize_objective_space=True, **kwargs):
        # Check for deprecated parameters
        if "max_budget" in kwargs or "min_budget" in kwargs:
            raise TypeError("Parameters min_budget and max_budget have been deprecated since " \
                            "v0.1.0. Please use the new parameters min_fidelity and max_fidelity " \
                            "or downgrade to a version prior to v0.1.0")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)
        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Miscellaneous
        self._setup_logger(resume, kwargs)
        self.config_repository = ConfigRepository()

        # Benchmark related variables
        self.cs = cs
        self.use_configspace = True if isinstance(self.cs, ConfigSpace.ConfigurationSpace) else False
        if self.use_configspace:
            self.cs.seed(self._original_seed)
            self.dimensions = len(list(self.cs.values()))
        elif dimensions is None or not isinstance(dimensions, (int, np.integer)):
            assert "Need to specify `dimensions` as an int when `cs` is not available/specified!"
        else:
            self.dimensions = dimensions
        self.f = f

        # MaNSGA-II related variables
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.objective_number = objective_number
        if trade_off_param is None:
            if objective_number <= 2:
                trade_off_param = 0.0
            elif objective_number <= 5:
                trade_off_param = 0.01
            else:
                trade_off_param = 0.10
        if not (0.0 <= trade_off_param <= 1.0):
            raise ValueError("trade_off_param must be in [0, 1]")
        self.trade_off_param = trade_off_param
        self.normalize_objective_space = normalize_objective_space
        self.strategy = strategy
        self.fix_type = boundary_fix_type
        self.max_age = max_age
        self.mansga_params = {
            "mutation_factor": self.mutation_factor,
            "crossover_prob": self.crossover_prob,
            "objective_number": self.objective_number,
            "trade_off_param": self.trade_off_param,
            "strategy": self.strategy,
            "configspace": self.use_configspace,
            "boundary_fix_type": self.fix_type,
            "max_age": self.max_age,
            "normalize_objective_space": self.normalize_objective_space,
            "cs": self.cs,
            "dimensions": self.dimensions,
            "f": f,
        }

        # Hyperband related variables
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        if self.max_fidelity <= self.min_fidelity:
            self.logger.error("Only (Max Fidelity > Min Fidelity) is supported for DEHB.")
            if self.max_fidelity == self.min_fidelity:
                self.logger.error(
                    "If you have a fixed fidelity, " \
                    "you can instead run adapted_MaNSGA_II. For more information checkout: " \
                    "https://automl.github.io/DEHB/references/de")
            raise AssertionError()
        self.eta = eta
        self.min_clip = min_clip
        self.max_clip = max_clip

        # Precomputing fidelity spacing and number of configurations for HB iterations
        self._pre_compute_fidelity_spacing()

        # Updating MaNSGA-II parameter list
        self.mansga_params.update({"output_path": self.output_path})

        # Global trackers
        self.population = None
        self.fitness = None
        # a dictionary of lists of configs used to keep tracks of the results from the solutions of active brackets
        #self.result_buffer = {}
        #a list of fitness values corresponding to the configs of inc_configs
        self.inc_score = []
        # instead of one config, save the first front (nondominated configs)
        self.inc_configs = None
        self.history = []

        self.full_budget_configs = []
        self.full_budget_fitness = []
        self.full_budget_ids = []

        # Keeps track of weighted (by fidelity) fevals
        self.current_fevals = 0.0

    def _setup_logger(self, resume, kwargs):
        """Sets up the logger."""
        log_level = kwargs["log_level"] if "log_level" in kwargs else "WARNING"
        _logger_props["level"] = log_level
        logger.configure(handlers=[{"sink": sys.stdout, "level": log_level}])
        self.output_path = Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        # Only append to log if resuming an optimization run, else overwrite
        _logger_props["mode"] = "a" if resume else "w"
        self.log_filename = f"{self.output_path}/dehb.log"
        self.logger.add(
            self.log_filename,
            **_logger_props,
        )

    def _pre_compute_fidelity_spacing(self):
        self.max_SH_iter = None
        self.fidelities = None
        if self.min_fidelity is not None and \
           self.max_fidelity is not None and \
           self.eta is not None:
            self.max_SH_iter = -int(np.log(self.min_fidelity / self.max_fidelity) / np.log(self.eta)) + 1
            self.fidelities = self.max_fidelity * np.power(self.eta,
                                                     -np.linspace(start=self.max_SH_iter - 1,
                                                                  stop=0, num=self.max_SH_iter))

    def reset(self, *, reset_seeds: bool = True):
        self.inc_score = []
        self.inc_configs = None
        self.population = None
        self.fitness = None
        self.traj = []
        self.runtime = []
        self.history = []
        self.full_budget_configs = []
        self.full_budget_fitness = []
        self.full_budget_ids = []
        self.current_fevals = 0.0
        self._bracket_rung_results = {}
        self._promotion_state = {}
        self.mo_buffers = {}
        if reset_seeds:
            if isinstance(self.cs, ConfigSpace.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)
        self.logger.info("\n\nRESET at {}\n\n".format(time.strftime("%x %X %Z")))

    def _init_population(self):
        raise NotImplementedError("Redefine!")

    def _get_next_iteration(self, iteration: int) -> Tuple[np.array, np.array]:
        """Computes the Successive Halving spacing.

        Given the iteration index, computes the fidelity spacing to be used and
        the number of configurations to be used for the SH iterations.

        Args:
            iteration (int): Iteration index.

        Returns:
            A tuple containing number of configurations in the bracket
            and the respective fidelities
        """
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # fidelity spacing for this iteration
        fidelities = self.fidelities[(-s-1):]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, fidelities

    # def _get_next_iteration(self, iteration: int):
    #     """Computes the Successive Halving spacing.

    #     Given the iteration index, computes the fidelity spacing to be used and
    #     the number of configurations to be used for the SH iterations.

    #     Args:
    #         iteration (int): Iteration index.

    #     Returns:
    #         A tuple containing number of configurations in the bracket
    #         and the respective fidelities
    #     """
    #     # Hyperband fidelity logic
    #     s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
    #     fidelities = self.fidelities[(-s - 1):]

    #     # Population scaling parameters 
    #     POP_MAX = 100
    #     POP_MIN = 20
    #     ALPHA = 1.0

    #     ns = []
    #     for fid in fidelities:
    #         scale = (fid / self.max_fidelity) ** ALPHA
    #         pop = int(POP_MIN + scale * (POP_MAX - POP_MIN))
    #         pop = max(pop, 1)
    #         ns.append(pop)

    #     ns = sorted(ns, reverse=True)

    #     return ns, fidelities


    def get_incumbents(self) -> Tuple[List[Union[dict, "ConfigSpace.Configuration"]], List]:
        """Return incumbent Pareto front as (configs_list, scores_list).

        - Always returns lists (even for a single incumbent).
        - Calls vector_to_configspace(v) for each vector when use_configspace is True.
        """
        if self.inc_configs is None:
            return [], []

        # normalize inc_configs -> list of vectors
        inc = self.inc_configs
        if isinstance(inc, np.ndarray):
            if inc.ndim == 1:
                configs_list = [inc]
            else:
                configs_list = [inc[i] for i in range(inc.shape[0])]
        elif isinstance(inc, (list, tuple)):
            configs_list = list(inc)
        else:
            configs_list = [inc]

        # convert to ConfigSpace objects if requested
        if self.use_configspace:
            cs_list = [self.vector_to_configspace(v) for v in configs_list]
            configs_out = cs_list
        else:
            configs_out = [np.asarray(v) for v in configs_list]

        # normalize scores -> list of objective vectors
        scores = self.inc_score
        if isinstance(scores, np.ndarray):
            if scores.ndim == 1:
                scores_list = [scores.tolist()]
            else:
                scores_list = [scores[i].tolist() for i in range(scores.shape[0])]
        elif isinstance(scores, (list, tuple)):
            scores_list = [list(s) if isinstance(s, (list, tuple, np.ndarray)) else s for s in scores]
        else:
            scores_list = [scores]

        return configs_out, scores_list

    def _f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class MaDEHB(MaDEHBBase):
    def __init__(self, cs=None, objective_number=1, trade_off_param=None, f=None, dimensions=None, mutation_factor=0.5,
                 crossover_prob=0.5, strategy="rand1_bin", min_fidelity=None,
                 max_fidelity=None, eta=3, min_clip=None, max_clip=None, seed=None,
                 configspace=True, boundary_fix_type="random", max_age=np.inf, n_workers=None,
                 client=None, async_strategy="immediate", save_freq="incumbent", resume=False,
                 normalize_objective_space=True, mo_selection_batch_size=1, **kwargs):
        super().__init__(cs=cs, objective_number=objective_number, trade_off_param=trade_off_param, f=f, dimensions=dimensions, mutation_factor=mutation_factor,
                         crossover_prob=crossover_prob, strategy=strategy, min_fidelity=min_fidelity,
                         max_fidelity=max_fidelity, eta=eta, min_clip=min_clip, max_clip=max_clip,
                         normalize_objective_space=normalize_objective_space,
                         seed=seed, configspace=configspace, boundary_fix_type=boundary_fix_type,
                         max_age=max_age, resume=resume, **kwargs)
        self.mansga_params.update({"async_strategy": async_strategy})
        self.iteration_counter = -1
        self.mansga = {}
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects
        self.traj = []
        self.runtime = []
        self.history = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.start = None
        if save_freq not in ["incumbent", "step", "end"] and save_freq is not None:
            self.logger.warning(f"Save frequency {save_freq} unknown. Resorting to using 'end'.")
            save_freq = "end"
        self.save_freq = "end" if save_freq is None else save_freq

        # one buffer per fidelity
        self.mo_buffers = {}

        self.mo_selection_batch_size = max(1, int(mo_selection_batch_size))
        self._bracket_rung_results = {}
        self._promotion_state = {}

        # Dask variables
        if n_workers is None and client is None:
            raise ValueError("Need to specify either 'n_workers'(>0) or 'client' (a Dask client)!")
        if client is not None and isinstance(client, Client):
            self.client = client
            self.n_workers = len(client.ncores())
        else:
            self.n_workers = n_workers
            if self.n_workers > 1:
                self.client = Client(
                    n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
                )  # port 0 makes Dask select a random free port
            else:
                self.client = None
        self.futures = []
        self.shared_data = None

        # Initializing MaNSGA-II subpopulations
        self._get_pop_sizes()
        self._init_subpop()
        self.config_repository.initial_configs = self.config_repository.configs.copy()

        # Misc.
        self.available_gpus = None
        self.gpu_usage = None
        self.single_node_with_gpus = None

        self._time_budget_exhausted = False
        self._runtime_budget_timer = None

        # Setup logging and potentially reload state
        if resume:
            self.logger.info("Loading checkpoint...")
            success = self._load_checkpoint(self.output_path)
            if not success:
                self.logger.error("Checkpoint could not be loaded. " \
                                  "Please refer to the prior warning in order to " \
                                  "identifiy the problem.")
                raise AttributeError("Checkpoint could not be loaded. Check the logs" \
                                     "for more information")
        elif (self.output_path / "dehb_state.json").exists():
            self.logger.warning("A checkpoint already exists, " \
                                "results could potentially be overwritten.")

    def __getstate__(self):
        """Allows the object to picklable while having Dask client as a class attribute."""
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        d["_runtime_budget_timer"] = None # hack to allow timer object to be a class attribute
        return d

    def __del__(self):
        """Ensures a clean kill of the Dask client and frees up a port."""
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def _f_objective(self, job_info):
        """Wrapper to call MaNSGA-II's objective function."""
        # check if job_info appended during job submission self.submit_job() includes "gpu_devices"
        if "gpu_devices" in job_info and self.single_node_with_gpus:
            # should set the environment variable for the spawned worker process
            # reprioritising a CUDA device order specific to this worker process
            os.environ.update({"CUDA_VISIBLE_DEVICES": job_info["gpu_devices"]})

        config, config_id = job_info["config"], job_info["config_id"]
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        bracket_id = job_info["bracket_id"]
        kwargs = job_info["kwargs"]
        res = self.mansga[fidelity].f_objective(config, fidelity, **kwargs)
        #TODO verify correctness
        info = res["info"] if "info" in res else {}
        run_info = {
            "job_info": {
                "config": config,
                "config_id": config_id,
                "fidelity": fidelity,
                "parent_id": parent_id,
                "bracket_id": bracket_id,
            },
            "result": {
                "fitness": res["fitness"],
                "cost": res["cost"],
                "info": info,
            },
        }

        if "gpu_devices" in job_info:
            # important for GPU usage tracking if single_node_with_gpus=True
            device_id = int(job_info["gpu_devices"].strip().split(",")[0])
            run_info.update({"device_id": device_id})
        return run_info

    def _create_cuda_visible_devices(self, available_gpus: List[int], start_id: int) -> str:
        """Generates a string to set the CUDA_VISIBLE_DEVICES environment variable.

        Given a list of available GPU device IDs and a preferred ID (start_id), the environment
        variable is created by putting the start_id device first, followed by the remaining devices
        arranged randomly. The worker that uses this string to set the environment variable uses
        the start_id GPU device primarily now.
        """
        assert start_id in available_gpus
        available_gpus = deepcopy(available_gpus)
        available_gpus.remove(start_id)
        self.rng.shuffle(available_gpus)
        final_variable = [str(start_id)] + [str(_id) for _id in available_gpus]
        final_variable = ",".join(final_variable)
        return final_variable

    def _distribute_gpus(self):
        """Function to create a GPU usage tracker dict.

        The idea is to extract the exact GPU device IDs available. During job submission, each
        submitted job is given a preference of a GPU device ID based on the GPU device with the
        least number of active running jobs. On retrieval of the result, this gpu usage dict is
        updated for the device ID that the finished job was mapped to.
        """
        try:
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            available_gpus = available_gpus.strip().split(",")
            self.available_gpus = [int(_id) for _id in available_gpus]
        except KeyError as e:
            print("Unable to find valid GPU devices. "
                  f"Environment variable {str(e)} not visible!")
            self.available_gpus = []
        self.gpu_usage = dict()
        for _id in self.available_gpus:
            self.gpu_usage[_id] = 0

    def _timeout_handler(self) -> None:
        self.logger.warning("Runtime budget exhausted. Saving optimization checkpoint now.")
        self.save()
        # Important to set this flag to true after saving
        self._time_budget_exhausted = True

    def vector_to_configspace(self, config: np.array) -> ConfigSpace.Configuration:
        """Converts numpy representation to `Configuration`.

        Args:
            config (np.array): Configuration to convert.

        Returns:
            ConfigSpace.Configuration: Converted configuration
        """
        assert hasattr(self, "mansga")
        assert len(self.fidelities) > 0
        return self.mansga[self.fidelities[0]].vector_to_configspace(config)

    def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.array:
        """Converts `Configuration` to numpy array.

        Args:
            config (ConfigSpace.Configuration): Configuration to convert

        Returns:
            np.array: Converted configuration
        """
        assert hasattr(self, "mansga")
        assert len(self.fidelities) > 0
        return self.mansga[self.fidelities[0]].configspace_to_vector(config)

    def reset(self, *, reset_seeds: bool = True):
        super().reset(reset_seeds=reset_seeds)
        if self.n_workers > 1 and hasattr(self, "client") and isinstance(self.client, Client):
            self.client.restart()
        else:
            self.client = None
        self.futures = []
        self.shared_data = None
        self.iteration_counter = -1
        self.mansga = {}
        self._max_pop_size = None
        self.start = None
        self.active_brackets = []
        self.traj = []
        self.runtime = []
        self.history = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.config_repository.reset()
        self._get_pop_sizes()
        self._init_subpop()
        self.available_gpus = None
        self.gpu_usage = None
        self._time_budget_exhausted = False
        self._runtime_budget_timer = None

    def _init_population(self, pop_size):
        if self.use_configspace:
            population = self.cs.sample_configuration(size=pop_size)
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            population = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        return population

    def _clean_inactive_brackets(self):
        """Removes brackets from the active list if it is done as communicated by Bracket Manager."""
        if len(self.active_brackets) == 0:
            return
        self.active_brackets = [
            bracket for bracket in self.active_brackets if ~bracket.is_bracket_done()
        ]
        return

    def _update_trackers(self, traj, runtime, history):
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _update_incumbents(self, configs, score, info):
        self.inc_configs = configs
        self.inc_score = score
        self.inc_info = info

    def _get_pop_sizes(self):
        """Determines maximum pop size for each fidelity."""
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self._get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    # def _get_pop_sizes(self):
    #     """
    #     Set MaNSGA-II population sizes based on:
    #     - dimensionality
    #     - number of objectives
    #     - fidelity level (mild scaling)
    #     """
    #     base = max(20, 4 * self.dimensions)        # evolutionary minimum
    #     obj_factor = max(1.0, self.objective_number / 2)

    #     self._max_pop_size = {}

    #     for f in self.fidelities:
    #         fidelity_scale = 0.5 + 0.5 * (f / self.max_fidelity)
    #         pop = int(base * obj_factor * fidelity_scale)
    #         pop = max(pop, 20)
    #         self._max_pop_size[f] = pop


    # def _init_subpop(self):
    #     """List of MaNSGA-II objects corresponding to the fidelities."""
    #     self.mansga = {}
    #     seeds = self.rng.integers(0, 2**32 - 1, size=len(self._max_pop_size))
    #     for (i, f), _seed in zip(enumerate(self._max_pop_size.keys()), seeds):
    #         self.mansga[f] = AsyncMa_NSGA_II(**self.mansga_params, pop_size=self._max_pop_size[f],
    #                              config_repository=self.config_repository, seed=int(_seed))
    #         self.mansga[f].population = self.mansga[f].init_population(pop_size=self._max_pop_size[f])
    #         self.mansga[f].population_ids = self.config_repository.announce_population(self.mansga[f].population, f)
    #         self.mansga[f].fitness = np.full((self._max_pop_size[f], self.objective_number), np.inf)
    #         # adding attributes to MaDEHB objects to allow communication across subpopulations
    #         self.mansga[f].parent_counter = 0
    #         self.mansga[f].promotion_pop = None
    #         self.mansga[f].promotion_pop_ids = None
    #         self.mansga[f].promotion_fitness = None

    #         # one buffer per fidelity
    #         self.mo_buffers = {
    #             f: {
    #                 "configs": [],
    #                 "fitness": [],
    #                 "config_ids": [],
    #             }
    #             for f in self.mansga.keys()
    #         }
    #         # buffer sizes for MaNSGA-II selection. full generational replacement
    #         self.mo_buffer_size = {
    #             f: self.mansga[f].pop_size
    #             for f in self.mansga.keys()
    #         }
    #         #inital "incumbent" score
    #     self.update_global_incumbent()

    def _init_subpop(self):
        """List of MaNSGA-II objects corresponding to the fidelities."""
        self.mansga = {}
        seeds = self.rng.integers(0, 2**32 - 1, size=len(self._max_pop_size))

        # --- create MaNSGA-II instances ---
        for (i, f), _seed in zip(enumerate(self._max_pop_size.keys()), seeds):
            pop_size = self._max_pop_size[f]

            self.mansga[f] = AsyncMa_NSGA_II(
                **self.mansga_params,
                pop_size=pop_size,
                config_repository=self.config_repository,
                seed=int(_seed)
            )

            self.mansga[f].population = self.mansga[f].init_population(pop_size=pop_size)
            self.mansga[f].population_ids = self.config_repository.announce_population(
                self.mansga[f].population, f
            )
            self.mansga[f].fitness = np.full((pop_size, self.objective_number), np.inf)

            # cross-subpopulation communication
            self.mansga[f].parent_counter = 0
            self.mansga[f].promotion_pop = None
            self.mansga[f].promotion_pop_ids = None
            self.mansga[f].promotion_fitness = None

        # buffers for selection
        self.mo_buffers = {
            f: {"configs": [], "fitness": [], "config_ids": []}
            for f in self.mansga.keys()
        }

        self.mo_buffer_size = {
            f: min(self.mo_selection_batch_size, self.mansga[f].pop_size)
            for f in self.mansga.keys()
        }

        self.update_global_incumbent()



    def _concat_pops(self, exclude_fidelity=None):
        """Concatenates all subpopulations."""
        fidelities = list(self.fidelities)
        if exclude_fidelity is not None:
            fidelities.remove(exclude_fidelity)
        pop = []
        for fidelity in fidelities:
            pop.extend(self.mansga[fidelity].population.tolist())
        return np.array(pop)

    def _start_new_bracket(self):
        """Starts a new bracket based on Hyperband."""
        # start new bracket
        self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, fidelities = self._get_next_iteration(self.iteration_counter)
        bracket = SHBracketManager(
            n_configs=n_configs, fidelities=fidelities, bracket_id=self.iteration_counter
        )
        self.active_brackets.append(bracket)
        self._bracket_rung_results[bracket.bracket_id] = {
            float(f): [] for f in fidelities
        }
        return bracket

    def _get_worker_count(self):
        if isinstance(self.client, Client):
            return len(self.client.ncores())
        else:
            return 1

    def _is_worker_available(self):
        """Checks if at least one worker is available to run a job."""
        if self.n_workers == 1 or self.client is None or not isinstance(self.client, Client):
            # in the synchronous case, one worker is always available
            return True
        workers = self._get_worker_count()  # len(self.client.ncores())
        if len(self.futures) >= workers:
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def _get_promotion_candidate(self, low_fidelity, high_fidelity, n_configs, bracket_id=None):
        """Select promotion candidates for the current Hyperband bracket.
        """
        key = (int(bracket_id), float(high_fidelity)) if bracket_id is not None else None
        if key is not None:
            state = self._promotion_state.setdefault(
                key, {"initialized": False, "queue": [], "promoted_ids": set()}
            )
            if not state["initialized"]:
                records = self._bracket_rung_results.get(int(bracket_id), {}).get(
                    float(low_fidelity), []
                )
                # Keep only candidates that have not already been promoted in this bracket.
                promoted = state["promoted_ids"]
                records = [r for r in records if int(r[2]) not in promoted]

                if records:
                    pop = np.asarray([r[0] for r in records])
                    fit = np.asarray([r[1] for r in records], dtype=float)
                    ids = np.asarray([r[2] for r in records], dtype=np.int64)
                    n_select = min(int(n_configs), len(records))
                    selected = self.mansga[low_fidelity].environmental_selection(
                        pop, fit, pop_size=n_select, update_points=True
                    )
                    for idx in selected.tolist():
                        state["queue"].append((pop[idx].copy(), int(ids[idx])))
                    state["initialized"] = True

            if state["queue"]:
                config, config_id = state["queue"].pop(0)
                state["promoted_ids"].add(int(config_id))
                return config, config_id

            records = self._bracket_rung_results.get(int(bracket_id), {}).get(
                float(low_fidelity), []
            )
            remaining = [r for r in records if int(r[2]) not in state["promoted_ids"]]
            if remaining:
                r = remaining[0]
                state["promoted_ids"].add(int(r[2]))
                return np.asarray(r[0]).copy(), int(r[2])

        evaluated = np.where(
            np.isfinite(self.mansga[low_fidelity].fitness).all(axis=1)
        )[0]
        if evaluated.size == 0:
            idx = self.rng.integers(self.mansga[low_fidelity].population.shape[0])
            return (
                self.mansga[low_fidelity].population[idx],
                self.mansga[low_fidelity].population_ids[idx],
            )

        pop = self.mansga[low_fidelity].population[evaluated]
        fit = self.mansga[low_fidelity].fitness[evaluated]
        ids = self.mansga[low_fidelity].population_ids[evaluated]
        selected = self.mansga[low_fidelity].environmental_selection(
            pop, fit, pop_size=min(int(n_configs), len(pop)), update_points=True
        )
        # Return the first selected candidate. Hyperband normally asks only
        # for one candidate per scheduling call; the bracket state caches the
        # full selection when bracket_id is available.
        idx = int(selected[0])
        return pop[idx], ids[idx]

    def _get_next_parent_for_subpop(self, fidelity):
        """Maintains a looping counter over a subpopulation, to iteratively select a parent."""
        parent_id = self.mansga[fidelity].parent_counter
        self.mansga[fidelity].parent_counter += 1
        self.mansga[fidelity].parent_counter = self.mansga[fidelity].parent_counter % self._max_pop_size[fidelity]
        return parent_id

    def _acquire_config(self, bracket, fidelity):
        """Generate a configuration to evaluate at the given fidelity."""

        # 1. Select parent from current fidelity population
        parent_id = self._get_next_parent_for_subpop(fidelity)
        target = self.mansga[fidelity].population[parent_id]

        # 2. Check if this rung requires promotions from lower fidelity
        lower_fidelity, num_configs = bracket.get_lower_fidelity_promotions(fidelity)

        if self.iteration_counter < self.max_SH_iter:
            # If not the first rung, promote configs from lower fidelity
            if fidelity != bracket.fidelities[0]:
                config, config_id = self._get_promotion_candidate(
                    lower_fidelity, fidelity, num_configs, bracket_id=bracket.bracket_id
                )
                return config, config_id, parent_id

        # 3. No promotion - generate a new offspring by variation only
        #    (selection is handled in tell())

        source_fidelity = lower_fidelity if lower_fidelity in self.mansga else fidelity
        evaluated = np.where(np.isfinite(self.mansga[source_fidelity].fitness).all(axis=1))[0]
        source_pop = self.mansga[source_fidelity].population[evaluated]
        source_fit = self.mansga[source_fidelity].fitness[evaluated]

        if num_configs is not None and len(source_pop) > int(num_configs) > 0:
            elite_idx = self.mansga[source_fidelity].environmental_selection(
                source_pop, source_fit, pop_size=int(num_configs), update_points=False,
            )
            mutation_pop = source_pop[elite_idx]
        else:
            mutation_pop = source_pop

        # Ensure minimum population size for mutation
        if len(mutation_pop) < self.mansga[fidelity]._min_pop_size:
            filler = self.mansga[fidelity]._min_pop_size - len(mutation_pop) + 1
            new_pop = self.mansga[fidelity]._init_mutant_population(
                pop_size=filler,
                population=self._concat_pops(),
                target=target,
                best=None,
            )
            mutation_pop = np.concatenate((mutation_pop, new_pop))

        # 4. Variation: mutation + crossover
        mutant = self.mansga[fidelity].mutation(
            current=target, best=None, alt_pop=mutation_pop
        )
        config = self.mansga[fidelity].crossover(target=target, mutant=mutant)
        config = self.mansga[fidelity].boundary_check(config)

        # 5. Announce new config
        config_id = self.config_repository.announce_config(config, fidelity)
        return config, config_id, parent_id


    def _get_next_bracket(self, only_id=False):
        """Used to retrieve what bracket the bracket for the next job.

        Optionally, a new bracket is started, if there are no more pending jobs or
        when all active brackets are waiting.

        Args:
            only_id (bool): Only returns the id of the next bracket

        Returns:
            SHBracketmanager or int: bracket or bracket ID of next job
        """
        bracket = None
        start_new_bracket = False
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            start_new_bracket = True
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow MaDEHB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                start_new_bracket = True

        if only_id:
            return self.iteration_counter + 1 if start_new_bracket else bracket.bracket_id

        return self._start_new_bracket() if start_new_bracket else bracket

    def _get_next_job(self):
        """Loads a configuration and fidelity to be evaluated next.

        Returns:
            dict: Dicitonary containing all necessary information of the next job.
        """
        bracket = self._get_next_bracket()
        # fidelity that the SH bracket allots
        fidelity = bracket.get_next_job_fidelity()
        config, config_id, parent_id = self._acquire_config(bracket, fidelity)

        # transform config to proper representation
        if self.use_configspace:
            # converts [0, 1] vector to a ConfigSpace object
            config = self.mansga[fidelity].vector_to_configspace(config)

        # notifies the Bracket Manager that a single config is to run for the fidelity chosen
        job_info = {
            "config": config,
            "config_id": config_id,
            "fidelity": fidelity,
            "parent_id": parent_id,
            "bracket_id": bracket.bracket_id,
        }

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info["bracket_id"]:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info["fidelity"])
                break
        return job_info

    def ask(self, n_configs: int=1) -> Union[dict, List[dict]]:
        """Get the next configuration to run from the optimizer.

        The retrieved configuration can then be evaluated by the user.
        After evaluation use `tell` to report the results back to the optimizer.
        For more information, please refer to the description of `tell`.

        Args:
            n_configs (int, optional): Number of configs to ask for. Defaults to 1.

        Returns:
            dict or list of dict: Job info(s) of next configuration to evaluate.
        """
        jobs = []
        if n_configs == 1:
            jobs = self._get_next_job()
            self._ask_counter += 1
        else:
            for _ in range(n_configs):
                jobs.append(self._get_next_job())
                self._ask_counter += 1

        return jobs

    def _get_gpu_id_with_low_load(self):
        candidates = []
        for k, v in self.gpu_usage.items():
            if v == min(self.gpu_usage.values()):
                candidates.append(k)
        device_id = self.rng.choice(candidates)
        # creating string for setting environment variable CUDA_VISIBLE_DEVICES
        gpu_ids = self._create_cuda_visible_devices(
            self.available_gpus, device_id,
        )
        # updating GPU usage
        self.gpu_usage[device_id] += 1
        self.logger.debug(f"GPU device selected: {device_id}")
        self.logger.debug(f"GPU device usage: {self.gpu_usage}")
        return gpu_ids

    def _submit_job(self, job_info, **kwargs):
        """Asks a free worker to run the objective function on config and fidelity."""
        job_info["kwargs"] = self.shared_data if self.shared_data is not None else kwargs
        # submit to Dask client
        if self.n_workers > 1 or isinstance(self.client, Client):
            if self.single_node_with_gpus:
                # managing GPU allocation for the job to be submitted
                job_info.update({"gpu_devices": self._get_gpu_id_with_low_load()})
            self.futures.append(
                self.client.submit(self._f_objective, job_info)
            )
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

    def _fetch_results_from_workers(self):
        """Iterate over futures and collect results from finished workers."""
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                f"Collecting {len(done_list)} of the {len(self.futures)} job(s) active.",
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    # updating GPU usage
                    self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug("GPU device released: {}".format(run_info["device_id"]))
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # tell result
            self.tell(run_info["job_info"], run_info["result"])
        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()

    def _adjust_budgets(self, fevals=None, brackets=None):
        # only update budgets if it is not the first run
        if fevals is not None and len(self.traj) > 0:
            #fevals = len(self.traj) + fevals
            fevals = self.current_fevals + fevals
        elif brackets is not None and self.iteration_counter > -1:
            brackets = self.iteration_counter + brackets + 1

        return fevals, brackets

    def _get_state(self):
        state = {}
        # MaNSGA-II parameters
        serializable_mansga_params = self.mansga_params.copy()
        serializable_mansga_params.pop("cs", None)
        serializable_mansga_params.pop("rng", None)
        serializable_mansga_params.pop("f", None)
        serializable_mansga_params["output_path"] = str(serializable_mansga_params["output_path"])
        state["MaNSGA_params"] = serializable_mansga_params
        # Hyperband variables
        hb_dict = {}
        hb_dict["min_fidelity"] = self.min_fidelity
        hb_dict["max_fidelity"] = self.max_fidelity
        hb_dict["min_clip"] = self.min_clip
        hb_dict["max_clip"] = self.max_clip
        hb_dict["eta"] = self.eta
        state["HB_params"] = hb_dict
        # Save DEHB interals
        dehb_internals = {}
        dehb_internals["initial_configs"] = self.config_repository.get_serialized_initial_configs()
        state["internals"] = dehb_internals
        return state

    def _to_serializable(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
        
    def _save_state(self):
        state = self._get_state()
        try:
            state_path = self.output_path / "dehb_state.json"
            with state_path.open("w") as f:
                json.dump(state, f, indent=2, default=self._to_serializable)
        except Exception as e:
            self.logger.warning(f"State not saved: {e!r}")



    def _is_run_budget_exhausted(self, fevals=None, brackets=None):
        """Checks if the DEHB run should be terminated or continued."""
        if fevals is not None:
            #TODO make sure its fine
            #if len(self.traj) >= fevals:
            if int(self.current_fevals) >= fevals:
                return True
        elif brackets is not None:
            future_iteration_counter = self._get_next_bracket(only_id=True)
            if future_iteration_counter >= brackets:
                for bracket in self.active_brackets:
                    # waits for all brackets < iteration_counter to finish by collecting results
                    if bracket.bracket_id < future_iteration_counter and \
                            not bracket.is_bracket_done():
                        return False
                return True
        else:
            return self._time_budget_exhausted
        return False

    #TODO Experimental. needs testing.
    def _save_incumbent(self):
        if self.inc_configs is None:
            return

        def to_json(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.floating, np.bool_)):
                return x.item()
            if isinstance(x, (list, tuple)):
                return [to_json(i) for i in x]
            if isinstance(x, dict):
                return {k: to_json(v) for k, v in x.items()}
            return x

        try:
            # Normalize configs into a list
            inc = self.inc_configs
            if isinstance(inc, np.ndarray):
                configs_list = inc if inc.ndim > 1 else [inc]
            else:
                configs_list = list(inc)

            # Convert configs
            if self.use_configspace:
                configs_json = [dict(self.vector_to_configspace(v)) for v in configs_list]
            else:
                configs_json = [np.asarray(v).tolist() for v in configs_list]

            res = {
                "configs": configs_json,
                "score": to_json(self.inc_score),
                "info": to_json(getattr(self, "inc_info", None)),
                "timestamp": int(time.time()),
            }

            with (self.output_path / "incumbent.json").open("w") as f:
                json.dump(res, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Incumbent not saved: {e!r}")


    def _save_history(self, name="madehb_history.parquet.gzip"):
        # Return early if there is no history yet
        if self.history is None:
            return
        try:
            history_path = self.output_path / name
            history_df = pd.DataFrame(self.history, columns=["config_id", "config", "fitness",
                                                             "cost", "fidelity", "info"])
            # Check if the 'info' column is empty or contains only None values
            if history_df["info"].apply(lambda x: (isinstance(x, dict) and len(x) == 0)).all():
                # Drop the 'info' column
                history_df = history_df.drop(columns=["info"])
            history_df.to_parquet(history_path, compression="gzip")
        except Exception as e:
            self.logger.warning(f"History not saved: {e!r}")

    def _log_debug(self):
        for bracket in self.active_brackets:
            self.logger.debug(f"Bracket ID {bracket.bracket_id}:\n{bracket!s}")

    def _log_runtime(self, fevals, brackets, total_cost):
        if fevals is not None:
            remaining = (int(self.current_fevals), fevals, "function evaluation(s) done")
        elif brackets is not None:
            _suffix = f"bracket(s) started; # active brackets: {len(self.active_brackets)}"
            remaining = (self.iteration_counter + 1, brackets, _suffix)
        else:
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_cost, "seconds elapsed")
        self.logger.info(
            f"{remaining[0]}/{remaining[1]} {remaining[2]}",
        )

    def _log_job_submission(self, job_info: dict):
        fidelity = job_info["fidelity"]
        config_id = job_info["config_id"]
        self.logger.info(
            "Evaluating configuration {} with fidelity {} under "
            "bracket ID {}".format(config_id, fidelity, job_info["bracket_id"]),
        )
        self.logger.info(
            f"Best scores seen/Incumbent scores: {self.inc_score}",
        )

    def _load_checkpoint(self, run_dir: str):
        # Check if path exists, otherwise give warning
        run_dir = Path(run_dir)
        if not Path.exists(run_dir):
            self.logger.warning("Path to run directory does not exist.")
            return False
        # Load dehb state
        dehb_state_path = run_dir / "dehb_state.json"
        with dehb_state_path.open() as f:
            dehb_state = json.load(f)
        # Convert output_path of checkpoint to Path
        dehb_state["MaNSGA_params"]["output_path"] = Path(dehb_state["MaNSGA_params"]["output_path"])
        if not all(dehb_state["MaNSGA_params"][key] == self.mansga_params[key]
                   for key in dehb_state["MaNSGA_params"]):
            self.logger.warning("Initialized MaNSGA parameters do not match saved parameters.")
            return False
        self.mansga_params.update(dehb_state["MaNSGA_params"])

        hb_vars = dehb_state["HB_params"]
        if self.min_fidelity != hb_vars["min_fidelity"]:
            self.logger.warning("Initialized min_fidelity does not match saved parameters.")
            return False
        self.min_fidelity = hb_vars["min_fidelity"]

        if self.max_fidelity != hb_vars["max_fidelity"]:
            self.logger.warning("Initialized max_fidelity does not match saved parameters.")
            return False
        self.max_fidelity = hb_vars["max_fidelity"]

        if self.min_clip != hb_vars["min_clip"]:
            self.logger.warning("Initialized min_clip does not match saved parameters.")
            return False
        self.min_clip = hb_vars["min_clip"]

        if self.max_clip != hb_vars["max_clip"]:
            self.logger.warning("Initialized max_clip does not match saved parameters.")
            return False
        self.max_clip = hb_vars["max_clip"]

        if self.eta != hb_vars["eta"]:
            self.logger.warning("Initialized eta does not match saved parameters.")
            return False
        self.eta = hb_vars["eta"]

        # Load history
        history_path = run_dir / "madehb_history.parquet.gzip"
        history = pd.read_parquet(history_path)

        # Replay history
        for _, row in history.iterrows():
            job_info = {
                "fidelity": row["fidelity"],
                "config_id": row["config_id"],
                "config": np.array(row["config"]),
            }
            result = {
                "fitness": row["fitness"],
                "cost": row["cost"],
                "info": row.get("info", {}),
            }

            self.tell(job_info, result, replay=True)
        # Clean inactive brackets
        self._clean_inactive_brackets()
        return True

    def save(self):
        """Saves the current incumbent, history and state to disk."""
        self.logger.info("Saving state to disk...")
        if self._time_budget_exhausted:
            self.logger.info("Runtime budget exhausted. Resorting to only saving overtime history.")
            self._save_history(name="overtime_history.parquet.gzip")
        else:
            self._save_incumbent()
            self._save_history()
            self._save_state()

    def _flush_mo_buffer(self, fidelity):
        """Assimilate any pending evaluations for a fidelity via MaNSGA-II."""
        if fidelity not in self.mo_buffers or not self.mo_buffers[fidelity]["configs"]:
            return
        buf = self.mo_buffers[fidelity]
        comb_pop = np.vstack((self.mansga[fidelity].population, np.asarray(buf["configs"])))
        comb_fit = np.vstack((self.mansga[fidelity].fitness, np.asarray(buf["fitness"], dtype=float)))
        comb_ids = np.hstack((self.mansga[fidelity].population_ids, np.asarray(buf["config_ids"])))
        uniq_pop, idx = np.unique(comb_pop, axis=0, return_index=True)
        if len(idx) >= self.mansga[fidelity].pop_size:
            comb_pop, comb_fit, comb_ids = uniq_pop, comb_fit[idx], comb_ids[idx]
        selected = self.mansga[fidelity].environmental_selection(
            comb_pop, comb_fit, pop_size=self.mansga[fidelity].pop_size, update_points=True
        )
        self.mansga[fidelity].population = comb_pop[selected]
        self.mansga[fidelity].fitness = comb_fit[selected]
        self.mansga[fidelity].population_ids = comb_ids[selected]
        self.mo_buffers[fidelity] = {"configs": [], "fitness": [], "config_ids": []}

    def update_global_incumbent(self):
        """Update the Pareto archive, prioritizing all full-fidelity evaluations."""
        if self.full_budget_fitness:
            fit_f = np.asarray(self.full_budget_fitness, dtype=float)
            pop_f = np.asarray(self.full_budget_configs)
            fronts = self.mansga[self.max_fidelity].fast_nondominated_sort(fit_f)
            first_front = np.asarray(fronts[0], dtype=int)
            self.inc_configs = pop_f[first_front].copy()
            self.inc_score = fit_f[first_front].copy()
            infos = []
            for idx in first_front:
                try:
                    cid = int(self.full_budget_ids[int(idx)])
                    infos.append(self.config_repository.configs[cid].results[self.max_fidelity].info)
                except Exception:
                    infos.append({})
            self.inc_info = infos
            return

        for f in reversed(self.fidelities):
            pop = self.mansga[f].population
            fit = self.mansga[f].fitness
            ids = self.mansga[f].population_ids
            if pop is None or len(pop) == 0:
                continue
            mask = np.isfinite(fit).all(axis=1)
            if not np.any(mask):
                continue
            pop_f = pop[mask]
            fit_f = fit[mask]
            ids_f = ids[mask]
            fronts = self.mansga[f].fast_nondominated_sort(fit_f)
            first_front = np.asarray(fronts[0], dtype=int)
            self.inc_configs = pop_f[first_front].copy()
            self.inc_score = fit_f[first_front].copy()
            infos = []
            for cid in ids_f[first_front]:
                try:
                    infos.append(self.config_repository.configs[int(cid)].results[f].info)
                except Exception:
                    infos.append({})
            self.inc_info = infos
            return

        self.inc_configs = []
        self.inc_score = []
        self.inc_info = []


    def tell(self, job_info: dict, result: dict, replay: bool = False) -> None:
        """Feed a result back to the optimizer (many-objective version).

        In order to correctly interpret the results, the `job_info` dict, retrieved by `ask`,
        has to be given. Moreover, the `result` dict has to contain the keys `fitness` and `cost`.
        `fitness` resembles the objectives you are trying to optimize, e.g. validation losses.
        `cost` resembles the computational cost for computing the result, e.g. the wallclock time
        for training and validating a neural network to achieve the validation losses specified in
        `fitness`. It is also possible to add the field `info` to the `result` in order to store
        additional, user-specific information.

        Modified to work in many-objective cases:
        - Maintain a per-fidelity buffer of offspring.
        - Once the buffer for a fidelity reaches a batch size (typically pop_size),
        perform MaNSGA-II selection on parents + offspring.
        """
        if replay:
            self.config_repository.restore_config(
                job_info["config_id"], job_info["config"], job_info["fidelity"]
            )
            job_info = {
                "fidelity": job_info["fidelity"],
                "config": job_info["config"],
                "config_id": job_info["config_id"],
                "parent_id": -1,
                "bracket_id": -1,
            }
            self._ask_counter += 1

        if self._tell_counter >= self._ask_counter:
            raise NotImplementedError(
                "Called tell() more often than ask(). "
                "Warmstarting with tell is not supported."
            )
        self._tell_counter += 1

        # Preserve user-provided information and augment it with elapsed time.
        # During checkpoint replay ``start`` is not set yet.
        elapsed = 0.0 if self.start is None else time.time() - self.start
        result_info = result.get("info", {})
        result["info"] = dict(result_info) if isinstance(result_info, dict) else {}
        result["info"]["elapsed_time"] = float(elapsed)

        # unpack result and job info
        fitness = np.array(result["fitness"], dtype=float)
        cost = float(result["cost"])
        info = result["info"] if "info" in result else {}

        fidelity = job_info["fidelity"]
        parent_id = job_info["parent_id"]
        config = job_info["config"]
        config_id = job_info["config_id"]
        bracket_id = job_info["bracket_id"]

        # update bracket information (synchronous SH bookkeeping)
        for bracket in self.active_brackets:
            if bracket.bracket_id == bracket_id:
                bracket.complete_job(fidelity)
                break

        # store result in config repository
        self.config_repository.tell_result(config_id, fidelity, fitness, cost, info)

        self.current_fevals += float(fidelity) / float(self.max_fidelity)

        if self.use_configspace:
            config = np.asarray(self.config_repository.get(config_id), dtype=float).copy()
        else:
            config = np.asarray(config, dtype=float).copy()

        if bracket_id >= 0:
            rung = self._bracket_rung_results.setdefault(int(bracket_id), {})
            rung.setdefault(float(fidelity), []).append(
                (config.copy(), fitness.copy(), int(config_id))
            )
        #print("tell(): writing result for config_id =", config_id)

        #TODO Experimental
        if fidelity == self.max_fidelity:
            self.full_budget_configs.append(np.asarray(config).copy())
            self.full_budget_fitness.append(fitness.copy())
            self.full_budget_ids.append(int(config_id))
            # The final archive should contain every evaluated full-budget
            # candidate, not just those that happened to survive the latest
            # MaNSGA-II population replacement.
            self.update_global_incumbent()

        # MaNSGA-II: per-fidelity result buffer 
        if fidelity not in self.mo_buffers:
            self.mo_buffers[fidelity] = {
                "configs": [],
                "fitness": [],
                "config_ids": [],
            }
            #full generational replacement
            if fidelity not in self.mo_buffer_size:
                self.mo_buffer_size[fidelity] = min(
                    self.mo_selection_batch_size, self.mansga[fidelity].pop_size
                )

        buf = self.mo_buffers[fidelity]
        buf["configs"].append(config)
        buf["fitness"].append(fitness)
        buf["config_ids"].append(config_id)

        # temporary incumbent before selection
        if self.inc_configs is None or len(self.inc_configs) == 0:
            self.inc_configs = [config]
            self.inc_score  = [fitness]
            self.inc_info   = [info]

        elif len(self.inc_configs) == 1:
            inc = self.inc_score[0]
            if np.all(fitness <= inc) and np.any(fitness < inc):
                self.inc_configs = [config]
                self.inc_score  = [fitness]
                self.inc_info   = [info]

        # when all the offspring is accumulated, perform MaNSGA-II selection
        if len(buf["configs"]) >= self.mo_buffer_size[fidelity]:
            offspring_pop = np.array(buf["configs"])
            offspring_fit = np.array(buf["fitness"])
            offspring_ids = np.array(buf["config_ids"])

            # combine parents + offspring
            comb_pop = np.vstack((self.mansga[fidelity].population, offspring_pop))
            comb_fitness = np.vstack((self.mansga[fidelity].fitness, offspring_fit))
            comb_ids = np.hstack((self.mansga[fidelity].population_ids, offspring_ids))

            uniq_pop, idx = np.unique(comb_pop, axis=0, return_index=True)
            if len(idx) >= self.mansga[fidelity].pop_size:
                comb_pop = uniq_pop
                comb_fitness = comb_fitness[idx]
                comb_ids = comb_ids[idx]

            selected_indices = self.mansga[fidelity].environmental_selection(
                comb_pop,
                comb_fitness,
                pop_size=self.mansga[fidelity].pop_size,
                update_points=True,
            )

            self.mansga[fidelity].population = comb_pop[selected_indices]
            self.mansga[fidelity].fitness = comb_fitness[selected_indices]
            self.mansga[fidelity].population_ids = comb_ids[selected_indices]

            # clear buffer for this fidelity
            self.mo_buffers[fidelity] = {
                "configs": [],
                "fitness": [],
                "config_ids": [],
            }
            self.update_global_incumbent()
        # incumbents test implementation
        # book-keeping
        self._update_trackers(
            traj=self.inc_score,
            runtime=cost,
            history=(
                config_id,
                config.tolist(),
                fitness,
                float(cost),
                float(fidelity),
                info,
            ),
        )

        if self.save_freq == "step" and not replay:
            self.save()


    @logger.catch(reraise=True)
    def run(self, fevals=None, brackets=None, total_cost=None, single_node_with_gpus=False,
            **kwargs) -> Tuple[np.array, np.array, np.array]:
        """Main interface to run optimization by DEHB.

        This function waits on workers and if a worker is free, asks for a configuration and a
        fidelity to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low): <br>
        1) Number of function evaluations (fevals) <br>
        2) Number of Successive Halving brackets run under Hyperband (brackets) <br>
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)

        !!! note "Using `tell` under the hood."

            Please note, that `run` uses `tell` under the hood, therefore please have a
            look at the documentation of `tell` for more information e.g. about the result format.

        !!! note "Adjusting verbosity"

            The verbosity of DEHB logs can be adjusted via adding the `log_level` parameter to DEHBs
            initialization. As we use loguru, the logging levels can be found on [their website](https://loguru.readthedocs.io/en/stable/api/logger.html#levels).

        Args:
            fevals (int, optional): Number of functions evaluations to run. Defaults to None.
            brackets (int, optional): Number of brackets to run. Defaults to None.
            total_cost (int, optional): Wallclock budget in seconds. Defaults to None.
            single_node_with_gpus (bool): Workers get assigned different GPUs. Default to False.

        Returns:
            Trajectory, runtime and optimization history.
        """
        # Warn if users use old state saving frequencies
        if "save_history" in kwargs or "save_intermediate" in kwargs or "name" in kwargs:
            logger.warning("The run parameters 'save_history', 'save_intermediate' and 'name' are "\
                           "deprecated, since the changes in v0.1.1. Please use the 'saving_freq' "\
                           "parameter in the constructor to adjust when to save DEHBs state " \
                           "(including history). Please use the 'output_path' parameter to adjust "\
                           "where the state and logs should be saved.")
            raise TypeError("Used deprecated parameters 'save_history', 'save_intermediate' " \
                            "and/or 'name'. Please check the logs for more information.")
        if "verbose" in kwargs:
            logger.warning("The run parameters 'verbose' is deprecated since the changes in v0.1.2. "\
                           "Please use the 'log_level' parameter when initializing DEHB.")
            raise TypeError("Used deprecated parameter 'verbose'. "\
                            "Please check the logs for more information.")
        # check if run has already been called before
        if self.start is not None:
            logger.warning("DEHB has already been run. Calling 'run' twice could lead to unintended"
                           + " behavior. Please restart DEHB with an increased compute budget"
                           + " instead of calling 'run' twice.")
            self._time_budget_exhausted = False

        # checks if a Dask client exists
        if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
            self.shared_data = self.client.scatter(kwargs, broadcast=True)

        self.single_node_with_gpus = single_node_with_gpus
        if self.single_node_with_gpus:
            self._distribute_gpus()

        self.start = self.start = time.time()
        self.logger.info("\nLogging at {} for optimization starting at {}\n".format(
            Path.cwd() / self.log_filename,
            time.strftime("%x %X %Z", time.localtime(self.start)),
        ))

        delimiters = [fevals, brackets, total_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
        fevals, brackets = self._adjust_budgets(fevals, brackets)
        # Set alarm for specified runtime budget
        if total_cost is not None:
            self._runtime_budget_timer = Timer(total_cost, self._timeout_handler)
            self._runtime_budget_timer.start()
        while True:
            if self._is_run_budget_exhausted(fevals, brackets):
                break
            if self._is_worker_available():
                next_bracket_id = self._get_next_bracket(only_id=True)
                if brackets is not None and next_bracket_id >= brackets:
                    pass
                else:
                    if self.n_workers > 1 or isinstance(self.client, Client):
                        self.logger.debug("{}/{} worker(s) available.".format(
                            self._get_worker_count() - len(self.futures), self._get_worker_count(),
                        ))
                    # Ask for new job_info
                    job_info = self.ask()
                    # Submit job_info to a worker for execution
                    self._submit_job(job_info, **kwargs)
                    self._log_runtime(fevals, brackets, total_cost)
                    self._log_job_submission(job_info)
                    self._log_debug()
            self._fetch_results_from_workers()
            self._clean_inactive_brackets()
        for _fidelity in self.mansga:
            self._flush_mo_buffer(_fidelity)
        self.update_global_incumbent()
        time_taken = time.time() - self.start
        self.logger.info("End of optimisation! Total duration: {}; Total fevals: {}\n".format(
            time_taken, len(self.traj),
        ))
        # Multi-objective incumbent logging (Pareto front)
        self.logger.info("Pareto-front incumbent scores:")
        for i, score in enumerate(self.inc_score):
            self.logger.info(f"  [{i}] {score}")

        self.logger.info("Pareto-front incumbent configs:")
        if self.use_configspace:
            for i, cfg in enumerate(self.inc_configs):
                cs_cfg = self.vector_to_configspace(cfg)
                self.logger.info(f"  [{i}]")
                for k, v in dict(cs_cfg).items():
                    self.logger.info(f"    {k}: {v}")
        else:
            for i, cfg in enumerate(self.inc_configs):
                self.logger.info(f"  [{i}] {cfg}")

        self.save()
        # cancel timer
        if self._runtime_budget_timer:
            self._runtime_budget_timer.cancel()
        # reset waiting jobs of active bracket to allow for continuation
        self.active_brackets = []
        if len(self.active_brackets) > 0:
            for active_bracket in self.active_brackets:
                active_bracket.reset_waiting_jobs()
        return [self.full_budget_configs,self.full_budget_fitness], self.traj, np.array(self.runtime), np.array(self.history, dtype=object), time.time()-self.start


