import os
from pathlib import Path
from typing import List

import ConfigSpace
import ConfigSpace.util
import numpy as np
"""Dask is optional for serial optimization and external ask/tell workers."""
try:
    from distributed import Client
except (ImportError, OSError) as exc:
    _dask_import_error = str(exc)

    class Client:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Dask is unavailable in this environment: " + _dask_import_error
                + ". Use n_workers=1 or distribute ask()/tell() externally. "
            )

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ..utils import ConfigRepository


class MaNSGA_II_base():
    '''Base class for adapted MaNSGA-II
    '''
    def __init__(self, cs=None, objective_number=1,trade_off_param=None, f=None, dimensions=None, pop_size=None, max_age=None,
                 mutation_factor=None, crossover_prob=None, strategy=None, normalize_objective_space=True,
                 boundary_fix_type='random', config_repository=None, seed=None, **kwargs):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)

        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(list(self.cs.values()))
        else:
            self.dimensions = dimensions

        # MaNSGA-II related variables
        self.pop_size = pop_size
        self.max_age = max_age
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
        self.alpha = trade_off_param*objective_number /(1 - trade_off_param + (trade_off_param*objective_number))
        self.objective_number = objective_number
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.fix_type = boundary_fix_type
        self.normalize_objective_space = normalize_objective_space

        self.ideal_point = np.full(self.objective_number, np.inf)
        self.nadir_point = np.full(self.objective_number, -np.inf)

        # Miscellaneous
        self.configspace = True if isinstance(self.cs, ConfigSpace.ConfigurationSpace) else False
        self.hps = dict()
        if self.configspace:
            self.cs.seed(self._original_seed)
            for i, hp in enumerate(list(cs.values())):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i
        self.output_path = Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        self.output_path.mkdir(parents=True, exist_ok=True)

        if config_repository:
            self.config_repository = config_repository
        else:
            self.config_repository = ConfigRepository()

        # Global trackers
        # There is no best configuration in MO context
            #self.inc_score : float
            #self.inc_config : np.ndarray[float]
            #self.inc_id : int

        self.population : np.ndarray[np.ndarray[float]]
        self.population_ids :np.ndarray[int]
        self.ideal_point: np.ndarray[float]
        #2D array that saves fitness for every objective for every solution
        self.fitness : np.ndarray[np.ndarray[float]]
        self.age : int
        #TODO maybe disable the history?
        self.history : list[object]
        self.reset()

    def reset(self, *, reset_seeds: bool = True):
        #self.inc_score = np.inf
        #self.inc_config = None
        #self.inc_id = -1
        self.population = None
        self.population_ids = None
        self.fitness = None
        self.unmodified_fitness = None
        self.age = None

        if reset_seeds:
            if isinstance(self.cs, ConfigSpace.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)

        self.history = []

    #TODO Figure out what to do with the _shiffe_pop fucntion. May guess is that it is irrelevant for MaNSGA-II
    # def _shuffle_pop(self):
    #     pop_order = np.arange(len(self.population))
    #     self.rng.shuffle(pop_order)
    #     self.population = self.population[pop_order]
    #     self.fitness = self.fitness[pop_order]
    #     self.age = self.age[pop_order]

    #DE sorting replaced by fast nondominated sorting to account for MO
    # def _sort_pop(self):
    #     pop_order = np.argsort(self.fitness)
    #     self.rng.shuffle(pop_order)
    #     self.population = self.population[pop_order]
    #     self.fitness = self.fitness[pop_order]
    #     self.age = self.age[pop_order]

    def dominates(self, a, b) -> bool:
        """Pareto dominance: a dominates b."""
        return np.all(a <= b) and np.any(a < b)
    
    def fast_nondominated_sort(self,F: np.ndarray[np.ndarray[float]]):
        nds = NonDominatedSorting()  # minimization
        fronts = nds.do(F, return_rank=False)
        return [list(front) for front in fronts]
    
    def crowding_distance(self, F):
        F = np.array(F)
        N, M = F.shape
        dist = np.zeros(N)

        for m in range(M):
            idx = np.argsort(F[:, m])
            dist[idx[0]] = dist[idx[-1]] = np.inf

            fmin, fmax = F[idx[0], m], F[idx[-1], m]
            denom = fmax - fmin

            # safe denominator
            if abs(denom) < 1e-12:
                continue

            for i in range(1, N - 1):
                dist[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / denom

        return dist

    
    def distance_selection(self, first_F, front_indices, ideal_point, pop_size):
        N, M = first_F.shape

        if N <= pop_size:
            return list(front_indices)

        # # compute nadir
        # nadir_p = np.max(first_F, axis=0)
        # den = nadir_p - ideal_point

        # compute nadir
        nadir_p = np.max(first_F, axis=0)
        den = nadir_p - ideal_point

        if np.allclose(den, 0):
            # fully degenerate front
            normalized_F = first_F - ideal_point
        else:
            # per-dimension safe normalization
            safe_den = np.where(np.abs(den) < 1e-12, 1.0, den)
            normalized_F = (first_F - ideal_point) / safe_den


        # avoid division by zero in projection
        row_sums = np.sum(normalized_F, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        projected = normalized_F / row_sums

        # distance to ideal point
        dist_to_ideal = np.linalg.norm(normalized_F, axis=1)
        first_idx = np.argmin(dist_to_ideal)

        selected = [first_idx]
        v = np.zeros(N, dtype=bool)
        v[first_idx] = True

        # initialize distances to nearest selected point
        d = np.full(N, np.inf)
        for k in range(N):
            if not v[k]:
                d[k] = np.linalg.norm(projected[k] - projected[first_idx])

        # greedy selection
        while len(selected) < pop_size:
            remaining = np.where(~v)[0]
            j = remaining[np.argmax(d[remaining])]

            selected.append(j)
            v[j] = True
            d[j] = np.inf

            for k in remaining:
                if not v[k]:
                    dist = np.linalg.norm(projected[k] - projected[j])
                    d[k] = min(d[k], dist)

        return [front_indices[i] for i in selected]


    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def init_population(self, pop_size: int) -> List:
        if self.configspace:
            # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
            population = self.cs.sample_configuration(size=int(pop_size))
            if not isinstance(population, List):
                population = [population]
            # the population is maintained in a list-of-vector form where each ConfigSpace
            # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            # if no ConfigSpace representation available, uniformly sample from [0, 1]
            population = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))

        return np.array(population)

    def sample_population(self, size: int = 3, alt_pop: List = None) -> List:
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population (alt_pop)
        '''
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = self.rng.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]

    def boundary_check(self, vector: np.ndarray) -> np.ndarray:
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        if fix_type == 'random', the values are replaced with a random sampling from (0,1)
        if fix_type == 'clip', the values are clipped to the closest limit from {0, 1}

        Parameters
        ----------
        vector : array

        Returns
        -------
        array
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.fix_type == 'random':
            vector[violations] = self.rng.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    # Required by MaNSGA-II to reduce the effect of dominance-resistant configurations
    def modify_objective_values(self, fitness : np.array) -> np.array:
        norm_fitness = np.zeros_like(fitness, dtype=float)

        if self.normalize_objective_space:
            denom = self.nadir_point - self.ideal_point
            # eps = 1e-6 only when division by 0 would occur
            eps = np.where(np.isclose(denom, 0.0), 1e-6, 0.0)
            norm_fitness = (fitness - self.ideal_point) / (denom + eps)
        else:
            norm_fitness = fitness

        avg = np.average(norm_fitness)
        #apply smoothing of all objectives
        return (1 - self.alpha) * norm_fitness + self.alpha * avg


    def modify_objective_values_batch(self, fitness: np.ndarray) -> np.ndarray:
        """Vectorized MaNSGA-II objective modification.

        The transformation is applied to all candidates using the same
        ideal/nadir estimates, avoiding tiny candidate-dependent numerical
        differences from repeatedly calling the scalar implementation.
        """
        fitness = np.asarray(fitness, dtype=float)
        if fitness.ndim == 1:
            fitness = fitness.reshape(1, -1)

        if self.normalize_objective_space:
            denom = np.asarray(self.nadir_point, dtype=float) - np.asarray(self.ideal_point, dtype=float)
            safe_den = np.where(np.abs(denom) > 1e-12, denom, 1.0)
            norm_fitness = (fitness - self.ideal_point) / safe_den
        else:
            norm_fitness = fitness.copy()

        avg = np.mean(norm_fitness, axis=1, keepdims=True)
        return (1.0 - self.alpha) * norm_fitness + self.alpha * avg

    def environmental_selection(self, population, fitness, pop_size=None, update_points=True):

        population = np.asarray(population)
        fitness = np.asarray(fitness, dtype=float)
        if fitness.ndim == 1:
            fitness = fitness.reshape(1, -1)
        if pop_size is None:
            pop_size = self.pop_size
        pop_size = int(pop_size)
        n_total = len(population)

        finite_mask = np.isfinite(fitness).all(axis=1)
        valid_indices = np.flatnonzero(finite_mask)
        invalid_indices = np.flatnonzero(~finite_mask)

        if valid_indices.size == 0:

            return np.arange(min(pop_size, n_total))

        valid_fitness = fitness[valid_indices]
        target_size = min(pop_size, len(valid_indices))

        if update_points:
            self.ideal_point = np.minimum(self.ideal_point, valid_fitness.min(axis=0))
            self.nadir_point = valid_fitness.max(axis=0)

        mod_fitness = self.modify_objective_values_batch(valid_fitness)
        fronts = self.fast_nondominated_sort(mod_fitness)

        cumulative = np.cumsum([len(front) for front in fronts])
        k = int(np.searchsorted(cumulative, target_size, side="left"))

        if k > 0:
            selected_local = []
            for front_idx in range(k):
                selected_local.extend(fronts[front_idx])
            remaining = target_size - len(selected_local)
            if remaining > 0:
                partial_front = fronts[k]
                crowd = self.crowding_distance(mod_fitness[partial_front])
                order = np.argsort(-crowd, kind="mergesort")[:remaining]
                selected_local.extend(np.asarray(partial_front)[order].tolist())
        elif len(fronts[0]) == target_size:
            selected_local = list(fronts[0])
        else:
            first_front = fronts[0]
            selected_local = self.distance_selection(
                first_F=valid_fitness[first_front],
                front_indices=first_front,
                ideal_point=self.ideal_point,
                pop_size=target_size,
            )

        selected = valid_indices[np.asarray(selected_local, dtype=int)]

        shortfall = min(pop_size, n_total) - len(selected)
        if shortfall > 0 and invalid_indices.size > 0:
            pad = invalid_indices[:shortfall]
            selected = np.concatenate([selected, pad])

        return selected

    #TODO ensure it is valid. No loss of precision due to float arithmetics?
    def vector_to_configspace(self, vector: np.ndarray) -> ConfigSpace.Configuration:
        '''Converts numpy array to ConfigSpace object

        Works when self.cs is a ConfigSpace object and the input vector is in the domain [0, 1].
        '''
        # creates a ConfigSpace object dict with all hyperparameters present, the inactive too
        new_config = dict(ConfigSpace.util.impute_inactive_values(
            self.cs.get_default_configuration()
        ))
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(list(self.cs.values())):
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.sequence))
                param_value = hyper.sequence[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                ranges = np.arange(start=0, stop=1, step=1/len(hyper.choices))
                param_value = hyper.choices[np.where((vector[i] < ranges) == False)[0][-1]]
            elif type(hyper) == ConfigSpace.Constant:
                param_value = hyper.default_value
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if type(hyper) == ConfigSpace.UniformIntegerHyperparameter:
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        new_config = ConfigSpace.util.deactivate_inactive_hyperparameters(
            configuration = new_config, configuration_space=self.cs
        )
        return new_config

    def configspace_to_vector(self, config: ConfigSpace.Configuration) -> np.ndarray:
        '''Converts ConfigSpace object to numpy array scaled to [0,1]

        Works when self.cs is a ConfigSpace object and the input config is a ConfigSpace object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        '''
        # the imputation replaces illegal parameter values with their default
        config = ConfigSpace.util.impute_inactive_values(config)
        dimensions = len(list(self.cs.values()))
        vector = [np.nan for i in range(dimensions)]
        for name in config:
            i = self.hps[name]
            hyper = self.cs[name]
            if type(hyper) == ConfigSpace.OrdinalHyperparameter:
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.CategoricalHyperparameter:
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            elif type(hyper) == ConfigSpace.Constant:
                vector[i] = 0 # set constant to 0, so that it wont be affected by mutation
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)

    def f_objective(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        raise NotImplementedError("The function needs to be defined in the sub class.")


class Ma_NSGA_II(MaNSGA_II_base):
    def __init__(self, cs=None, objective_number=1, trade_off_param=None, f=None, dimensions=None, pop_size=20, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin', encoding=False,
                 dim_map=None, seed=None, config_repository=None, normalize_objective_space=True, **kwargs):
        super().__init__(cs=cs, objective_number=objective_number, trade_off_param=trade_off_param, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, seed=seed, config_repository=config_repository,
                         **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None

        self.encoding = encoding
        self.dim_map = dim_map
        self._set_min_pop_size()
        self.ideal_point = np.full(self.objective_number, np.inf)
        self.nadir_point = np.full(self.objective_number, -np.inf)

        self.normalize_objective_space = normalize_objective_space
    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self.client, Client):
            self.client.close()

    def reset(self, *, reset_seeds: bool = True):
        super().reset(reset_seeds=reset_seeds)
        self.traj = []
        self.runtime = []
        self.history = []

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def map_to_original(self, vector):
        dimensions = len(self.dim_map.keys())
        new_vector = self.rng.uniform(size=dimensions)
        for i in range(dimensions):
            new_vector[i] = np.max(np.array(vector)[self.dim_map[i]])
        return new_vector

    # Used to evalutate a configuration.
    def f_objective(self, x, fidelity=None, **kwargs):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        if self.encoding:
            x = self.map_to_original(x)

        # Only convert config if configspace is used + configuration has not been converted yet
        if self.configspace:
            if not isinstance(x, ConfigSpace.Configuration):
                # converts [0, 1] vector to a ConfigSpace object
                config = self.vector_to_configspace(x)
            else:
                config = x
        else:
            config = x.copy()

        if fidelity is not None:  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, fidelity=fidelity, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert "fitness" in res
        assert "cost" in res

        #MaNSGA-II related
        #res["fitness"] = np.array(self.modify_objective_values(res["fitness"]),dtype=float)
        return res

    def init_eval_pop(self, fidelity=None, eval=True, **kwargs):
        '''Creates new population of 'pop_size' and evaluates individuals.
        '''
        self.population = self.init_population(self.pop_size)
        self.population_ids = self.config_repository.announce_population(self.population, fidelity)
        self.fitness = np.full((self.pop_size, self.objective_number), np.inf)
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.pop_size):
            config = self.population[i]
            config_id = self.population_ids[i]
            res = self.f_objective(config, fidelity, **kwargs)
            self.fitness[i], cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            #TODO No best solution is defined in MO context, but maybe an alternative could be found
            # if self.fitness[i] < self.inc_score:
            #     self.inc_score = self.fitness[i]
            #     self.inc_config = config
            #     self.inc_id = config_id

            #TODO adjust the config repositiory if needed
            self.config_repository.tell_result(config_id, float(fidelity or 0), res["fitness"], res["cost"], info)
            #traj.append(self.inc_score)
            runtime.append(cost)
            #TODO ensure that a change here does not cause bugs
            history.append((config.tolist(), self.fitness[i], float(fidelity or 0), info))

        return traj, runtime, history

    def eval_pop(self, population=None, population_ids=None, fidelity=None, **kwargs):
        '''Evaluates a population

        If population=None, the current population's fitness will be evaluated
        If population!=None, this population will be evaluated
        '''
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            #TODO why set global fitnesses twice????
            if population is None:
                self.fitness[i] = fitness
            #if fitness <= self.inc_score:
            #     self.inc_score = fitness
            #     self.inc_config = pop[i]
            #     self.inc_id = pop_ids[i]
            self.config_repository.tell_result(pop_ids[i], float(fidelity or 0),fitness, cost, info)
            #traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), fitness, float(fidelity or 0), info))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        if population is None:
            self.fitness = np.array(fitnesses)
            return traj, runtime, history
        else:
            return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    #TODO these might not work in MO context, so mayber remove them later
    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation_polynomial(self, x, eta=20):
        mutant = x.copy()
        for i in range(self.dimensions):
            if self.rng.random() < 1.0 / self.dimensions:
                u = self.rng.random()
                if u < 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                mutant[i] += delta
        return self.boundary_check(mutant)


    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5, alt_pop=alt_pop)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        elif self.mutation_strategy == 'poly':
            mutant = self.mutation_polynomial(current)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.rng.random(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.rng.integers(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = self.rng.integers(0, self.dimensions)
        L = 0
        while ((self.rng.random() < self.crossover_prob) and L < self.dimensions):
            idx = (n+L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover_sbx(self, parent1, parent2, eta=15):
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(self.dimensions):
            if self.rng.random() <= self.crossover_prob:
                x1, x2 = parent1[i], parent2[i]
                if abs(x1 - x2) > 1e-14:
                    u = self.rng.random()
                    beta = 1.0 + (2.0 * min(x1, x2))
                    alpha = 2.0 - beta**(-(eta+1))
                    if u <= 1.0/alpha:
                        betaq = (u * alpha)**(1.0/(eta+1))
                    else:
                        betaq = (1.0/(2.0 - u*alpha))**(1.0/(eta+1))

                    c1 = 0.5*((x1 + x2) - betaq*abs(x2 - x1))
                    c2 = 0.5*((x1 + x2) + betaq*abs(x2 - x1))

                    child1[i] = c1
                    child2[i] = c2

        return self.boundary_check(child1), self.boundary_check(child2)

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        elif self.crossover_strategy == 'sbx':
            offspring, _ = self.crossover_sbx(target, mutant)

        return offspring

    #TODO the interesting part. What is the most optimal approach? 
    def selection(self, trials, trial_ids, fidelity=None, **kwargs):
        """Evaluate offspring and apply the canonical MaNSGA-II selection."""
        traj = []
        runtime = []
        history = []

        _, runtime, history, trial_fitness, trial_age = self.eval_pop(
            trials, trial_ids, fidelity=fidelity
        )
        comb_pop = np.vstack((self.population, trials))
        comb_ids = np.hstack((self.population_ids, trial_ids))
        comb_fitness = np.vstack((self.fitness, trial_fitness))
        comb_age = np.hstack((self.age, trial_age))

        _, idx = np.unique(comb_pop, axis=0, return_index=True)
        if len(idx) >= self.pop_size:
            comb_pop = comb_pop[idx]
            comb_ids = comb_ids[idx]
            comb_fitness = comb_fitness[idx]
            comb_age = comb_age[idx]

        selected_indices = self.environmental_selection(
            comb_pop,
            comb_fitness,
            pop_size=self.pop_size,
            update_points=True,
        )
        self.population = comb_pop[selected_indices]
        self.fitness = comb_fitness[selected_indices]
        self.population_ids = comb_ids[selected_indices]
        self.age = comb_age[selected_indices]

        return traj, runtime, history

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        '''Performs a complete evolution cycle: mutation -> crossover -> selection
        '''
        trials = []
        trial_ids = []
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=None, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
            trials.append(trial)
            trial_ids.append(trial_id)
        trials = np.array(trials)
        trial_ids = np.array(trial_ids)
        traj, runtime, history = self.selection(trials, trial_ids, fidelity, **kwargs)
        return traj, runtime, history

    def sample_mutants(self, size, population=None):
        '''Generates 'size' mutants from the population using rand1
        '''
        if population is None:
            population = self.population
        elif len(population) < 3:
            population = np.vstack((self.population, population))

        old_strategy = self.mutation_strategy
        self.mutation_strategy = 'rand1'
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            mutant = self.mutation(current=None, best=None, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)
        self.mutation_strategy = old_strategy

        return mutants

    def run(self, generations=1, verbose=False, fidelity=None, reset=True, **kwargs):
        # checking if a run exists
        if not hasattr(self, 'traj') or reset:
            self.reset()
            if verbose:
                print("Initializing and evaluating new population...")
            self.traj, self.runtime, self.history = self.init_eval_pop(fidelity=fidelity, **kwargs)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, 0.1))
            traj, runtime, history = self.evolve_generation(fidelity=fidelity, **kwargs)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object),np.array(self.fitness)
    
class AsyncMa_NSGA_II(Ma_NSGA_II):
    def __init__(self, cs=None, objective_number=1, trade_off_param=0.1, f=None, dimensions=None, pop_size=None, max_age=np.inf,
                 mutation_factor=None, crossover_prob=None, strategy='rand1_bin',
                 async_strategy='immediate', seed=None, rng=None, config_repository=None, **kwargs):
        '''Extends DE to be Asynchronous with variations

        Parameters
        ----------
        async_strategy : str
            'deferred' - target will be chosen sequentially from the population
                the winner of the selection step will be included in the population only after
                the entire population has had a selection step in that generation
            'immediate' - target will be chosen sequentially from the population
                the winner of the selection step is included in the population right away
            'random' - target will be chosen randomly from the population for mutation-crossover
                the winner of the selection step is included in the population right away
            'worst' - the worst individual will be chosen as the target
                the winner of the selection step is included in the population right away
            {immediate, worst, random} implement Asynchronous-DE
        '''
        super().__init__(cs=cs, objective_number=objective_number, trade_off_param=trade_off_param, f=f, dimensions=dimensions, pop_size=pop_size, max_age=max_age,
                         mutation_factor=mutation_factor, crossover_prob=crossover_prob,
                         strategy=strategy, seed=seed, rng=rng, config_repository=config_repository,
                         **kwargs)
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.async_strategy = async_strategy
        assert self.async_strategy in ['immediate', 'random', 'worst', 'deferred'], \
                "{} is not a valid choice for type of DE".format(self.async_strategy)

    def _add_random_population(self, pop_size, population=None, fitness=[], age=[]):
        '''Adds random individuals to the population
        '''
        new_pop = self.init_population(pop_size=pop_size)
        new_fitness = np.full((pop_size, self.objective_number), np.inf)
        new_age = np.array([self.max_age] * pop_size)

        if population is None:
            population = self.population
            fitness = self.fitness
            age = self.age

        population = np.concatenate((population, new_pop))
        fitness = np.concatenate((fitness, new_fitness))
        age = np.concatenate((age, new_age))

        return population, fitness, age

    def _init_mutant_population(self, pop_size, population, target=None, best=None):
        '''Generates pop_size mutants from the passed population
        '''
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, best=best, alt_pop=population)
        return mutants

    def _sample_population(self, size=3, alt_pop=None, target=None):
        '''Samples 'size' individuals for mutation step

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        '''
        population = None
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]  # checks if all individuals are valid
            if any(idx):
                # default to the object's initialized population
                population = self.population
            else:
                # choose the passed population
                population = alt_pop
        else:
            # default to the object's initialized population
            population = self.population

        if target is not None and len(population) > 1:
            for i, pop in enumerate(population):
                if all(target == pop):
                    population = np.concatenate((population[:i], population[i + 1:]))
                    break
        if len(population) < self._min_pop_size:
            # compensate if target was part of the population and deleted earlier
            filler = self._min_pop_size - len(population)
            new_pop = self.init_population(pop_size=filler)  # chosen in a uniformly random manner
            population = np.concatenate((population, new_pop))

        selection = self.rng.choice(np.arange(len(population)), size, replace=False)
        return population[selection]

    # def eval_pop(self, population=None, population_ids=None, fidelity=None, **kwargs):
    #     pop = self.population if population is None else population
    #     pop_ids = self.population_ids if population_ids is None else population_ids
    #     pop_size = self.pop_size if population is None else len(pop)
    #     traj = []
    #     runtime = []
    #     history = []
    #     fitnesses = []
    #     costs = []
    #     ages = []
    #     for i in range(pop_size):
    #         res = self.f_objective(pop[i], fidelity, **kwargs)
    #         fitness, cost = res["fitness"], res["cost"]
    #         info = res["info"] if "info" in res else dict()
    #         if population is None:
    #             self.fitness[i] = fitness
    #         if fitness <= self.inc_score:
    #             self.inc_score = fitness
    #             self.inc_config = pop[i]
    #             self.inc_id = pop_ids[i]
    #         self.config_repository.tell_result(pop_ids[i], float(fidelity or 0), fitness, cost, info)
    #         traj.append(self.inc_score)
    #         runtime.append(cost)
    #         history.append((pop[i].tolist(), float(fitness), float(fidelity or 0), info))
    #         fitnesses.append(fitness)
    #         costs.append(cost)
    #         ages.append(self.max_age)
    #     return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self._sample_population(size=5, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self._sample_population(size=4, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        elif self.mutation_strategy == 'poly':
            mutant = self.mutation_polynomial(current)

        return mutant

    def sample_mutants(self, size, population=None):
        '''Samples 'size' mutants from the population
        '''
        if population is None:
            population = self.population

        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            j = self.rng.choice(np.arange(len(population)))
            mutant = self.mutation(current=population[j], best=self.inc_config, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)

        return mutants

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        '''Performs a complete DE evolution, mutation -> crossover -> selection
        '''
        traj = []
        runtime = []
        history = []

        if self.async_strategy == "deferred":
            trials = []
            trial_ids = []
            for j in range(self.pop_size):
                target = self.population[j]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
                trials.append(trial)
                trial_ids.append(trial_id)
            # selection takes place on a separate trial population only after
            # one iteration through the population has taken place
            trials = np.array(trials)
            traj, runtime, history = self.selection(trials, trial_ids, fidelity, **kwargs)
            return traj, runtime, history

        # elif self.async_strategy == "immediate":
        #     for i in range(self.pop_size):
        #         target = self.population[i]
        #         donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
        #         trial = self.crossover(target, donor)
        #         trial = self.boundary_check(trial)
        #         trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
        #         # evaluating a single trial population for the i-th individual
        #         de_traj, de_runtime, de_history, fitnesses, costs = \
        #             self.eval_pop(trial.reshape(1, self.dimensions),
        #                           np.array([trial_id]), fidelity=fidelity, **kwargs)
        #         # one-vs-one selection
        #         ## can replace the i-the population despite not completing one iteration
        #         if fitnesses[0] <= self.fitness[i]:
        #             self.population[i] = trial
        #             self.population_ids[i] = trial_id
        #             self.fitness[i] = fitnesses[0]
        #         traj.extend(de_traj)
        #         runtime.extend(de_runtime)
        #         history.extend(de_history)
        #     return traj, runtime, history

        # else:  # async_strategy == 'random' or async_strategy == 'worst':
        #     for count in range(self.pop_size):
        #         # choosing target individual
        #         if self.async_strategy == "random":
        #             i = self.rng.choice(np.arange(self.pop_size))
        #         else:  # async_strategy == 'worst'
        #             i = np.argsort(-self.fitness)[0]
        #         target = self.population[i]
        #         mutant = self.mutation(current=target, best=best, alt_pop=alt_pop)
        #         trial = self.crossover(target, mutant)
        #         trial = self.boundary_check(trial)
        #         trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
        #         # evaluating a single trial population for the i-th individual
        #         de_traj, de_runtime, de_history, fitnesses, costs = \
        #             self.eval_pop(trial.reshape(1, self.dimensions), np.array([trial_id]),
        #                            fidelity=fidelity, **kwargs)
        #         # one-vs-one selection
        #         ## can replace the i-the population despite not completing one iteration
        #         if fitnesses[0] <= self.fitness[i]:
        #             self.population[i] = trial
        #             self.fitness[i] = fitnesses[0]
        #         traj.extend(de_traj)
        #         runtime.extend(de_runtime)
        #         history.extend(de_history)

        return traj, runtime, history

    def run(self, generations=1, verbose=False, fidelity=None, reset=True, **kwargs):
        # checking if a run exists
        print("Async MaNSGA-II is not executable by itself currently. Use Ma_NSGA_II class instead if you want" \
        "Evolutionary many objective optimization without fidelity definition. Else use MaDEHB itself.")
        # if not hasattr(self, "traj") or reset:
        #     self.reset()
        #     if verbose:
        #         print("Initializing and evaluating new population...")
        #     self.traj, self.runtime, self.history = self.init_eval_pop(fidelity=fidelity, **kwargs)

        # if verbose:
        #     print("Running evolutionary search...")
        # for i in range(generations):
        #     if verbose:
        #         print("Generation {:<2}/{:<2} -- {:<0.7}".format(i+1, generations, self.inc_score))
        #     traj, runtime, history = self.evolve_generation(fidelity=fidelity,
        #                                                     best=self.inc_config, **kwargs)
        #     self.traj.extend(traj)
        #     self.runtime.extend(runtime)
        #     self.history.extend(history)

        # if verbose:
        #     print("\nRun complete!")

        # return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)
# Keep the supplied operators and standalone API; replace the selection and
# unsafe DE helpers through a shared mixin used by both public classes.
"""Numerically safe MaNSGA-II
"""
import numpy as np


def unique_candidates(population, fitness, keys=None):
    """Prefer the latest finite observation, never an unevaluated duplicate.
    """
    chosen = {}
    for i, x in enumerate(population):
        key = tuple(x) if keys is None else keys[i]
        old = chosen.get(key)
        if old is None or np.isfinite(fitness[i]).all():
            chosen[key] = i
    return np.fromiter(chosen.values(), dtype=int)


class Selection:
    def selection(self, trials, trial_ids, fidelity=None, **kwargs):
        traj, runtime, history, trial_fitness, trial_age = self.eval_pop(
            trials, trial_ids, fidelity=fidelity, **kwargs)
        X = np.vstack((self.population, trials))
        F = np.vstack((self.fitness, trial_fitness))
        ids = np.concatenate((self.population_ids, trial_ids))
        ages = np.concatenate((self.age, trial_age))
        indices = unique_candidates(X, F)
        chosen = self.environmental_selection(X[indices], F[indices], self.pop_size)
        selected = indices[chosen]
        if len(selected) < self.pop_size:
            selected = np.resize(selected, self.pop_size)
        self.population, self.fitness = X[selected], F[selected]
        self.population_ids, self.age = ids[selected], ages[selected]
        return traj, runtime, history

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        trials, ids = [], []
        for current in self.population.copy():
            mutant = self.mutation(current=current, best=best, alt_pop=alt_pop)
            trial = self.boundary_check(self.crossover(current, mutant))
            trials.append(trial)
            ids.append(self.config_repository.announce_config(trial, fidelity))
        return self.selection(np.asarray(trials), np.asarray(ids), fidelity, **kwargs)

    def reset(self, *, reset_seeds=True):
        super().reset(reset_seeds=reset_seeds)
        self.ideal_point = np.full(self.objective_number, np.inf)
        self.nadir_point = np.full(self.objective_number, -np.inf)

    def crowding_distance(self, F):
        F = np.asarray(F, dtype=float)
        n, m = F.shape
        distance = np.zeros(n)
        if n <= 2:
            return np.full(n, np.inf)
        for j in range(m):
            order = np.argsort(F[:, j], kind="stable")
            span = F[order[-1], j] - F[order[0], j]
            if span <= 0:
                continue  # A constant objective has no boundary or distance.
            distance[order[[0, -1]]] = np.inf
            distance[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
        return distance

    def distance_selection(self, first_F, front_indices, ideal_point, pop_size):
        F = np.asarray(first_F, dtype=float)
        n = len(F)
        if pop_size <= 0:
            return []
        if n <= pop_size:
            return list(front_indices)
        span = F.max(axis=0) - ideal_point
        Z = (F - ideal_point) / np.where(span > 0, span, 1.0)
        norms = np.linalg.norm(Z, axis=1)
        sums = Z.sum(axis=1, keepdims=True)
        projected = np.divide(Z, sums, out=np.zeros_like(Z), where=sums > 0)
        # Random ties are reproducible through the optimizer's seeded RNG.
        first = int(self.rng.choice(np.flatnonzero(norms == norms.min())))
        selected = [first]
        distances = np.sum((projected - projected[first]) ** 2, axis=1)
        distances[first] = -np.inf
        while len(selected) < pop_size:
            tied = np.flatnonzero(distances == distances.max())
            # If directions coincide, retain the better converged candidate.
            best_norm = norms[tied].min()
            j = int(self.rng.choice(tied[norms[tied] == best_norm]))
            selected.append(j)
            distances = np.minimum(distances, np.sum((projected - projected[j]) ** 2, axis=1))
            distances[selected] = -np.inf
        return np.asarray(front_indices)[selected].tolist()

    def modify_objective_values(self, fitness):
        return self.modify_objective_values_batch(np.asarray(fitness)[None, :])[0]

    def modify_objective_values_batch(self, fitness):
        F = np.atleast_2d(np.asarray(fitness, dtype=float))
        if self.normalize_objective_space:
            span = self.nadir_point - self.ideal_point
            F = (F - self.ideal_point) / np.where(span > 0, span, 1.0)
        return (1 - self.alpha) * F + self.alpha * F.mean(axis=1, keepdims=True)

    def environmental_selection(self, population, fitness, pop_size=None, update_points=True):
        X, F = np.asarray(population), np.asarray(fitness, dtype=float)
        if F.ndim != 2 or F.shape != (len(X), self.objective_number):
            raise ValueError("fitness must have shape (n_candidates, objective_number)")
        size = min(int(self.pop_size if pop_size is None else pop_size), len(X))
        if size < 0:
            raise ValueError("pop_size must be nonnegative")
        if size == 0:
            return np.empty(0, dtype=int)
        valid = np.flatnonzero(np.isfinite(F).all(axis=1))
        invalid = np.flatnonzero(~np.isfinite(F).all(axis=1))
        if not len(valid):
            return invalid[:size]
        ideal = np.minimum(self.ideal_point, F[valid].min(axis=0))
        nadir = F[valid].max(axis=0)
        if update_points:
            self.ideal_point, self.nadir_point = ideal, nadir
        G = F[valid]
        if self.normalize_objective_space:
            span = nadir - ideal
            G = (G - ideal) / np.where(span > 0, span, 1.0)
        G = (1 - self.alpha) * G + self.alpha * G.mean(axis=1, keepdims=True)
        permutation = self.rng.permutation(len(valid))
        valid, G = valid[permutation], G[permutation]
        fronts = self.fast_nondominated_sort(G)
        target = min(size, len(valid))
        if len(fronts[0]) > target:
            local = self.distance_selection(F[valid[fronts[0]]], fronts[0], ideal, target)
        else:
            local = []
            for front in fronts:
                remaining = target - len(local)
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    local.extend(front)
                else:
                    crowd = self.crowding_distance(G[front])
                    local.extend(np.asarray(front)[np.argsort(-crowd, kind="stable")[:remaining]])
        selected = valid[np.asarray(local, dtype=int)]
        return np.concatenate((selected, invalid[:size - len(selected)]))

    def crossover_exp(self, target, mutant):
        result = np.array(target, copy=True)
        start = int(self.rng.integers(self.dimensions))
        count = 1  # DE exponential crossover MUST take at least one donor gene.
        while count < self.dimensions and self.rng.random() < self.crossover_prob:
            count += 1
        indices = (start + np.arange(count)) % self.dimensions
        result[indices] = mutant[indices]
        return result

    def _sample_population(self, size=3, alt_pop=None, target=None):
        pool = self.population if alt_pop is None else np.asarray(alt_pop)
        pool = np.unique(np.asarray(pool), axis=0)
        if target is not None:
            pool = pool[~np.all(pool == target, axis=1)]
        if len(pool) < size:
            pool = np.vstack((pool, self.init_population(size - len(pool))))
        return pool[self.rng.choice(len(pool), size, replace=False)]

    def mutation(self, current=None, best=None, alt_pop=None):
        if best is None and 'best' in self.mutation_strategy:
            valid = np.flatnonzero(np.isfinite(self.fitness).all(axis=1))
            if len(valid):
                selected = self.environmental_selection(
                    self.population[valid], self.fitness[valid],
                    pop_size=min(len(valid), max(2, self.objective_number)), update_points=False)
                best = self.population[valid[int(self.rng.choice(selected))]]
            else:
                best = current
        return super().mutation(current=current, best=best, alt_pop=alt_pop)

_LegacyMaNSGAII = Ma_NSGA_II
_LegacyAsyncMaNSGAII = AsyncMa_NSGA_II

class Ma_NSGA_II(Selection, _LegacyMaNSGAII):
    pass

class AsyncMa_NSGA_II(Selection, _LegacyAsyncMaNSGAII):
    run = _LegacyMaNSGAII.run

