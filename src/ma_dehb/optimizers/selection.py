"""Numerically safe MaNSGA-II (Pang et al., Algorithms 1 and 2).

All fitness arrays contain ORIGINAL minimization objectives. Transformations
exist only inside selection. No hypervolume or scalarization replaces survival.
"""
import numpy as np


def unique_candidates(population, fitness):
    """Prefer the latest finite observation, never an unevaluated duplicate.

    Stable decision-space ordering avoids sorting the population by variable 0.
    For repeated noisy evaluations this is a latest-observation policy, not an
    optimistic best-observation policy. Replicate aggregation belongs in f().
    """
    chosen = {}
    for i, x in enumerate(population):
        key = tuple(x)
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
        # The standalone API evolves one whole generation. The supplied async
        # implementation silently did zero evaluations unless set to deferred.
        # MaDEHB's asynchronous ask/tell survival is managed separately.
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
        # Every selection uses bounds from its own candidate set. Read-only
        # ranking must not poison another fidelity's normalization state.
        ideal = np.minimum(self.ideal_point, F[valid].min(axis=0))
        nadir = F[valid].max(axis=0)
        if update_points:
            self.ideal_point, self.nadir_point = ideal, nadir
        G = F[valid]
        if self.normalize_objective_space:
            span = nadir - ideal
            G = (G - ideal) / np.where(span > 0, span, 1.0)
        G = (1 - self.alpha) * G + self.alpha * G.mean(axis=1, keepdims=True)
        # Shuffle before ranking to avoid permanent parent / lexicographic tie
        # advantage. This does not change any strict dominance comparison.
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
