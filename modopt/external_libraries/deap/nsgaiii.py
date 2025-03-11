import time
import numpy as np
import random
from deap import base, creator, tools
from modopt import Optimizer
from itertools import combinations

class NSGAIII(Optimizer):

    def initialize(self):
        """Initialize NSGA-III Algorithm within ModOpt."""
        self.solver_name = 'NSGA-III'

        self.obj = self.problem._compute_objectives

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)
        self.options.declare('initialPopulationSize', default=200, types=int)
        self.options.declare('rangeLow', default=-4.0, types=float)
        self.options.declare('rangeHigh', default=4.0, types=float)
        self.options.declare('mutationRateGene', default=0.1, types=float)
        self.options.declare('alpha', default=0.5, types=float)
        self.options.declare('tournsize', default=3, types=int)
        self.options.declare('cxProb', default=0.5, types=float)
        self.options.declare('mutationRateInd', default=0.2, types=float)
        self.options.declare('numRefPoints', default=12, types=int)  # Reference points for NSGA-III

        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            'x': (float, (self.problem.nx,)),
            'opt': float,
            'time': float,
        }

    def generate_reference_points(self, num_obj, num_ref):
        """Generate Das & Dennis (2011) reference points for NSGA-III."""
        ref_points = []
        ref_levels = [i / num_ref for i in range(num_ref + 1)]

        for comb in combinations(ref_levels, num_obj):
            if sum(comb) == 1.0:
                ref_points.append(comb)

        return np.array(ref_points)

    def setup(self):
        """Setup NSGA-III Algorithm with DEAP"""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * self.problem.n_obj)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        IND_SIZE = self.problem.nx
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, self.options['rangeLow'], self.options['rangeHigh'])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=self.options['alpha'])
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(self.options['rangeHigh'] - self.options['rangeLow']) * 0.1, indpb=self.options['mutationRateGene'])
        self.toolbox.register("select", tools.selNSGA3, ref_points=self.generate_reference_points(self.problem.n_obj, self.options['numRefPoints']))

        def modopt_evaluate(individual):
            return tuple(self.problem._compute_objectives(individual))  # Ensure tuple return

        self.toolbox.register("evaluate", modopt_evaluate)

    def solve(self):
        """Solve optimization problem using NSGA-III Algorithm"""
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj

        start_time = time.time()

        x_k = x * 1.
        f_k = obj(x_k)

        itr = 0
        opt = float('inf')

        self.update_outputs(
            itr=0,
            x=np.atleast_1d(x_k).tolist(),
            obj=float(np.mean(f_k)) if isinstance(f_k, (list, np.ndarray)) else float(f_k),
            opt=float(np.min(f_k)),
            time=float(time.time() - start_time)
        )

        pop = self.toolbox.population(n=self.options['initialPopulationSize'])
        CXPB, MUTPB, NGEN = self.options['cxProb'], self.options['mutationRateInd'], maxiter

        # Evaluate initial population
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        while (opt > opt_tol and itr < NGEN):
            offspring = list(map(self.toolbox.clone, pop))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    # Ensure children stay within bounds
                    for i in range(len(child1)):
                        child1[i] = np.clip(child1[i], self.options['rangeLow'], self.options['rangeHigh'])
                        child2[i] = np.clip(child2[i], self.options['rangeLow'], self.options['rangeHigh'])
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    # Ensure mutation does not exceed bounds
                    for i in range(len(mutant)):
                        mutant[i] = np.clip(mutant[i], self.options['rangeLow'], self.options['rangeHigh'])
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # The population is entirely replaced by the offspring
            combined_pop = pop + offspring
            pop[:] = self.toolbox.select(combined_pop, len(pop))

            # Output the best Pareto front
            best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

            x_k = [np.array(ind) for ind in best_inds]
            f_k = [ind.fitness.values for ind in best_inds]

            itr += 1

            self.update_outputs(
                itr=itr,
                x=np.atleast_1d(x_k).tolist(),
                obj=float(np.mean(f_k)) if isinstance(f_k, (list, np.ndarray)) else float(f_k),
                opt=float(np.min(f_k)),
                time=float(time.time() - start_time)
            )

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k,
            'optimality': min(min(obj) for obj in f_k),
            'itr': itr,
            'time': self.total_time
        }

        self.run_post_processing()

        return self.results, x_k, f_k, pop, itr, self.total_time