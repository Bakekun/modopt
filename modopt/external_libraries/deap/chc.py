import random
import time
import numpy as np
from deap import base, creator, tools
from modopt import Optimizer

class CHC(Optimizer):

    def initialize(self):
        """Initialize CHC Algorithm within ModOpt."""
        self.solver_name = 'DEAP_CHC'

        self.obj = self.problem._compute_objective

        # Define hyperparameters
        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)
        self.options.declare('initialPopulationSize', default=200, types=int)
        self.options.declare('rangeLow', default=-4.0, types=float)
        self.options.declare('rangeHigh', default=4.0, types=float)
        self.options.declare('diversity_threshold', default=3.0, types=float)  # Hamming distance threshold
        self.options.declare('restart_fraction', default=0.35, types=float)  # Fraction to reinitialize
        self.options.declare('stagnation_limit', default=10, types=int)  # Restart after N generations of no improvement

        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            'x': (float, (self.problem.nx,)),
            'opt': float,
            'time': float,
        }

    @staticmethod
    def hux_crossover(ind1, ind2):
        """Half Uniform Crossover (HUX): Swaps 50% of differing genes."""
        diff_indices = [i for i in range(len(ind1)) if ind1[i] != ind2[i]]
        num_swaps = len(diff_indices) // 2  # Swap half of differing bits
        
        for idx in random.sample(diff_indices, num_swaps):
            ind1[idx], ind2[idx] = ind2[idx], ind1[idx]

        return ind1, ind2

    @staticmethod
    def hamming_distance(ind1, ind2):
        """Compute Hamming distance between two individuals."""
        return sum(1 for a, b in zip(ind1, ind2) if a != b)

    def setup(self):
        """Setup CHC Algorithm with DEAP"""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        IND_SIZE = self.problem.nx

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, self.options['rangeLow'], self.options['rangeHigh'])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", self.hux_crossover)
        self.toolbox.register("select", tools.selBest)

        def modopt_evaluate(individual):
            return (self.problem._compute_objective(individual),)

        self.toolbox.register("evaluate", modopt_evaluate)

    def solve(self):
        """Solve optimization problem using CHC Algorithm"""
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']
        stagnation_limit = self.options['stagnation_limit']
        diversity_threshold = self.options['diversity_threshold']
        restart_fraction = self.options['restart_fraction']

        obj = self.obj

        start_time = time.time()

        x_k = x * 1.
        f_k = obj(x_k)

        itr = 0
        opt = float('inf')
        stagnation_counter = 0  # Track stagnation

        self.update_outputs(itr=0, x=x_k, obj=f_k, opt=opt, time=time.time() - start_time)

        pop = self.toolbox.population(n=self.options['initialPopulationSize'])

        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]

        while opt > opt_tol and itr < maxiter:
            # Sort by fitness
            pop = tools.selBest(pop, len(pop))
            offspring = pop[:]

            # Apply HUX crossover
            for i in range(0, len(offspring) - 1, 2):
                if self.hamming_distance(offspring[i], offspring[i+1]) >= diversity_threshold:
                    offspring[i], offspring[i+1] = self.toolbox.mate(offspring[i], offspring[i+1])

            # Evaluate new individuals
            for ind in offspring:
                del ind.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Update population
            pop[:] = tools.selBest(pop + offspring, len(pop))

            # Check best fitness
            new_best_ind = tools.selBest(pop, 1)[0]
            new_best_fitness = new_best_ind.fitness.values[0]

            if new_best_fitness < best_fitness:
                best_fitness = new_best_fitness
                best_ind = new_best_ind
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            # **Restart mechanism (Cataclysmic Mutation)**
            if stagnation_counter >= stagnation_limit:
                print("Restarting due to stagnation...")
                num_reset = int(len(pop) * restart_fraction)
                for i in range(num_reset):
                    pop[i] = self.toolbox.individual()
                stagnation_counter = 0

            x_k = np.array(best_ind)
            f_k = best_fitness
            opt = f_k
            itr += 1

            # Log progress
            self.update_outputs(itr=itr, x=x_k.tolist(), obj=f_k, opt=opt, time=time.time() - start_time)

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k,
            'optimality': opt,
            'itr': itr,
            'time': self.total_time
        }

        self.run_post_processing()
        return self.results