import time
import numpy as np
from modopt import Optimizer
from deap import base, creator, tools
import random

class NSGAII(Optimizer):


    def initialize(self):

        # Name your algorithm
        self.solver_name = 'NSGA-II'

        self.obj = self.problem._compute_objectives

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)
        self.options.declare('initialPopulationSize', default=200, types=int)
        self.options.declare('rangeLow', default = -4.0, types = float)
        self.options.declare('rangeHigh', default =  4.0, types = float)
        self.options.declare('mutationRateGene', default = 0.1, types = float)
        self.options.declare('alpha', default = 0.5, types = float)
        self.options.declare('tournsize', default = 3, types = int)
        self.options.declare('cxProb' , default =  0.5, types = float)
        self.options.declare('mutationRateInd', default =   0.2, types = float)

        # Enable user to specify, as a list, which among the available outputs
        # need to be written to output files
        self.options.declare('readable_outputs', types=list, default=[])

        # Specify format of outputs available from your optimizer after each iteration
        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
        }

    def setup(self):
        if not hasattr(creator, "FitnessMin"):
            # -1.0 Weight means minimization, 1.0 for Maximization
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        IND_SIZE = self.problem.nx

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, self.options['rangeLow'], self.options['rangeHigh'])  # Adjusted range
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                 self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=self.options['alpha'])
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(self.options['rangeHigh'] - self.options['rangeLow'])*0.1, indpb=self.options['mutationRateGene'])
        self.toolbox.register("select", tools.selNSGA2)
        def modopt_evaluate(individual):
            return np.array(self.problem._compute_objectives(individual))  # Ensure it returns multiple objectives

        self.toolbox.register("evaluate", modopt_evaluate)

    def solve(self):
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)

        # Iteration counter
        itr = 0

        # Optimality
        opt = float('inf')

        # Initializing outputs
        self.update_outputs(
                itr=itr,
                x=np.atleast_1d(x_k).tolist(),  # Ensure x_k is a list
                obj=float(np.mean(f_k)) if isinstance(f_k, (list, np.ndarray)) else float(f_k),  # Convert to single value
                opt = float(np.min(f_k)),
                time=float(time.time() - start_time)
            )


        pop = self.toolbox.population(n=self.options['initialPopulationSize'])
        CXPB, MUTPB, NGEN = self.options['cxProb'], self.options['mutationRateInd'], maxiter

        # Evaluate the entire population
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

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # The population is entirely replaced by the offspring
            combined_pop = pop + offspring

            pop[:] = self.toolbox.select(combined_pop, len(pop))

            # Output the best solution(s)
            best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

            # Convert Pareto front to NumPy arrays
            x_k = [np.array(ind) for ind in best_inds]
            f_k = [ind.fitness.values for ind in best_inds]

            itr += 1

            # Append arrays inside outputs dict with new values from the current iteration
            self.update_outputs(
                itr=itr,
                x=np.atleast_1d(x_k).tolist(),  # Ensure x_k is a list
                obj=float(np.mean(f_k)) if isinstance(f_k, (list, np.ndarray)) else float(f_k),  # Convert to single value
                opt = float(np.min(f_k)),
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

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results, x_k, f_k, pop, itr, self.total_time