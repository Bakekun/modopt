import time
import numpy as np
from modopt import Optimizer
from deap import base, creator, tools
import random

class DEAPGeneric(Optimizer):

    def initialize(self):

        # Name your algorithm
        self.solver_name = 'DEAP_Genetic_Algorithm'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-12, types=float)
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
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
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
        self.toolbox.register("select", tools.selTournament, tournsize=self.options['tournsize'])
        def modopt_evaluate(individual):
            return (self.problem._compute_objective(individual),)

        self.toolbox.register("evaluate", modopt_evaluate)

    def solve(self):
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Iteration counter
        itr = 0

        # Optimality
        opt = float('inf')

        # Initializing outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time)

        pop = self.toolbox.population(n=self.options['initialPopulationSize'])
        CXPB, MUTPB, NGEN = self.options['cxProb'], self.options['mutationRateInd'], maxiter

        # Evaluate the entire population
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        while (opt > opt_tol and itr < NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            val = self.obj(pop[0])                # Might be float (SO) or array/tuple (MO)
            val_arr = np.atleast_1d(val)          # Ensures at least 1D
            
            if len(val_arr) == 1:
                best_ind = tools.selBest(pop, 1)[0]  # Single-objective
            else:
                best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
                best_ind = random.choice(best_inds)  # Multi-objective


            f_k = best_ind.fitness.values[0]
            opt = f_k

            x_k = np.array(best_ind)

            itr += 1
            
            # Append arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time)

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k,
            'optimality': opt,
            'itr': itr,
            'time': self.total_time
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results