import time
import random
import numpy as np
from deap import base, creator, tools, algorithms
from modopt import Optimizer

class SPEA2(Optimizer):
    def initialize(self):
        # Name the algorithm
        self.solver_name = "SPEA2"

        self.obj = self.problem._compute_objectives

        # Declare optimizer parameters
        self.options.declare("maxiter", default=1000, types=int)
        self.options.declare("opt_tol", default=1e-5, types=float)
        self.options.declare("initialPopulationSize", default=200, types=int)
        self.options.declare("archiveSize", default=50, types=int)
        self.options.declare("rangeLow", default=-4.0, types=float)
        self.options.declare("rangeHigh", default=4.0, types=float)
        self.options.declare("mutationRate", default=0.1, types=float)
        self.options.declare("cxProb", default=0.7, types=float)
        self.options.declare("tournsize", default=3, types=int)
        self.options.declare('readable_outputs', types=list, default=[])


        # Available outputs
        self.available_outputs = {
            "itr": int,
            "obj": float,
            "x": (float, (self.problem.nx,)),
            "opt": float,
            "time": float,
        }

    def setup(self):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        IND_SIZE = self.problem.nx
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, self.options["rangeLow"], self.options["rangeHigh"])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=self.options["mutationRate"])
        self.toolbox.register("select", tools.selSPEA2)  # Use SPEA2 selection

        def modopt_evaluate(individual):
            return np.array(self.problem._compute_objectives(individual))

        self.toolbox.register("evaluate", modopt_evaluate)

    def solve(self):
        x = self.problem.x0
        opt_tol = self.options["opt_tol"]
        maxiter = self.options["maxiter"]

        start_time = time.time()

        # Initialize population
        pop = self.toolbox.population(n=self.options["initialPopulationSize"])
        archive = []  # External archive

        # Evaluate population
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        itr = 0
        opt = float("inf")

        while (opt > opt_tol and itr < maxiter):
            offspring = list(map(self.toolbox.clone, pop))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.options["cxProb"]:
                    self.toolbox.mate(child1, child2)

                    # Clip values after crossover
                    for i in range(len(child1)):
                        child1[i] = np.clip(child1[i], self.options["rangeLow"], self.options["rangeHigh"])
                        child2[i] = np.clip(child2[i], self.options["rangeLow"], self.options["rangeHigh"])

                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.options["mutationRate"]:
                    self.toolbox.mutate(mutant)

                    # Clip values after mutation
                    for i in range(len(mutant)):
                        mutant[i] = np.clip(mutant[i], self.options["rangeLow"], self.options["rangeHigh"])

                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Combine population + archive
            combined_pop = pop + archive + offspring

            # SPEA2 selection
            pop = self.toolbox.select(combined_pop, self.options["initialPopulationSize"])
            archive = self.toolbox.select(combined_pop, self.options["archiveSize"])

            # Extract best solutions
            best_inds = tools.sortNondominated(archive, len(archive), first_front_only=True)

            if len(best_inds) == 0 or len(best_inds[0]) == 0:
                raise ValueError("No valid Pareto-optimal solutions found. Check population settings.")
            x_k = [np.array(ind) for ind in best_inds[0]] if best_inds else []
            f_k = [ind.fitness.values for ind in best_inds[0]] if best_inds else []

            opt = min(min(obj) for obj in f_k)

            # Store iteration results
            self.update_outputs(
                itr=itr,
                x=np.atleast_1d(x_k).tolist(),
                obj=float(np.mean(f_k)),
                opt=opt,
                time=float(time.time() - start_time),
            )

            itr += 1

        self.total_time = time.time() - start_time

        self.results = {
            "x": x_k,
            "objective": f_k,
            "optimality": opt,
            "itr": itr,
            "time": self.total_time,
        }

        self.run_post_processing()
        return self.results, x_k, f_k, archive, itr, self.total_time