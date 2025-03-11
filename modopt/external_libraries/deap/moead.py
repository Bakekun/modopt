import numpy as np
import random
import time
from deap import base, creator, tools
from modopt import Optimizer

class MOEAD(Optimizer):

    def initialize(self):
        # Name your algorithm
        self.solver_name = "DEAP_MOEAD"
        self.obj = self.problem._compute_objectives

        # Declare hyperparameters
        self.options.declare("maxiter", default=500, types=int)
        self.options.declare("opt_tol", default=1e-5, types=float)
        self.options.declare("initialPopulationSize", default=100, types=int)
        self.options.declare("rangeLow", default=-4.0, types=float)
        self.options.declare("rangeHigh", default=4.0, types=float)
        self.options.declare("neighborhood_size", default=10, types=int)  # Number of neighbors per individual
        self.options.declare("mutation_std", default=0.1, types=float)
        self.options.declare("crossover_rate", default=0.9, types=float)
        self.options.declare("decomposition", default="tchebycheff", types=str)  # Scalarization method
        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,  # Must be a single scalar
            'x': (float, (self.problem.nx,)),
            'opt': float,
            'time': float,
        }

    def setup(self):
        if not hasattr(creator, "FitnessMin"):
            # Multi-objective: 2 objectives => weights=(-1.0, -1.0)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        IND_SIZE = self.problem.nx

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, self.options["rangeLow"], self.options["rangeHigh"])
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Evaluate function must return a tuple => multi-objective
        self.toolbox.register("evaluate", lambda ind: tuple(self.problem._compute_objectives(ind)))

        # Initialize weight vectors (decomposition approach)
        self.num_objs = len(self.problem._compute_objectives(self.toolbox.individual()))
        self.weight_vectors = self.generate_weight_vectors(self.options["initialPopulationSize"], self.num_objs)
        
        # Neighborhood structure (based on Euclidean distance between weight vectors)
        self.neighborhoods = self.compute_neighborhoods(self.weight_vectors, self.options["neighborhood_size"])

    def generate_weight_vectors(self, num_vectors, num_objs):
        """ Generate weight vectors uniformly distributed on the simplex. """
        weights = np.random.dirichlet(np.ones(num_objs), size=num_vectors)
        return weights.tolist()

    def compute_neighborhoods(self, weight_vectors, size):
        """ Compute neighborhoods based on Euclidean distance between weight vectors. """
        distances = np.linalg.norm(np.expand_dims(weight_vectors, axis=1) - weight_vectors, axis=2)
        return np.argsort(distances, axis=1)[:, :size]

    def scalarization(self, individual, weight_vector, ideal_point):
        """ Tchebycheff scalarization function. """
        objectives = np.array(self.problem._compute_objectives(individual))
        return np.max(weight_vector * np.abs(objectives - ideal_point))

    def solve(self):
        opt_tol = self.options["opt_tol"]
        maxiter = self.options["maxiter"]
        pop_size = self.options["initialPopulationSize"]
        
        start_time = time.time()

        # Initialize population
        pop = self.toolbox.population(n=pop_size)

        # Evaluate fitness
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        # Initialize ideal point (min values across objectives)
        ideal_point = np.min([ind.fitness.values for ind in pop], axis=0)

        itr = 0
        final_front_x = []
        final_front_f = []

        while itr < maxiter:
            for i, ind in enumerate(pop):
                # Select two parents from the neighborhood
                neighbors = [pop[idx] for idx in self.neighborhoods[i]]
                parent1, parent2 = random.sample(neighbors, 2)

                if random.random() < self.options["crossover_rate"]:
                    child1, child2 = tools.cxBlend(parent1, parent2, alpha=0.5)
    
                    # Apply mutation
                    tools.mutGaussian(child1, mu=0, sigma=self.options["mutation_std"], indpb=0.2)
                    tools.mutGaussian(child2, mu=0, sigma=self.options["mutation_std"], indpb=0.2)

                    # Clip values to stay within [rangeLow, rangeHigh]
                    for i in range(len(child1)):
                        child1[i] = np.clip(child1[i], self.options["rangeLow"], self.options["rangeHigh"])
                        child2[i] = np.clip(child2[i], self.options["rangeLow"], self.options["rangeHigh"])

                    del child1.fitness.values, child2.fitness.values
                else:
                    child1, child2 = parent1, parent2


                # Evaluate offspring
                for child in (child1, child2):
                    child.fitness.values = self.toolbox.evaluate(child)

                # Update ideal point
                ideal_point = np.minimum(ideal_point, child1.fitness.values)
                ideal_point = np.minimum(ideal_point, child2.fitness.values)

                # Replace individual if the new child has a better scalarized objective
                if self.scalarization(child1, self.weight_vectors[i], ideal_point) < self.scalarization(ind, self.weight_vectors[i], ideal_point):
                    pop[i] = child1
                if self.scalarization(child2, self.weight_vectors[i], ideal_point) < self.scalarization(ind, self.weight_vectors[i], ideal_point):
                    pop[i] = child2

            # Extract Pareto front
            best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            x_k = [np.array(ind) for ind in best_inds]
            f_k = [ind.fitness.values for ind in best_inds]

            final_front_x = x_k
            final_front_f = f_k

            itr += 1

            # Store outputs
            avg_sum_obj = float(np.mean([sum(objs) for objs in f_k]))
            self.update_outputs(
                itr=itr,
                x=x_k,
                obj=avg_sum_obj,
                opt=float(np.min([sum(obj) for obj in f_k])),
                time=time.time() - start_time
            )

        self.total_time = time.time() - start_time
        self.results = {
            "x": final_front_x,
            "objective": final_front_f,
            "optimality": np.min([sum(obj) for obj in final_front_f]),
            "itr": itr,
            "time": self.total_time
        }

        self.run_post_processing()
        return self.results, final_front_x, final_front_f, pop, itr, self.total_time