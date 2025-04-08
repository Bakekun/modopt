import time
import numpy as np
from modopt import Optimizer
from deap import base, creator, tools
import random

class PenaltyDEAPGeneric(Optimizer):

    def initialize(self):

        # Name your algorithm
        self.solver_name = 'Penalty_DEAP_Genetic_Algorithm'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

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
        self.options.declare('penaltyWeight', default =   1e3, types = float)

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
        self.toolbox.register("select", lambda inds, k: self.constraint_tournament(inds, k))
        def modopt_evaluate(ind):
            return self.constraint_evaluate(ind)  # use penalty-based evaluation
        self.toolbox.register("evaluate", modopt_evaluate)

    def constraint_evaluate(self, individual):
        """ Evaluate function with a simple penalty for constraint violation. """
        f_val = self.problem._compute_objective(individual)
        viol = self.get_constraint_violation(individual)

        # e.g. penalty_weight * viol
        penalty_weight = self.options['penaltyWeight']
        return (f_val + penalty_weight * viol,)

    
    def get_constraint_violation(self, individual):
        """
        Compute a single scalar measure of constraint violation for the given individual.
        c_lower[i], c_upper[i] might be -inf or +inf for unconstrained sides.
        If c_lower[i] == c_upper[i], we interpret it as an equality constraint.
        """
        # 1. Evaluate the constraints
        c_vals = self.problem._compute_constraints(individual)  # shape (nc, )

        # 2. Access the Problem's c_lower, c_upper
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper
        violation = 0.0

        for i in range(self.problem.nc):
            val = c_vals[i]
            low = c_lower[i]
            high = c_upper[i]

            # If c_lower == c_upper => equality constraint => measure abs difference
            if np.isfinite(low) and np.isfinite(high) and abs(low - high) < 1e-14:
                # equality constraint
                violation += abs(val - low)
            else:
                # inequality constraint => val should lie in [low, high]
                # if val < low => violation is (low - val)
                if val < low:
                    violation += (low - val)
                # if val > high => violation is (val - high)
                if val > high:
                    violation += (val - high)

        return violation

    def constraint_tournament(self, individuals, k):
        """Tournament selection that prefers feasible solutions, 
           else picks the least violating solution."""
        chosen = []
        for _ in range(k):
            aspirants = random.sample(individuals, self.options["tournsize"])
            feasible = [ind for ind in aspirants if self.get_constraint_violation(ind) < 1e-14]
            if feasible:
                chosen.append(tools.selBest(feasible, 1)[0])
            else:
                chosen.append(min(aspirants, key=self.get_constraint_violation))
        return chosen


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
            ind.fitness.values = self.constraint_evaluate(ind)


        while (opt > opt_tol and itr < NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

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
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = sorted(offspring,
                            key=lambda ind: (self.get_constraint_violation(ind), ind.fitness.values[0]))

            val = self.obj(pop[0])          # Might be float or array/tuple
            val_arr = np.atleast_1d(val)    # Convert to 1D array if float
            if val_arr.size == 1:
                # Single-objective
                best_ind = tools.selBest(pop, 1)[0]
            else:
                # Multi-objective
                best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
                best_ind = random.choice(best_inds)



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