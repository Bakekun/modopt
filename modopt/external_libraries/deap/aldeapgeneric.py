import time
import random
import numpy as np

from modopt import Optimizer
from modopt.merit_functions import AugmentedLagrangian
from deap import base, creator, tools

class ALDEAPGeneric(Optimizer):
    def initialize(self):
        self.solver_name = 'AL_DEAP_Genetic_Algorithm'

        # Problem dimensions
        self.nx = self.problem.nx

        # Objective function
        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.con_in = self.problem._compute_constraints if self.problem.constrained else None
        self.jac_in = self.problem._compute_constraint_jacobian if self.problem.constrained else None

        # Declare optimizer settings
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
        self.options.declare('rho_init', default=1e2, types=float)  # Initial penalty parameter
        self.options.declare('lambda_init', default=0.0, types=float)  # Initial Lagrange multipliers

        self.options.declare('readable_outputs', types=list, default=[])

        # Define available outputs
        self.available_outputs = {
            'itr': int,                    # Iteration count
            'obj': float,                   # Objective function value
            'x': (float, (self.problem.nx,)),  # Design variables
            'opt': float,                   # Best fitness value at each iteration
            'time': float,                   # Total elapsed time
            'lag_mult': (float, (self.problem.nc,)),  # Lagrange multipliers
            'constraints': (float, (self.problem.nc,)),  # Constraint violations
            'merit': float  # Augmented Lagrangian Merit function value
            }

    def setup(self):
        """Set up the genetic algorithm (GA) components and constraint handling."""
        self.setup_constraints()

        # Problem dimensions
        nx = self.nx
        nc = self.nc
        nc_e = self.nc_e

        self.available_outputs['lag_mult'] = (float, (nc,))
        self.available_outputs['constraints'] = (float, (nc,))

        # Initialize Lagrange multipliers and penalty
        self.lambda_vals = np.full(nc, self.options['lambda_init'])
        self.rho = np.full(nc, self.options['rho_init'])

        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian

        

        # Initialize Augmented Lagrangian function
        self.MF = AugmentedLagrangian(nx=nx,
                                      nc=nc,
                                      nc_e=nc_e,
                                      f=self.obj,
                                      c=self.con,
                                      g=self.grad,
                                      j=self.jac)
        self.MF.set_rho(self.rho)

        # Set up DEAP-based genetic algorithm
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform,
                              self.options['rangeLow'], self.options['rangeHigh'])
        self.toolbox.register("individual", tools.initRepeat,
                              creator.Individual, self.toolbox.attribute, n=nx)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=self.options['alpha'])
        self.toolbox.register("mutate", tools.mutGaussian,
                              mu=0, sigma=(self.options['rangeHigh'] - self.options['rangeLow']) * 0.1,
                              indpb=self.options['mutationRateGene'])
        self.toolbox.register("select", lambda inds, k: self.constraint_tournament(inds, k))
        self.toolbox.register("evaluate", self.constraint_evaluate)

    def setup_constraints(self):
        """Adapt constraints into all-inequality form: C_e(x) = 0, C_i(x) >= 0."""

        # Handle variable bounds: x_lower and x_upper
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        self.eq_bound_indices = np.where(xl == xu)[0]
        self.lower_bound_indices = np.where((xl != -np.inf) & (xl != xu))[0]
        self.upper_bound_indices = np.where((xu != np.inf) & (xl != xu))[0]

        self.eq_bounded = len(self.eq_bound_indices) > 0
        self.lower_bounded = len(self.lower_bound_indices) > 0
        self.upper_bounded = len(self.upper_bound_indices) > 0

        # Handle problem constraints
        if self.problem.constrained:
            cl = self.problem.c_lower
            cu = self.problem.c_upper

            self.eq_constraint_indices = np.where(cl == cu)[0]
            self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
            self.upper_constraint_indices = np.where((cu != np.inf) & (cl != cu))[0]
        else:
            self.eq_constraint_indices = np.array([])
            self.lower_constraint_indices = np.array([])
            self.upper_constraint_indices = np.array([])

        self.eq_constrained = len(self.eq_constraint_indices) > 0
        self.lower_constrained = len(self.lower_constraint_indices) > 0
        self.upper_constrained = len(self.upper_constraint_indices) > 0

        # Compute total constraint counts
        self.nc_e = len(self.eq_bound_indices) + len(self.eq_constraint_indices)
        self.nc_i = (
            len(self.lower_bound_indices) + len(self.upper_bound_indices)
            + len(self.lower_constraint_indices) + len(self.upper_constraint_indices)
        )
        self.nc = self.nc_e + self.nc_i  # Total number of constraints

    def con(self, x):
        """
        Return a single array of constraints, c_out, of shape (nc,).
        The first self.nc_e entries correspond to equality constraints (both
        variable bounds and problem constraints).
        The remaining self.nc_i entries correspond to inequality constraints.
        """
        ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices

        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Evaluate the original problem constraints
            c_in = self.con_in(x)
        else:
            # If unconstrained, just return an empty array
            return np.array([])

        # ------------------------
        # 1) Build EQUALITY part
        # ------------------------
        c_eq = np.array([])

        # (a) eq_bounded => x[ebi] == x_lower[ebi]
        if self.eq_bounded:
            c_eq = np.append(c_eq, x[ebi] - self.problem.x_lower[ebi])

        # (b) eq_constrained => c_in[eci] == c_lower[eci]
        if self.eq_constrained:
            c_eq = np.append(c_eq, c_in[eci] - self.problem.c_lower[eci])

        # ------------------------
        # 2) Build INEQUALITY part
        #     We want all c_i(x) >= 0
        # ------------------------
        c_ineq = np.array([])

        # (c) lower_bounded => x[lbi] >= x_lower[lbi]
        if self.lower_bounded:
            c_ineq = np.append(c_ineq, x[lbi] - self.problem.x_lower[lbi])

        # (d) upper_bounded => x[ubi] <= x_upper[ubi], i.e. x_upper[ubi] - x[ubi] >= 0
        if self.upper_bounded:
            c_ineq = np.append(c_ineq, self.problem.x_upper[ubi] - x[ubi])

        # (e) lower_constrained => c_in[lci] >= c_lower[lci]
        if self.lower_constrained:
            c_ineq = np.append(c_ineq, c_in[lci] - self.problem.c_lower[lci])

        # (f) upper_constrained => c_in[uci] <= c_upper[uci], i.e. c_upper[uci] - c_in[uci] >= 0
        if self.upper_constrained:
            c_ineq = np.append(c_ineq, self.problem.c_upper[uci] - c_in[uci])

        # Concatenate eq + ineq
        c_out = np.concatenate([c_eq, c_ineq])

        # c_out should now have shape == self.nc
        return c_out

    def jac(self, x):
        nx = self.nx
        ebi = self.eq_bound_indices
        lbi = self.lower_bound_indices
        ubi = self.upper_bound_indices

        if self.problem.constrained:
            eci = self.eq_constraint_indices
            lci = self.lower_constraint_indices
            uci = self.upper_constraint_indices
            # Compute original problem constraint Jacobian
            j_in = self.jac_in(x)

        j_out = np.empty((1, nx), dtype=float)

        # Handle Jacobians for equality constraints
        if self.eq_bounded:
            j_out = np.append(j_out, np.identity(nx)[ebi], axis=0)
        if self.eq_constrained:
            j_out = np.append(j_out, j_in[eci], axis=0)

        # Convert variable bounds and inequalities
        if self.lower_bounded:
            j_out = np.append(j_out, np.identity(nx)[lbi], axis=0)
        if self.upper_bounded:
            j_out = np.append(j_out, -np.identity(nx)[ubi], axis=0)
        if self.lower_constrained:
            j_out = np.append(j_out, j_in[lci], axis=0)
        if self.upper_constrained:
            j_out = np.append(j_out, -j_in[uci], axis=0)

        return j_out[1:]  # Remove first empty row

    def repair(self, individual):
        # Example: clamp each variable between [rangeLow, rangeHigh]
        for i in range(len(individual)):
            individual[i] = max(self.options['rangeLow'], 
                                min(self.options['rangeHigh'], individual[i]))
        return individual

    def constraint_evaluate(self, individual):
        """Evaluate the individual using the Augmented Lagrangian Merit function."""
        x = np.array(individual, dtype=float)
        f_k = self.obj(x)
        c_k = self.con(x)

        s_k = np.zeros(self.nc - self.nc_e)  # Should match the number of inequality constraints

        scaled_f = f_k * 1e-3

        merit = self.MF.evaluate_function(x, self.lambda_vals, s_k, scaled_f, c_k)
        return (merit,)

    def get_constraint_violation(self, individual):
        """
        Compute a scalar measure of constraint violation.
        - For equality constraints, the violation is |c - 0|
        - For inequality constraints c >= 0, the violation is max(0, -c).
        """
        x = np.array(individual, dtype=float)
        c_vals = self.con(x)
        
        violation = 0.0
        for i in range(self.nc):
            val = c_vals[i]
            if i < self.nc_e:  # Equality constraint
                violation += abs(val)
            else:  # Inequality constraint
                violation += max(0, -val)
    
        return violation

    def constraint_tournament(self, individuals, k):
        """Tournament selection that prefers feasible individuals."""
        chosen = []
        for _ in range(k):
            aspirants = random.sample(individuals, self.options["tournsize"])
            
            # Separate feasible from infeasible individuals
            feasible = [ind for ind in aspirants if self.get_constraint_violation(ind) < 1e-14]
    
            if feasible:
                # Select the best among feasible candidates
                chosen.append(tools.selBest(feasible, 1)[0])
            else:
                # Otherwise, select the one with the least constraint violation
                chosen.append(min(aspirants, key=self.get_constraint_violation))
        
        return chosen

    def update_multipliers(self, population):
        """Update Lagrange multipliers and penalty parameters."""
        violations = np.zeros(self.nc)
        for i in range(self.nc):
            vals_i = [max(0, self.con(np.array(ind, dtype=float))[i]) for ind in population]
            violations[i] = np.mean(vals_i)

        self.lambda_vals += self.rho * violations
        avg_violation = np.mean(violations)
        if avg_violation > self.options['opt_tol']:
            self.rho *= 2

        self.MF.set_rho(self.rho)

    def solve(self):
        """Run the genetic algorithm with Augmented Lagrangian for constrained optimization."""
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']
        start_time = time.time()

        pi_k = np.full((self.nc, ), 0.)

        x_k = x * 1.0
        f_k = self.obj(x_k)
        g_k = self.grad(x_k)

        c_k = self.con(x_k)  # Compute constraints
        s_k = np.zeros(self.nc - self.nc_e)

        itr = 0
        opt = float('inf')

        # Initialize outputs at iteration 0
        self.update_outputs(
            itr=0,
            x=x_k,
            obj=f_k,
            opt=float('inf'),  # Initial best fitness (will update later)
            time=time.time() - start_time,
            lag_mult=self.lambda_vals,
            constraints=self.con(x_k),
            merit=self.MF.evaluate_function(x_k, self.lambda_vals, s_k, f_k, c_k)
        )


        # Generate initial population
        pop = self.toolbox.population(n=self.options['initialPopulationSize'])
        CXPB = self.options['cxProb']
        MUTPB = self.options['mutationRateInd']
        NGEN = maxiter

        # Evaluate initial population using the Augmented Lagrangian
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        while np.abs(opt) > opt_tol and itr < NGEN:
            offspring = self.toolbox.select(pop, len(pop))
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

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Sort population: prioritize feasibility, then fitness
            pop[:] = sorted(offspring, key=lambda ind: (self.get_constraint_violation(ind), ind.fitness.values[0]))

            # Update Lagrange multipliers and penalty parameter
            self.update_multipliers(pop)

            self.rho = np.minimum(self.rho * 2, 1e5)  # Prevent runaway penalty growth

            # Select the best individual using DEAP's `selBest`
            best_ind = tools.selBest(pop, 1)[0]
            f_k = best_ind.fitness.values[0]
            opt = f_k
            x_k = np.array(best_ind)

            itr += 1

            true_obj_value = self.obj(x_k)  # The real objective at best_ind
            al_merit_value = best_ind.fitness.values[0]

            # Update outputs during each iteration
            self.update_outputs(
                itr=itr,
                x=x_k,
                obj=true_obj_value,
                opt=al_merit_value,  # Best fitness found so far
                time=time.time() - start_time,
                lag_mult=self.lambda_vals,
                constraints=self.con(x_k),
                merit=self.MF.evaluate_function(x_k, self.lambda_vals, np.zeros(self.nc - self.nc_e), f_k, self.con(x_k))
            )


        # Store final results after completion
        self.total_time = time.time() - start_time
        self.results = {
            'x': x_k,
            'objective': true_obj_value,
            'optimality': al_merit_value,  # Matches `opt` in outputs
            'itr': itr,
            'time': self.total_time,
            'lag_mult': self.lambda_vals.tolist(),
            'constraints': self.con(x_k),
            'merit': self.MF.evaluate_function(x_k, self.lambda_vals, np.zeros(self.nc - self.nc_e), f_k, self.con(x_k))
        }


        self.run_post_processing()
        return self.results
