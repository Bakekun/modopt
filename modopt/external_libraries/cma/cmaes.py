import time
import numpy as np
import cma 
from modopt import Optimizer

class CMAES(Optimizer):
    def initialize(self):
        """Initialize CMA-ES parameters and declare options."""
        self.solver_name = "CMA-ES"
        # Assume a single-objective function that returns a tuple (we use index 0)
        self.obj = self.problem._compute_objective  
        self.options.declare('maxiter', default=300, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)
        self.options.declare('init_sigma', default=0.5, types=float)
        self.options.declare('popsize', default=50, types=int)
        
        # Ensure modOpt's output options are declared
        self.options.declare('readable_outputs', types=list, default=[])
        self.available_outputs = {
            'itr': int,
            'obj': float,
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
        }

    def setup(self):
        """Set up the CMA-ES strategy using the cma package."""
        dim = self.problem.nx
        # If no initial guess is provided, default to a zero vector.
        if self.problem.x0 is None:
            self.problem.x0 = np.zeros(dim)
        # Initialize CMA-ES; note that CMAEvolutionStrategy expects a list.
        self.es = cma.CMAEvolutionStrategy(self.problem.x0.tolist(),
                                           self.options['init_sigma'],
                                           {'popsize': self.options['popsize']})

    def solve(self):
        """Run the CMA-ES optimization."""
        start_time = time.time()
        itr = 0
        best_f = float('inf')
        best_x = None

        # Main optimization loop
        while not self.es.stop() and itr < self.options['maxiter']:
            # Generate a batch of candidate solutions
            solutions = self.es.ask()
            # Evaluate each candidate.
            # (Assuming self.obj returns a tuple; we take the first element.)
            fitnesses = [float(self.obj(sol)) if np.isscalar(self.obj(sol)) else self.obj(sol)[0] for sol in solutions]
            # Update the distribution with the evaluated fitnesses
            self.es.tell(solutions, fitnesses)
            itr += 1
            # Track the best solution so far (for logging and optimality measure)
            current_best = min(fitnesses)
            if current_best < best_f:
                best_f = current_best
                best_x = solutions[np.argmin(fitnesses)]
            # Log outputs (update modOpt outputs)
            self.update_outputs(
                itr=itr,
                x=best_x,  # best_x is a list (the candidate solution)
                obj=best_f,
                opt=best_f,
                time=float(time.time() - start_time)
            )
        self.total_time = time.time() - start_time

        # Store final results
        self.results = {
            'x': best_x,
            'objective': best_f,
            'optimality': best_f,
            'itr': itr,
            'time': self.total_time
        }
        self.run_post_processing()
        return self.results