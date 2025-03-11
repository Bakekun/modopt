import numpy as np
import time
from modopt import Optimizer
from scipy.stats.qmc import Sobol

class SimpleGA(Optimizer):

    def initialize(self):
        # Name your algorithm
        self.solver_name = 'Simple_Genetic_Algorithm'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)
        self.options.declare('initialPopulationSize', default = 2**9, types=int)
        self.options.declare('rangeLow', default = -4.0, types = float)
        self.options.declare('rangeHigh', default= 4.0, types = float)
        self.options.declare('mutationRate', default = 0.2, types = float)
        self.options.declare('mutationStd', default = self.options['rangeHigh'] - self.options['rangeHigh'], types = float)

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
        # Instantiate any modules you need for the algorithm
        pass

    def Populate(self):
        n = self.options['initialPopulationSize']

        # Generate Sobol sequence in [0,1]
        qrng = Sobol(d=self.problem.nx, scramble=True)
        matrix = qrng.random(n)

        # Scale to [-4,4]
        population = self.options['rangeLow'] + (self.options['rangeHigh']- self.options['rangeLow']) * matrix
        return population  
    
    def Evaluate(self, population):
        return np.array([self.obj(ind) for ind in population])

    def Fitness_Selection(self,population):
        N, D = population.shape 
        Evaluation = self.Evaluate(population)
        
        population_with_scores = np.column_stack((population, Evaluation))
        population_with_scores = population_with_scores[population_with_scores[:, -1].argsort()]
        
        num_elites = N // 2  # Select half the population
        tournament_size = max(2, N // 5)  # Set tournament size

        selected_indices = []
        for _ in range(num_elites):
            tournament_contestants = np.random.choice(N, tournament_size, replace=False)
            best_index = tournament_contestants[np.argmin(Evaluation[tournament_contestants])]
            selected_indices.append(best_index)

        elites = population[selected_indices]

        return elites 

    def Crossover(self,elites):
        np.random.shuffle(elites)
        N = len(elites)

        offspring = []

        for i in range(0,N,2):
            if i + 1 < N:
                parent1, parent2 = elites[i], elites[i+1]

                var_index = np.random.choice([0,1])

                alpha1 = np.random.uniform(0, 1)
                alpha2 = np.random.uniform(0, 1)

                child1, child2 = parent1.copy(), parent2.copy()
                child1[var_index] = alpha1 * parent1[var_index] + (1 - alpha1) * parent2[var_index]
                child2[var_index] = alpha2 * parent1[var_index] + (1 - alpha2) * parent2[var_index]

                offspring.extend([child1, child2])

        return offspring

    def Mutation(self, elites, offspring):
        newPopulation = np.vstack((elites,offspring))

        N, D = newPopulation.shape

        mutation_mask = np.random.rand(N, D) < self.options['mutationRate']

        gaussian_noise = np.random.normal(0, self.options['mutationStd'], (N, D))
        mutatedPopulation = np.copy(newPopulation)
        mutatedPopulation += mutation_mask * gaussian_noise

        mutatedPopulation = np.clip(mutatedPopulation, self.options['rangeLow'], self.options['rangeHigh'])

        return newPopulation

    def solve(self):
        nx = self.problem.nx
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

        population = self.Populate()

        while (opt > opt_tol and itr < maxiter):
            itr_start = time.time()
            itr += 1
            elites = self.Fitness_Selection(population)

            offspring = self.Crossover(elites)

            population = self.Mutation(elites,offspring)

            final_evaluations = self.Evaluate(population)
            optimal_index = np.argmin(final_evaluations)
            f_k = final_evaluations[optimal_index]
            opt = f_k
            x_k = population[optimal_index]

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