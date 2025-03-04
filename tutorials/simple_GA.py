""" 
Tristan Pham
MAE 270 Project
2/26/25

This is a basic implementation of a Gentic algorithm for 2d optimization. 
Factors to consider are using a mix of real, discrete, and integer design variables, 
using a high number (>1000) of design variables, how to handle boundless design variables,
and handling various types of optimization problems.
"""

import numpy as np
from scipy.stats.qmc import Sobol
import time

class simpleGA:
    Optimization_name = "Simple Genetic Algorithm"

    def __init__(self, objective, initialPopulationSize, rangeLow, rangeHigh, numDesignVars, mutationRate, mutationStd, generations):
        self.objective = objective
        self.initialPopulationSize = initialPopulationSize # Must be an even number
        self.rangeLow = rangeLow
        self.rangeHigh = rangeHigh
        self.mutationRate = mutationRate
        self.numDesignVars = numDesignVars
        self.generations = generations
        self.mutationStd = mutationStd                     # Can be scaled to the variable range or reduced over generations for fine tuning

    def Populate(self):
        n = self.initialPopulationSize

        # Generate Sobol sequence in [0,1]
        qrng = Sobol(d=self.numDesignVars, scramble=True)
        matrix = qrng.random(n)

        # Scale to [-4,4]
        population = self.rangeLow + (self.rangeHigh - self.rangeLow) * matrix
        return population  
    
    def Evaluate(self, population):
        Evaluation = np.apply_along_axis(lambda row: self.objective(*row), 1, population)
        return Evaluation

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

        mutation_mask = np.random.rand(N, D) < self.mutationRate

        gaussian_noise = np.random.normal(0, self.mutationStd, (N, D))
        mutatedPopulation = np.copy(newPopulation)
        mutatedPopulation += mutation_mask * gaussian_noise

        mutatedPopulation = np.clip(mutatedPopulation, self.rangeLow, self.rangeHigh)

        return newPopulation

    def Optimize(self):
        population = self.Populate()

        count = 0

        while count < self.generations:
            elites = self.Fitness_Selection(population)

            offspring = self.Crossover(elites)

            population = self.Mutation(elites,offspring)

            count += 1

        final_evaluations = self.Evaluate(population)
        optimal_index = np.argmin(final_evaluations)
        optimality = final_evaluations[optimal_index]
        optimal_design = population[optimal_index]
        return optimality, optimal_design

if __name__ == '__main__':

    Rosenbrock2D = lambda x,y: (1-x)**2 + 100*(y-x**2)**2
    Goldstien_Price = lambda x,y: (1+((x+y+1)**2)*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+((2*x-3*y)**2)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    Himmelblau = lambda x,y: (x**2+y-11)**2+(x+y**2-7)**2+0.1*x + 0.1*y
    McCormick = lambda x,y: np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5 *y +1
    Matyas = lambda x,y: 0.26*(x**2+y**2)-0.48*x*y
    Beale = lambda x,y: (1.5-x+x*y)**2 + (2.25 -x +x*y**2)**2 + (2.625-x+x*y**3)**2
    Ackley = lambda x,y: -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20

    myGA_Rosenbrock = simpleGA(Rosenbrock2D, 2**9, -4, 4, 2, 0.2, 1, 5000)
    start_time = time.time()
    optimality, optimal_design = myGA_Rosenbrock.Optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    print(optimality)
    print(optimal_design)

    myGA_Gold = simpleGA(Goldstien_Price, 2**9, -4, 4, 2, 0.2, 1, 5000)
    start_time = time.time()
    optimality, optimal_design = myGA_Gold.Optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    print(optimality)
    print(optimal_design)

    myGA_Himme = simpleGA(Himmelblau, 2**9, -4, 4, 2, 0.2, 1, 5000)
    start_time = time.time()
    optimality, optimal_design = myGA_Himme.Optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    print(optimality)
    print(optimal_design)