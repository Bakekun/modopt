from deap import base, creator, tools
import random

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

IND_SIZE = 2

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, -5, 5)  # Adjusted range
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    x, y = individual
    objective1 = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    objective2 = x**2 + y**2
    return (objective1, objective2)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=200)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 700

    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)


    for g in range(NGEN):
        offspring = list(map(toolbox.clone, pop))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # The population is entirely replaced by the offspring
        combined_pop = pop + offspring

        pop[:] = toolbox.select(combined_pop, len(pop))

    # Retrieve and print the best individual found
    best_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    print("\nPareto Front Solutions:")
    for ind in best_inds:
        print(f"Solution: {ind}, Fitness: {ind.fitness.values}")


    return pop

if __name__ == "__main__":
    main()