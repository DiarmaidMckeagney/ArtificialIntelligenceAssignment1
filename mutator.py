import random

def mutate(population, mutation_rate):
    for individual in population:
        if random.random() < mutation_rate:
            for value in range(len(individual)):
                if random.random() < len(individual) / 100:
                    individual[value] = 1 - individual[value]
    print("mutated")
    return population