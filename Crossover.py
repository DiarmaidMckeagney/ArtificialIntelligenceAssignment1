import random

def crossover(population, crossover_rate, isEqualSplit):
    for i in range(0, len(population) // 2):
        if random.random() < crossover_rate:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            crossover_point = len(parent1) // 2 if isEqualSplit else random.randint(1, len(parent1) - 1)

            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            population[population.index(parent1)] = child1
            population[population.index(parent2)] = child2
    return population
