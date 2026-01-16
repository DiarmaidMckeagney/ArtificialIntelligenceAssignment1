import random

def setup_population(populationCount: int, numOfOnes: int) -> list:
    population = []

    for x in range(populationCount):
        pop_values = []
        for i in range(numOfOnes):
            randomNumber = random.randint(1, 10000)
            if randomNumber % 2 == 0:
                pop_values.append(0)
            else:
                pop_values.append(1)

        population.append(pop_values)

    return population