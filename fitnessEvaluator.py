def evaluateFitness(obj):
    # Evaluate the fitness of a given object
    fitness_score = 0

    for i in range(len(obj)):
        fitness_score += obj[i]

    return fitness_score / len(obj)