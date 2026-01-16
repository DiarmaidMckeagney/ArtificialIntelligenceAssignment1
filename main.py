import math
import numpy as np
import setup
import fitnessEvaluator
import mutator
import Crossover

if __name__ == '__main__':
    populationCount = 100
    individualLength = 10
    generationCount = 10
    mutationRate = 0.05
    crossover_rate = 0.10

    generation = setup.setup_population(populationCount, individualLength)

    for i in range(generationCount):
        fitnesses = []

        for pop in generation:
            fitness = fitnessEvaluator.evaluateFitness(pop)
            fitnesses.append(fitness)

        fitnesses = np.array(fitnesses, dtype=float)
        fitness_mean = np.mean(fitnesses)
        fitness_std = np.std(fitnesses)
        if fitness_std == 0:
            z_scores = np.zeros_like(fitnesses)
        else:
            z_scores = (fitnesses - fitness_mean) / fitness_std

        # Map z-scores to survival probabilities using the normal CDF:
        erf_vec = np.vectorize(math.erf)
        survival_prob = 0.5 * (1.0 + erf_vec(z_scores / np.sqrt(2.0)))
        weights = np.asarray(survival_prob, dtype=float)
        sum_weights = np.sum(weights)
        normalized_weights = (weights / sum_weights)
        for weights in normalized_weights:
            if weights > 0.05:
                weights = weights - 0.04
        next_generation = []

        for i in range(populationCount):
            parent_idxs = np.random.choice(populationCount, size=1, replace=True, p=normalized_weights)
            p1 = generation[parent_idxs[0]]
            child = p1.copy()
            next_generation.append(child)

        next_generation = mutator.mutate(next_generation, mutationRate)
        next_generation = Crossover.crossover(next_generation,crossover_rate,True)
        generation = next_generation

    print(generation)