import math
import numpy as np
import setup
import fitnessEvaluator
import mutator
import Crossover

def run_flow(genCount, population, mutation_rate, crossoverRate, crossSplit):
    for i in range(genCount):
        fitnesses = []

        for pop in population:
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
        for weight in normalized_weights:
            if weight > 0.05:
                normalized_weights[normalized_weights.index(weight)] = weight - 0.04
        next_generation = []

        for i in range(populationCount):
            parent_idxs = np.random.choice(populationCount, size=1, replace=True, p=normalized_weights)
            p1 = population[parent_idxs[0]]
            child = p1.copy()
            next_generation.append(child)

        next_generation = mutator.mutate(next_generation, mutation_rate)
        next_generation = Crossover.crossover(next_generation,crossoverRate,crossSplit)
        population = next_generation

    final_fitness = []
    for pop in population:
        fitness = fitnessEvaluator.evaluateFitness(pop)
        final_fitness.append(fitness)

    result = np.mean(final_fitness)
    return result

if __name__ == '__main__':
    populationCount = 100
    individualLength = 10
    generationCount = 10
    mutationRate = [0.01,0.05,0.1,0.2]
    crossover_rate = [0.01,0.05,0.1,0.2]

    generation = setup.setup_population(populationCount, individualLength)
    means = []
    for mutationR in mutationRate:
        for crossoverR in crossover_rate:
            mean1 = run_flow(generationCount, generation, mutationR, crossoverR,True)
            mean2 = run_flow(generationCount, generation, mutationR, crossoverR,False)
            means.append(mean1)
            means.append(mean2)

    # find index of best mean and map back to hyperparameters
    max_idx = int(np.argmax(means))  # flattened index in means
    m_len = len(mutationRate)
    c_len = len(crossover_rate)

    pair_idx = max_idx // 2  # index among (mutation,crossover) pairs
    split_flag = max_idx % 2  # 0 => crossSplit True (first appended), 1 => False

    i_mut = pair_idx // c_len
    i_cross = pair_idx % c_len

    best_mutation = mutationRate[i_mut]
    best_crossover = crossover_rate[i_cross]
    best_crossSplit = True if split_flag == 0 else False

    print("max mean index:", max_idx)
    print("max mean:", np.max(means))
    print("best mutationRate:", best_mutation)
    print("best crossover_rate:", best_crossover)
    print("best crossSplit:", best_crossSplit)
