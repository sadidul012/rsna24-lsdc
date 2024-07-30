import numpy as np
import tqdm

from single_test import calculate_condition_metrics
from single_train import set_random_seed, SEED

set_random_seed(SEED)


# Define the fitness function (negative of the objective function)
def maximize_fitness(individual):
    # print(individual, individual ** 2, -np.sum(individual ** 2))
    # print(np.sum(ground_truth - individual))
    return np.sum(individual)


# Generate an initial population
def generate_population(size, dim):
    return np.random.rand(size, dim)


# Genetic algorithm
def genetic_algorithm(population, fitness_func, n_generations=100, mutation_rate=0.01):
    for _ in tqdm.tqdm(range(n_generations)):
        population = sorted(population, key=fitness_func, reverse=True)
        next_generation = population[:len(population) // 2].copy()
        while len(next_generation) < len(population):
            parents_indices = np.random.choice(len(next_generation), 2, replace=False)
            parent1, parent2 = next_generation[parents_indices[0]], next_generation[parents_indices[1]]
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            if np.random.rand() < mutation_rate:
                mutate_point = np.random.randint(len(child))
                child[mutate_point] = np.random.rand()
            next_generation.append(child)
        population = np.array(next_generation)
    return population[0]


def run_genetic_algorithm(fitness_func, population_size=10, dimension=3, n_generations=100, mutation_rate=0.01):
    # Initialize population
    population = generate_population(population_size, dimension)
    # Run genetic algorithm
    best_individual = genetic_algorithm(population, fitness_func, n_generations, mutation_rate)

    print("Best individual:", best_individual)
    print("Best fitness:", -fitness_func(best_individual))  # Convert back to positive for the objective value
    return best_individual


def optimize(solution, _submission, condition, value_column, population_size=100, n_generations=10, mutation_rate=0.3):
    def fitness_function(individual):
        _sub = _submission.copy()
        _sub[value_column] = _sub[value_column].values * individual
        accuracy, _ = calculate_condition_metrics(solution, _sub, condition, True)
        return accuracy

    best = run_genetic_algorithm(fitness_function, population_size=population_size, n_generations=n_generations, mutation_rate=mutation_rate)
    tuning_sub = _submission.copy()
    tuning_sub[value_column] = tuning_sub[value_column].values * best
    acc, c_m = calculate_condition_metrics(solution, tuning_sub, condition, True)
    print("Accuracy", acc)

    return c_m


def main():
    best = run_genetic_algorithm(maximize_fitness, 100, dimension=3, n_generations=100, mutation_rate=0.01)
    print(np.sum(best), best)


if __name__ == '__main__':
    main()
