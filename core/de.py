import numpy as np
import random
from core.fitness import fitness
from core.fitness import generate_chromosome
from core.experiments import population_diversity

# ENCODE / DECODE

def encode(chromosome):
    # [(x,y), (x,y)] → [x1,y1,x2,y2,...]
    return np.array([coord for point in chromosome for coord in point])


def decode(vector):
    # [x1,y1,x2,y2,...] → [(x,y),(x,y)]
    return [(int(vector[i]), int(vector[i+1]))
            for i in range(0, len(vector), 2)]


# REPAIR FUNCTION

def repair_de(individual, grid_shape, num_sensors):

    n, m = grid_shape

    repaired = []
    for x, y in individual:
        x = int(np.clip(x, 0, n - 1))
        y = int(np.clip(y, 0, m - 1))
        repaired.append((x, y))

    repaired = list(dict.fromkeys(repaired))

    while len(repaired) < num_sensors:
        repaired.append((random.randint(0, n - 1),
                        random.randint(0, m - 1)))

    return repaired[:num_sensors]


# MUTATION (DE)
def de_mutation(population_vectors, i, F=0.5):

    idxs = list(range(len(population_vectors)))
    idxs.remove(i)

    a, b, c = random.sample(idxs, 3)

    x1 = population_vectors[a]
    x2 = population_vectors[b]
    x3 = population_vectors[c]

    mutant = x1 + F * (x2 - x3)

    return mutant




# CROSSOVER (DE)

def de_crossover(target, mutant, CR=0.7):

    trial = np.copy(target)

    for i in range(len(target)):
        if random.random() < CR:
            trial[i] = mutant[i]

    return trial


# DE MAIN LOOP
def run_de(grid,
        pop_size=30,
        num_sensors=10,
        generations=30,
        seed=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = [
        encode(generate_chromosome(num_sensors, grid.shape))
        for _ in range(pop_size)
    ]

    best_history = []
    avg_history = []
    diversity_history = []
    snapshots = []

    best_chromosome = None
    best_fitness_so_far = -float("inf")

    dim = num_sensors * 2

    for gen in range(generations):

        F = 0.3 + 0.4 * random.random()
        CR = 0.6 + 0.3 * (gen / generations)

        new_population = []

        for i in range(pop_size):

            x = population[i]

            idxs = list(range(pop_size))
            idxs.remove(i)

            a, b, c = random.sample(idxs, 3)

            x1 = population[a]
            x2 = population[b]
            x3 = population[c]

            mutant = x1 + F * (x2 - x3)

            trial = np.copy(x)

            for j in range(dim):
                if random.random() < CR:
                    trial[j] = mutant[j]

            trial_chrom = repair_de(decode(trial), grid.shape, num_sensors)
            target_chrom = repair_de(decode(x), grid.shape, num_sensors)

            if fitness(trial_chrom, grid) > fitness(target_chrom, grid):
                new_population.append(trial)
            else:
                new_population.append(x)

        population = new_population

        # FITNESS
        fitnesses = [
            fitness(repair_de(decode(ind), grid.shape, num_sensors), grid)
            for ind in population
        ]

        best = max(fitnesses)
        avg = np.mean(fitnesses)

        best_history.append(best)
        avg_history.append(avg)

        # DIVERSITY
        diversity_history.append(population_diversity(
            [repair_de(decode(ind), grid.shape, num_sensors)
            for ind in population]
        ))

        # BEST UPDATE
        best_idx = np.argmax(fitnesses)

        if fitnesses[best_idx] > best_fitness_so_far:
            best_fitness_so_far = fitnesses[best_idx]
            best_chromosome = repair_de(
                decode(population[best_idx]),
                grid.shape,
                num_sensors
            )

        # SNAPSHOT
        snapshots.append((gen, best_chromosome))

        print(f"[DE] Gen {gen:02d} | Best={best:.4f} | Avg={avg:.4f}")

    return max(best_history), best_history, avg_history, best_chromosome, diversity_history, snapshots