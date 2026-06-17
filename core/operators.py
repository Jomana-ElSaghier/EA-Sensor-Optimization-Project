import random
import numpy as np
from core.fitness import fitness
# INITIALIZATION

def random_init(pop_size, num_sensors, grid_shape):
    n, m = grid_shape

    return [
        [(random.randint(0, n-1), random.randint(0, m-1))
         for _ in range(num_sensors)]
        for _ in range(pop_size)
    ]


def heuristic_init(pop_size, num_sensors, grid):

    flat = []

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            flat.append((i, j, grid[i][j]))

    flat.sort(key=lambda x: x[2], reverse=True)

    top_cells = [(i, j) for i, j, _ in flat[:100]]

    population = []

    for _ in range(pop_size):
        chromosome = random.sample(top_cells, num_sensors)
        population.append(chromosome)

    return population


# SELECTION METHODS

def tournament_selection(population, fitnesses, k=3):

    selected = []

    for _ in range(len(population)):

        idx = random.sample(range(len(population)), k)

        best = idx[0]

        for i in idx:
            if fitnesses[i] > fitnesses[best]:
                best = i

        selected.append(population[best])

    return selected


def roulette_selection(population, fitnesses):

    min_fit = min(fitnesses)
    shifted = [f - min_fit + 1e-6 for f in fitnesses]

    total = sum(shifted)

    if total == 0:
        return random.choices(population, k=len(population))

    probs = [f / total for f in shifted]

    return random.choices(population, weights=probs, k=len(population))


def sus_selection(population, fitnesses):

    min_fit = min(fitnesses)
    shifted = [f - min_fit + 1e-6 for f in fitnesses]

    total = sum(shifted)

    if total == 0:
        return random.sample(population, len(population))

    probs = [f / total for f in shifted]

    cum = np.cumsum(probs)

    N = len(population)
    start = random.random() / N
    pointers = [start + i*(1/N) for i in range(N)]

    selected = []

    for p in pointers:
        for i in range(len(cum)):
            if p <= cum[i]:
                selected.append(population[i])
                break

    return selected


def over_selection(population, fitnesses, top_ratio=0.5, elite_pressure=0.8):

    idx = np.argsort(fitnesses)[::-1]
    sorted_pop = [population[i] for i in idx]

    split = int(len(population) * top_ratio)

    top_group = sorted_pop[:split]
    bottom_group = sorted_pop[split:]

    selected = []

    for _ in range(len(population)):

        if random.random() < elite_pressure:
            selected.append(random.choice(top_group))
        else:
            selected.append(random.choice(bottom_group))

    return selected



# ELITISM


def elitism(population, fitnesses, elite_size=2):

    idx = np.argsort(fitnesses)[::-1]

    elite = [population[i] for i in idx[:elite_size]]

    return elite



# SURVIVOR SELECTION


def rank_based_survivor(population, fitnesses):

    idx = np.argsort(fitnesses)
    sorted_pop = [population[i] for i in idx]

    weights = np.arange(1, len(population)+1)
    probs = weights / sum(weights)

    selected_idx = np.random.choice(
        len(population),
        size=len(population),
        p=probs
    )

    return [sorted_pop[i] for i in selected_idx]


# UTILITIES


def evaluate_population(population, grid, R=5):

    return [fitness(ind, grid, R) for ind in population]


def adaptive_tournament_k(gen, generations):

    return max(2, int(5 * (1 - gen / generations)))

# REPAIR FUNCTION


def repair(individual, grid_shape, num_sensors):

    n, m = grid_shape

    repaired = [(int(x), int(y)) for x, y in individual]

    repaired = list(dict.fromkeys(repaired))  # remove duplicates only

    while len(repaired) < num_sensors:
        repaired.append(random.choice(repaired))

    return repaired[:num_sensors]


# CROSSOVER
def one_point_crossover(p1, p2, num_sensors, grid):

    point = num_sensors // 2

    child = p1[:point] + p2[point:]

    return repair(child, grid.shape, num_sensors)


def uniform_crossover(p1, p2, num_sensors, grid):

    child = []

    for i in range(num_sensors):

        gene = p1[i] if random.random() < 0.5 else p2[i]

        if gene not in child:
            child.append(gene)

    return repair(child, grid.shape, num_sensors)


# MUTATION

def random_shift(individual, grid, mutation_rate=0.2):

    new_ind = individual.copy()

    for i in range(len(new_ind)):

        if random.random() < mutation_rate:

            x, y = new_ind[i]

            nx = max(0, min(grid.shape[0] - 1, x + random.randint(-2, 2)))
            ny = max(0, min(grid.shape[1] - 1, y + random.randint(-2, 2)))

            new_ind[i] = (nx, ny)

    return repair(new_ind, grid.shape, len(individual))


def random_reset(individual, grid, mutation_rate=0.2):

    new_ind = individual.copy()

    for i in range(len(new_ind)):

        if random.random() < mutation_rate:

            nx = random.randint(0, grid.shape[0] - 1)
            ny = random.randint(0, grid.shape[1] - 1)

            new_ind[i] = (nx, ny)

    return repair(new_ind, grid.shape, len(individual))


# DIVERSITY

def diversity_basic(population, threshold=0.8):

    new_pop = []

    for ind in population:

        keep = True

        for kept in new_pop:

            similarity = len(set(ind) & set(kept)) / len(ind)

            if similarity > threshold:
                keep = False
                break

        if keep:
            new_pop.append(ind)

    return new_pop if len(new_pop) > 0 else population


def diversity_strong(population, threshold=0.6):

    new_pop = []

    for ind in population:

        keep = True

        for kept in new_pop:

            similarity = len(set(ind) & set(kept)) / len(ind)

            if similarity > threshold:
                keep = False
                break

        if keep:
            new_pop.append(ind)

    return new_pop if len(new_pop) >= 2 else population


# ADAPTIVE CONTROL

def adaptive_crossover(gen, generations):

    start = 0.9
    end = 0.5

    return start - (start - end) * (gen / generations)


def adaptive_mutation(gen, generations):

    start = 0.2
    end = 0.1

    return start - (start - end) * (gen / generations)


def generate_chromosome(num_sensors, grid_shape):
    n, m = grid_shape
    return [
        (random.randint(0, n - 1), random.randint(0, m - 1))
        for _ in range(num_sensors)
    ]