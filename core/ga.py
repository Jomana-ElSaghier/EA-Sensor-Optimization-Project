import numpy as np
from core.operators import *
from config import PARAMS
from core.fitness import fitness
from core.experiments import population_diversity
from core.operators import (
    random_init,
    heuristic_init,
    tournament_selection,
    roulette_selection,
    sus_selection,
    over_selection,
    elitism,
    rank_based_survivor,
    evaluate_population,
    adaptive_tournament_k,
    repair,
    one_point_crossover,
    uniform_crossover,
    random_shift,
    random_reset,
    diversity_basic,
    diversity_strong,
    adaptive_crossover,
    adaptive_mutation,
    generate_chromosome
)
from core.de import (
    encode,
    decode,
    de_mutation,
    de_crossover,
    repair_de
)

# 5] GA LOOP

def log_parameters(gen, mutation_rate, crossover_rate, k):

    print(f"""
    Generation {gen}
    Mutation Rate   : {mutation_rate:.3f}
    Crossover Rate  : {crossover_rate:.3f}
    Tournament K    : {k}
    """)


def run_ga(
        grid,
        pop_size=30,
        num_sensors=10,
        generations=30,

        selection_method="tournament",
        crossover_type="uniform",
        mutation_type="shift",
        diversity_type="basic",
        survivor_method="elitism",
        init_method="heuristic",
        # USER CONTROLLED RATES
        crossover_rate=0.7,
        mutation_rate=0.1,

        # OPTIONAL DE INTEGRATION
        use_de=False,
        de_F=0.5,
        de_CR=0.7,
        de_apply_rate=0.5,

        seed=None
):

    # SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # INIT
    if init_method == "random":
        population = random_init(pop_size, num_sensors, grid.shape)
    else:
        population = heuristic_init(pop_size, num_sensors, grid)

    snapshots = []
    best_history = []
    avg_history = []
    diversity_history = []
    best_chromosome = None

    elite_size = 4

    # MAIN LOOP
    for gen in range(generations):

        # HYBRID CONTROL (USER + ADAPTIVE)
        adaptive_mut = adaptive_mutation(gen, generations)
        adaptive_cross = adaptive_crossover(gen, generations)

        mutation_rate_gen = mutation_rate * adaptive_mut
        crossover_rate_gen = crossover_rate * adaptive_cross

        k = max(2, min(10, adaptive_tournament_k(gen, generations) + 2))

        log_parameters(gen, mutation_rate_gen, crossover_rate_gen, k)

        # FITNESS
        fitnesses = evaluate_population(population, grid, R=PARAMS["radius"])

        # SELECTION
        if selection_method == "tournament":
            parents = tournament_selection(population, fitnesses, k)
        elif selection_method == "roulette":
            parents = roulette_selection(population, fitnesses)
        elif selection_method == "sus":
            parents = sus_selection(population, fitnesses)
        elif selection_method == "over":
            parents = over_selection(population, fitnesses)
        else:
            parents = tournament_selection(population, fitnesses, k)

        # ELITES
        current_elites = elitism(population, fitnesses, elite_size)

        # OFFSPRING
        offspring_generated = []

        while len(offspring_generated) < pop_size - elite_size:

            if len(parents) < 2:
                p1 = random.choice(parents) if parents else generate_chromosome(num_sensors, grid.shape)
                p2 = random.choice(parents) if parents else generate_chromosome(num_sensors, grid.shape)

                if len(parents) == 1:
                    p2 = random.choice(population)
            else:
                p1, p2 = random.sample(parents, 2)

            # CROSSOVER
            if random.random() < crossover_rate_gen:
                if crossover_type == "onepoint":
                    child = one_point_crossover(p1, p2, num_sensors, grid)
                else:
                    child = uniform_crossover(p1, p2, num_sensors, grid)
            else:
                child = p1.copy()

            # MUTATION
            if mutation_type == "shift":
                child = random_shift(child, grid, mutation_rate_gen)
            else:
                child = random_reset(child, grid, mutation_rate_gen)

            offspring_generated.append(child)

        # NEXT POPULATION
        candidate_next_population = offspring_generated + current_elites

        # DIVERSITY CONTROL
        if diversity_type == "basic":
            candidate_next_population = diversity_basic(candidate_next_population)
        else:
            candidate_next_population = diversity_strong(candidate_next_population)

        # fill population if needed
        while len(candidate_next_population) < pop_size:
            if parents:
                candidate_next_population.append(random.choice(parents))
            else:
                candidate_next_population.append(generate_chromosome(num_sensors, grid.shape))

        # SURVIVOR SELECTION
        if survivor_method == "elitism":
            population = candidate_next_population[:pop_size]

        elif survivor_method == "rank":
            candidate_fitnesses = evaluate_population(candidate_next_population, grid, R=PARAMS["radius"])
            population = rank_based_survivor(candidate_next_population, candidate_fitnesses)[:pop_size]

        else:
            population = candidate_next_population[:pop_size]

        # OPTIONAL DE INTEGRATION
        if use_de:
            vector_pop = [encode(ind) for ind in population]

            if len(vector_pop) >= 4:
                n_de = int(de_apply_rate * len(vector_pop))
                n_de = min(n_de, len(vector_pop))

                for i in random.sample(range(len(vector_pop)), n_de):
                    mutant = de_mutation(vector_pop, i, de_F)
                    trial = de_crossover(vector_pop[i], mutant, de_CR)

                    trial_chrom = repair_de(decode(trial), grid.shape, num_sensors)
                    target_chrom = repair_de(decode(vector_pop[i]), grid.shape, num_sensors)

                    if fitness(trial_chrom, grid) > fitness(target_chrom, grid):
                        vector_pop[i] = trial

                population = [
                    repair_de(decode(v), grid.shape, num_sensors)
                    for v in vector_pop
                ]


        # DIVERSITY METRIC

        div_val = population_diversity(population)
        diversity_history.append(div_val)

        print(f"Diversity = {div_val:.4f}")

        # FINAL EVALUATION
        fitnesses = evaluate_population(population, grid, R=PARAMS["radius"])

        best_gen_fitness = max(fitnesses)
        avg_gen_fitness = np.mean(fitnesses)
        best_gen_idx = np.argmax(fitnesses)

        if not best_history or best_gen_fitness > max(best_history):
            best_chromosome = population[best_gen_idx]

        best_history.append(best_gen_fitness)
        avg_history.append(avg_gen_fitness)

        # snapshots
        if gen in [0, generations // 2, generations - 1]:
            snapshots.append((gen, best_chromosome))

        print(
            f"[GA{'+DE' if use_de else ''}] Gen {gen:02d} | "
            f"Best={best_gen_fitness:.4f} | Avg={avg_gen_fitness:.4f}"
        )

    # RETURN
    return (
        max(best_history),
        best_history,
        avg_history,
        best_chromosome,
        diversity_history,
        snapshots
    )