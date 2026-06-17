import numpy as np
import random
from joblib import Parallel, delayed

from core.fitness import fitness


# HELPERS

def get_coverage_map(chromosome, grid, R):
    n, m = grid.shape
    covered = set()

    for (si, sj) in chromosome:
        for i in range(max(0, si - R), min(n, si + R + 1)):
            for j in range(max(0, sj - R), min(m, sj + R + 1)):
                if np.linalg.norm([si - i, sj - j]) <= R:
                    covered.add((i, j))

    return covered


def coverage_ratio(chromosome, grid, R):
    covered = get_coverage_map(chromosome, grid, R)
    total_cells = grid.shape[0] * grid.shape[1]
    return len(covered) / total_cells


def population_diversity(population):
    arr = np.array([np.array(ind).flatten() for ind in population])
    return np.mean(np.std(arr, axis=0))


# PARALLEL RUNNERS

def single_run_ga_full(
    grid, sel, cross, mut, div, init, surv,
    sensors, algo, seed, run_ga
):

    best, hist, avg, best_chrom, div_hist, snapshots = run_ga(
        grid,
        selection_method=sel,
        crossover_type=cross,
        mutation_type=mut,
        diversity_type=div,
        survivor_method=surv,
        init_method=init,
        num_sensors=sensors,
        use_de=(algo == "HYBRID"),
        seed=seed
    )

    cov = coverage_ratio(best_chrom, grid, R=5)

    return best, hist, div_hist, snapshots, cov


def single_run_de_full(grid, sensors, seed, run_de):

    best, hist, avg, best_chrom, div_hist, snapshots = run_de(
        grid,
        num_sensors=sensors,
        seed=seed
    )

    cov = coverage_ratio(best_chrom, grid, R=5)

    return best, hist, div_hist, snapshots, cov


# MAIN EXPERIMENT FUNCTION

def run_experiments(grid, run_ga, run_de, runs=10):

    results = []

    selection_methods = ["tournament", "roulette", "sus", "over"]
    crossover_types = ["onepoint", "uniform"]
    mutation_types = ["shift", "reset"]
    diversity_types = ["basic", "strong"]

    init_methods = ["random", "heuristic"]
    survivor_methods = ["elitism", "rank"]
    sensor_values = [5, 10, 15]

    algorithms = ["GA", "DE", "HYBRID"]

    all_histories = {"GA": [], "DE": [], "HYBRID": []}
    all_snapshots = {"GA": [], "DE": [], "HYBRID": []}
    all_diversity = {"GA": [], "DE": [], "HYBRID": []}

    for algo in algorithms:

        print("\n" + "#" * 70)
        print(f"ALGORITHM: {algo}")
        print("#" * 70)

        # GA & HYBRID

        if algo in ["GA", "HYBRID"]:

            for sel in selection_methods:
                for cross in crossover_types:
                    for mut in mutation_types:
                        for div in diversity_types:
                            for init in init_methods:
                                for surv in survivor_methods:
                                    for sensors in sensor_values:

                                        print(f"\nCONFIG: {sel}|{cross}|{mut}|{div}|{init}|{surv}|S={sensors}")

                                        seeds = [random.randint(0, 100000) for _ in range(runs)]

                                        results_parallel = Parallel(n_jobs=-1)(
                                            delayed(single_run_ga_full)(
                                                grid, sel, cross, mut, div, init, surv,
                                                sensors, algo, seed, run_ga
                                            )
                                            for seed in seeds
                                        )

                                        scores = [r[0] for r in results_parallel]
                                        hist_collection = [r[1] for r in results_parallel]
                                        div_collection = [r[2] for r in results_parallel]
                                        snapshots_collection = [r[3] for r in results_parallel]

                                        avg_hist = np.mean(hist_collection, axis=0)

                                        all_histories[algo].append(avg_hist)
                                        all_diversity[algo].append(np.mean(div_collection, axis=0))

                                        all_snapshots[algo].append({
                                            "config": (sel, cross, mut, div, init, surv, sensors),
                                            "runs": snapshots_collection
                                        })

                                        print(f"Mean = {np.mean(scores):.4f} | Max = {np.max(scores):.4f}")

                                        results.append({
                                            "algorithm": algo,
                                            "selection": sel,
                                            "crossover": cross,
                                            "mutation": mut,
                                            "diversity": div,
                                            "init": init,
                                            "survivor": surv,
                                            "sensors": sensors,
                                            "mean": np.mean(scores),
                                            "std": np.std(scores),
                                            "max": np.max(scores),
                                            "seeds": seeds
                                        })

        # DE ONLY

        else:

            for sensors in sensor_values:

                print(f"\nCONFIG: DE | Sensors={sensors}")

                seeds = [random.randint(0, 100000) for _ in range(runs)]

                results_parallel = Parallel(n_jobs=-1)(
                    delayed(single_run_de_full)(
                        grid, sensors, seed, run_de
                    )
                    for seed in seeds
                )

                scores = [r[0] for r in results_parallel]
                hist_collection = [r[1] for r in results_parallel]
                div_collection = [r[2] for r in results_parallel]

                avg_hist = np.mean(hist_collection, axis=0)

                all_histories[algo].append(avg_hist)
                all_diversity[algo].append(np.mean(div_collection, axis=0))

                print(f"Mean = {np.mean(scores):.4f} | Max = {np.max(scores):.4f}")

                results.append({
                    "algorithm": "DE",
                    "sensors": sensors,
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "max": np.max(scores),
                    "seeds": seeds
                })

    return results, all_histories, all_diversity, all_snapshots