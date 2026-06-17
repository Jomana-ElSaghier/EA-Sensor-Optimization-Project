import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import combinations

# COVERAGE FUNCTION
def coverage(chromosome, grid, R):

    n, m = grid.shape

    covered = set()
    coverage_count = [[0] * m for _ in range(n)]
    sensor_covering = {}

    for (si, sj) in chromosome:

        sensor_cells = set()

        for i in range(max(0, si - R), min(n, si + R + 1)):
            for j in range(max(0, sj - R), min(m, sj + R + 1)):

                d = ((si - i) ** 2 + (sj - j) ** 2) ** 0.5

                if d <= R:
                    covered.add((i, j))
                    sensor_cells.add((i, j))
                    coverage_count[i][j] += 1

        sensor_covering[(si, sj)] = sensor_cells

    total_coverage = sum(grid[i][j] for (i, j) in covered)

    return total_coverage, coverage_count, covered, sensor_covering


# OVERLAP

def compute_overlap(grid, coverage_count):

    overlap = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if coverage_count[i][j] > 1:
                overlap += grid[i][j]

    return overlap


# UNCOVERED
def compute_uncovered(grid, covered):

    uncovered = 0

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) not in covered:
                uncovered += grid[i][j]

    return uncovered

# FITNESS FUNCTION
def fitness(chromosome, grid, R=5,
            alpha=0.6, beta=0.2, gamma=0.1, delta=0.1):

    total_coverage, coverage_count, covered, _ = coverage(chromosome, grid, R)

    overlap = compute_overlap(grid, coverage_count)
    uncovered = compute_uncovered(grid, covered)

    fitness_value = (
        alpha * total_coverage
        - beta * overlap
        - gamma * len(chromosome)
        - delta * uncovered
    )

    return max(0, fitness_value)


# CHROMOSOME GENERATION
def generate_chromosome(num_sensors, grid_shape):

    n, m = grid_shape

    return [
        (random.randint(0, n - 1), random.randint(0, m - 1))
        for _ in range(num_sensors)
    ]


# VISUALIZATION (optional for debugging only)

def visualize(chromosome, covered, overlap_map):

    sx = [x for x, y in chromosome]
    sy = [y for x, y in chromosome]

    cx = [x for x, y in covered]
    cy = [y for x, y in covered]

    # overlap_map is dict -> convert safely
    zx = [x for x, _ in overlap_map.keys()]
    zy = [y for _, y in overlap_map.keys()]

    plt.figure(figsize=(8, 8))

    plt.scatter(cx, cy, s=5, alpha=0.4, label="Covered")
    plt.scatter(sx, sy, c='red', s=80, label="Sensors")
    plt.scatter(zx, zy, c='black', s=80, label="Overlap Sensors")

    plt.title("Sensor Coverage")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()