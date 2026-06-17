import numpy as np


# COVERAGE MAP

def get_coverage_map(chromosome, grid, R):

    covered = set()

    for (si, sj) in chromosome:
        for i in range(max(0, si-R), min(grid.shape[0], si+R+1)):
            for j in range(max(0, sj-R), min(grid.shape[1], sj+R+1)):

                if np.linalg.norm([si-i, sj-j]) <= R:
                    covered.add((i, j))

    return covered


# COVERAGE RATIO

def coverage_ratio(chromosome, grid, R):

    covered = get_coverage_map(chromosome, grid, R)
    total = grid.shape[0] * grid.shape[1]

    return len(covered) / total



# VISUAL PREP (optional)

def extract_xy(chromosome):
    x = [p[0] for p in chromosome]
    y = [p[1] for p in chromosome]
    return x, y