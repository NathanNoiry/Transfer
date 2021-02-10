import numpy as np


n_loop = 100
n_repet = 100


def matrix(Z):
    W = np.array([[Z[0]**2,0],
                  [0, Z[2]**2]])
    return W


prob_grid = np.zeros((6,6))

prob_grid[0] = [0.05, 0.08, 0.04, 0.04, 0.01, 0.00]
prob_grid[1] = [0.02, 0.04, 0.19, 0.14, 0.01, 0.00]
prob_grid[2] = [0.04, 0.00, 0.13, 0.01, 0.01, 0.00]
prob_grid[3] = [0.03, 0.00, 0.02, 0.02, 0.01, 0.01]
prob_grid[4] = [0.03, 0.00, 0.00, 0.02, 0.01, 0.01]
prob_grid[5] = [0.01, 0.00, 0.00, 0.00, 0.01, 0.01]

prob_grid_antirot = np.rot90(prob_grid,-1)
weight = prob_grid_antirot.flatten()