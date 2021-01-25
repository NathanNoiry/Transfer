import numpy as np


mc_size = int(10e5)
w = np.array([2.,0.8,1.3])
alpha_true = np.array([1.,1.,2.])

n_repet = 100
sample_size = 50000
sub_sample_size = 10000

def matrix_init(z):
    M = np.array([[z[0]**2,0,0],
                 [0,4*z[1]**2,0],
                 [0,0,2*z[2]**2] ])
    return M 