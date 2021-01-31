import numpy as np

n_loop = 100

mc_size = int(10e5)
w = np.array([2.,0.8,1.3])
alpha_true = np.array([1.,1.,2.])

n_repet = 100
sample_size = 10000
sub_sample_size = 2000

#choose the optimization method
optim_choices = ['boot_on_sample','boot_on_init']

optim_method = optim_choices[0]


#choose the ml algorithm
ml_algo = 'ols'

#parameters for ml_algo = 'rf'
n_estimators = 100
max_depth = 5

def matrix_init(z):
    M = np.array([[z[0]**2,0,0],
                 [0,4*z[1]**2,0],
                 [0,0,2*z[2]**2] ])
    return M 