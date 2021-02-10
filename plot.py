from utils import Generator, Optimization
import parameters as param 
import numpy as np


# Seed initialization
np.random.seed(seed=1)

mc_size = param.mc_size
w = param.w
alpha_true = param.alpha_true
matrix_init = param.matrix_init

#instantiate the generator class
generator = Generator(param.mc_size, 
	                  param.w, 
	                  param.alpha_true,
	                  param.matrix_init)

#generate the data used for the Monte-Carlo simulation
generator.generate_data()

#compute the normalization factor
generator.compute_norm_factor()

#compute the weights
generator.compute_weights()

#plot marginals
generator.plot_marginals(100000)
