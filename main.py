from utils import Generator, Optimization
import parameters as param 
import numpy as np


# Seed initialization
np.random.seed(seed=1)

#############################################################
##################### IMPORT PARAMETERS #####################
#############################################################

mc_size = param.mc_size
w = param.w
alpha_true = param.alpha_true

n_repet = param.n_repet
sample_size = param.sample_size
sub_sample_size = param.sub_sample_size

matrix_init = param.matrix_init

#############################################################
################## STEP 0: DATA GENERATION ##################
#############################################################

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

#compute the moments
generator.compute_moments()

#############################################################
################## STEP 1: ALPHA ESTIMATION #################
#############################################################

#instantiate the class
optim = Optimization(generator.mu) 

Z_S = generator.prob_source(sample_size)
matrix = generator.matrix_normalized

alpha_emp = optim.estimation(Z_S,
	                         matrix,
	                         sample_size,
	                         sub_sample_size,
	                         n_repet)

print(alpha_emp)


#############################################################
######################## STEP 2: ERM ########################
#############################################################


#plot marginals
#generator.plot_marginals(100000)