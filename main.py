from utils import Generator, Optimization
import parameters as param 
import numpy as np

#import the parameters
mc_size = param.mc_size
w = param.w
alpha_true = param.alpha_true

matrix_init = param.matrix_init


# Seed initialization
np.random.seed(seed=1)

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

#plot marginals
#generator.plot_marginals(100000)


#print(generator.mu, generator.norm_factor)

#instantiate the class
optim = Optimization(generator.mu)

matrix = generator.matrix_normalized

n_repet = 100
sample_size = 200000
sub_sample_size = 40000
alpha_est = []
psi_emp_est = []

Z_S = generator.prob_source(sample_size)

for i in range(n_repet):
    idx = np.random.randint(Z_S.shape[0], size=sub_sample_size)
    Z_boot = Z_S[idx]
    #Z_S = generator.prob_source(sample_size)
    optim.compute_empirical_moments(Z_boot,matrix)
    res = optim.estimation()
    alpha_est.append(res)
    psi_emp_est.append(optim.psi_emp(res))

x = np.array(alpha_est)
x = np.abs(x)
print(x.mean(axis=0),x.std(axis=0),np.median(x,axis=0))

idx = np.argmin(psi_emp_est)
print(alpha_est[idx])


