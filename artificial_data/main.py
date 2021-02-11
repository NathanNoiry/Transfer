from utils import Generator, Optimization
import parameters as param 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from time import time


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

optim_method = param.optim_method
ml_algo = param.ml_algo

# Define the lists to be saved
list_alpha_erm = []

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

#Tracking times of execution
list_time = []

#################### START THE REPETITIONS ##################

for i in range(param.n_loop):

#############################################################
################## STEP 1: ALPHA ESTIMATION #################
#############################################################
	
	time_generation0 = time()

	#draw a sample from the source
	Z_S = generator.prob_source(sample_size)
	matrix = generator.matrix_normalized

	#instantiate the class and compute moments based on Z_S
	optim = Optimization(generator.mu)
	optim.compute_empirical_moments(Z_S,matrix)

    #another instance of the class for the bootstrap procedure
	optim_boot = Optimization(generator.mu)

	time_generation1 = time() - time_generation0

	alpha_est = []
	psi_emp_est = []

	time_est_alpha0 = time()
	for n in range(n_repet):
		if optim_method == 'boot_on_sample':
			idx = np.random.randint(Z_S.shape[0], 
				                    size=sub_sample_size)
			z_boot = Z_S[idx]
			optim_boot.compute_empirical_moments(z_boot,matrix)
			res = optim_boot.estimation()
			alpha_est.append(res)
			psi_emp_est.append(optim.psi_emp(res))

		elif optim_method == 'boot_on_init':
		    res = optim.estimation()
		    alpha_est.append(res)
		    psi_emp_est.append(optim.psi_emp(res))

	idx = np.argmin(psi_emp_est)
	alpha_emp = alpha_est[idx]
	time_est_alpha1 = time() - time_est_alpha0

#############################################################
######################## STEP 2: ERM ########################
#############################################################
	
	time_weight0 = time()
	#compute empirical weights
	weight_emp = np.array([ alpha_emp.T @ 
		                    generator.matrix_normalized(elem) @ 
	                        alpha_emp for elem in Z_S ])
	time_weight1 = time() - time_weight0

	#source sample
	X_S, y_S = Z_S[:,:2], Z_S[:,2]

	#target sample
	Z_T = generator.prob_target(sample_size)
	X_T, y_T = Z_T[:,:2], Z_T[:,2]

	#test sample
	Z_test = generator.prob_target(int(sample_size/20)) 
	X_test, y_test = Z_test[:,:2], Z_test[:,2]

	time_erm_fit0 = time()

	if ml_algo == 'svr':
		svr1 = SVR()
		svr1.fit(X_S,y_S,weight_emp)
		time_fit_Rw = time() - time_erm_fit0

		svr2 = SVR()
		svr2.fit(X_T,y_T)
		time_fit_T = time() - time_fit_Rw

		svr3 = SVR()
		svr3.fit(X_S,y_S)
		time_fit_S = time() - time_fit_T

		time_erm_predict0 = time()
		y_pred_1 = svr1.predict(X_test)
		time_predict_Rw = time() - time_erm_predict0

		y_pred_2 = svr2.predict(X_test)
		time_predict_T = time() - time_predict_Rw

		y_pred_3 = svr3.predict(X_test)
		time_predict_S = time() - time_predict_T

		mse1 = mean_squared_error(y_test,y_pred_1)
		mse2 = mean_squared_error(y_test,y_pred_2)
		mse3 = mean_squared_error(y_test,y_pred_3)

	elif ml_algo == 'rf':
		rf1 = RandomForestRegressor(param.n_estimators, 
									 criterion='mse', 
									 max_depth=param.max_depth)
		rf1.fit(X_S,y_S,weight_emp)
		time_fit_Rw = time() - time_erm_fit0

		rf2 = RandomForestRegressor(param.n_estimators, 
									 criterion='mse', 
									 max_depth=param.max_depth)
		rf2.fit(X_T,y_T)
		time_fit_T = time() - time_fit_Rw + time_erm_fit0

		rf3 = RandomForestRegressor(param.n_estimators, 
									 criterion='mse', 
									 max_depth=param.max_depth)
		rf3.fit(X_S,y_S)
		time_fit_S = time() - time_fit_T - time_fit_Rw + time_erm_fit0

		time_erm_predict0 = time()
		y_pred_1 = rf1.predict(X_test)
		time_predict_Rw = time() - time_erm_predict0

		y_pred_2 = rf2.predict(X_test)
		time_predict_T = time() - time_predict_Rw + time_erm_predict0

		y_pred_3 = rf3.predict(X_test)
		time_predict_S = time() - time_predict_T - time_predict_T  + time_erm_predict0

		mse1 = mean_squared_error(y_test,y_pred_1)
		mse2 = mean_squared_error(y_test,y_pred_2)
		mse3 = mean_squared_error(y_test,y_pred_3)

	elif ml_algo == 'ols':
		lin1 = LinearRegression()
		lin1.fit(X_S,y_S,weight_emp)
		time_fit_Rw = time() - time_erm_fit0

		lin2 = LinearRegression()
		lin2.fit(X_T,y_T)
		time_fit_T = time() - time_fit_Rw - time_erm_fit0

		lin3 = LinearRegression()
		lin3.fit(X_S,y_S)
		time_fit_S = time() - time_fit_T - time_fit_Rw - time_erm_fit0

		time_erm_predict0 = time()
		y_pred_1 = lin1.predict(X_test)
		time_predict_Rw = time() - time_erm_predict0

		y_pred_2 = lin2.predict(X_test)
		time_predict_T = time() - time_predict_Rw - time_erm_predict0

		y_pred_3 = lin3.predict(X_test)
		time_predict_S = time() - time_predict_T - time_predict_Rw - time_erm_predict0

		mse1 = mean_squared_error(y_test,y_pred_1)
		mse2 = mean_squared_error(y_test,y_pred_2)
		mse3 = mean_squared_error(y_test,y_pred_3)

	list_alpha_erm.append([alpha_emp[0], 
						   alpha_emp[1],
						   alpha_emp[2],
						   mse1, mse2, mse3])

	list_time.append([time_generation1, time_est_alpha1, time_weight1, 
		              time_fit_Rw, time_fit_T, time_fit_S,
		              time_predict_Rw, time_predict_T, time_predict_S])

	print(i)

##################### END THE REPETITIONS ###################


df = pd.DataFrame(list_alpha_erm)
df.columns = ['alpha_best_1',
			  'alpha_best_2',
			  'alpha_best_3',
			  'mse_w_S','mse_T','mse_S']

df_time = pd.DataFrame(list_time)
df_time.columns = ['data_gen', 'est_alpha', 'time_weight',
				   'fit_Rw', 'fit_T', 'fit_S',
				   'pred_Rw', 'pred_T', 'pred_S']

df.to_csv('./results/transfer_{}_{}_{}_{}_{}_{}.csv'.format(ml_algo,
	mc_size,sample_size,sub_sample_size,param.n_loop,n_repet), 
	index=False)
df_time.to_csv('./results/transfer_times_{}_{}_{}_{}_{}_{}.csv'.format(ml_algo,
	mc_size,sample_size,sub_sample_size,param.n_loop,n_repet), 
	index=False)
