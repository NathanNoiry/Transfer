import numpy as np
import pandas as pd
from expec_utils import compute_matrix, prob_source, Optimization
import expec_param as param
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Seed initialization
np.random.seed(seed=1)

#import parameters
n_loop = param.n_loop
n_repet = param.n_repet
weight = param.weight
matrix = param.matrix


#Load data and preprocess
df = pd.read_csv('Life_Expectancy_Data.csv')
df_clean = df[['Adult Mortality', 'Alcohol', 'Life expectancy ']]
df_clean.columns = ['adult_mortality', 'alcohol', 'life_expec']

#Delete rows containing NaN values
df_clean = df_clean.dropna()

#Normalize the DataFrame
normalized_df=(df_clean-df_clean.mean())/df_clean.std()
size_data = normalized_df.shape[0]

#Compute moments
mu = np.zeros(3)
mu[0] = 1
mu[1] = normalized_df['adult_mortality'].mean()
mu[2] = normalized_df['life_expec'].mean()

#Compute quantile
q_df = normalized_df[['adult_mortality','life_expec']].quantile([0.25,0.325,0.5,0.625,0.75])
arr_quantile = np.array(q_df)
arr_quantile = np.vstack((np.array([-10,-10]),arr_quantile,np.array([100,100])))

#Start the loop
scores_lin = []
scores_svr = []
for n in range(n_loop):

	#Divide in train/test
	idx_test = np.random.choice(size_data,size=int(size_data*0.1),replace=False)
	idx_used = np.delete(np.arange(size_data),idx_test)
	df_test = normalized_df.iloc[idx_test]
	df_used = normalized_df.iloc[idx_used]

	#Count the number of observations in each box of the quantile grid
	matrix_count = np.zeros((6,6))
	list_idx = []
	arr = np.array(df_used[['adult_mortality','life_expec']])
	for i in range(6):
		for j in range(6):
		    matrix_count[i,j], elem = compute_matrix(arr_quantile,arr,i,j)
		    list_idx.append(elem)
	matrix_prop = matrix_count / matrix_count.sum()

	#Data sampling
	vec = np.array(df_used)
	Z_S = prob_source(vec, df_used.shape[0], weight, list_idx)
	Z_T = np.array(df_used)
	Z_test = np.array(df_test)

	#Instantiate Optimization class
	optim = Optimization(mu)

	#compute empirical moments
	optim.compute_empirical_moments(Z_S,matrix)

	#Estimate alpha
	alpha_list = []
	psi_list = []
	for i in range(n_repet):
		res = optim.estimation()
		alpha_list.append(res)
		psi_list.append(optim.psi_emp(res))
		idx = np.argmin(psi_list)
		alpha_emp = alpha_list[idx]

	#Compute radon weights
	weight_emp = np.array([ alpha_emp.T @ matrix(elem) @ alpha_emp for elem in Z_S ])

	#Data for learning
	X_S, y_S = Z_S[:,:2], Z_S[:,2]
	X_T, y_T = Z_T[:,:2], Z_T[:,2]
	X_test, y_test = Z_test[:,:2], Z_test[:,2]

	#Learning linear regression
	lin1 = LinearRegression()
	lin1.fit(X_S,y_S,weight_emp)

	lin2 = LinearRegression()
	lin2.fit(X_T,y_T)

	lin3 = LinearRegression()
	lin3.fit(X_S,y_S)

	y_pred_1 = lin1.predict(X_test)
	y_pred_2 = lin2.predict(X_test)
	y_pred_3 = lin3.predict(X_test)

	mse1 = mean_squared_error(y_test,y_pred_1)
	mse2 = mean_squared_error(y_test,y_pred_2)
	mse3 = mean_squared_error(y_test,y_pred_3)

	scores_lin.append([mse1,mse2,mse3])

	#learning SVR with different C parameters
	grid = [0.001,0.01,0.1,1.,10.]
	for C in grid:
		svm1 = SVR(C=C)
		svm1.fit(X_S,y_S,weight_emp)

		svm2 = SVR(C=C)
		svm2.fit(X_T,y_T)

		svm3 = SVR(C=C)
		svm3.fit(X_S,y_S)

		y_pred_1 = svm1.predict(X_test)
		y_pred_2 = svm2.predict(X_test)
		y_pred_3 = svm3.predict(X_test)

		mse1 = mean_squared_error(y_test,y_pred_1)
		mse2 = mean_squared_error(y_test,y_pred_2)
		mse3 = mean_squared_error(y_test,y_pred_3)

		scores_svr.append([C,mse1,mse2,mse3])

	print(n)

df_scores_lin = pd.DataFrame(scores_lin)
df_scores_svr = pd.DataFrame(scores_svr)

df_scores_lin.to_csv('transfer_expec_lin_{}_{}_difmat.csv'.format(n_loop,n_repet), index=False)
df_scores_svr.to_csv('transfer_expec_svr_{}_{}_difmat.csv'.format(n_loop,n_repet), index=False)


