import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


grid_param = ['1000_200','2000_400',
              '4000_800','8000_1600',
              '16000_3200','32000_6400']

#plot running time
list_mean_time = []
list_std_time = []
for param in grid_param:
    df = pd.read_csv('./results/transfer_times_ols_1000000_'+param+'_100_100.csv')
    list_mean_time.append(list(df.mean()))
    list_std_time.append(list(df.std()))
list_mean_time = np.array(list_mean_time)
list_std_time = np.array(list_std_time)

mean = list_mean_time[:,1]
std = list_std_time[:,1]
mean_p = mean + std
mean_m = mean - std

plt.figure(figsize=(5,5))

x = np.arange(mean.shape[0])

plt.xticks(np.arange(6), (str(250*2**(i+2)) for i in range(6)) )
plt.xlabel('Training size')
plt.ylabel('Time in seconds')

plt.plot(mean)
plt.plot(mean_p,color='lightblue')
plt.plot(mean_m,color='lightblue')
plt.fill_between(x, mean_m, mean_p, color='lightblue', alpha=0.6)
plt.show()


grid_param = ['500_100',
              '1000_200','2000_400',
              '4000_800','8000_1600',
              '16000_3200','32000_6400']

#plt.figure(figsize=(15,8))
#MSE scores with ols
plt.figure(figsize=(5,5))

list_mse_mean_ols = []
list_mse_std_ols = []
for param in grid_param:
    df = pd.read_csv('./results/transfer_ols_1000000_'+param+'_100_100_binit.csv')
    list_mse_mean_ols.append(list(df.mean()))
    list_mse_std_ols.append(list(df.std()))
    
list_mse_mean_ols = np.array(list_mse_mean_ols)
list_mse_std_ols = np.array(list_mse_std_ols)

mean_wS, std_wS = list_mse_mean_ols[:,3], list_mse_std_ols[:,3]
mean_T, std_T = list_mse_mean_ols[:,4], list_mse_std_ols[:,4]
mean_S, std_S = list_mse_mean_ols[:,5], list_mse_std_ols[:,5]

x = np.arange(mean_wS.shape[0])
plt.xticks(np.arange(7), (str(250*2**(i+1)) for i in range(7)))
plt.xlabel('Training size')
plt.ylabel('MSE score')

plt.plot(mean_wS, color='red')
plt.plot(mean_T, color='blue')
plt.plot(mean_S, color='green')

plt.plot(mean_wS + std_wS, color='tomato')
plt.plot(mean_T + std_T, color='lightblue')
plt.plot(mean_S + std_S, color='lightgreen')

plt.plot(mean_wS - std_wS, color='tomato')
plt.plot(mean_T - std_T, color='lightblue')
plt.plot(mean_S - std_S, color='lightgreen')

plt.fill_between(x, mean_wS - std_wS, mean_wS + std_wS, color='tomato', alpha=0.3)
plt.fill_between(x, mean_T - std_T, mean_T + std_T, color='lightblue', alpha=0.3)
plt.fill_between(x, mean_S - std_S, mean_S + std_S, color='lightgreen', alpha=0.3)

plt.plot()

#MSE scores with svr
plt.figure(figsize=(5,5))

list_mse_mean_svr = []
list_mse_std_svr = []
for param in grid_param:
    df = pd.read_csv('./results/transfer_svm_1000000_'+param+'_100_100_binit.csv')
    list_mse_mean_svr.append(list(df.mean()))
    list_mse_std_svr.append(list(df.std()))
    
list_mse_mean_svr = np.array(list_mse_mean_svr)
list_mse_std_svr = np.array(list_mse_std_svr)

mean_wS, std_wS = list_mse_mean_svr[:,3], list_mse_std_svr[:,3]
mean_T, std_T = list_mse_mean_svr[:,4], list_mse_std_svr[:,4]
mean_S, std_S = list_mse_mean_svr[:,5], list_mse_std_svr[:,5]

x = np.arange(mean_wS.shape[0])
plt.xticks(np.arange(7), (str(250*2**(i+1)) for i in range(7)))
plt.xlabel('Training size')
#plt.ylabel('MSE score')

plt.plot(mean_wS, color='red')
plt.plot(mean_T, color='blue')
plt.plot(mean_S, color='green')

plt.plot(mean_wS + std_wS, color='tomato')
plt.plot(mean_T + std_T, color='lightblue')
plt.plot(mean_S + std_S, color='lightgreen')

plt.plot(mean_wS - std_wS, color='tomato')
plt.plot(mean_T - std_T, color='lightblue')
plt.plot(mean_S - std_S, color='lightgreen')

plt.fill_between(x, mean_wS - std_wS, mean_wS + std_wS, color='tomato', alpha=0.3)
plt.fill_between(x, mean_T - std_T, mean_T + std_T, color='lightblue', alpha=0.3)
plt.fill_between(x, mean_S - std_S, mean_S + std_S, color='lightgreen', alpha=0.3)

plt.plot()

#MSE scores with rf
plt.figure(figsize=(5,5))

list_mse_mean_rf = []
list_mse_std_rf = []
for param in grid_param:
    df = pd.read_csv('./results/transfer_rf_1000000_'+param+'_100_100_binit.csv')
    list_mse_mean_rf.append(list(df.mean()))
    list_mse_std_rf.append(list(df.std()))
    
list_mse_mean_rf = np.array(list_mse_mean_rf)
list_mse_std_rf = np.array(list_mse_std_rf)

mean_wS, std_wS = list_mse_mean_rf[:,3], list_mse_std_rf[:,3]
mean_T, std_T = list_mse_mean_rf[:,4], list_mse_std_rf[:,4]
mean_S, std_S = list_mse_mean_rf[:,5], list_mse_std_rf[:,5]

x = np.arange(mean_wS.shape[0])
plt.xticks(np.arange(7), (str(250*2**(i+1)) for i in range(7)))
plt.xlabel('Training size')
#plt.ylabel('MSE score')

plt.plot(mean_wS, color='red')
plt.plot(mean_T, color='blue')
plt.plot(mean_S, color='green')

plt.plot(mean_wS + std_wS, color='tomato')
plt.plot(mean_T + std_T, color='lightblue')
plt.plot(mean_S + std_S, color='lightgreen')

plt.plot(mean_wS - std_wS, color='tomato')
plt.plot(mean_T - std_T, color='lightblue')
plt.plot(mean_S - std_S, color='lightgreen')

plt.fill_between(x, mean_wS - std_wS, mean_wS + std_wS, color='tomato', alpha=0.3)
plt.fill_between(x, mean_T - std_T, mean_T + std_T, color='lightblue', alpha=0.3)
plt.fill_between(x, mean_S - std_S, mean_S + std_S, color='lightgreen', alpha=0.3)

plt.show()

#optimization algorithms

grid_param = ['8000_1600','10000_2000',
              '16000_3200','32000_6400']

list_opti1_mean = []
list_opti2_mean = []
list_opti1_std = []
list_opti2_std = []
for param in grid_param:
    df1 = pd.read_csv('./results/transfer_ols_1000000_'+param+'_100_100.csv')
    df2 = pd.read_csv('./results/transfer_ols_1000000_'+param+'_100_100_binit.csv')
    list_opti1_mean.append(list(df1.mean()))
    list_opti2_mean.append(list(df2.mean()))
    list_opti1_std.append(list(df1.std()))
    list_opti2_std.append(list(df2.std()))
    
list_opti1_mean = np.array(list_opti1_mean)
list_opti2_mean = np.array(list_opti2_mean)
list_opti1_std = np.array(list_opti1_std)
list_opti2_std = np.array(list_opti2_std)

mean_wS_1 = list_opti1_mean[:,3]
mean_wS_2 = list_opti2_mean[:,3]

mean_T_1 = list_opti1_mean[:,4]
mean_T_2 = list_opti2_mean[:,4]

plt.figure(figsize=(10,10))

x = np.arange(mean_wS_1.shape[0])
plt.xticks(np.arange(4), ['8000','10000','16000','32000'] )
plt.xlabel('Training size')
plt.ylabel('Ratio of MSE scores')

plt.plot(x,mean_wS_2/mean_T_2,color='blue',label='Algorithm 1')
plt.plot(x,mean_wS_1/mean_T_1,color='red',label='Algorithm 2')
plt.legend(loc='best')

plt.show()

#first step time

grid_param = ['1000_200','2000_400',
              '4000_800','8000_1600',
              '16000_3200','32000_6400']

list_mean_time_1 = []
list_std_time_1 = []
list_mean_time_2 = []
list_std_time_2 = []
for param in grid_param:
    df1 = pd.read_csv('./results/transfer_times_ols_1000000_'+param+'_100_100_binit.csv')
    df2 = pd.read_csv('./results/transfer_times_ols_1000000_'+param+'_100_100.csv')
    list_mean_time_1.append(list(df1.mean()))
    list_std_time_1.append(list(df1.std()))
    list_mean_time_2.append(list(df2.mean()))
    list_std_time_2.append(list(df2.std()))
    
list_mean_time_1 = np.array(list_mean_time_1)
list_std_time_1 = np.array(list_std_time_1)
list_mean_time_2 = np.array(list_mean_time_2)
list_std_time_2 = np.array(list_std_time_2)

mean_alpha_time_1, std_alpha_time_1 = list_mean_time_1[:,1], list_std_time_1[:,1]
mean_alpha_time_2, std_alpha_time_2 = list_mean_time_2[:,1], list_std_time_2[:,1]

mean_1, mean_2 = mean_alpha_time_1, mean_alpha_time_2
mean_1_p, mean_2_p = mean_alpha_time_1 + std_alpha_time_1, mean_alpha_time_2 + std_alpha_time_2
mean_1_m, mean_2_m = mean_alpha_time_1 - std_alpha_time_1, mean_alpha_time_2 - std_alpha_time_2

plt.figure(figsize=(5,5))

x = np.arange(mean_alpha_time_1.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')
plt.ylabel('Time in seconds')

plt.plot(mean_1)
plt.plot(mean_1_p,color='lightblue')
plt.plot(mean_1_m,color='lightblue')
plt.fill_between(x, mean_1_m, mean_1_p, color='lightblue', alpha=0.6)
plt.show()

plt.figure(figsize=(5,5))

x = np.arange(mean_alpha_time_2.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')
plt.ylabel('Time in seconds')

plt.plot(mean_2)
plt.plot(mean_2_p,color='lightblue')
plt.plot(mean_2_m,color='lightblue')
plt.fill_between(x, mean_2_m, mean_2_p, color='lightblue', alpha=0.6)
plt.show()

#training time

times_mean = []
times_std = []
for param in grid_param:
    df = pd.read_csv('./results/transfer_times_ols_1000000_'+param+'_100_100_binit.csv')
    times_mean.append(list(df.mean()))
    times_std.append(list(df.mean()))
times_mean = np.array(times_mean)
times_std = np.array(times_std)

time_alpha_mean, time_alpha_std = times_mean[:,1], times_std[:,1]
time_weight_mean, time_weight_std = times_mean[:,2], times_std[:,2]

time_fit_Rw_mean, time_fit_Rw_std = times_mean[:,3], times_std[:,3]
time_fit_T_mean, time_fit_T_std = times_mean[:,4], times_std[:,4]

time_pred_Rw_mean, time_pred_Rw_std = times_mean[:,6], times_std[:,6]
time_pred_T_mean, time_pred_T_std = times_mean[:,7], times_std[:,7]

time_tot_train_Rw_mean = time_weight_mean + time_fit_Rw_mean
time_tot_train_Rw_std = time_alpha_std + time_fit_Rw_std


plt.figure(figsize=(6.5,6.5))

x = np.arange(time_alpha_mean.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')
plt.ylabel('Running time in seconds')

plt.plot(x,time_fit_Rw_mean,color='red')
plt.plot(x,time_fit_T_mean,color='blue')

plt.show()

plt.figure(figsize=(6.5,6.5))

x = np.arange(time_alpha_mean.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')

plt.plot(x,time_weight_mean,color='red')

plt.show()

plt.figure(figsize=(6.5,6.5))

x = np.arange(time_alpha_mean.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')


plt.plot(x,time_tot_train_Rw_mean,color='red')
#plt.plot(x,time_tot_train_Rw_mean+time_tot_train_Rw_std,color='lightblue')
#plt.plot(x,time_tot_train_Rw_mean-time_tot_train_Rw_std,color='lightblue')
#plt.fill_between(x,time_tot_train_Rw_mean-time_tot_train_Rw_std,
	#time_tot_train_Rw_mean+time_tot_train_Rw_std,color='lightblue',alpha=0.3)

plt.show()

