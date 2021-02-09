import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


grid_param = ['250_50','500_100',
              '1000_200','2000_400',
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
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')
plt.ylabel('Time in seconds')

plt.plot(mean)
plt.plot(mean_p,color='lightblue')
plt.plot(mean_m,color='lightblue')
plt.fill_between(x, mean_m, mean_p, color='lightblue', alpha=0.6)
plt.show()


plt.figure(figsize=(15,8))
#MSE scores with ols
plt.subplot(1,3,1)

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
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)), rotation=45)
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

#MSE scores with svr
plt.subplot(1,3,2)

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
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)), rotation=45)
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

#MSE scores with rf
plt.subplot(1,3,3)

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
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)), rotation=45)
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

plt.figure(figsize=(5,5))

x = np.arange(mean_wS_1.shape[0])
plt.xticks(np.arange(8), (str(250*2**i) for i in range(8)) )
plt.xlabel('Training size')
plt.ylabel('Ratio of MSE scores')

plt.plot(x,mean_wS_2/mean_T_2,color='blue',label='Algorithm 1')
plt.plot(x,mean_wS_1/mean_T_1,color='red',label='Algorithm 2')
plt.legend(loc='best')

plt.show()

#tunning time
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