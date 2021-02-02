import numpy as np
import pandas as pd
import expec_param as param
from expec_utils import compute_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# Seed initialization
np.random.seed(seed=1)

#import parameter
weight = param.weight
prob_grid = param.prob_grid

#Load data and preprocess
df = pd.read_csv('Life_Expectancy_Data.csv')
df_clean = df[['Adult Mortality','Alcohol', 'Life expectancy ']]
df_clean.columns = ['adult_mortality','alcohol', 'life_expec']

df_clean = df_clean.dropna()

normalized_df=(df_clean-df_clean.mean())/df_clean.std()
size_data = normalized_df.shape[0]

#quantile
q_df = normalized_df[['adult_mortality','life_expec']].quantile([0.25,0.325,0.5,0.625,0.75])
arr_quantile = np.array(q_df)
arr_quantile = np.vstack((np.array([-10,-10]),arr_quantile,np.array([100,100])))

#count
matrix_count = np.zeros((6,6))
list_idx = []
arr = np.array(normalized_df[['adult_mortality','life_expec']])
for i in range(6):
	for j in range(6):
	    matrix_count[i,j], elem = compute_matrix(arr_quantile,arr,i,j)
	    list_idx.append(elem)
matrix_prop = matrix_count / matrix_count.sum()
matrix_rot = np.rot90(matrix_prop)

#plot
fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(8,4), 
                  gridspec_kw={"width_ratios":[1,1, 0.05]})
fig.subplots_adjust(wspace=0.3)

im  = ax.imshow(prob_grid, cmap='Reds', alpha=0.7)
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_ylabel('Life Expectancy')
ax.set_xlabel('Adult Mortality Rate')
ax.set_title("Source")

im2 = ax2.imshow(matrix_rot, cmap='Reds', alpha=0.7)

ax2.set_xticks([])
ax2.axes.yaxis.set_visible(False)
ax2.set_title("Target")
ax2.set_xlabel('Adult Mortality Rate')

ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
cax.set_axes_locator(ip)

fig.colorbar(im, cax=cax)

plt.show()