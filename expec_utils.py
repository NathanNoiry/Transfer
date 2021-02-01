import numpy as np 


def compute_matrix(q, x, i, j):
    cnt = 0
    l = []
    for k in range(x.shape[0]):
        if (q[i,0] < x[k,0] <= q[i+1,0]) and (q[j,1] < x[k,1] <= q[j+1,1]):
            cnt += 1
            l.append(k)
    return cnt, l


def prob_source(nb_samples, weight):
    l = []
    for n in range(nb_samples):       
        x = np.random.choice(36, p=weight)
        idx = np.random.choice(list_idx[x])
        l.append(arr_df[idx])
    return np.array(l)


def psi_emp(alpha,M0,M1,M2,mu):
    term0 = alpha.T @ M0 @ alpha
    term1 = alpha.T @ M1 @ alpha
    term2 = alpha.T @ M2 @ alpha
    return (term0-1)**2 + (term1-mu[1])**2 + (term2-mu[2])**2


def grad_psi_emp(alpha,M0,M1,M2,mu):
    term0 = 2 * (alpha.T @ M0 @ alpha - 1) * (M0 @ alpha).T
    term1 = 2 * (alpha.T @ M1 @ alpha - mu[1]) * (M1 @ alpha).T
    term2 = 2 * (alpha.T @ M2 @ alpha - mu[2]) * (M2 @ alpha).T
    return term0 + term1 + term2