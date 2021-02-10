import numpy as np 
from scipy.optimize import minimize


def compute_matrix(q, x, i, j):
    cnt = 0
    l = []
    for k in range(x.shape[0]):
        if (q[i,0] < x[k,0] <= q[i+1,0]) and (q[j,1] < x[k,1] <= q[j+1,1]):
            cnt += 1
            l.append(k)
    return cnt, l


def prob_source(vec, nb_samples, weight, list_idx):
    l = []
    for n in range(nb_samples):       
        x = np.random.choice(36, p=weight)
        idx = np.random.choice(list_idx[x])
        l.append(vec[idx])
    return np.array(l)


class Optimization(object):
    """
    A class used to generate data

    ...

    Attributes
    ----------    
    mu: the moments of the target

    M0, M1, M2: empirical moments to be computed


    Methods
    -------
    psi_emp

    grad_psi_emp

    estimation

    """

    def __init__(self,mu):
        self.mu = mu

        self.M0 = None
        self.M1 = None
        self.M2 = None


    def compute_empirical_moments(self,z,matrix):
        self.M0 = np.array( list(map(matrix,z))).mean(axis=0)
        self.M1 = np.array([ elem[0]*matrix(elem) for elem in z ]).mean(axis=0)
        self.M2 = np.array([ elem[1]*matrix(elem) for elem in z ]).mean(axis=0)


    def psi_emp(self,alpha): 
        term0 = alpha.T @ self.M0 @ alpha
        term1 = alpha.T @ self.M1 @ alpha
        term2 = alpha.T @ self.M2 @ alpha
        return (term0-1)**2 + (term1-self.mu[1])**2 + (term2-self.mu[2])**2


    def grad_psi_emp(self,alpha):
        term0 = 2 * (alpha.T @ self.M0 @ alpha - 1) * (self.M0 @ alpha).T
        term1 = 2 * (alpha.T @ self.M1 @ alpha - self.mu[1]) * (self.M1 @ alpha).T
        term2 = 2 * (alpha.T @ self.M2 @ alpha - self.mu[2]) * (self.M2 @ alpha).T
        return term0 + term1 + term2


    def estimation(self):
        res = minimize(self.psi_emp,np.random.randn(2),jac=self.grad_psi_emp,method='BFGS')
        return np.abs(res.x)