import numpy as np
from scipy.stats import gamma, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Generator(object):
    """
    A class used to generate data

    ...

    Attributes
    ----------
    mc_size: number of samples used for the Monte-Carlo simulation

    w: weight vectors used for the definition of the target variable

    alpha: the true alpha parameter 

    matrix_init: the initial matrix transform to be normalized

    mu: the moments of the target to be computed

    z: the dataset used for the Monte-Carlo simulation ; z.shape = (mc_size,dim_features(x)+dim_target(y))

    norm_factor: the normalization factor computed on z in order to renormalize the initial function func_init

    matrix_normalized: matrix_init renormalized by norm_factor

    radon_weights: the weights to be computed on z ; radon_weights[i] = func_normalized(alpha,z[i])


    Methods
    -------
    generate_data

    compute_norm_factor

    transform

    compute_weights
    
    prob_source
    
    prob_target

    compute_moments

    plot_marginals

    """

    def __init__(self, mc_size, w, alpha, matrix):
        self.mc_size = mc_size
        self.w = w
        self.alpha = alpha
        self.matrix_init = matrix
        self.mu = np.zeros(3)

        self.z = None
        self.norm_factor = None
        self.matrix_normalized = None
        self.radon_weights = None


    def generate_data(self):
        x = np.zeros((self.mc_size, 2))
        x[:, 0] = np.random.randn(self.mc_size)
        x[:, 1] = np.random.gamma(2, size=self.mc_size)

        eps = np.random.randn(self.mc_size)

        y = 0.5*(self.w[0] * x[:, 0] + self.w[1] * x[:, 1] + self.w[2] * x[:, 0] * x[:, 1]) + eps
        self.z = np.c_[x, y]


    def compute_norm_factor(self):
        self.norm_factor = self.alpha.T @ np.array(list(map(self.matrix_init,self.z))).mean(axis=0) @ self.alpha

        def func_norm(z):
            return self.matrix_init(z) / self.norm_factor

        self.matrix_normalized = func_norm


    def transform(self,elem):
        return self.alpha.T @ self.matrix_normalized(elem) @ self.alpha


    def compute_weights(self):
        self.radon_weights =  np.array([self.transform(elem) for elem in self.z]) * (1 / self.mc_size)


    def prob_source(self, nb_samples):
        idx = np.random.randint(self.z.shape[0], size=nb_samples)
        return self.z[idx]


    def prob_target(self, nb_samples):
        idx = np.random.choice(self.z.shape[0], size=nb_samples, p=self.radon_weights)
        return self.z[idx]


    def compute_moments(self):
        idx = np.random.choice(self.z.shape[0], size=self.mc_size, p=self.radon_weights)
        z_t = self.z[idx]
        self.mu[0] = 1
        self.mu[1] = z_t[:,0].mean()
        self.mu[2] = z_t[:,1].mean()


    def plot_marginals(self,nb_samples):
        Z_S = self.prob_source(nb_samples)
        Z_T = self.prob_target(nb_samples)
        plt.figure(figsize=(20,5))

        plt.subplot(1,3,1)
        plt.xlim(-4.5,4.5)
        plt.xlabel('Feature first marginal')
        plt.ylabel('Density')
        plt.hist(Z_S[:,0],bins=35,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,0],bins=35,rwidth=0.8,alpha=0.5,density=True)

        plt.subplot(1,3,2)
        plt.xlim(-0.5,14)
        plt.xlabel('Feature second marginal')
        plt.hist(Z_S[:,1],bins=35,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,1],bins=35,rwidth=0.8,alpha=0.5,density=True)

        plt.subplot(1,3,3)
        plt.xlim(-18,24)
        plt.xlabel('Target marginal')
        plt.hist(Z_S[:,2],bins=35,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,2],bins=35,rwidth=0.8,alpha=0.5,density=True)

        #plt.savefig('img_transfer')

        plt.show()


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
    compute_empirical_moments

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
        res = minimize(self.psi_emp,np.random.randn(3),jac=self.grad_psi_emp,method='BFGS')
        return np.abs(res.x)

        




