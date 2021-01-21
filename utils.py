import numpy as np


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

    """

    def __init__(self, mc_size, w, alpha, matrix):
        self.mc_size = mc_size
        self.w = w
        self.alpha = alpha
        self.matrix_init = matrix

        self.z = None
        self.norm_factor = None
        self.matrix_normalized = None
        self.radon_weights = None


    def generate_data(self):
        x = np.zeros((self.mc_size, 2))
        x[:, 0] = np.random.randn(self.mc_size)
        x[:, 1] = np.random.gamma(2, size=self.mc_size)

        eps = np.random.randn(self.mc_size)

        y = self.w[0] * x[:, 0] + self.w[1] * x[:, 1] + self.w[2] * x[:, 0] * x[:, 1] + eps
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


    def plot_marginals(self,nb_samples):
        z_s = generator.prob_source(nb_samples)
        Z_T = generator.prob_target(nb_samples)
        plt.figure(figsize=(20,5))

        plt.subplot(1,3,1)
        plt.hist(Z_S[:,0],bins=40,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,0],bins=40,rwidth=0.8,alpha=0.5,density=True)

        plt.subplot(1,3,2)
        plt.hist(Z_S[:,1],bins=40,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,1],bins=40,rwidth=0.8,alpha=0.5,density=True)

        plt.subplot(1,3,3)
        plt.hist(Z_S[:,2],bins=40,rwidth=0.8,alpha=0.5,density=True)
        plt.hist(Z_T[:,2],bins=40,rwidth=0.8,alpha=0.5,density=True)

        #plt.savefig('img_transfer')

        plt.show()
