import numpy as np


class Generator(object):
    """
    A class used to generate data

    ...

    Attributes
    ----------
    mc_size: number of samples used for the Monte-Carlo simulation

    w: weight vectors used for the definition of the target variable

    z: the dataset used for the Monte-Carlo simulation ; z.shape = (mc_size,dim_features(x)+dim_target(y))


    Methods
    -------
    generate_data
    
    prob_source
    
    prob_target

    """

    def __init__(self, mc_size, w):
        self.mc_size = mc_size
        self.w = w
        self.z = None

    def generate_data(self):
        x = np.zeros((self.mc_size, 2))
        x[:, 0] = np.random.randn(self.mc_size)
        x[:, 1] = np.random.gamma(2, size=self.mc_size)

        eps = np.random.randn(self.mc_size)

        y = self.w[0] * x[:, 0] + self.w[1] * x[:, 1] + self.w[2] * x[:, 0] * x[:, 1] + eps
        self.z = np.c_[x, y]

    def prob_source(self, nb_samples):
        idx = np.random.randint(self.z.shape[0], size=nb_samples)
        return self.z[idx]

    def prob_target(self, nb_samples, weights):
        idx = np.random.choice(self.z.shape[0], size=nb_samples, p=weights)
        return self.z[idx]


class Weight(object):
        """
    A class used to generate data

    ...

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self,alpha,func):
        self.alpha = alpha
        self.matrix = matrix
        self.func_init = func
        self.weights = None
        self.norm_factor = None
        self.func_norm = None

    def compute_weights(self,z):
        self.norm_factor = alpha_true.T @ np.array(list(map(self.func_init,z))).mean(axis=0) @ alpha_true

        def func_norm(z):
            return self.func_init(z) / self.norm_factor

        self.func_norm = func_norm

    def transform(self,z):
        return self.alpha.T @ self.func_norm(z) @ self.alpha

    def 





def w_matrix(z):
    m = np.array([[z[0]**2, 0, 0],
                 [0, 4*z[1]**2, 0],
                 [0, 0, 2*z[2]**2]])
    return m
