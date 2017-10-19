import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log, exp, pi
from matplotlib.colors import LogNorm
from random as rd import uniform
from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

#Initialisation
n_components = 2
likelihood = 3
n_iterations = 5
phi = 1/2
mu = 1/2
sigma = 1/2
W = np.zeros((len(X_train), 2))




class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma1, sigma2):
        #mean and standard deviation
        self.mu1 = [rd.randint(-10, 20), rd.randint(-5, 25)]
        self.mu2 = [rd.randint(-10, 20), rd.randint(-5, 25)]
        self.sigma1 = sigma1
        self.sigma2 = sigma2


    #probability density function
    def pdf(self, datapoint, i):
        "Probability of a data point given the current parameters"
        if(i == 1):
            u = (datapoint - self.mu1)
            y = ((1 / ((2 * pi) * pow(np.linalg.det(self.sigma1, 1 / 2))) * exp(-np.transpose(u) / self.sigma1 * u / 2))
        else:
            u = (datapoint - self.mu2)
            y = ((1 / ((2 * pi) * pow(np.linalg.det(self.sigma2, 1 / 2))) * exp(-np.transpose(u) / self.sigma2 * u / 2))
        return y


class GaussianMixture:
    "Model mixture of two univariate Gaussians and their EM estimation"
    def __init__(self, X_train, mu_min=min(X_train), mu_max=max(X_train), sigma_min=.1, sigma_max=1, phi=.5):
        self.X_train = X_train
        # init with multiple gaussians
        self.one = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))
        self.two = Gaussian(uniform(mu_min, mu_max),
                            uniform(sigma_min, sigma_max))

        # as well as how much to mix them
        self.phi = phi

    def Estep(self):
        "Perform an E(stimation)-step, freshening up self.loglike in the process"
        # compute weights
        self.loglike = 0.  # = log(p = 1)
        for datapoint in self.X_train:
            # unnormalized weights
            wp1 = self.one.pdf(datapoint, 1) * self.phi
            wp2 = self.two.pdf(datapoint, 2) * (1. - self.phi)
            # compute denominator
            den = wp1 + wp2
            # normalize
            wp1 /= den
            wp2 /= den
            # add into loglike
            self.loglike += log(wp1 + wp2)
            # yield weight tuple
            yield (wp1, wp2)

    def Mstep(self, weights):
        "Perform an M(aximization)-step"
        # compute denominators
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)
        # compute new means
        self.one.mu = sum(w * d / one_den for (w, d) in zip(left, X_train))
        self.two.mu = sum(w * d / two_den for (w, d) in zip(rigt, X_train))
        # compute new sigmas
        self.one.sigma = sqrt(sum(w * ((d - self.one.mu) ** 2)
                                  for (w, d) in zip(left, X_train)) / one_den)
        self.two.sigma = sqrt(sum(w * ((d - self.two.mu) ** 2)
                                  for (w, d) in zip(rigt, X_train)) / two_den)
        # compute new mix
        self.mix = one_den / len(X_train)

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then compute log-likelihood"

    def pdf(self, x):
        return (self.mix) * self.one.pdf(x) + (1 - self.mix) * self.two.pdf(x)

    def __repr__(self):
        return 'GaussianMixture({0}, {1}, mix={2.03})'.format(self.one,
                                                              self.two,
                                                              self.mix)

    def __str__(self):
        return 'Mixture: {0}, {1}, mix={2:.03})'.format(self.one,
                                                        self.two,
                                                        self.mix)


# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()

