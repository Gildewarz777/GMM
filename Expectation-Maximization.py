import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import sqrt, log, exp, pi
import matplotlib as mpl
import itertools
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

mu = np.random.rand(2, 2) * 30
sigma = np.array([[0.25, 0.30], [0.3, 1]])
sigma = np.stack((sigma, sigma), 2)
phi = 0.5
weights = np.zeros((len(X_train), 2))

# training models and choosing the best one
best_gmm = None
best_loglike = float("-inf")

for i in range(0, 3):
    ##### E-step #####
    loglike = 0
    for i in range(0, len(X_train)):
        # Unnormalized weights
        wp1 = 1 / ((2 * pi) * sqrt(np.linalg.det(sigma[:, :, 0]))) * np.exp(-0.5 * np.matmul(np.matmul((X_train[i] - mu[0]), np.linalg.inv(sigma[:, :, 0])), (X_train[i] - mu[0]))) * phi
        wp2 = 1 / ((2 * pi) * sqrt(np.linalg.det(sigma[:, :, 1]))) * np.exp(-0.5 * np.matmul(np.matmul((X_train[i] - mu[1]), np.linalg.inv(sigma[:, :, 1])), (X_train[i] - mu[1]))) * (1 - phi)

        # Denominator
        den = (wp1 + wp2)

        # Incrementing loglike
        loglike += log(wp1 + wp2)

        # Weight tuple added
        weights[i] = np.array([wp1 / den, wp2 / den])

    ##### M-step #####
    # Phi update
    phi = (1 / len(X_train)) * np.sum(weights, 0)[0]

    # Mu update
    mu[0] = np.sum(weights[:, 0, None] * X_train, 0) / np.sum(weights, 0)[0]
    mu[1] = np.sum(weights[:, 1, None] * X_train, 0) / np.sum(weights, 0)[1]

    # Sigma update
    numerator = np.zeros((2, 2, 2))
    for i in range(0, len(X_train)):
        numerator[0] += weights[i, 0] * (X_train[i] - mu[0]).reshape((2, 1)) * (X_train[i] - mu[0])
        numerator[1] += weights[i, 1] * (X_train[i] - mu[1]).reshape((2, 1)) * (X_train[i] - mu[1])
    sigma[0] = numerator[0] / np.sum(weights, 0)[0]
    sigma[1] = numerator[1] / np.sum(weights, 0)[1]


# Printing results
colors = ['lightcoral', 'slateblue']
color_iter = itertools.cycle(['red', 'blue', 'cornflowerblue', 'gold', 'darkorange'])
labels = [colors[int(np.argmax(res))] for res in weights]

print("\nLog-likelihood of the best GMM = " + str(loglike) + "\nPhi of the best GMM = " + str(phi))


predicted = np.array([int(np.argmax(weights[i])) for i in range(0, len(X_train))])
subplot = plt.subplot(2, 1, 1)

for i, (mean, covar, color) in enumerate(zip(mu, sigma, color_iter)):
    v, w = np.linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    # as the DP will not use every component it has access to
    # unless it needs it, we shouldn't plot the redundant
    # components.
    if not np.any(predicted == i):
        continue
    plt.scatter(X_train[predicted == i, 0], X_train[predicted == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(subplot.bbox)
    ell.set_alpha(0.5)
    subplot.add_artist(ell)

plt.xlim(-10, 23)
plt.ylim(-10, 23)
plt.xticks(())
plt.yticks(())
plt.title('Negative log-likelihood predicted by a GMM - My method')

plt.show()


# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

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

plt.title('Negative log-likelihood predicted by a GMM - The sklearn method')
plt.axis('tight')
plt.show()
