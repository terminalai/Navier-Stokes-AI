# large portion of code in this file is stolen from https://peterroelants.github.io/posts/gaussian-process-tutorial/

import scipy
import numpy as np


def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -200*scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """

    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance


def generate_random_functions(num, resolution=1000, points=np.array(((0, 0), (1, 0)))):
    """
    Generates random initial conditions which the model will be given as input
    :param num: The number of random functions to generate
    :param resolution: The resolution of these functions
    :param points: The points that this function must pass through
    :return: Returns an array of these random functions
    """
    # Sample observations (X1, y1) on the function
    X1 = points[:, 0, np.newaxis]
    y1 = points[:, 1, np.newaxis]

    # Predict points at uniform spacing to capture function
    X2 = np.linspace(0, 1, resolution).reshape(-1, 1)

    # Compute posterior mean and covariance
    μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)

    # Draw some samples of the posterior
    y2 = np.random.multivariate_normal(mean=μ2[:, 0], cov=Σ2, size=num)
    return y2
