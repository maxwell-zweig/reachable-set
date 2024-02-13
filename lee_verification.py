from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
from sympy import symbols, Eq
from sympy.plotting import plot_implicit

"""
Takes in a positive semi-definite matrix M and a parameter m that determines shape and orientation of the ellipsoid. 
This satisfies x^TMx \leq m. Assuming further that M is not a degenerate ellipsoid in any dimensions (ie. initial velocity in one direction is not just zero but a range of possible values)
"""


import numpy as np
from scipy.linalg import cholesky  # computes upper triangle by default, matches paper


def sample(S, z_hat, m_FA, Gamma_Threshold=1.0):

    nz = S.shape[0]
    z_hat = z_hat.reshape(nz, 1)

    X_Cnz = np.random.normal(size=(nz, m_FA))

    rss_array = np.sqrt(np.sum(np.square(X_Cnz), axis=0))
    kron_prod = np.kron(np.ones((nz, 1)), rss_array)

    X_Cnz = X_Cnz / kron_prod  # Points uniformly distributed on hypersphere surface

    R = np.ones((nz, 1)) * (np.power(np.random.rand(1, m_FA), (1.0 / nz)))

    unif_sph = R * X_Cnz
    # m_FA points within the hypersphere
    T = np.asmatrix(cholesky(S))  # Cholesky factorization of S => S=T’T

    unif_ell = T.H * unif_sph
    # Hypersphere to hyperellipsoid mapping

    # Translation and scaling about the center
    z_fa = unif_ell * np.sqrt(Gamma_Threshold) + (z_hat * np.ones((1, m_FA)))

    return np.array(z_fa)


def plot_elipsoid(M, m):
    pass


#  print(equation)
#  plot_implicit(equation)


if __name__ == "__main__":
    # generating positive semi-definite matrix
    matrixSize = 6
    A = np.random.rand(matrixSize, matrixSize)
    M = np.dot(A, A.transpose())

    m = 100
    # plot_elipsoid(M, m)
    initial_reachable_set_sampling(M, m)
