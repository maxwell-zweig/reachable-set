from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
import matplotlib.animation as animation


def phi_calc(A, B, th):
    uvec = np.array([np.cos(theta), np.sin(theta)])
    invA_vec = np.linalg.solve(A, uvec)
    invB_vec = np.linalg.solve(B, uvec)
    return np.linalg.norm(invB_vec)/np.linalg.norm(invA_vec)



def reachable_set_calcs(t2, t1):
    print(f"Computing statistics for terminal time: {t2}, thrust time: {t1}")

    STM = lambda t, n: expm(
        [
            [0, 0, 0, 1 * t, 0, 0],
            [0, 0, 0, 0, 1 * t, 0],
            [0, 0, 0, 0, 0, 1 * t],
            [3 * t * n**2, 0, 0, 0, 2 * n * t, 0],
            [0, 0, 0, -2 * n * t, 0, 0],
            [0, 0, -t * n**2, 0, 0, 0],
        ]
    )

    ITM = lambda t, n: [
        [(1 - np.cos(t * n)) / n**2, -2 * (-t * n + np.sin(t * n)) / n**2, 0],
        [
            2 * (-t * n + np.sin(t * n)) / n**2,
            (-3 * t**2) / 2 + 4 * (1 - np.cos(t * n)) / n**2,
            0,
        ],
        [0, 0, (1 - np.cos(t * n)) / n**2],
        [np.sin(t * n) / n, 2 * (-1 + np.cos(t * n)) / n, 0],
        [2 * (-1 + np.cos(t * n)) / n, -3 * t + 4 * np.sin(t * n) / n, 0],
        [0, 0, np.sin(t * n) / n],
    ]

    a_ = np.array(STM(t2, 1)[0:2, 3:5])
    aT = STM(t2, 1)[0:2, 3:5].T

    b_ = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1)
    bT = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1).T

    w, v = eigh(
        np.linalg.inv(aT) @ np.linalg.inv(a_), np.linalg.inv(bT) @ np.linalg.inv(b_)
    )

    v0 = v[:, 0]
    v1 = v[:, 1]
    # print(np.sqrt(w))
    # print(v0, v1)

    t0 = np.arctan2(v0[1], v0[0])
    t1 = np.arctan2(v1[1], v1[0])

    # print(t0, t1)
    #
    #   plt.plot(phis, res)
    #   plt.axvline(x=t0)
    #   plt.axvline(x=t1)
    #   plt.show()
    #   print()
    return np.sqrt(w), t0, t1
print(reachable_set_calcs(1.1*2.*np.pi, .1*2.*np.pi))