from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh


def plot_reachable_sets(A, B):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol("theta")
    q1 = A @ Matrix([cos(theta), sin(theta)])
    q2 = B @ Matrix([cos(theta), sin(theta)])

    plot_parametric(
        (q1[0], q1[1]), (q2[0], q2[1]), (theta, 0, 2 * math.pi), xlabel="X", ylabel="Y"
    )


def phi_calc(A, B, th):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol("theta")

    res_vec_B = B.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)
    res_vec_A = A.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)

    return sqrt(res_vec_B.dot(res_vec_B)) / sqrt(res_vec_A.dot(res_vec_A))


t2 = 10
t1 = 0.1


def reachable_set_calcs(t2, t1):

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

    U_1, S_1, V_1 = svd(STM(t2, 1)[0:2, 3:5])

    transform_mat_1 = [U_1.T[0] * S_1[0], U_1.T[1] * S_1[1]]
    transform_mat_1 = np.array(transform_mat_1).T

    itm = np.array(ITM(t1, 1))
    stm = np.array(STM(t2 - t1, 1))

    U_2, S_2, V_2 = svd((stm[0:2, 0:6] @ itm)[:2, :2] / t1)

    transform_mat_2 = [U_2.T[0] * S_2[0], U_2.T[1] * S_2[1]]
    transform_mat_2 = np.array(transform_mat_2).T

    plot_reachable_sets(STM(t2, 1)[0:2, 3:5], (stm[0:2, 0:6] @ itm)[:2, :2] / t1)
    plot_reachable_sets(transform_mat_1, transform_mat_2)

    phis = np.linspace(0, 2 * math.pi, 628)
    res = [phi_calc(transform_mat_1, transform_mat_2, phi) for phi in phis]
    res_min = min(res)
    res_max = max(res)

    a_ = np.array(STM(t2, 1)[0:2, 3:5])
    aT = STM(t2, 1)[0:2, 3:5].T

    b_ = np.array((stm[0:2, 0:6] @ itm)[:2, :2] / t1)
    bT = np.array((stm[0:2, 0:6] @ itm)[:2, :2] / t1).T

    w, v = eigh(
        a=np.linalg.inv(aT) @ np.linalg.inv(a_), b=np.linalg.inv(bT) @ np.linalg.inv(b_)
    )

    v0 = v[:, 0]
    v1 = v[:, 1]
    print(np.sqrt(w))
    print(v0, v1)

    t0 = np.arctan2(v0[1], v0[0])
    t1 = np.arctan2(v1[1], v1[0])

    print(t0, t1)

    plt.plot(phis, res)
    plt.axvline(x=t0)
    plt.axvline(x=t1)
    plt.show()
    print()
    return np.sqrt(w), t0, t1


# reachable_set_calcs(10, 0.1)
# reachable_set_calcs(10, 0.01)
# reachable_set_calcs(10, 0.001)
# reachable_set_calcs(10, 0.0001)
# quit()
"""

reachable_set_calcs(10, 0.1)
reachable_set_calcs(10, 0.01)
reachable_set_calcs(10, 0.001)
reachable_set_calcs(10, 0.0001)

print('next t1')

reachable_set_calcs(10, 1)
reachable_set_calcs(9, 1)
reachable_set_calcs(8, 1)
reachable_set_calcs(7, 1)
reachable_set_calcs(6, 1)
reachable_set_calcs(5, 1)
reachable_set_calcs(4, 1)
reachable_set_calcs(3, 1)
reachable_set_calcs(2, 1)
reachable_set_calcs(1, 1)

print('next t1')

reachable_set_calcs(10, 2)
reachable_set_calcs(9, 2)
reachable_set_calcs(8, 2)
reachable_set_calcs(7, 2)
reachable_set_calcs(6, 2)
reachable_set_calcs(5, 2)
reachable_set_calcs(4, 2)
reachable_set_calcs(3, 2)
reachable_set_calcs(2, 2)

"""


l1 = [reachable_set_calcs(i / 10, 0.05) for i in range(1, 100)]
l2 = [reachable_set_calcs(i / 10, 0.1) for i in range(1, 100)]
l3 = [reachable_set_calcs(i / 10, 0.15) for i in range(1, 100)]

val_mins_l1 = [ele[0][0] for ele in l1]
val_maxs_l1 = [ele[0][1] for ele in l1]

val_mins_l2 = [ele[0][0] for ele in l2]
val_maxs_l2 = [ele[0][1] for ele in l2]

val_mins_l3 = [ele[0][0] for ele in l3]
val_maxs_l3 = [ele[0][1] for ele in l3]


theta_min_l1 = [ele[1] for ele in l1]
theta_max_l1 = [ele[2] for ele in l1]
theta_min_l2 = [ele[1] for ele in l2]
theta_max_l2 = [ele[2] for ele in l2]
theta_min_l3 = [ele[1] for ele in l3]
theta_max_l3 = [ele[2] for ele in l3]


plt.plot([i / 10 for i in range(1, 100)], val_mins_l1, label="l1 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l1, label="l1 max")
plt.plot([i / 10 for i in range(1, 100)], val_mins_l2, label="l2 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l2, label="l2 max")
plt.plot([i / 10 for i in range(1, 100)], val_mins_l3, label="l3 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l3, label="l3 max")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Reacahble set min/max Ratios")
plt.title("Reachable Set min/max Ratios vs t2")
plt.yscale("log")
plt.figure()
plt.plot([i / 10 for i in range(1, 100)], theta_min_l1, label="l1 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l1, label="l1 max")
plt.plot([i / 10 for i in range(1, 100)], theta_min_l2, label="l2 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l2, label="l2 max")
plt.plot([i / 10 for i in range(1, 100)], theta_min_l3, label="l3 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l3, label="l3 max")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Reacahble set min/max thetas")
plt.title("Reachable Set min/max thetas vs t2")


plt.show()
