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

    print(np.linalg.inv(aT) @ np.linalg.inv(a_) @ v0 / np.linalg.norm(np.linalg.inv(aT) @ np.linalg.inv(a_) @ v0))
    print(np.linalg.inv(bT) @ np.linalg.inv(b_) @ v0 / np.linalg.norm(np.linalg.inv(bT) @ np.linalg.inv(b_) @ v0))

    print(np.linalg.inv(aT) @ np.linalg.inv(a_) @ v1 / np.linalg.norm(np.linalg.inv(aT) @ np.linalg.inv(a_) @ v1))
    print(np.linalg.inv(bT) @ np.linalg.inv(b_) @ v1 / np.linalg.norm(np.linalg.inv(bT) @ np.linalg.inv(b_) @ v1))

    theta0 = np.arctan(v0[1]/v0[0])
    theta1 = np.arctan(v1[1]/v1[0])

    return np.sqrt(w), theta0, theta1
print(reachable_set_calcs(.25*2.*np.pi, .00001*2.*np.pi))

dt = .00001*2.*np.pi
tofs = np.linspace(.1*2.*np.pi, 2.01*2*np.pi,100)
derivative1 = []
derivative2 = []
angle1 = []
angle2 = [] 
for tof in tofs:
    ratios, theta1, theta2 = reachable_set_calcs(tof, dt)
    derivative1.append((ratios[0]-1.)/dt)
    derivative2.append((ratios[1]-1.)/dt)
    angle1.append(theta1)
    angle2.append(theta2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(tofs, derivative1, label="Ratio 1", linewidth=4)
ax.plot(tofs, derivative2, label="Ratio 2", linewidth=4)
ax.legend(fontsize=12)
ax.set_xlabel("Time of Flight (rad)", fontsize=18)
ax.tick_params(axis="y", labelsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.set_ylabel("Ratio Time Derivative (1/rad)", fontsize=18)
ax.set_ylim([-5, 5])
plt.plot(tofs, np.zeros(len(tofs)), linestyle="dashed", color="gray")
plt.savefig(f"plots/derivatives_plot.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(tofs, angle1, label="Ratio 1", linewidth=4)
ax.plot(tofs, angle2, label="Ratio 2", linewidth=4)
ax.legend(fontsize=12)
ax.set_xlabel("Time of Flight (rad)", fontsize=18)
ax.tick_params(axis="y", labelsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.set_ylabel("Angle (rad)", fontsize=18)
plt.savefig(f"plots/angles_plot.png")
plt.close()

