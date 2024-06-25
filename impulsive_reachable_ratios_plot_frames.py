from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
import matplotlib.animation as animation
from itertools import cycle


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
    
    ITM_ECI = lambda t, n: [[((-1 + 3 * np.cos(n * t)) * np.sin((n * t)/2)**2)/
          n**2, -((2 * n * t - 8 * np.sin(n * t) + 3 * np.sin(2 * n * t))/(4 * n**2)), 
          0], [((1 + 3 * np.cos(n * t)) * (-1 * n * t + np.sin(n * t)))/n**2, (
          1 - np.cos(n * t) - 3 * n * t * np.sin(n * t) + 3 * np.sin(n * t)**2)/n**2, 0], [0, 0, (
          1 - np.cos(n * t))/n**2], [-(t/2) + (3 * np.sin(2 * n * t))/(4 * n), (
          3 * np.sin(n * t)**2)/(2 * n), 0], [-((6 * np.cos(n * t) * np.sin((n * t)/2)**2)/n), (
          n * t - 3 * np.sin(n * t) + 3/2 * np.sin(2 * n * t))/n, 0], [0, 0, np.sin(n * t)/n]
    ]
    
    ITM_ECI_centered = lambda t, n: [[(np.sin((n * t)/2) * (-n * t + 3 * np.sin(n * t)))/(2 * n**2), (
          8 * np.sin((n * t)/2) - np.cos((n * t)/2) * (n * t + 3 * np.sin(n * t)))/(2 * n**2), 
          0], [(-8 * n * t * np.cos((n * t)/2) + 7 * np.sin((n * t)/2) + 3 * np.sin((3 * n * t)/2))/(
          2 * n**2), (np.sin((n * t)/2) * (-2 * n * t + 3 * np.sin(n * t)))/n**2, 0], [0, 0, (
          1 - np.cos(n * t))/n**2], [(np.cos((n * t)/2) * (-1. * n * t + 3 * np.sin(n * t)))/(2 * n), (
          np.sin((n * t)/2) * (n * t + 3 * np.sin(n * t)))/(2 * n), 0], [(
          np.sin((n * t)/2) * (n * t - 3 * np.sin(n * t)))/n, 
          t * np.cos((n * t)/2) - (6 * np.sin((n * t)/2)**3)/n, 0], [0, 0, np.sin(n * t)/n]
    ]
    
    

    a_ = np.array(STM(t2, 1)[0:2, 3:5])
    aT = STM(t2, 1)[0:2, 3:5].T

    b_ = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1)
    bT = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1).T
    
    c_ = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM_ECI(t1, 1))[:2, :2] / t1)
    cT = np.array((STM(t2-t1, 1)[0:2, 0:6] @ ITM_ECI(t1, 1))[:2, :2] / t1).T
    
    d_ = np.array((STM(t2-t1/2, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1)
    dT = np.array((STM(t2-t1/2, 1)[0:2, 0:6] @ ITM(t1, 1))[:2, :2] / t1).T
    
    e_ = np.array((STM(t2-t1/2, 1)[0:2, 0:6] @ ITM_ECI_centered(t1, 1))[:2, :2] / t1)
    eT = np.array((STM(t2-t1/2, 1)[0:2, 0:6] @ ITM_ECI_centered(t1, 1))[:2, :2] / t1).T

    w, _ = eigh(
        np.linalg.inv(aT) @ np.linalg.inv(a_), np.linalg.inv(bT) @ np.linalg.inv(b_)
    )
    
    w1, _ = eigh(
        np.linalg.inv(aT) @ np.linalg.inv(a_), np.linalg.inv(cT) @ np.linalg.inv(c_)
    )
    
    w2, _ = eigh(
        np.linalg.inv(aT) @ np.linalg.inv(a_), np.linalg.inv(dT) @ np.linalg.inv(d_)
    )
    
    w3, _ = eigh(
        np.linalg.inv(aT) @ np.linalg.inv(a_), np.linalg.inv(eT) @ np.linalg.inv(e_)
    )


    return np.sqrt(w),np.sqrt(w1),np.sqrt(w2),np.sqrt(w3) 
print(reachable_set_calcs(.25*2.*np.pi, .00001*2.*np.pi))

dt = .05*2.*np.pi
tofs = np.linspace(.1*2.*np.pi, 2.01*2*np.pi,100)
derivative1 = []
derivative2 = []
derivative11 = []
derivative12 = []
derivative21 = []
derivative22 = []
derivative31 = []
derivative32 = []
for tof in tofs:
    ratios, ratios1, ratios2, ratios3 = reachable_set_calcs(tof, dt)
    derivative1.append((ratios[0]-1.)/dt)
    derivative2.append((ratios[1]-1.)/dt)
    derivative11.append((ratios1[0]-1.)/dt)
    derivative12.append((ratios1[1]-1.)/dt)
    derivative21.append((ratios2[0]-1.)/dt)
    derivative22.append((ratios2[1]-1.)/dt)
    derivative31.append((ratios3[0]-1.)/dt)
    derivative32.append((ratios3[1]-1.)/dt)



fig, ax = plt.subplots(figsize=(8, 6))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = cycle(prop_cycle.by_key()['color'])

#ax.plot(tofs, derivative1, label="RIC", linewidth=4)
#ax.plot(tofs, derivative2, label="RIC", linewidth=4)
#ax.plot(tofs, derivative11, label="ECI", linewidth=4)
#ax.plot(tofs, derivative12, label="ECI", linewidth=4)
#ax.plot(tofs, derivative21, label="RIC Centered", linewidth=4)
#ax.plot(tofs, derivative22, label="RIC Centered", linewidth=4)
col = next(colors)
ax.plot(tofs, np.array(derivative31)*dt+1, label="ECI Centered", linewidth=4, color=col)
ax.plot(tofs, np.array(derivative32)*dt+1, linewidth=4, color=col)

col = next(colors)
ax.plot(tofs, np.array(derivative11)*dt+1, label="ECI", linewidth=4, color=col)
ax.plot(tofs, np.array(derivative12)*dt+1, linewidth=4, color=col)

col = next(colors)
ax.plot(tofs, np.array(derivative21)*dt+1, label="RIC Centered", linewidth=4, color=col)
ax.plot(tofs, np.array(derivative22)*dt+1, linewidth=4, color=col)

col = next(colors)
ax.plot(tofs, np.array(derivative1)*dt+1, label="RIC", linewidth=4, color=col)
ax.plot(tofs, np.array(derivative2)*dt+1, linewidth=4, color=col)
print(derivative31)
print(derivative32)
ax.legend(fontsize=12)
ax.set_xlabel("Time of Flight (rad)", fontsize=18)
ax.tick_params(axis="y", labelsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.set_ylabel("Ratio Time Derivative (1/rad)", fontsize=18)
ax.set_ylim([0, 2])
plt.plot(tofs, np.ones(len(tofs)), linestyle="dashed", color="gray")
plt.savefig(f"plots/derivatives_plot.png")
plt.close()


