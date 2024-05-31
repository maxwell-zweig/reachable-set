from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
import matplotlib.animation as animation

    
def plot_reachable_sets_matplotlib(A, Bs, title):
    ang = np.linspace(0,2.*np.pi,200)
    vecs = np.vstack((np.cos(ang), np.sin(ang)))
    Areachable = A[:2,:2] @ vecs
    allbreachable = Bs[0][:2,:2] @ vecs
    for B in Bs:
        Breachable = B[:2,:2] @ vecs
        allbreachable = np.hstack((allbreachable, Breachable))
        print(allbreachable)
    i = 1
    for B in Bs:
        Breachable = B[:2,:2] @ vecs
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(Areachable[1], Areachable[0], label="Impulsive", linewidth=4)
        ax.plot(Breachable[1], Breachable[0], label="Constant Thrust",linewidth=4)
        ax.legend(fontsize=12)
        ax.set_xlabel("y (in-track)", fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel("x (radial)", fontsize=18)
        ax.set_xlim([1.1*np.min(allbreachable[1]), 1.1*np.max(allbreachable[1])])
        ax.set_ylim([1.1*np.min(allbreachable[0]), 1.1*np.max(allbreachable[0])])
        plt.savefig(f"plots/{title}_{i}_impulsive_v_constant_RIC.png")
        plt.close()
        i += 1
    
    
    
def plotsaver(title, terminal_time=.75 * 2 * math.pi, max_thrust_time=.1 * 2 * math.pi, num_plots=20):

    thrust_times = np.linspace(0.001, max_thrust_time, num_plots)
    itms = []
    for i in range(len(thrust_times)):

        cur_thrust_time = thrust_times[i]
        tt = terminal_time

        print(
            f"Computing statistics for terminal time: {tt}, thrust time: {cur_thrust_time}"
        )

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

        itm = np.array(ITM(cur_thrust_time, 1))
        stm = np.array(STM(tt - cur_thrust_time, 1))

        itms.append((stm[0:2, 0:6] @ itm)[:2, :2] / cur_thrust_time)
        stm = STM(tt, 1)[0:2, 3:5]
    plot_reachable_sets_matplotlib(stm, itms, title)
        
plotsaver("1.6rev", 1.6*2*math.pi, .1*2*math.pi, 50)
plotsaver("1.1rev", 1.1*2*math.pi, .1*2*math.pi, 50)
plotsaver("0.25rev", .25*2*math.pi, .1*2*math.pi, 50)

#plot_reachable_sets_matplotlib(np.identity(3), [2.*np.identity(2), 3.*np.identity(2)])