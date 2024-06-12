from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
import matplotlib.animation as animation

    
def plot_reachable_sets_matplotlib(As, Bs, title):
    ang = np.linspace(0,2.*np.pi,200)
    vecs = np.vstack((np.cos(ang), np.sin(ang)))
    allAreachable = As[0][:2,:2] @ vecs
    allBreachable = Bs[0][:2,:2] @ vecs
    for j in range(len(As)):
        A = As[j]
        B = Bs[j]
        Areachable = A[:2,:2] @ vecs
        Breachable = B[:2,:2] @ vecs
        allAreachable = np.hstack((allAreachable, Areachable))
        allBreachable = np.hstack((allBreachable, Breachable))
    i = 1
    for j in range(len(As)):
        Areachable = As[j][:2,:2] @ vecs
        Breachable = Bs[j][:2,:2] @ vecs
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(Areachable[1], Areachable[0], label="Impulsive", linewidth=4)
        ax.plot(Breachable[1], Breachable[0], label="Constant Thrust",linewidth=4)
        ax.legend(fontsize=12)
        ax.set_xlabel("y (in-track)", fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.tick_params(axis="x", labelsize=16)
        ax.set_ylabel("x (radial)", fontsize=18)
        ax.set_xlim([1.1*np.min(allBreachable[1]), 1.1*np.max(allBreachable[1])])
        ax.set_ylim([1.1*np.min(allBreachable[0]), 1.1*np.max(allBreachable[0])])
        plt.savefig(f"plots/{title}_{i}_impulsive_v_constant_RIC_terminal_time_varying.png")
        plt.close()
        i += 1
    
    
    
def plotsaver(title, max_terminal_time=2 * 2 * math.pi, thrust_time=.01 * 2 * math.pi, num_plots=20):

    terminal_times = np.linspace(0.001, max_terminal_time, num_plots)
    itms = []
    stms = []
    for i in range(len(terminal_times)):

        cur_terminal_time = terminal_times[i]

        print(
            f"Computing statistics for terminal time: {cur_terminal_time}, thrust time: {thrust_time}"
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

        itm = np.array(ITM(thrust_time, 1))
        stm = np.array(STM(cur_terminal_time - thrust_time, 1))

        itms.append((stm[0:2, 0:6] @ itm)[:2, :2] / thrust_time)
        stm = STM(cur_terminal_time, 1)[0:2, 3:5]
        stms.append(stm)
    plot_reachable_sets_matplotlib(stms, itms, title)
        
plotsaver("2rev", 2*2*math.pi, .01*2*math.pi, 100)

#plot_reachable_sets_matplotlib(np.identity(3), [2.*np.identity(2), 3.*np.identity(2)])