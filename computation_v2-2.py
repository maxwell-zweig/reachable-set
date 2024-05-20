from sympy import *
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eigh
import matplotlib.animation as animation
from spb import *


def plot_reachable_sets(A, B, thrusttime, terminaltime, l1, integer):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol("theta")
    q1 = A @ Matrix([cos(theta), sin(theta)])
    q2 = B @ Matrix([cos(theta), sin(theta)])

    """
    p = plot_parametric(
        (q1[0], q1[1]),
        (q2[0], q2[1]),
        (theta, 0, 2 * math.pi),
        xlabel="X Reachable Set",
        ylabel="Y Reachable Set",
        show=False,
        title="Impulsive vs Constant Thrust Reachable Sets",
    )
    p.save(f"plots/{round(thrusttime, 3)}tt{round(terminaltime, 3)}{l1}.png")
    """

    p = graphics(
        line_parametric_2d(
            q1[0],
            q1[1],
            (theta, 0, 2 * math.pi),
            use_cm=False,
            label="Impulsive RRS",
        ),
        line_parametric_2d(
            q2[0],
            q2[1],
            (theta, 0, 2 * math.pi),
            use_cm=False,
            label="Constant Thrust RRS",
        ),
        xlabel="x (meters)",
        ylabel="y (meters)",
        legend=True,
        show=False,
        title="X-Y Position Relative Reachable Set",
    )
    p.save(f"plots/{integer}tt{l1}.png")


def phi_calc(A, B, th):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol("theta")

    res_vec_B = B.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)
    res_vec_A = A.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)

    return sqrt(res_vec_B.dot(res_vec_B)) / sqrt(res_vec_A.dot(res_vec_A))


t2 = 10
t1 = 0.1


def reachable_set_calcs(t2, t1, l1, i):
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

    U_1, S_1, V_1 = svd(STM(t2, 1)[0:2, 3:5])

    transform_mat_1 = [U_1.T[0] * S_1[0], U_1.T[1] * S_1[1]]
    transform_mat_1 = np.array(transform_mat_1).T

    itm = np.array(ITM(t1, 1))
    stm = np.array(STM(t2 - t1, 1))

    U_2, S_2, V_2 = svd((stm[0:2, 0:6] @ itm)[:2, :2] / t1)

    transform_mat_2 = [U_2.T[0] * S_2[0], U_2.T[1] * S_2[1]]
    transform_mat_2 = np.array(transform_mat_2).T

    # plot_reachable_sets(STM(t2, 1)[0:2, 3:5], (stm[0:2, 0:6] @ itm)[:2, :2] / t1)
    plot_reachable_sets(transform_mat_1, transform_mat_2, t1, t2, l1, i)

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


def plotsaver(terminal_time=4 * math.pi, max_thrust_time=3 * math.pi, num_plots=200):

    for i in range(np.linspace(0.01, max_thrust_time, num_plots).shape[0]):

        cur_thrust_time = np.linspace(0.01, max_thrust_time, num_plots)[i]
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

        U_1, S_1, V_1 = svd(STM(tt, 1)[0:2, 3:5])

        transform_mat_1 = [U_1.T[0] * S_1[0], U_1.T[1] * S_1[1]]
        transform_mat_1 = np.array(transform_mat_1).T

        itm = np.array(ITM(cur_thrust_time, 1))
        stm = np.array(STM(tt - cur_thrust_time, 1))

        U_2, S_2, V_2 = svd((stm[0:2, 0:6] @ itm)[:2, :2] / cur_thrust_time)

        transform_mat_2 = [U_2.T[0] * S_2[0], U_2.T[1] * S_2[1]]
        transform_mat_2 = np.array(transform_mat_2).T

        plot_reachable_sets(
            STM(t2, 1)[0:2, 3:5],
            (stm[0:2, 0:6] @ itm)[:2, :2] / cur_thrust_time,
            cur_thrust_time,
            tt,
        )
        # plot_reachable_sets(transform_mat_1, transform_mat_2, cur_thrust_time, tt)


# plotsaver()


l1 = [reachable_set_calcs(8, i / 20, "l1", i) for i in range(1, 200)]
l2 = [reachable_set_calcs(10, i / 20, "l2", i) for i in range(1, 200)]
l3 = [reachable_set_calcs(12, i / 20, "l3", i) for i in range(1, 200)]
"""

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


plt.legend()
plt.xlabel("Time")
plt.ylabel("Reacahble set min/max Ratios")
plt.title("Reachable Set min/max Ratios vs t2")
plt.yscale("log")

plt.plot([i / 10 for i in range(1, 100)], val_mins_l1, label="l1 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l1, label="l1 max")
plt.savefig("figs/ratiosvstimel1.png")

plt.plot([i / 10 for i in range(1, 100)], val_mins_l2, label="l2 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l2, label="l2 max")

plt.savefig("figs/ratiosvstimel2.png")
plt.plot([i / 10 for i in range(1, 100)], val_mins_l3, label="l3 min")
plt.plot([i / 10 for i in range(1, 100)], val_maxs_l3, label="l3 max")

plt.savefig("figs/ratiosvstimel3.png")

plt.figure()

plt.legend()
plt.xlabel("Time")
plt.ylabel("Reacahble set min/max thetas")
plt.title("Reachable Set min/max thetas vs t2")
plt.yscale("log")


plt.plot([i / 10 for i in range(1, 100)], theta_min_l1, label="l1 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l1, label="l1 max")
plt.savefig("figs/thetassvstimel1.png")


plt.plot([i / 10 for i in range(1, 100)], theta_min_l2, label="l2 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l2, label="l2 max")

plt.savefig("figs/thetassvstimel2.png")


plt.plot([i / 10 for i in range(1, 100)], theta_min_l3, label="l3 min")
plt.plot([i / 10 for i in range(1, 100)], theta_max_l3, label="l3 max")

plt.savefig("figs/thetasvstimel3.png")

"""

"""


llist = []
for j in range(1, 2):
    varryingt1 = []
    for i in range(1, 20):
        if i / 10 > j / 10:
            stats = reachable_set_calcs(i / 10, j / 10)
            varryingt1.append(stats)
    llist.append(varryingt1)

# print(len(llist))
# print(llist)
valmins = []
for j in range(1, 2):
    varryingt1 = []
    for i in range(1, 20):
        if i / 10 > j / 10:
            listlength = len(llist[j - 1])
            stats = llist[j - 1][i - (19 - listlength) - 1][0][0]
            varryingt1.append(stats)
    valmins.append(varryingt1)

valmaxs = []
for j in range(1, 2):
    varryingt1 = []
    for i in range(1, 20):
        if i / 10 > j / 10:
            listlength = len(llist[j - 1])
            stats = llist[j - 1][i - (19 - listlength) - 1][0][1]
            varryingt1.append(stats)
    valmaxs.append(varryingt1)

thetamins = []
for j in range(1, 2):
    varryingt1 = []
    for i in range(1, 20):
        if i / 10 > j / 10:
            listlength = len(llist[j - 1])
            stats = llist[j - 1][i - (19 - listlength) - 1][1]
            varryingt1.append(stats)
    thetamins.append(varryingt1)

thetamaxs = []
for j in range(1, 2):
    varryingt1 = []
    for i in range(1, 20):
        if i / 10 > j / 10:
            listlength = len(llist[j - 1])
            stats = llist[j - 1][i - (19 - listlength) - 1][2]
            varryingt1.append(stats)
    thetamaxs.append(varryingt1)

The first dimension of each of these lists varies as thrust time varies. The second time varies as final time varies. 
So since the first dimension of the list has size 9, there are going to be 9 different animations associated with each thrust time. Each animation contains the change in statistics as terminal time varies. 
fig, ax = plt.subplots()
ims = []
for i in range(len(thetamaxs[0])):
    x = np.linspace(20 - len(thetamaxs[0]), 20, 18)
    x /= 10
    print(len(thetamaxs[0]))
    print(len(x))
    im = ax.imshow(x, thetamaxs[0], animated=True)
    if i == 0:
        ax.imshow(x, thetamaxs[0])  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)


# To save the animation, use e.g.
#
# ani.save("movie.mp4")


for j in range(9):

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Reacahble set min/max thetas")
    plt.title("Reachable Set min/max thetas vs t2")
    plt.yscale("log")

    plt.plot([i / 10 for i in range(1, 100)], thetamins[j], label="l1 min")
    plt.plot([i / 10 for i in range(1, 100)], thetamaxs[j], label="l1 max")
    plt.savefig("figs/thetassvsthrusttimel1.png")

    plt.figure()
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Reacahble set min/max ratios")
    plt.title("Reachable Set min/max ratios vs t2")
    plt.yscale("log")

    plt.plot([i / 10 for i in range(1, 100)], valmins[j], label="l1 min")
    plt.plot([i / 10 for i in range(1, 100)], valmaxs[j], label="l1 max")
"""
