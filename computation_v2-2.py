from sympy import * 
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.linalg import expm
from scipy.linalg import eig


def plot_reachable_sets(A, B):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol('theta') 
    q1 = A @ Matrix([cos(theta), sin(theta)])
    q2 = B @ Matrix([cos(theta), sin(theta)])

    plot_parametric((q1[0], q1[1]), (q2[0], q2[1]), (theta, 0, 2 * math.pi), xlabel='X', ylabel='Y')

def phi_calc(A, B, th):
    A = Matrix(A)
    B = Matrix(B)
    theta = Symbol('theta')

    res_vec_B = B.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)
    res_vec_A = A.inv() @ Matrix([cos(theta), sin(theta)]).subs(theta, th)

    return sqrt(res_vec_B.dot(res_vec_B)) / sqrt(res_vec_A.dot(res_vec_A))

t0 = 9.9
t1 = 0.1

STM = lambda t, n : expm([[0,0,0,1 * t,0,0], [0,0,0,0,1 * t,0], [0,0,0,0,0,1 * t], [3 * t * n ** 2, 0, 0, 0, 2 * n * t, 0], [0, 0, 0, -2 * n * t, 0, 0], [0, 0, -t * n ** 2, 0, 0, 0]])

ITM = lambda t, n: [[(1 - np.cos(t * n)) / n ** 2, -2 * (- t * n + np.sin(t * n)) / n ** 2, 0], 
       [2 * (-t * n + np.sin(t * n)) / n ** 2, (-3 * t ** 2) / 2 + 4 * ( 1 - np.cos(t * n)) / n ** 2, 0],
       [0, 0, (1 - np.cos(t * n)) / n ** 2], 
       [np.sin(t * n) / n, 2 * (-1 + np.cos(t * n)) / n, 0],
       [2 * (-1 + np.cos(t * n)) / n, -3 * t + 4 * np.sin(t * n) / n, 0],
       [0, 0, np.sin(t * n) / n]]


U_1, S_1, V_1 = svd(STM(10, 1)[0 : 2, 3 : 5])

transform_mat_1 = [U_1.T[0] * S_1[0], U_1.T[1] * S_1[1]]
transform_mat_1 = np.array(transform_mat_1).T


itm = np.array(ITM(0.1, 1))
stm = np.array(STM(9.9, 1))


U_2, S_2, V_2 = svd((stm[0: 2, 0 : 6] @ itm)[:2, :2] / 0.1)

transform_mat_2 = [U_2.T[0] * S_2[0], U_2.T[1] * S_2[1]]
transform_mat_2 = np.array(transform_mat_2).T

plot_reachable_sets(STM(10, 1)[0 : 2, 3 : 5], (stm[0: 2, 0 : 6] @ itm)[:2, :2] / 0.1)
plot_reachable_sets(transform_mat_1, transform_mat_2)

phis = np.linspace(0, 2 * math.pi, 628)
res = [phi_calc(transform_mat_1, transform_mat_2, phi) for phi in phis]
res_min = min(res)
res_max = max(res)


a_ = np.array(STM(10, 1)[0 : 2, 3 : 5])
aT = STM(10, 1)[0 : 2, 3 : 5].T

b_ = np.array((stm[0: 2, 0 : 6] @ itm) [:2, :2] / 0.1)
bT = np.array((stm[0: 2, 0 : 6] @ itm) [:2, :2] / 0.1).T


print(a_.shape)
print(b_.shape)



w, v = eig(a = a_ @ aT, b = b_ @ bT)

v0 = v[:, 0]
v1 = v[:, 1]
print(np.sqrt(w))
print(v0, v1)

t0 = np.arctan2(v0[1] , v0[0])
t1 = np.arctan2(v1[1] , v1[0])

print(t0, t1)

t0_f = phi_calc(a_, b_, t0 + math.pi / 2)
t1_f = phi_calc(a_, b_, t1 + math.pi / 2)





plt.plot(phis, res)
plt.axvline(x=t0 + math.pi / 2)
plt.axvline(x=t1 + math.pi /2 )
plt.show()

