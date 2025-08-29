# from julia_first_setup import *

from DDAfunctions import *
import numpy as np
import matplotlib.pyplot as plt

NrSyst = 1
ROS = np.array(
    [[0, 0, 2], [0, 0, 3], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 3], [2, 1, 3]]
)

MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

a = 0.2
c = 5.7
dt = 0.05
X0 = np.random.rand(DIM, 1)
L = 10000
TRANS = 5000

b = 0.45
MOD_par = [-1, -1, 1, a, b, -c, 1]
# DO NOT FORGET: "chmod +x i_ODE_general_BIG" in linux!
CH_list = list(range(1, 4))
DELTA = 1

X = integrate_ODE_general_BIG(
    MOD_nr, MOD_par, dt, L, DIM, ODEorder, X0, "", CH_list, DELTA, TRANS
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], color="blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

input("Make png file and continue? ")
plt.savefig("Roessler_0.45.png")

b = 1
MOD_par = [-1, -1, 1, a, b, -c, 1]
X = integrate_ODE_general_BIG(
    MOD_nr, MOD_par, dt, L, DIM, ODEorder, X0, "", CH_list, DELTA, TRANS
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], color="blue")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

input("Make png file and continue? ")
plt.savefig("Roessler_1.png")
