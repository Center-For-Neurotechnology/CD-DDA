import numpy as np
import matplotlib.pyplot as plt
from DDAfunctions import make_MOD_nr, integrate_ODE_general_BIG

# 1 single system
NrSyst = 1

# Single Roessler system
ROS = np.array(
    [[0, 0, 2], [0, 0, 3], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 3], [2, 1, 3]]
)

# Encoding of the Roessler system
MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

# Choice of parameters
a = 0.2
c = 5.7
dt = 0.05
X0 = np.random.rand(DIM)
L = 10000  # integration length
TRANS = 5000  # transient

# First case: b = 0.45
b = 0.45
MOD_par = [-1, -1, 1, a, b, -c, 1]  # parameters
CH_list = list(range(1, 4))
DELTA = 1

# Integrate system
X = integrate_ODE_general_BIG(
    MOD_nr, MOD_par, dt, L, DIM, ODEorder, X0, "", CH_list, DELTA, TRANS
)

# Plot the attractor
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], color="blue", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"Roessler Attractor (b={b})")
plt.savefig("Roessler_0.45.png")
plt.show()

# Second case: b = 1
b = 1
MOD_par = [-1, -1, 1, a, b, -c, 1]  # parameters
X = integrate_ODE_general_BIG(
    MOD_nr, MOD_par, dt, L, DIM, ODEorder, X0, "", CH_list, DELTA, TRANS
)

# Plot the attractor
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], color="blue", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"Roessler Attractor (b={b})")
plt.savefig("Roessler_1.png")
plt.show()
