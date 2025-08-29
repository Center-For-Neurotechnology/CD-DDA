from DDAfunctions import *
import numpy as np
import matplotlib.pyplot as plt
import os

WL = 2000
WS = 500 
WN = 500

DDA_DIR = "DDA"
dir_exist(DDA_DIR)
DATA_DIR = "DATA"
dir_exist(DATA_DIR)
FIG_DIR = "FIG"
dir_exist(FIG_DIR)

NrSyst = 7
ROS = np.array([[0, 0, 2],
                [0, 0, 3],
                [1, 0, 1],
                [1, 0, 2],
                [2, 0, 0],
                [2, 0, 3],
                [2, 1, 3]])

MOD_nr, DIM, ODEorder, P = make_MOD_nr(ROS, NrSyst)

a123 = 0.21
a456 = 0.20
a7 = 0.18
b1 = 0.2150
b2 = 0.2020
b3 = 0.2041
b4 = 0.4050
b5 = 0.3991
b6 = 0.4100
b7 = 0.5000
c = 5.7
c7 = 6.8

MOD_par = np.array([
    [-1, -1, 1, a123, b1, -c, 1],
    [-1, -1, 1, a123, b2, -c, 1],
    [-1, -1, 1, a123, b3, -c, 1],
    [-1, -1, 1, a456, b4, -c, 1],
    [-1, -1, 1, a456, b5, -c, 1],
    [-1, -1, 1, a456, b6, -c, 1],
    [-1, -1, 1, a7, b7, -c7, 1]
])
MOD_par = MOD_par.T.reshape(-1, order='F')  # Julia's reshape(MOD_par', size(ROS,1)*NrSyst)' with Fortran order

FromTo1 = np.array([])

FromTo2 = np.array([[4, 0, 0, 1, 7, 0, 0, 1],
                    [5, 0, 0, 1, 7, 0, 0, 1],
                    [6, 0, 0, 1, 7, 0, 0, 1]])

FromTo3 = np.array([[7, 0, 0, 1, 4, 0, 0, 1],
                    [7, 0, 0, 1, 5, 0, 0, 1],
                    [7, 0, 0, 1, 6, 0, 0, 1]])

I2 = make_MOD_nr_Coupling(FromTo2, DIM, P)
I3 = make_MOD_nr_Coupling(FromTo3, DIM, P)

II = [np.array([]), I2, I3]

epsilon = 0.15

MOD_par_add2 = np.tile([epsilon, -epsilon], FromTo2.shape[0])
MOD_par_add3 = np.tile([epsilon, -epsilon], FromTo3.shape[0])

MOD_par_add = [np.array([]), MOD_par_add2, MOD_par_add3]

TAU = [32, 9]
TM = max(TAU)
dm = 4

LL = [WS * (WN - 1) + WL + TM + dm,
      WS * WN,
      WS * WN + dm - 1]

DELTA = 2
CH_list = list(range(0, DIM * NrSyst, DIM))  # only x (0-based indexing)
TRANS = 20000
dt = 0.05

CASE = ["i", "ii", "iii"]

for n_CASE in range(len(CASE)):
    FN = f"{DATA_DIR}{SL}CD_DDA_data__WL{WL}_WS{WS}_WN{WN}__case_{CASE[n_CASE]}.ascii"
    
    if not os.path.exists(FN):
        X0 = np.random.rand(DIM * NrSyst, 1)
        
        if len(II[n_CASE]) > 0:
            M1 = np.concatenate([MOD_nr, II[n_CASE]])
            M2 = np.concatenate([MOD_par, MOD_par_add[n_CASE]])
        else:
            M1 = MOD_nr
            M2 = MOD_par
        
        integrate_ODE_general_BIG(M1, M2, dt, LL[n_CASE], DIM * NrSyst, ODEorder, X0,
                                 FN, CH_list, DELTA, TRANS)

X = None
for n_CASE in range(len(CASE)):
    FN = f"{DATA_DIR}{SL}CD_DDA_data__WL{WL}_WS{WS}_WN{WN}__case_{CASE[n_CASE]}.ascii"
    
    if n_CASE == 0:
        X = np.loadtxt(FN)
    else:
        X = np.vstack([X, np.loadtxt(FN)])

# Make plot of delay embeddings
fig, axes = plt.subplots(len(CASE), NrSyst, figsize=(21, 8))

for n_CASE in range(len(CASE)):
    for n_SYST in range(NrSyst):
        row = n_CASE
        col = n_SYST
        
        # Julia: X[((20000:24000) .+ (n_CASE-1)*LL[n_CASE]), n_SYST]
        # Convert to 0-based indexing: 19999:23999 + n_CASE * LL[n_CASE]
        offset = n_CASE * LL[n_CASE]
        indices = np.arange(19999, 24000) + offset  # Julia 20000:24000 -> Python 19999:23999 inclusive
        indices_delayed = indices - 10
        
        axes[row, col].plot(X[indices, n_SYST], 
                           X[indices_delayed, n_SYST])

plt.tight_layout()
plt.savefig(f"{DATA_DIR}{SL}Roessler_7syst_NoNoise.png")
plt.show()

# Add noise
SNR = 15
Y = X.copy()

for n_CASE in range(len(CASE)):
    for n_SYST in range(NrSyst):
        start_idx = n_CASE * LL[n_CASE]
        end_idx = (n_CASE + 1) * LL[n_CASE]
        Y[start_idx:end_idx, n_SYST] = add_noise(Y[start_idx:end_idx, n_SYST], SNR)

# Make plot of delay embeddings with noise
fig, axes = plt.subplots(len(CASE), NrSyst, figsize=(21, 8))

for n_CASE in range(len(CASE)):
    for n_SYST in range(NrSyst):
        row = n_CASE
        col = n_SYST
        
        # Julia: Y[((20000:24000) .+ (n_CASE-1)*LL[n_CASE]), n_SYST]
        # Convert to 0-based indexing: 19999:23999 + n_CASE * LL[n_CASE]
        offset = n_CASE * LL[n_CASE]
        indices = np.arange(19999, 24000) + offset  # Julia 20000:24000 -> Python 19999:23999 inclusive
        indices_delayed = indices - 10
        
        axes[row, col].plot(Y[indices, n_SYST], 
                           Y[indices_delayed, n_SYST])

plt.tight_layout()
plt.savefig(f"{DATA_DIR}{SL}Roessler_7syst_15dB.png")
plt.show()

# Save data
FN = f"{DATA_DIR}{SL}CD_DDA_data_NoNoise__WL{WL}_WS{WS}_WN{WN}.ascii"
np.savetxt(FN, X, fmt='%.15f', delimiter=' ')

FN = f"{DATA_DIR}{SL}CD_DDA_data_15dB__WL{WL}_WS{WS}_WN{WN}.ascii"
np.savetxt(FN, Y, fmt='%.15f', delimiter=' ')

# Clean up memory
Y = None
X = None