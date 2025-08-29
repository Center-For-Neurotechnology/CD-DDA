import numpy as np
import os
import subprocess
import platform
from itertools import combinations

nr_delays = 2

if platform.system() == "Windows":
    SL = "\\"
else:
    SL = "/"


def index(DIM, ORDER):
    B = np.ones((DIM**ORDER, ORDER), dtype=int)

    if DIM > 1:
        # Julia: for i = 2:(DIM^ORDER)  ->  Python: for i in range(1, DIM**ORDER)
        for i in range(1, DIM**ORDER):  # i corresponds to Julia's i (1-based)
            # Julia: if B[i-1, ORDER] < DIM  ->  Python: if B[i-1, ORDER-1] < DIM
            if B[i - 1, ORDER - 1] < DIM:
                B[i, ORDER - 1] = B[i - 1, ORDER - 1] + 1

            # Julia: for i_DIM = 1:ORDER-1  ->  Python: for i_DIM in range(1, ORDER)
            for i_DIM in range(1, ORDER):
                if round(((i + 1) / DIM**i_DIM - np.floor((i + 1) / DIM**i_DIM)) * DIM**i_DIM) == 1:
                    # Julia: if B[i-DIM^i_DIM, ORDER-i_DIM] < DIM
                    if B[i - DIM**i_DIM, ORDER - i_DIM - 1] < DIM:
                        # Julia: for j = 0:DIM^i_DIM-1
                        for j in range(DIM**i_DIM):
                            # Julia: B[i+j, ORDER-i_DIM] = B[i+j-DIM^i_DIM, ORDER-i_DIM] + 1
                            B[i + j, ORDER - i_DIM - 1] = B[i + j - DIM**i_DIM, ORDER - i_DIM - 1] + 1

        i_BB = 0
        BB = []
        for i in range(B.shape[0]):
            jn = 1
            # Julia: for j = 2:ORDER  ->  Python: for j in range(1, ORDER)
            for j in range(1, ORDER):
                # Julia: if B[i, j] >= B[i, j-1]  ->  Python: if B[i, j] >= B[i, j-1]
                if B[i, j] >= B[i, j - 1]:
                    jn += 1
            if jn == ORDER:
                BB.append(B[i, :])
                i_BB += 1
    else:
        print("DIM=1!!!")

    return np.array(BB).T


def monomial_list(nr_delays, order):
    P_ODE = index(nr_delays + 1, order).T
    P_ODE = P_ODE - np.ones(P_ODE.shape, dtype=int)
    P_ODE = P_ODE[1:, :]
    return P_ODE


def make_MOD_nr(SYST, NrSyst):
    DIM = len(np.unique(SYST[:, 0]))
    order = SYST.shape[1] - 1

    P = np.vstack([np.array([[0, 0]]), monomial_list(DIM * NrSyst, order)])

    MOD_nr = np.zeros((SYST.shape[0] * NrSyst, 2), dtype=int)
    for n in range(NrSyst):
        for i in range(SYST.shape[0]):
            II = SYST[i, 1:].copy()
            II[II > 0] += DIM * n

            Nr = i + SYST.shape[0] * n
            # Match Julia's logic: findall(sum(abs.(repeat(II, size(P, 1), 1) - P), dims=2)' .== 0)[1][2] - 1
            diff = np.abs(P - II).sum(axis=1)
            match_indices = np.where(diff == 0)[0]
            if len(match_indices) > 0:
                MOD_nr[Nr, 1] = match_indices[0]  # Julia uses 1-based, but we need 0-based index for P
            else:
                raise ValueError(f"No matching polynomial found for system {n}, equation {i}")
            MOD_nr[Nr, 0] = SYST[i, 0] + DIM * n

    # Julia: MOD_nr = reshape(MOD_nr', size(SYST, 1) * NrSyst * 2)'
    # Julia uses column-major order (Fortran-style) for reshape
    MOD_nr = MOD_nr.T.reshape(-1, order='F')  # Column-major flatten after transpose
    return MOD_nr, DIM, order, P


def integrate_ODE_general_BIG(
    MOD_nr, MOD_par, dt, L, DIM, ODEorder, X0, FNout, CH_list, DELTA, TRANS=None
):
    if TRANS is None:
        TRANS = 0

    if platform.system() == "Windows":
        if not os.path.isfile("i_ODE_general_BIG.exe"):
            import shutil

            shutil.copy("i_ODE_general_BIG", "i_ODE_general_BIG.exe")
        CMD = ".\\i_ODE_general_BIG.exe"
    else:
        CMD = "./i_ODE_general_BIG"

    MOD_NR = " ".join(map(str, MOD_nr))
    CMD = f"{CMD} -MODEL {MOD_NR}"
    MOD_PAR = " ".join(map(str, MOD_par))
    CMD = f"{CMD} -PAR {MOD_PAR}"
    ANF = " ".join(map(str, X0.flatten()))
    CMD = f"{CMD} -ANF {ANF}"
    CMD = f"{CMD} -dt {dt}"
    CMD = f"{CMD} -L {L}"
    CMD = f"{CMD} -DIM {DIM}"
    CMD = f"{CMD} -order {ODEorder}"
    if TRANS > 0:
        CMD = f"{CMD} -TRANS {TRANS}"
    if len(FNout) > 0:
        CMD = f"{CMD} -FILE {FNout}"
    CMD = f"{CMD} -DELTA {DELTA}"
    CMD = f"{CMD} -CH_list {' '.join(map(str, CH_list))}"

    if len(FNout) > 0:
        subprocess.run(CMD, shell=True)
    else:
        result = subprocess.run(CMD, shell=True, capture_output=True, text=True)
        X = result.stdout.strip().split("\n")
        X = np.array([list(map(float, row.split())) for row in X])
        return X


def make_MOD_nr_Coupling(FromTo, DIM, P):
    order = P.shape[1]
    II = np.zeros((FromTo.shape[0], 4), dtype=int)

    for j in range(II.shape[0]):
        n1 = FromTo[j, 0]
        k1 = FromTo[j, 1] + 1
        range1 = slice(2, 2 + order)
        n2 = FromTo[j, 1 + order + 1]
        k2 = FromTo[j, 2 + order + 1] + 1
        # Julia: range2 = range1 .+ range1[end] where range1=[3,4] (1-based), range1[end]=4
        # So range2 = [3,4] + 4 = [7,8] (1-based) = [6,7] (0-based)
        range2 = slice(6, 6 + order)

        JJ = FromTo[j, range1].copy()
        JJ[JJ > 0] += DIM * (n1 - 1)
        diff = np.abs(P - JJ).sum(axis=1)
        match_indices = np.where(diff == 0)[0]
        if len(match_indices) > 0:
            II[j, 3] = match_indices[0]
        else:
            raise ValueError(f"No matching polynomial found for coupling {j}, first part")

        JJ = FromTo[j, range2].copy()
        JJ[JJ > 0] += DIM * (n2 - 1)
        diff = np.abs(P - JJ).sum(axis=1)
        match_indices = np.where(diff == 0)[0]
        if len(match_indices) > 0:
            II[j, 1] = match_indices[0]
        else:
            raise ValueError(f"No matching polynomial found for coupling {j}, second part")

        II[j, 0] = DIM * n2 - (DIM - k2) - 1
        II[j, 2] = DIM * n2 - (DIM - k1) - 1

    II = II.T.flatten()
    return II


def dir_exist(DIR):
    if not os.path.isdir(DIR):
        os.mkdir(DIR)


def add_noise(s, SNR):
    N = len(s)
    n = np.random.randn(N)
    n = (n - np.mean(n)) / np.std(n)
    c = np.sqrt(np.var(s) * 10 ** (-SNR / 10))
    s_out = s + c * n
    return s_out


def number_to_string(n):
    return f"{n:.15f}"


def make_MODEL_new(MOD, SSYM, mm):
    MODEL = np.where(MOD[mm, :] == 1)[0]
    L_AF = len(MODEL) + 1
    SYM = SSYM[mm, :]
    model = "_".join([f"{x:02d}" for x in MODEL])
    return MODEL, SYM, model, L_AF


def make_MODEL(SYST):
    order = SYST.shape[1]
    nr_delays = 2

    P_ODE = monomial_list(nr_delays, order)

    MODEL = np.zeros(SYST.shape[0], dtype=int)
    for i in range(SYST.shape[0]):
        II = SYST[i, :]
        diff = np.abs(P_ODE - II).sum(axis=1)
        MODEL[i] = np.where(diff == 0)[0][0]

    L_AF = len(MODEL) + 1
    return MODEL, L_AF, order


def deriv_all(data, dm, order=None, dt=None):
    if order is None:
        order = 2
    if dt is None:
        dt = 1

    t = np.arange(dm, len(data) - dm)
    L = len(t)

    if order == 2:
        ddata = np.zeros(L)
        for n1 in range(1, dm + 1):
            ddata += (data[t + n1] - data[t - n1]) / n1
        ddata /= dm / dt

    if order == 3:
        ddata = np.zeros(L)
        d = 0
        for n1 in range(1, dm + 1):
            for n2 in range(n1 + 1, dm + 1):
                d += 1
                ddata -= (
                    (data[t - n2] - data[t + n2]) * n1**3
                    - (data[t - n1] - data[t + n1]) * n2**3
                ) / (n1**3 * n2 - n1 * n2**3)
        ddata /= d / dt

    return ddata


def make_MOD_new_new(N_MOD, nr_delays, order):
    if nr_delays != 2:
        print("only nr_delays=2 supported")
        nr_delays = 2

    P_DDA = monomial_list(nr_delays, order)
    L = P_DDA.shape[0]

    PP = -P_DDA.copy()
    PP[PP == -1] = 2
    PP[PP == -2] = 1
    PP = np.sort(PP, axis=1)

    f = np.zeros((P_DDA.shape[0], 2), dtype=int)
    for k1 in range(P_DDA.shape[0]):
        f[k1, 0] = k1
        diff = np.abs(P_DDA - PP[k1, :]).sum(axis=1)
        ff = np.where(diff == 0)[0]
        if len(ff) > 0:
            f[k1, 1] = ff[0]

    MOD = np.zeros((1, P_DDA.shape[0]), dtype=int)
    for n_N in range(len(N_MOD)):
        N = N_MOD[n_N]
        C = list(combinations(range(L), N))
        C = np.array(C)
        M = np.zeros((len(C), L), dtype=int)

        for c in range(len(C)):
            M[c, C[c]] = 1

        M1 = np.sort(M * np.arange(1, L + 1), axis=1)[:, -N:]
        M2 = -M1.copy()

        for k1 in range(f.shape[0]):
            M2[M2 == -f[k1, 0]] = f[k1, 1]
        M2 = np.sort(M2, axis=1)

        f2 = np.zeros((M1.shape[0], 2), dtype=int)
        for k1 in range(M1.shape[0]):
            f2[k1, 0] = k1
            diff = np.abs(M1 - M2[k1, :]).sum(axis=1)
            ff = np.where(diff == 0)[0]
            if len(ff) > 0:
                f2[k1, 1] = ff[0]

        f2 = np.sort(f2, axis=1)
        f2 = np.unique(f2, axis=0)
        f2 = f2[f2[:, 0] != f2[:, 1], 1]
        f2 = np.setdiff1d(np.arange(M1.shape[0]), f2)

        MOD = np.vstack([MOD, M[f2, :]])

    MOD = MOD[1:, :]

    SSYM = np.full((MOD.shape[0], 2), -1, dtype=int)
    for n_M in range(MOD.shape[0]):
        p = P_DDA[MOD[n_M, :] == 1, :]

        SSYM[n_M, 0] = len(np.unique(p[p > 0]))

        p = p.astype(float)
        p[p == 0] = np.nan
        p1 = (p + 2) % 2 + 1
        p1[np.isnan(p1)] = 0
        p1 = p1.astype(int)
        p[np.isnan(p)] = 0
        p = p.astype(int)

        p1 = np.sort(p1, axis=1)
        p1 = p1[np.lexsort(p1.T)]

        if np.sum(np.abs(p - p1)) == 0:
            SSYM[n_M, 1] = 1
        else:
            SSYM[n_M, 1] = 0

    return MOD, P_DDA, SSYM


def make_TAU_ALL(SSYM, DELAYS):
    uSYM = np.unique(SSYM, axis=0)
    for k in range(uSYM.shape[0]):
        s = uSYM[k, :]
        nr = s[0]
        sym = s[1]

        FN = f"TAU_ALL__{s[0]}_{s[1]}"
        with open(FN, "w") as fid:
            if nr == 1:
                for tau1 in range(len(DELAYS)):
                    fid.write(f"{DELAYS[tau1]}\n")
            elif nr == 2:
                if sym == 0:
                    for tau1 in range(len(DELAYS)):
                        for tau2 in range(len(DELAYS)):
                            if tau1 != tau2:
                                fid.write(f"{DELAYS[tau1]} {DELAYS[tau2]}\n")
                elif sym == 1:
                    for tau1 in range(len(DELAYS)):
                        for tau2 in range(len(DELAYS)):
                            if tau1 < tau2:
                                fid.write(f"{DELAYS[tau1]} {DELAYS[tau2]}\n")
