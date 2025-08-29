# from run_DDA_Roessler import *
from DDAfunctions import *
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, permutations
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import networkx as nx

def show_roessler_results():
    """Main function to analyze and display Roessler results."""
    
    WL = 2000
    WS = 500
    WN = 500

    DDA_DIR = "DDA"
    dir_exist(DDA_DIR)
    DATA_DIR = "DATA"
    FIG_DIR = "FIG"
    dir_exist(FIG_DIR)

    NOISE = ["NoNoise", "15dB"]

    NrSyst = 7
    DIM = 3
    NrCH = NrSyst
    CH = list(range(1, NrCH + 1))  # Keep 1-based for combinations logic

    DDAmodel = np.array([[0, 0, 1],
                         [0, 0, 2],
                         [1, 1, 1]])
    MODEL, L_AF, DDAorder = make_MODEL(DDAmodel)

    LIST = list(combinations(CH, 2))
    LL1 = np.array(LIST).flatten()
    LIST = np.array(LIST)
    
    for n_NOISE in range(len(NOISE)):
        noise = NOISE[n_NOISE]
        
        FN_DDA = f"{DDA_DIR}{SL}{noise}__WL{WL}_WS{WS}_WN{WN}.DDA"
        
        E = np.full((WN, NrSyst, NrSyst, 3), np.nan)
        C = np.full((WN, NrSyst, NrSyst, 3), np.nan)
        
        ST = np.loadtxt(f"{FN_DDA}_ST")
        T = ST[:, :2]
        ST = ST[:, 2:]
        ST = ST[:, L_AF-1::L_AF]
        ST = ST.reshape(WN, 3, NrSyst)
        
        CT = np.loadtxt(f"{FN_DDA}_CT")
        CT = CT[:, 2:]
        CT = CT[:, L_AF-1::L_AF]
        CT = CT.reshape(WN, 3, LIST.shape[0])
        
        for l in range(LIST.shape[0]):
            ch1 = LIST[l, 0] - 1  # Convert to 0-based indexing
            ch2 = LIST[l, 1] - 1
            E[:, ch1, ch2, :] = np.abs(np.mean(ST[:, :, [ch1, ch2]], axis=2) / CT[:, :, l] - 1)
            E[:, ch2, ch1, :] = E[:, ch1, ch2, :]
        
        CD = np.loadtxt(f"{FN_DDA}_CD_DDA_ST")
        CD = CD[:, 2:]
        CD = CD.reshape(WN, 3, 2, LIST.shape[0], order='F')  # Use Fortran order to match Julia
        
        for l in range(LIST.shape[0]):
            ch1 = LIST[l, 0] - 1
            ch2 = LIST[l, 1] - 1
            # Julia: CD[:,:,2,l] goes to C[:,ch1,ch2,:] (2 in Julia = 1 in Python 0-based)
            # Julia: CD[:,:,1,l] goes to C[:,ch2,ch1,:] (1 in Julia = 0 in Python 0-based)
            C[:, ch1, ch2, :] = CD[:, :, 1, l]  # CD dimension 2 (Julia) = 1 (Python)
            C[:, ch2, ch1, :] = CD[:, :, 0, l]  # CD dimension 1 (Julia) = 0 (Python)
        
        # Plot results - First plot (E matrices)
        # Julia layout: [a{0.7h} ; b c d] - one big plot on top, 3 smaller ones below
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 0.3], hspace=0.3, wspace=0.3)
        
        e = E.reshape(E.shape[0], NrSyst**2, 3)
        e = np.transpose(e, (0, 2, 1))  # Match Julia's permutedims([1,3,2]) = (WN, 3, 49)
        e = e.reshape(WN * 3, NrSyst**2).T
        
        N_indices = []
        for i in range(NrSyst):
            for j in range(NrSyst):
                if i != j and i < j:  # Lower triangular excluding diagonal
                    N_indices.append(i * NrSyst + j)
        
        S = [f"{LIST[i, 0]} {LIST[i, 1]}" for i in range(LIST.shape[0])]
        
        # Top subplot spanning all columns
        ax_top = fig.add_subplot(gs[0, :])
        im1 = ax_top.imshow(e[N_indices, :], cmap='viridis', aspect='auto')
        ax_top.set_yticks(range(len(S)))
        ax_top.set_yticklabels(S)
        ax_top.set_xticks([100])
        ax_top.set_xticklabels([" "])
        plt.colorbar(im1, ax=ax_top)
        
        e_mean = np.mean(E[19:, :, :, :], axis=0)
        for k in range(3):
            ax = fig.add_subplot(gs[1, k])
            im = ax.imshow(e_mean[:, :, k], cmap='jet')
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_title(f"({k+1})")
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        plt.savefig(f"{FIG_DIR}{SL}E__WL{WL}_WS{WS}_WN{WN}_{noise}.png")
        plt.close()  # Close instead of show to run non-interactively
        
        print(f"Saved E plot for {noise}")
        
        # Second plot for causality
        # Julia layout: [a{0.5h} ; b c d; e f g] - one plot on top, 3 heatmaps middle, 3 networks bottom
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 0.25, 0.25], hspace=0.3, wspace=0.3)
        
        c = C.reshape(C.shape[0], NrSyst**2, 3)
        c = np.transpose(c, (0, 2, 1))  # Match Julia's permutedims([1,3,2]) = (WN, 3, 49)
        c = c.reshape(WN * 3, NrSyst**2).T
        c_min = np.nanmin(c)
        c_max = np.nanmax(c)
        c = (c - c_min) / (c_max - c_min)
        
        N = list(range(NrSyst**2))
        diag_indices = [i * NrSyst + i for i in range(NrSyst)]
        N = [i for i in N if i not in diag_indices]
        
        S = []
        for i in range(NrSyst):
            for j in range(NrSyst):
                if i != j:
                    S.append(f"{i+1} {j+1}")
        
        # Top subplot spanning all columns
        ax_top = fig.add_subplot(gs[0, :])
        im1 = ax_top.imshow(c[N, :], cmap='viridis', aspect='auto')
        ax_top.set_yticks(range(len(S)))
        ax_top.set_yticklabels(S)
        ax_top.set_xticks([100])
        ax_top.set_xticklabels([" "])
        plt.colorbar(im1, ax=ax_top)
        
        # Match Julia's averaging and normalization exactly
        # Julia: c=dropdims(mean(C[50:end,:,:,:],dims=1),dims=1)
        c_mean = np.mean(C[49:, :, :, :], axis=0)  # 49: in Python = 50:end in Julia
        
        # Julia: c .= c .- minimum(filter(!isnan,c[:]))
        c_mean_min = np.nanmin(c_mean)
        c_mean = c_mean - c_mean_min
        
        # Julia: c .= c ./ maximum(filter(!isnan,c[:]))
        c_mean_max = np.nanmax(c_mean)  # Max AFTER subtraction
        c_mean = c_mean / c_mean_max
        
        # Second row: 3 heatmaps
        for k in range(3):
            ax = fig.add_subplot(gs[1, k])
            im = ax.imshow(c_mean[:, :, k], cmap='plasma', vmin=0, vmax=1)
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_title(f"({k+1})")
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        # Third row: Network plots
        q = c_mean.reshape(NrSyst * NrSyst, 3)
        for k in range(3):
            q_k = q[:, k].copy()
            q_k = q_k - np.nanmin(q_k)
            q_k = q_k / np.nanmax(q_k)
            q[:, k] = q_k
        
        q = q.reshape(NrSyst, NrSyst, 3)
        
        MS = [1, 1, 1, 2, 2, 2, 3]
        colors = ['plum', 'mistyrose', 'lavender']
        
        for k in range(3):
            A = q[:, :, k].copy()
            A[A < 0.25] = 0
            A[np.isnan(A)] = 0
            
            ax = fig.add_subplot(gs[2, k])
            
            G = nx.DiGraph()
            for i in range(NrSyst):
                G.add_node(i+1)
            
            for i in range(NrSyst):
                for j in range(NrSyst):
                    if A[i, j] > 0:
                        G.add_edge(i+1, j+1, weight=A[i, j])
            
            pos = nx.circular_layout(G)
            node_colors = [colors[MS[i]-1] for i in range(NrSyst)]
            
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                    node_size=500, font_size=12, arrows=True, arrowsize=20)
        
        plt.savefig(f"{FIG_DIR}{SL}C__WL{WL}_WS{WS}_WN{WN}_{noise}.png")
        plt.close()
        
        print(f"Saved C plot for {noise}")
        
        # Third plot for causality * ergodicity
        CE = C * E
        
        # Julia layout: [a{0.5h} ; b c d; e f g] - one plot on top, 3 heatmaps middle, 3 networks bottom
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 0.25, 0.25], hspace=0.3, wspace=0.3)
        
        c = CE.reshape(CE.shape[0], NrSyst**2, 3)
        c = np.transpose(c, (0, 2, 1))  # Match Julia's permutedims([1,3,2]) = (WN, 3, 49)
        c = c.reshape(WN * 3, NrSyst**2).T
        c_min = np.nanmin(c)
        c_max = np.nanmax(c)
        c = (c - c_min) / (c_max - c_min)
        
        # Top subplot spanning all columns
        ax_top = fig.add_subplot(gs[0, :])
        im1 = ax_top.imshow(c[N, :], cmap='viridis', aspect='auto')
        ax_top.set_yticks(range(len(S)))
        ax_top.set_yticklabels(S)
        ax_top.set_xticks([100])
        ax_top.set_xticklabels([" "])
        plt.colorbar(im1, ax=ax_top)
        
        # Match Julia's averaging and normalization exactly
        # Julia: c=dropdims(mean(CE[50:end,:,:,:],dims=1),dims=1)
        c_mean = np.mean(CE[49:, :, :, :], axis=0)  # 49: in Python = 50:end in Julia
        
        # Julia: c .= c .- minimum(filter(!isnan,c[:]))
        c_mean_min = np.nanmin(c_mean)
        c_mean = c_mean - c_mean_min
        
        # Julia: c .= c ./ maximum(filter(!isnan,c[:]))
        c_mean_max = np.nanmax(c_mean)  # Max AFTER subtraction
        c_mean = c_mean / c_mean_max
        
        # Second row: 3 heatmaps
        for k in range(3):
            ax = fig.add_subplot(gs[1, k])
            im = ax.imshow(c_mean[:, :, k], cmap='plasma', vmin=0, vmax=1)
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 6.5)
            ax.set_title(f"({k+1})")
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        # Third row: Network plots
        q = c_mean.reshape(NrSyst * NrSyst, 3)
        for k in range(3):
            q_k = q[:, k].copy()
            q_k = q_k - np.nanmin(q_k)
            q_k = q_k / np.nanmax(q_k)
            q[:, k] = q_k
        
        q = q.reshape(NrSyst, NrSyst, 3)
        
        for k in range(3):
            A = q[:, :, k].copy()
            A[A < 0.25] = 0
            A[np.isnan(A)] = 0
            
            ax = fig.add_subplot(gs[2, k])
            
            G = nx.DiGraph()
            for i in range(NrSyst):
                G.add_node(i+1)
            
            for i in range(NrSyst):
                for j in range(NrSyst):
                    if A[i, j] > 0:
                        G.add_edge(i+1, j+1, weight=A[i, j])
            
            pos = nx.circular_layout(G)
            node_colors = [colors[MS[i]-1] for i in range(NrSyst)]
            
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                    node_size=500, font_size=12, arrows=True, arrowsize=20)
        
        plt.savefig(f"{FIG_DIR}{SL}CE__WL{WL}_WS{WS}_WN{WN}_{noise}.png")
        plt.close()
        
        print(f"Saved CE plot for {noise}")

# If running as main script, execute the analysis
if __name__ == "__main__":
    show_roessler_results()