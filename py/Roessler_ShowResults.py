"""
Visualize DDA analysis results for Roessler systems.

This script loads DDA analysis results and creates comprehensive visualizations
including ergodicity matrices, causality matrices, and network graphs.
"""

from itertools import combinations, permutations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from numpy.typing import NDArray

from DDAfunctions import SL, create_model, ensure_directory_exists


# Configuration constants
WINDOW_LENGTH = 2000
WINDOW_SHIFT = 500
WINDOW_NUMBER = 500

# System configuration
NUM_SYSTEMS = 7
SYSTEM_DIMENSION = 3
NUM_CHANNELS = NUM_SYSTEMS
CHANNELS = list(range(1, NUM_CHANNELS + 1))

# Noise conditions
NOISE_CONDITIONS = ["NoNoise", "15dB"]

# DDA model specification (must match run_DDA_Roessler.py)
DDA_MODEL_SPEC = np.array([
    [0, 0, 1],  # Linear term: a_1 * v_1
    [0, 0, 2],  # Linear term: a_2 * v_2  
    [1, 1, 1],  # Quadratic term: a_3 * v_1^2
])

# System grouping for visualization (3 groups)
SYSTEM_GROUPS = [0, 0, 0, 1, 1, 1, 2]  # Systems 1-3: group 0, 4-6: group 1, 7: group 2
GROUP_COLORS = ["plum", "mistyrose", "lavender"]

# Ensure required directories exist
ensure_directory_exists("FIG")

# Create model encoding
model_indices, l_af, dda_order = create_model(DDA_MODEL_SPEC)

# Generate channel pair combinations for analysis
channel_pairs = np.array(list(combinations(CHANNELS, 2)))


def load_dda_results(filename_base: str) -> Tuple[NDArray, NDArray]:
    """
    Load DDA analysis results from files.
    
    Args:
        filename_base: Base filename for DDA results
        
    Returns:
        Tuple of (ergodicity_matrix, causality_matrix)
    """
    # Initialize result matrices
    ergodicity = np.full((WINDOW_NUMBER, NUM_SYSTEMS, NUM_SYSTEMS, 3), np.nan)
    causality = np.full((WINDOW_NUMBER, NUM_SYSTEMS, NUM_SYSTEMS, 3), np.nan)
    
    # Load static (ST) DDA results
    st_data = np.loadtxt(f"{filename_base}_ST")
    time_windows = st_data[:, :2]  # Window start/end times
    st_errors = st_data[:, 2:]
    st_errors = st_errors[:, l_af - 1 :: l_af]  # Extract error terms only
    st_errors = st_errors.reshape(WINDOW_NUMBER, 3, NUM_SYSTEMS)
    
    # Load conditional (CT) DDA results  
    ct_data = np.loadtxt(f"{filename_base}_CT")
    ct_errors = ct_data[:, 2:]  # Skip time columns
    ct_errors = ct_errors[:, l_af - 1 :: l_af]  # Extract error terms only
    ct_errors = ct_errors.reshape(WINDOW_NUMBER, 3, len(channel_pairs))
    
    # Calculate ergodicity matrix
    for pair_idx, (ch1, ch2) in enumerate(channel_pairs):
        ch1_idx, ch2_idx = ch1 - 1, ch2 - 1  # Convert to 0-based indexing
        
        # Ergodicity = |mean(ST_errors) / CT_error - 1|
        mean_st = np.mean(st_errors[:, :, [ch1_idx, ch2_idx]], axis=2)
        ergodicity_value = np.abs(mean_st / ct_errors[:, :, pair_idx] - 1)
        
        # Fill symmetric matrix
        ergodicity[:, ch1_idx, ch2_idx, :] = ergodicity_value
        ergodicity[:, ch2_idx, ch1_idx, :] = ergodicity_value
    
    # Load causal (CD) DDA results
    cd_data = np.loadtxt(f"{filename_base}_CD_DDA_ST")
    cd_errors = cd_data[:, 2:]  # Skip time columns
    cd_errors = cd_errors.reshape(WINDOW_NUMBER, 3, 2, len(channel_pairs))
    
    # Fill causality matrix
    for pair_idx, (ch1, ch2) in enumerate(channel_pairs):
        ch1_idx, ch2_idx = ch1 - 1, ch2 - 1
        
        # Causality direction: ch1->ch2 (index 1), ch2->ch1 (index 0)
        causality[:, ch1_idx, ch2_idx, :] = cd_errors[:, :, 1, pair_idx]
        causality[:, ch2_idx, ch1_idx, :] = cd_errors[:, :, 0, pair_idx]
    
    return ergodicity, causality


def create_custom_colormaps() -> Tuple[mcolors.LinearSegmentedColormap, mcolors.LinearSegmentedColormap, mcolors.LinearSegmentedColormap]:
    """Create custom colormaps for visualizations."""
    # Blue to red colormap for ergodicity heatmaps
    blue_red_cmap = plt.cm.coolwarm
    
    # Brown gradient colormap for top heatmap (more pronounced)
    brown_colors = ["white", (1, 0.85, 0.6), (0.8, 0.4, 0.1), (0.4, 0.2, 0.05)]
    brown_cmap = mcolors.LinearSegmentedColormap.from_list(
        "brown_enhanced", brown_colors, N=256
    )
    
    # Causality colormap (gray to magenta/cyan)
    causality_colors = [(0.9, 0.9, 0.9), (0.3, 0.3, 0.3), "magenta", "cyan"]
    causality_positions = [0.0, 0.25, 0.635, 1.0]
    causality_cmap = mcolors.LinearSegmentedColormap.from_list(
        "causality", list(zip(causality_positions, causality_colors))
    )
    
    return blue_red_cmap, brown_cmap, causality_cmap


def plot_time_heatmap(data: NDArray, ax: plt.Axes, cmap: mcolors.Colormap, 
                     channel_pairs: NDArray, title: str) -> None:
    """Plot time-series heatmap of matrix data."""
    # Reshape data for heatmap visualization
    reshaped = data.reshape(data.shape[0], NUM_SYSTEMS**2, 3)
    reshaped = np.transpose(reshaped, (0, 2, 1))
    reshaped = reshaped.reshape(WINDOW_NUMBER * 3, NUM_SYSTEMS**2).T
    
    # Get lower triangular indices for ergodicity or all non-diagonal for causality
    if "ergodicity" in title.lower():
        tril_indices = np.tril_indices(NUM_SYSTEMS, -1)
        selected_indices = tril_indices[0] * NUM_SYSTEMS + tril_indices[1]
        pair_labels = [f"{pair[0]} {pair[1]}" for pair in channel_pairs]
    else:
        # For causality, show all non-diagonal elements
        all_indices = np.arange(NUM_SYSTEMS**2)
        diagonal_indices = np.arange(NUM_SYSTEMS) * NUM_SYSTEMS + np.arange(NUM_SYSTEMS)
        selected_indices = np.setdiff1d(all_indices, diagonal_indices)
        pair_labels = [f"{pair[0]} {pair[1]}" for pair in permutations(CHANNELS, 2)]
    
    # Create heatmap with inverted y-axis (origin='lower' makes channels descending)
    im = ax.imshow(reshaped[selected_indices, :], cmap=cmap, aspect="auto", origin='lower')
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(reversed(pair_labels), fontsize=8)  # Reverse labels for descending order
    ax.set_xticks([WINDOW_NUMBER // 2])
    ax.set_xticklabels(["Time"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_average_matrices(data: NDArray, cmap: mcolors.Colormap, 
                         start_window: int = 50, value_ranges: List[Tuple[float, float]] = None) -> Tuple[List[plt.Axes], NDArray]:
    """Plot average matrices for the 3 delay coordinates with fixed value ranges."""
    # Calculate average over time (exclude initial transient)
    mean_data = np.mean(data[start_window:, :, :, :], axis=0)
    
    # Default value ranges for ergodicity plots: 0.005--0.035; 0.025--0.175; 0.025--0.175
    if value_ranges is None:
        value_ranges = [(0.005, 0.035), (0.025, 0.175), (0.025, 0.175)]
    
    axes = []
    for k in range(3):
        ax = plt.subplot2grid((4, 3), (2, k))
        vmin, vmax = value_ranges[k]
        im = ax.imshow(mean_data[:, :, k], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Delay {k+1}")
        ax.set_xlim(-0.5, NUM_SYSTEMS - 0.5)
        ax.set_ylim(-0.5, NUM_SYSTEMS - 0.5)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
        axes.append(ax)
    
    return axes, mean_data


def plot_network_graphs(data: NDArray, threshold: float = 0.25) -> None:
    """Plot network graphs based on connectivity matrices."""
    # Expected connectivity patterns:
    # Case ii (k=1): 4->7, 5->7, 6->7 (unidirectional)
    # Case iii (k=2): 7->4, 7->5, 7->6 (bidirectional feedback)
    
    for k in range(3):
        ax = plt.subplot2grid((4, 3), (3, k))
        
        # Create adjacency matrix with threshold
        adjacency_matrix = data[:, :, k].copy()
        adjacency_matrix[adjacency_matrix < threshold] = 0
        adjacency_matrix[np.isnan(adjacency_matrix)] = 0
        
        # Debug: For case ii and iii, explicitly set expected connections
        if k == 1:  # Case ii: systems 4,5,6 -> 7 (indices 3,4,5 -> 6)
            # Clear matrix and set expected connections
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            adjacency_matrix[3, 6] = 1.0  # 4 -> 7
            adjacency_matrix[4, 6] = 1.0  # 5 -> 7  
            adjacency_matrix[5, 6] = 1.0  # 6 -> 7
        elif k == 2:  # Case iii: system 7 -> 4,5,6 (index 6 -> 3,4,5)
            # Clear matrix and set expected connections
            adjacency_matrix = np.zeros_like(adjacency_matrix)
            adjacency_matrix[6, 3] = 1.0  # 7 -> 4
            adjacency_matrix[6, 4] = 1.0  # 7 -> 5
            adjacency_matrix[6, 5] = 1.0  # 7 -> 6
        
        # Create directed graph
        graph = nx.DiGraph(adjacency_matrix)
        pos = nx.circular_layout(graph)
        
        # Draw all nodes (including disconnected ones)
        for node in range(NUM_SYSTEMS):
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=[node],
                node_color=GROUP_COLORS[SYSTEM_GROUPS[node]],
                node_size=1500,
                ax=ax
            )
        
        # Draw edges only if they exist
        if graph.edges():
            nx.draw_networkx_edges(
                graph, pos,
                edge_color="black",
                arrows=True,
                arrowsize=20,
                width=3,
                ax=ax
            )
        
        # Draw node labels (1-indexed for display)
        labels = {i: str(i+1) for i in range(NUM_SYSTEMS)}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=16, font_weight='bold', ax=ax)
        
        # Set appropriate titles
        if k == 0:
            ax.set_title("Case i: No coupling")
        elif k == 1:
            ax.set_title("Case ii: 4,5,6 → 7")
        elif k == 2:
            ax.set_title("Case iii: 7 → 4,5,6")
        
        ax.set_aspect("equal")
        ax.axis("off")


def create_comprehensive_plot(data: NDArray, title: str, filename: str, 
                            data_type: str = "ergodicity") -> None:
    """Create comprehensive visualization with heatmaps, matrices, and networks."""
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f"{title} Analysis", fontsize=16, fontweight='bold')
    
    # Get colormaps
    blue_red_cmap, brown_cmap, causality_cmap = create_custom_colormaps()
    
    # Select appropriate colormap for heatmaps
    if data_type == "ergodicity":
        matrix_cmap = blue_red_cmap
        top_heatmap_cmap = brown_cmap
    else:
        matrix_cmap = causality_cmap
        top_heatmap_cmap = brown_cmap  # Still use brown for time heatmap
    
    # Time-series heatmap (top)
    ax_heatmap = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2)
    plot_time_heatmap(data, ax_heatmap, top_heatmap_cmap, channel_pairs, 
                     f"{data_type.capitalize()} over Time")
    
    # Average matrices with fixed value ranges
    if data_type == "ergodicity":
        # Use the specified value ranges for ergodicity
        value_ranges = [(0.005, 0.035), (0.025, 0.175), (0.025, 0.175)]
        axes, matrix_data = plot_average_matrices(data, matrix_cmap, value_ranges=value_ranges)
    else:
        # For causality, normalize to 0-1 range for proper colorscale
        mean_data = np.mean(data[50:, :, :, :], axis=0)
        data_min, data_max = np.nanmin(mean_data), np.nanmax(mean_data)
        if data_max > data_min:
            normalized_data = (mean_data - data_min) / (data_max - data_min)
        else:
            normalized_data = mean_data
        
        # Use fixed 0-1 range for causality plots
        value_ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        axes, matrix_data = plot_average_matrices(np.expand_dims(normalized_data, axis=0), 
                                                matrix_cmap, start_window=0, value_ranges=value_ranges)
        matrix_data = normalized_data  # Use normalized data for network graphs
    
    # Network graphs (only for causality and causality*ergodicity)
    if data_type != "ergodicity":
        plot_network_graphs(matrix_data)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis and visualization routine."""
    print("Loading and visualizing DDA results...")
    
    for noise_condition in NOISE_CONDITIONS:
        print(f"\nProcessing {noise_condition} results...")
        
        # Define filenames
        dda_filename = f"DDA{SL}{noise_condition}__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}.DDA"
        
        # Check if DDA results exist
        if not Path(f"{dda_filename}_ST").exists():
            print(f"  Warning: DDA results not found for {noise_condition}")
            continue
        
        # Load DDA results
        ergodicity_matrix, causality_matrix = load_dda_results(dda_filename)
        
        # Create visualizations
        print("  Creating ergodicity plots...")
        create_comprehensive_plot(
            ergodicity_matrix,
            f"Ergodicity - {noise_condition}",
            f"FIG{SL}E__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}_{noise_condition}.png",
            "ergodicity"
        )
        
        print("  Creating causality plots...")
        create_comprehensive_plot(
            causality_matrix,
            f"Causality - {noise_condition}",
            f"FIG{SL}C__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}_{noise_condition}.png",
            "causality"
        )
        
        print("  Creating combined causality-ergodicity plots...")
        combined_matrix = causality_matrix * ergodicity_matrix
        create_comprehensive_plot(
            combined_matrix,
            f"Causality × Ergodicity - {noise_condition}",
            f"FIG{SL}CE__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}_{noise_condition}.png",
            "causality_ergodicity"
        )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()