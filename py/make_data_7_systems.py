"""
Generate synthetic data for 7 coupled Roessler systems.

This script generates time series data for 7 coupled Roessler systems
with different coupling configurations (uncoupled, unidirectional, bidirectional).
The data is saved both with and without additive noise.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from DDAfunctions import (
    SL,
    add_noise,
    ensure_directory_exists,
    integrate_ode_general,
    create_mod_nr,
    create_coupling_mod_nr,
)


# Window parameters for data generation
WINDOW_LENGTH = 2000
WINDOW_SHIFT = 500
WINDOW_NUMBER = 500

# Alternative parameters (commented out)
# WINDOW_LENGTH = 4000
# WINDOW_SHIFT = 1000
# WINDOW_NUMBER = 2000

# Create necessary directories
DIRECTORIES = ["DDA", "DATA", "FIG"]
for directory in DIRECTORIES:
    ensure_directory_exists(directory)

# System configuration
NUM_SYSTEMS = 7

# Single Roessler system specification
# Format: [variable_index, power1, power2]
ROESSLER_SYSTEM = np.array(
    [
        [0, 0, 2],  # -y - z
        [0, 0, 3],  # -y - z
        [1, 0, 1],  # x + ay
        [1, 0, 2],  # x + ay
        [2, 0, 0],  # b + z(x - c)
        [2, 0, 3],  # b + z(x - c)
        [2, 1, 3],  # b + z(x - c)
    ]
)

# Generate encoding for 7 coupled systems
mod_nr, dimension, ode_order, monomial_array = create_mod_nr(
    ROESSLER_SYSTEM, NUM_SYSTEMS
)

# Model parameters for each Roessler system
ROESSLER_PARAMETERS = {
    "a_values": [0.21, 0.21, 0.21, 0.20, 0.20, 0.20, 0.18],
    "b_values": [0.2150, 0.2020, 0.2041, 0.4050, 0.3991, 0.4100, 0.5000],
    "c_values": [5.7, 5.7, 5.7, 5.7, 5.7, 5.7, 6.8],
}

# Build parameter matrix
parameter_matrix = []
for i in range(NUM_SYSTEMS):
    parameters = [
        -1,
        -1,
        1,  # Coefficients for -y - z + x
        ROESSLER_PARAMETERS["a_values"][i],
        ROESSLER_PARAMETERS["b_values"][i],
        -ROESSLER_PARAMETERS["c_values"][i],
        1,  # Coefficient for z term
    ]
    parameter_matrix.append(parameters)

model_parameters = np.array(parameter_matrix).T.flatten()

# Coupling configurations
# Case i: No coupling (empty)
coupling_case_i = np.array([])

# Case ii: Unidirectional coupling (systems 4,5,6 -> 7)
coupling_case_ii = np.array(
    [
        [4, 0, 0, 1, 7, 0, 0, 1],  # System 4 -> System 7
        [5, 0, 0, 1, 7, 0, 0, 1],  # System 5 -> System 7
        [6, 0, 0, 1, 7, 0, 0, 1],  # System 6 -> System 7
    ]
)

# Case iii: Bidirectional coupling (systems 4,5,6 <-> 7)
coupling_case_iii = np.array(
    [
        [7, 0, 0, 1, 4, 0, 0, 1],  # System 7 -> System 4
        [7, 0, 0, 1, 5, 0, 0, 1],  # System 7 -> System 5
        [7, 0, 0, 1, 6, 0, 0, 1],  # System 7 -> System 6
    ]
)

# Generate coupling MOD_nr
coupling_mod_ii = create_coupling_mod_nr(coupling_case_ii, dimension, monomial_array)
coupling_mod_iii = create_coupling_mod_nr(coupling_case_iii, dimension, monomial_array)

coupling_configurations = [np.array([]), coupling_mod_ii, coupling_mod_iii]

# Coupling strength
COUPLING_STRENGTH = 0.15

# Generate coupling parameters
coupling_params_ii = np.tile(
    [COUPLING_STRENGTH, -COUPLING_STRENGTH], (coupling_case_ii.shape[0], 1)
).flatten()
coupling_params_iii = np.tile(
    [COUPLING_STRENGTH, -COUPLING_STRENGTH], (coupling_case_iii.shape[0], 1)
).flatten()

coupling_parameters = [np.array([]), coupling_params_ii, coupling_params_iii]

# DDA parameters
DELAYS = [32, 9]
MAX_DELAY = max(DELAYS)
EMBEDDING_DIMENSION = 4

# Calculate data lengths for each case
data_lengths = [
    WINDOW_SHIFT * (WINDOW_NUMBER - 1)
    + WINDOW_LENGTH
    + MAX_DELAY
    + EMBEDDING_DIMENSION,
    WINDOW_SHIFT * WINDOW_NUMBER,
    WINDOW_SHIFT * WINDOW_NUMBER + EMBEDDING_DIMENSION - 1,
]

# Integration parameters
TIME_STEP = 0.05
SAMPLING_INTERVAL = 2  # Sample every 2nd point
TRANSIENT_STEPS = 20000

# Select channels (only x variables)
channel_list = list(range(1, dimension * NUM_SYSTEMS + 1, dimension))

# Case labels
CASES = ["i", "ii", "iii"]


def generate_data_for_case(case_index: int) -> None:
    """Generate data for a specific coupling case."""
    filename = f"DATA{SL}CD_DDA_data__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}__case_{CASES[case_index]}.ascii"

    if not Path(filename).exists():
        # Random initial conditions
        initial_conditions = np.random.rand(dimension * NUM_SYSTEMS)

        # Combine model and parameters with coupling if present
        if len(coupling_configurations[case_index]) > 0:
            combined_model = np.concatenate(
                [mod_nr, coupling_configurations[case_index]]
            )
            combined_params = np.concatenate(
                [model_parameters, coupling_parameters[case_index]]
            )
        else:
            combined_model = mod_nr
            combined_params = model_parameters

        # Integrate ODE system
        integrate_ode_general(
            combined_model,
            combined_params,
            TIME_STEP,
            data_lengths[case_index],
            dimension * NUM_SYSTEMS,
            ode_order,
            initial_conditions,
            filename,
            channel_list,
            SAMPLING_INTERVAL,
            TRANSIENT_STEPS,
        )


def plot_delay_embeddings(data: NDArray, title: str, filename: str) -> None:
    """Create delay embedding plots for all systems and cases."""
    fig, axes = plt.subplots(len(CASES), NUM_SYSTEMS, figsize=(21, 8))
    if len(CASES) == 1:
        axes = axes.reshape(1, -1)

    for case_idx in range(len(CASES)):
        for sys_idx in range(NUM_SYSTEMS):
            # Select data segment for plotting
            start_idx = 20000 + case_idx * data_lengths[case_idx]
            end_idx = 24000 + case_idx * data_lengths[case_idx]
            indices = list(range(start_idx, end_idx))
            delayed_indices = [i - 10 for i in indices]

            ax = axes[case_idx, sys_idx]
            ax.plot(
                data[indices, sys_idx], data[delayed_indices, sys_idx], linewidth=0.5
            )
            ax.set_aspect("equal")

            # Add labels to first row and column
            if case_idx == 0:
                ax.set_title(f"System {sys_idx + 1}")
            if sys_idx == 0:
                ax.set_ylabel(f"Case {CASES[case_idx]}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Generate data for all cases
print("Generating data for 7 coupled Roessler systems...")
for i in range(len(CASES)):
    print(f"  Case {CASES[i]}...")
    generate_data_for_case(i)

# Load all generated data
print("Loading generated data...")
all_data = None
for case_idx in range(len(CASES)):
    filename = f"DATA{SL}CD_DDA_data__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}__case_{CASES[case_idx]}.ascii"

    if case_idx == 0:
        all_data = np.loadtxt(filename)
    else:
        all_data = np.vstack([all_data, np.loadtxt(filename)])

# Plot noise-free data
print("Plotting noise-free delay embeddings...")
plot_delay_embeddings(
    all_data, "Roessler Systems - No Noise", f"DATA{SL}Roessler_7syst_NoNoise.png"
)

# Add noise to data
SNR_DB = 15  # Signal-to-noise ratio in dB
print(f"Adding noise (SNR = {SNR_DB} dB)...")

noisy_data = all_data.copy()
for case_idx in range(len(CASES)):
    for sys_idx in range(NUM_SYSTEMS):
        start_idx = case_idx * data_lengths[case_idx]
        end_idx = (case_idx + 1) * data_lengths[case_idx]
        indices = list(range(start_idx, end_idx))
        noisy_data[indices, sys_idx] = add_noise(noisy_data[indices, sys_idx], SNR_DB)

# Plot noisy data
print("Plotting noisy delay embeddings...")
plot_delay_embeddings(
    noisy_data,
    f"Roessler Systems - {SNR_DB}dB SNR",
    f"DATA{SL}Roessler_7syst_{SNR_DB}dB.png",
)

# Save processed data
print("Saving processed data...")
noise_free_filename = f"DATA{SL}CD_DDA_data_NoNoise__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}.ascii"
np.savetxt(noise_free_filename, all_data, fmt="%.15f", delimiter=" ")

noisy_filename = f"DATA{SL}CD_DDA_data_{SNR_DB}dB__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}.ascii"
np.savetxt(noisy_filename, noisy_data, fmt="%.15f", delimiter=" ")

# Clean up memory
del noisy_data
del all_data

print("Data generation complete!")
