"""
Run Delay Differential Analysis (DDA) on Roessler system data.

This script performs DDA analysis on previously generated Roessler system data,
including static (ST), conditional (CT), and causal (CD) analysis.
"""

import platform
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

from DDAfunctions import SL, create_model, ensure_directory_exists


# Data window parameters (must match data generation)
WINDOW_LENGTH = 2000
WINDOW_SHIFT = 500
WINDOW_NUMBER = 500

# Alternative parameters (commented out)
# WINDOW_LENGTH = 4000
# WINDOW_SHIFT = 1000
# WINDOW_NUMBER = 2000

# Ensure required directories exist
ensure_directory_exists("DDA")
ensure_directory_exists("FIG")

# Noise conditions to analyze
NOISE_CONDITIONS = ["NoNoise", "15dB"]

# System configuration
NUM_SYSTEMS = 7
SYSTEM_DIMENSION = 3
NUM_CHANNELS = NUM_SYSTEMS

# Analyze x-components of all systems (channels 1-7)
CHANNELS = list(range(1, NUM_CHANNELS + 1))

# Generate all pairwise channel combinations for cross-analysis
channel_pairs = list(combinations(CHANNELS, 2))
channel_pair_list = []
for pair in channel_pairs:
    channel_pair_list.extend(pair)

# DDA model parameters
NUM_DELAYS = 2
EMBEDDING_DIMENSION = 4

# DDA model specification for polynomial approximation
# Model: \dot{v} = a_1 * v_1 + a_2 * v_2 + a_3 * v_1^2
DDA_MODEL_SPEC = np.array(
    [
        [0, 0, 1],  # Linear term: a_1 * v_1
        [0, 0, 2],  # Linear term: a_2 * v_2
        [1, 1, 1],  # Quadratic term: a_3 * v_1^2
    ]
)

# Generate model encoding for DDA
model_indices, l_af, dda_order = create_model(DDA_MODEL_SPEC)

# Time delays for DDA analysis
DELAYS = [32, 9]  # Time delay values
MAX_DELAY = max(DELAYS)


def run_dda_analysis(data_filename: str, output_filename: str) -> None:
    """
    Run DDA analysis using external executable.

    Args:
        data_filename: Path to input data file
        output_filename: Base path for output files
    """
    # Check if analysis already completed
    if Path(f"{output_filename}_ST").exists():
        print(f"  Analysis already exists: {output_filename}")
        return

    # Determine executable based on platform
    if platform.system() == "Windows":
        executable = ".\\run_DDA_AsciiEdf.exe"
        # Copy executable if needed
        if not Path("run_DDA_AsciiEdf.exe").exists():
            import shutil

            shutil.copy("run_DDA_AsciiEdf", "run_DDA_AsciiEdf.exe")
    else:
        executable = "./run_DDA_AsciiEdf"

    # Build command line arguments
    cmd_parts = [
        executable,
        "-MODEL",
        " ".join(map(str, model_indices)),
        "-TAU",
        " ".join(map(str, DELAYS)),
        "-dm",
        str(EMBEDDING_DIMENSION),
        "-order",
        str(dda_order),
        "-nr_tau",
        str(NUM_DELAYS),
        "-DATA_FN",
        data_filename,
        "-OUT_FN",
        output_filename,
        "-WL",
        str(WINDOW_LENGTH),
        "-WS",
        str(WINDOW_SHIFT),
        "-SELECT",
        "1",
        "1",
        "1",
        "0",  # Enable ST, CT, CD analysis
        "-WL_CT",
        "2",  # Window length for cross-correlation
        "-WS_CT",
        "2",  # Window shift for cross-correlation
        "-CH_list",
        " ".join(map(str, channel_pair_list)),
    ]

    # Execute DDA analysis
    if platform.system() == "Windows":
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
    else:
        cmd = " ".join(cmd_parts)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Check for execution errors
    if result.returncode != 0:
        print(f"  Error running DDA analysis: {result.stderr}")
        return

    # Clean up info file if it exists
    info_file = f"{output_filename}.info"
    if Path(info_file).exists():
        Path(info_file).unlink()

    print(f"  Completed: {output_filename}")


def main():
    """Run DDA analysis for all noise conditions."""
    print("Running DDA analysis on Roessler system data...")

    for noise_condition in NOISE_CONDITIONS:
        print(f"\nProcessing {noise_condition} data...")

        # Define input and output filenames
        data_file = f"DATA{SL}CD_DDA_data_{noise_condition}__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}.ascii"
        output_file = f"DDA{SL}{noise_condition}__WL{WINDOW_LENGTH}_WS{WINDOW_SHIFT}_WN{WINDOW_NUMBER}.DDA"

        # Check if input data exists
        if not Path(data_file).exists():
            print(f"  Warning: Data file not found: {data_file}")
            continue

        # Run DDA analysis
        run_dda_analysis(data_file, output_file)

    print("\nDDA analysis complete!")


if __name__ == "__main__":
    main()
