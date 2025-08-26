# Python package installation script
# This file contains the Python equivalents of Julia packages needed for replicating the results

# Run this script to install all required packages:
# python julia_first_setup.py

import subprocess
import sys

packages = [
    "numpy",  # Linear algebra and array operations
    "pandas",  # DataFrames equivalent
    "matplotlib",  # Plotting
    "scipy",  # Statistics and scientific computing
    "networkx",  # Graphs equivalent
    "h5py",  # HDF5 file support (similar to JLD2)
    "mat73",  # MATLAB file support
    "seaborn",  # Additional plotting capabilities
]


def install_packages():
    """Install required Python packages"""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")


if __name__ == "__main__":
    print("Installing required Python packages for DDA project...")
    install_packages()
    print("\nInstallation complete!")
    print("\nNote: Some Julia-specific packages have Python equivalents:")
    print("- Combinatorics -> itertools (built-in)")
    print("- Printf -> f-strings or format() (built-in)")
    print("- Random -> numpy.random")
    print("- DelimitedFiles -> numpy.loadtxt/savetxt")
    print("- LaTeXStrings -> matplotlib supports LaTeX in labels")
    print("- GraphRecipes -> networkx + matplotlib")
    print("- Colors -> matplotlib.colors")
