from julia_first_setup import *
from DDAfunctions import *

# Import all the main components
# from make_data_7_systems import *  # Skip data generation - use existing Julia data
from run_DDA_Roessler import *
from Roessler_ShowResults import show_roessler_results

# Execute the analysis
show_roessler_results()
