from DDAfunctions import *
import subprocess
import platform
import os
from itertools import combinations

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
CH = list(range(1, NrCH + 1))

LIST = list(combinations(CH, 2))
LL1 = [item for sublist in LIST for item in sublist]  # Flatten list
LIST = np.array(LIST)

nr_delays = 2
dm = 4

DDAmodel = np.array([[0, 0, 1],
                     [0, 0, 2], 
                     [1, 1, 1]])

MODEL, L_AF, DDAorder = make_MODEL(DDAmodel)

TAU = [32, 9]
TM = max(TAU)

for n_NOISE in range(len(NOISE)):
    noise = NOISE[n_NOISE]
    
    FN_data = f"{DATA_DIR}{SL}CD_DDA_data_{noise}__WL{WL}_WS{WS}_WN{WN}.ascii"
    FN_DDA = f"{DDA_DIR}{SL}{noise}__WL{WL}_WS{WS}_WN{WN}.DDA"
    
    if not os.path.exists(f"{FN_DDA}_ST"):
        if platform.system() == "Windows":
            if not os.path.exists("run_DDA_AsciiEdf.exe"):
                import shutil
                shutil.copy("run_DDA_AsciiEdf", "run_DDA_AsciiEdf.exe")
            CMD = ".\\run_DDA_AsciiEdf.exe"
        else:
            CMD = "./run_DDA_AsciiEdf"
        
        CMD += f" -MODEL {' '.join(map(str, MODEL))}"
        CMD += f" -TAU {' '.join(map(str, TAU))}"
        CMD += f" -dm {dm} -order {DDAorder} -nr_tau {nr_delays}"
        CMD += f" -DATA_FN {FN_data} -OUT_FN {FN_DDA}"
        CMD += f" -WL {WL} -WS {WS}"
        CMD += " -SELECT 1 1 1 0"
        CMD += " -WL_CT 2 -WS_CT 2"
        CMD += f" -CH_list {' '.join(map(str, LL1))}"
        
        subprocess.run(CMD, shell=True)
        
        # Remove info file
        info_file = f"{FN_DDA}.info"
        if os.path.exists(info_file):
            os.remove(info_file)