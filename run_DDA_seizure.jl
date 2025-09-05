include("DDAfunctions.jl")

using MAT, DelimitedFiles, JLD2, Statistics, LinearAlgebra, Combinatorics, Plots, Printf

SR = 1024;
WL = Int(round(2 * SR)) # 2s windows
WS = Int(round(1 * SR)) # 1s overlap
global WN = 2048

seizure = "MG177Sz1"

DDA_DIR = "DDA"; dir_exist(DDA_DIR);
DATA_DIR = "DATA";
FIG_DIR = "FIG"; dir_exist(FIG_DIR);

function make_paths(mat_file::String, var_name::String="data_clean", out_dir::String="DATA_"*seizure,
    WL::Int=2048, WS::Int=1024, WN::Int=2048)
    dir_exist(out_dir)
    # load MATLAB and write ASCII for DDA binary
    M = matread(mat_file)
    X = M[var_name]["trial"][1] # nCh x time

    base = @sprintf("CD_DDA_%s_data__WL%d_WS%d_WN%d", seizure, WL, WS, WN)
    ascii_file = joinpath(out_dir, base * ".ascii")
    writedlm(ascii_file, permutedims(X, (2, 1)))
    
    FN_DATA = ascii_file
    FN_DDA  = joinpath(out_dir, base * ".DDA")
    FN_ALL  = joinpath(out_dir, base * ".jld2")
    return (; FN_DATA, FN_DDA, FN_ALL, X)
end

paths = make_paths("MG177_Sz1.mat")

NrSyst = size(paths.X, 1); DIM = 3;
NrCH = NrSyst; CH = collect(1:NrCH);

LIST = collect(combinations(CH, 2));
LL1 = vcat(LIST...)'; LIST=reduce(hcat, LIST)';

nr_delays = 2; dm = 4;

DDAmodel = [[0 0 0 1];
            [0 0 0 2];
            [1 1 1 1]];
(MODEL, L_AF, DDAorder) = make_MODEL(DDAmodel);

TAU = [7 10]; TM = maximum(TAU);

FN_data = paths.FN_DATA

base_DDA = @sprintf("%s%s%s__WL%d_WS%d_WN%d",
                    DDA_DIR, SL, seizure, WL, WS, WN)

DELTA = 200;
N = Int64(ceil(size(LIST, 1) / DELTA));

@threads for n_N in 1:N
    idx = ((n_N - 1) * DELTA + 1) : min(n_N * DELTA, size(LIST, 1))
    LL1_sub = LIST[idx, :]
    ch_arg = vec(permutedims(LL1_sub))

    FN_DDA_n = @sprintf("%s__%03d.DDA", base_DDA, n_N)

    if !isfile(string(FN_DDA_n, "_ST"))
        if Sys.iswindows()
          if !isfile("run_DDA_AsciiEdf.exe")
             cp("run_DDA_AsciiEdf","run_DDA_AsciiEdf.exe");
          end
    
          CMD=".\\run_DDA_AsciiEdf.exe";
       else
          CMD="./run_DDA_AsciiEdf";
       end

       CMD = "$CMD -MODEL $(join(MODEL," "))";                         # model
       CMD = "$CMD -TAU $(join(TAU," "))";                             # delays       
       CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays";       # DDA parameters
       CMD = "$CMD -DATA_FN $FN_data -OUT_FN $FN_DDA_n";                 # input and output files
       CMD = "$CMD -WL $WL -WS $WS";                                   # window length and shift
       CMD = "$CMD -SELECT 1 1 1 0";                                   # ST, CT, and CD DDA
       CMD = "$CMD -WL_CT 2 -WS_CT 2";                                 # take pairwise channels for CT and CD DDA
       CMD = "$CMD -CH_list $(join(string.(ch_arg), " "))";                     # list of channel pairs 
            
       if Sys.iswindows()                                              # run ST, CT, and CD DDA
          run(Cmd(string.(split(CMD, " "))));
       else
          run(`sh -c $CMD`);
       end
    
       rm(@sprintf("%s.info",FN_DDA_n));     
    end
end