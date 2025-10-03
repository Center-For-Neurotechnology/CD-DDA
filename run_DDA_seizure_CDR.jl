include("DDAfunctions.jl")

using MAT, DelimitedFiles, JLD2, Statistics, LinearAlgebra, Combinatorics, Plots, Printf, Colors, Base.Threads

SR = 1024;
WL = Int(round(2 * SR)) # 2s windows
WS = Int(round(1 * SR)) # 1s overlap
global WN = 2048

seizure = "MG109bSz2"

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

paths = make_paths("MG109b_Sz2.mat")

NrCH = size(paths.X, 1);
CH = 1:NrCH;
@info "There are $(NrCH) channels for this patient."

TAU=[7 10]; TM = maximum(TAU); dm=4;
WL=Int(round(2 * SR)); WS = Int(round(1 * SR));

nr_delays=2; 
DDAmodel=[[0 0 0 1];  
          [0 0 0 2]; 
          [1 1 1 1]];
(MODEL, L_AF, DDAorder)=make_MODEL(DDAmodel);

LIST=reduce(hcat,collect(combinations(CH,2)))';
DELTA = 200;
N = Int64(ceil(size(LIST, 1) / DELTA));

FN_DATA = paths.FN_DATA;
base = replace(basename(FN_DATA), r"\.[^.]+$" => "");
FN_DDA  = @sprintf("%s%s%s.DDA", DDA_DIR, SL, base);
FN_ALL  = @sprintf("%s%s%s.jld2", DDA_DIR, SL, base);

if !isfile(FN_ALL)
   @printf("%s\n",FN_ALL);

   if !isfile(join([FN_DDA,"_ST"]))
      if Sys.iswindows()
         if !isfile("run_DDA_AsciiEdf.exe")
            run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
         end

         CMD=".\\run_DDA_AsciiEdf.exe";
      else
         CMD="./run_DDA_AsciiEdf";
      end
   
      # CMD = "$CMD -EDF";                                 
      CMD = "$CMD -MODEL $(join(MODEL," "))"                    
      CMD = "$CMD -TAU $(join(TAU," "))"                        
      CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays"    
      CMD = "$CMD -DATA_FN $FN_DATA -OUT_FN $FN_DDA"        
      CMD = "$CMD -WL $WL -WS $WS";                            
      CMD = "$CMD -SELECT 1 0 0 0";                               # ST-DDA               
    
      if Sys.iswindows()
         run(Cmd(string.(split(CMD, " "))));
      else
         run(`sh -c $CMD`);
      end

      rm(@sprintf("%s.info",FN_DDA));     
   end

   @sync @threads for n_N=1:N
      FN_DDAn = @sprintf("%s%s%s__%03d.DDA", DDA_DIR, SL, base, n_N);
      # FN_DDAn=@sprintf("%s%s%s__%03d.DDA",DDA_DIR,SL,replace(EDF_file,".edf" => ""),n_N);
       
       n=collect(1:DELTA) .+ (n_N-1)*DELTA; n=n[n.<=size(LIST,1)];
       LL1=LIST[n,:]; LL1=vcat(LL1'...)';

       if !isfile(join([FN_DDAn,"_CD_DDA_ST"]))
          if Sys.iswindows()
             if !isfile("run_DDA_AsciiEdf.exe")
                run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
             end
 
             CMD=".\\run_DDA_AsciiEdf.exe";
          else
             CMD="./run_DDA_AsciiEdf";
          end
   
         #  CMD = "$CMD -EDF";                                 
          CMD = "$CMD -MODEL $(join(MODEL," "))"                    
          CMD = "$CMD -TAU $(join(TAU," "))"                        
          CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays"    
          CMD = "$CMD -DATA_FN $FN_DATA -OUT_FN $FN_DDAn"        
          CMD = "$CMD -WL $WL -WS $WS";                            
          CMD = "$CMD -SELECT 0 1 1 0";                           # CT-DDA and CD-DDA
          CMD = "$CMD -CH_list $(join(LL1," "))";                 # all pairwise channels
          CMD = "$CMD -WL_CT 2 -WS_CT 2";                         # pairwise channels for CT-DDA
        
          if Sys.iswindows()
             run(Cmd(string.(split(CMD, " "))));
          else
             run(`sh -c $CMD`);
          end
  
          rm(@sprintf("%s.info",FN_DDAn));     
       end
   end

   ST=readdlm(join([FN_DDA,"_ST"])); 
   T=ST[:,1:2]; ST=ST[:,3:end];
   WN=size(T,1);

   rhoS=ST[:,L_AF:L_AF:end];
   ST = nothing; GC.gc();
   
   E=fill(NaN,WN,NrCH,NrCH);
   for n_N=1:N
       @printf("%3d ",n_N)
       FN_DDAn = @sprintf("%s%s%s__%03d.DDA", DDA_DIR, SL, base, n_N)
      #  FN_DDAn=@sprintf("%s%s%s__%03d.DDA",DDA_DIR,SL,replace(EDF_file,".edf" => ""),n_N);
   
       n=collect(1:DELTA) .+ (n_N-1)*DELTA; n=n[n.<=size(LIST,1)];
       LL1=LIST[n,:] .- CH[1] .+ 1;
   
       CT=readdlm(join([FN_DDAn,"_CT"]));
       CT=CT[:,3:end];
       CT=CT[:,L_AF:L_AF:end];
   
       for l=1:size(LL1,1)
           ch1=LL1[l,1];ch2=LL1[l,2];
           E[:,ch1,ch2] = abs.( dropdims(mean(rhoS[:,[ch1,ch2]],dims=2),dims=2) ./ CT[:,l] .- 1 );
           E[:,ch2,ch1] = E[:,ch1,ch2];
       end
       CT = nothing; GC.gc();
   end
   @printf("\n");
       
   C=fill(NaN,WN,NrCH,NrCH);
   for n_N=1:N
       @printf("%3d ",n_N)
      #  FN_DDAn=@sprintf("%s%s%s__%03d.DDA",DDA_DIR,SL,replace(EDF_file,".edf" => ""),n_N);
       FN_DDAn=@sprintf("%s%s%s__%03d.DDA", DDA_DIR, SL, base, n_N);
   
       n=collect(1:DELTA) .+ (n_N-1)*DELTA; n=n[n.<=size(LIST,1)];
       LL1=LIST[n,:] .- CH[1] .+ 1;
   
       CD=readdlm(join([FN_DDAn,"_CD_DDA_ST"]));
       CD=CD[:,3:end];
       CD=reshape(CD,WN,2,size(LL1,1));
   
       for l=1:size(LL1,1)
           ch1=LL1[l,1];ch2=LL1[l,2];
           C[:,ch1,ch2] = CD[:,2,l];
           C[:,ch2,ch1] = CD[:,1,l];
       end
       CD = nothing; GC.gc();
   end
   @printf("\n\n");

   @save FN_ALL C E rhoS T WN
   E = nothing; C = nothing; GC.gc();
end

function select_windows_by_abs_seconds(FN_ALL::AbstractString, mat_file::AbstractString;
                                       t0_abs::Real, t1_abs::Real, SR::Real,
                                       var_name::String="data_clean", eps_sec::Real=0.0)
    # 1) Load segment's time vector start (absolute seconds)
    M = matread(mat_file)
    t = M[var_name]["time"][1]            # FieldTrip: 1×T cell, take first vector
    t0_seg = first(t)::Float64           # absolute time of the first sample in this segment
    t1_seg = last(t)::Float64

    # 2) Convert absolute bounds to segment-relative **seconds**
    lo_sec = min(t0_abs, t1_abs) - t0_seg
    hi_sec = max(t0_abs, t1_abs) - t0_seg

    # 3) Convert to **samples** to compare with T (which is in samples)
    eps_samp = round(Int, eps_sec * SR)
    lo_samp  = round(Int, lo_sec * SR) - eps_samp
    hi_samp  = round(Int, hi_sec * SR) + eps_samp

    # 4) Load T and filter
    T = nothing
    JLD2.@load FN_ALL T
    T === nothing && error("T not found in $FN_ALL (expected WN×2 start/end samples).")

    idx = findall(i -> (T[i,1] >= lo_samp) && (T[i,2] <= hi_samp), 1:size(T,1))

    @info "Segment time range = [$(t0_seg), $(t1_seg)] s; requested abs range = [$(t0_abs), $(t1_abs)] s" 
    @info "Relative range = [$(lo_sec), $(hi_sec)] s → [$(lo_samp), $(hi_samp)] samples"
    @info "Selected $(length(idx)) / $(size(T,1)) windows"
    return idx
end

# --- Color gradients (fixed stops; hard seam near 0.25) ---
const CG_E= cgrad([RGB(0.9,0.9,0.9), RGB(0.3,.3,0.3), :magenta, :cyan],    
              [0.0, 0.25, 0.2501, 0.635, 1],scale=:linear);  
const CG_CE = cgrad([RGB(0.9,0.9,0.9), RGB(0.3,.3,0.3), :magenta, :cyan],    
              [0.0, 0.25, 0.2501, 0.635, 1],scale=:linear);

# --- Finite-safe min/max for clim (no normalization) ---
_finite_minmax(A) = begin
    vals = vec(A[isfinite.(A)])
    isempty(vals) && return (0.0, 1.0)
    (minimum(vals), maximum(vals))
end

# --- Copy array and set non-finite entries to 0 for display only ---
finite_or_zero(A) = (B = copy(A); B[.!isfinite.(B)] .= 0.0; B)

# --- Select windows by *seconds* using T (which stores SAMPLE indices) ---
"""
    select_windows_seconds(FN_ALL; t0, t1, SR, eps=0)

Return indices of windows whose [start_sample, end_sample] fall within
[t0, t1] seconds (inclusive) after converting to samples with `SR`.
`eps` is an optional tolerance in samples (default 0).
"""
function select_windows_seconds(FN_ALL::AbstractString; t0::Real, t1::Real, SR::Real, eps::Real=0)
    T = nothing
    JLD2.@load FN_ALL T
    T === nothing && error("T not found in $FN_ALL (expected WN×2 start/end samples).")
    @assert size(T,2) == 2 "T must be WN×2 (start,end) in samples."

    t0_s = round(Int, t0*SR)
    t1_s = round(Int, t1*SR)
    lo = min(t0_s, t1_s) - round(Int, eps)
    hi = max(t0_s, t1_s) + round(Int, eps)

    return findall(i -> (T[i,1] >= lo) && (T[i,2] <= hi), 1:size(T,1))
end

# --- Single-window plotter (on-disk FN_ALL) ---
"""
    plot_C_E_CE_from_file(FN_ALL; w, out_dir="FIG", prefix="conn",
                          save=true, fix_clim=true, global_clim=nothing,
                          show_missing_as_zero=true)

Plots C, E and C⋅E for window `w` (1-based). If `save=true`, writes PNG.
- `fix_clim`: use finite min/max per plot (no normalization).
- `global_clim`: optional (lo,hi) for CE to enforce same scale across windows.
- `show_missing_as_zero`: display-only zeroing of NaN/Inf.
"""
function plot_C_E_CE_from_file(FN_ALL::AbstractString; w::Int,
                               out_dir::String="FIG", prefix::String="conn", save::Bool=true,
                               fix_clim::Bool=true, global_clim::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                               show_missing_as_zero::Bool=true)
    C = nothing; E = nothing
    JLD2.@load FN_ALL C E
    C === nothing && error("C not found in $FN_ALL")
    E === nothing && error("E not found in $FN_ALL")

    WN_local = size(C, 1)
    n        = size(C, 2)
    w = clamp(w, 1, WN_local)

    Cw  = @view C[w, 1:n, 1:n]
    Ew  = @view E[w, 1:n, 1:n]
    CEw = Cw .* Ew

    # display copies so NaNs/Infs don't blank the heatmaps
    Cw_disp  = show_missing_as_zero ? finite_or_zero(Cw)  : copy(Cw)
    Ew_disp  = show_missing_as_zero ? finite_or_zero(Ew)  : copy(Ew)
    CEw_disp = show_missing_as_zero ? finite_or_zero(CEw) : copy(CEw)

    # color limits
    climC  = fix_clim ? _finite_minmax(Cw)  : nothing
    climE  = fix_clim ? _finite_minmax(Ew)  : nothing
    climCE = fix_clim ? (global_clim === nothing ? _finite_minmax(CEw) : global_clim) : nothing
    if climC  !== nothing && climC[1]  == climC[2];  climC  = (climC[1]-1e-9,  climC[2]+1e-9);  end
    if climE  !== nothing && climE[1]  == climE[2];  climE  = (climE[1]-1e-9,  climE[2]+1e-9);  end
    if climCE !== nothing && climCE[1] == climCE[2]; climCE = (climCE[1]-1e-9, climCE[2]+1e-9); end

    # ticks every 20 nodes
    ticks = 1:20:n
    ticklabels = 0:20:(n-1)

    p1 = heatmap(Cw_disp;  title="C (w=$(w))",
                 aspect_ratio=1, framestyle=:box, clim=climC,
                 xlabel="Node", ylabel="Node",
                 yflip=true, xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

    p2 = heatmap(Ew_disp;  title="E (w=$(w))", c=CG_E, clim=climE,
                 aspect_ratio=1, framestyle=:box,
                 xlabel="Node", ylabel="Node",
                 yflip=true, xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

    p3 = heatmap(CEw_disp; title="C ⋅ E (w=$(w))", c=CG_CE, clim=climCE,
                 aspect_ratio=1, framestyle=:box,
                 xlabel="Node", ylabel="Node",
                 yflip=true, xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

    plt = plot(p1, p2, p3; layout=(1,3), size=(2100,700))
    if save
        isdir(out_dir) || mkpath(out_dir)
        savefig(plt, joinpath(out_dir, @sprintf("%s_w%04d.png", prefix, w)))
    end
    return plt
end

# --- Batch saver by window indices ---
"""
    save_all_windows_from_file(FN_ALL; windows=nothing, out_dir="FIG", prefix="conn",
                               fix_clim=true, global_clim=nothing, show_missing_as_zero=true)

Saves C, E, C⋅E panels for each window in `windows` (or all if `nothing`).
"""
function save_all_windows_from_file(FN_ALL::AbstractString; 
                                    windows=nothing,
                                    out_dir::String="FIG", 
                                    prefix::String="conn",
                                    fix_clim::Bool=true, 
                                    global_clim::Union{Nothing,Tuple{Float64,Float64}}=nothing,
                                    show_missing_as_zero::Bool=true,
                                    renumber::Bool=false)

    C = nothing; E = nothing
    JLD2.@load FN_ALL C E
    C === nothing && error("C not found in $FN_ALL")
    E === nothing && error("E not found in $FN_ALL")

    WN_local = size(C, 1)
    n        = size(C, 2)
    ws = windows === nothing ? (1:WN_local) : collect(windows)
    ws = filter(w -> 1 ≤ w ≤ WN_local, ws)

    isdir(out_dir) || mkpath(out_dir)
    ticks = 1:20:n
    ticklabels = 0:20:(n-1)

    for (j, w) in enumerate(ws)
        Cw  = @view C[w, 1:n, 1:n]
        Ew  = @view E[w, 1:n, 1:n]
        CEw = Cw .* Ew

        # finite-safe copies
        finite_or_zero(A) = (B = copy(A); B[.!isfinite.(B)] .= 0.0; B)
        Cw_disp  = show_missing_as_zero ? finite_or_zero(Cw)  : copy(Cw)
        Ew_disp  = show_missing_as_zero ? finite_or_zero(Ew)  : copy(Ew)
        CEw_disp = show_missing_as_zero ? finite_or_zero(CEw) : copy(CEw)

        # color limits
        _finite_minmax(A) = begin
            vals = vec(A[isfinite.(A)])
            isempty(vals) && return (0.0, 1.0)
            (minimum(vals), maximum(vals))
        end
        climC  = fix_clim ? _finite_minmax(Cw)  : nothing
        climE  = fix_clim ? _finite_minmax(Ew)  : nothing
        climCE = fix_clim ? (global_clim === nothing ? _finite_minmax(CEw) : global_clim) : nothing
        if climC  !== nothing && climC[1]  == climC[2];  climC  = (climC[1]-1e-9,  climC[2]+1e-9);  end
        if climE  !== nothing && climE[1]  == climE[2];  climE  = (climE[1]-1e-9,  climE[2]+1e-9);  end
        if climCE !== nothing && climCE[1] == climCE[2]; climCE = (climCE[1]-1e-9, climCE[2]+1e-9); end

        # renumbered index for title & filename
        idx = renumber ? j : w
        title_suffix = @sprintf "(w=%d)" idx

        p1 = heatmap(Cw_disp; title="C $title_suffix",
                     aspect_ratio=1, framestyle=:box, clim=climC,
                     xlabel="Node", ylabel="Node", yflip=true,
                     xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

        p2 = heatmap(Ew_disp; title="E $title_suffix", c=CG_E, clim=climE,
                     aspect_ratio=1, framestyle=:box,
                     xlabel="Node", ylabel="Node", yflip=true,
                     xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

        p3 = heatmap(CEw_disp; title="C ⋅ E $title_suffix", c=CG_CE, clim=climCE,
                     aspect_ratio=1, framestyle=:box,
                     xlabel="Node", ylabel="Node", yflip=true,
                     xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))

        plt = plot(p1, p2, p3; layout=(1,3), size=(2100,700))

        fn = joinpath(out_dir, @sprintf("%s_w%04d.png", prefix, idx))
        savefig(plt, fn)
    end
    return nothing
end


# 1) Plot a single window to screen (no save)
plt = plot_C_E_CE_from_file(FN_ALL; w=10, out_dir=FIG_DIR, prefix=base*"_C_E", save=false)
display(plt)

# 2) Save all windows in the file
t0_abs = 3524.4
t1_abs = 3604.1
ws = select_windows_by_abs_seconds(FN_ALL, "MG109b_Sz2.mat"; t0_abs=t0_abs, t1_abs=t1_abs, SR=SR)

# now save only those windows with your existing saver
save_all_windows_from_file(FN_ALL;
    windows=ws,
    out_dir=FIG_DIR,
    prefix=base,
    renumber=true)