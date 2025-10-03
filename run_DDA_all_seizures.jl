include("DDAfunctions.jl")

using MAT                    # matread, matwrite
using DelimitedFiles         # readdlm, writedlm
using Statistics, LinearAlgebra
using Combinatorics          # combinations
using Printf
using Base.Threads

# ------------------------ CONFIG ------------------------
const SR = 1024
const WL = Int(round(2 * SR))  # 2 s windows
const WS = Int(round(1 * SR))  # 1 s step
const TAU = [7, 10]
const dm = 4
const nr_delays = 2
const DDAmodel = [
    0 0 0 1;
    0 0 0 2;
    1 1 1 1
]

# Pair chunking for external DDA binary
const DELTA = 200

# Folder layout (relative to working directory)
const DATA_ROOT   = "data_mat"
const ASCII_ROOT  = "data_ascii"
const DDA_ROOT    = "dda"
const RES_SZ      = "results_seizure"
const RES_BL      = "results_baseline"

# Baseline definition
const BASELINE_SECS = 20.0  # 20 s before onset

# ------------------------ HELPERS ------------------------
is_win() = Sys.iswindows()

function ensure_dir(p::AbstractString)
    isdir(p) || mkpath(p)
    return p
end

# "EM23_Sz1_data_clean.mat" -> "EM23"
function patient_id_from_file(mat_path::AbstractString)
    base = basename(mat_path)
    split(base, "_"; limit=2)[1]
end

# "EM47_Sz15_data_clean.mat" -> "EM47_Sz15"   (strip extension and trailing "_data_clean")
function base_label(mat_path::AbstractString)
    base = replace(basename(mat_path), r"\.[^.]+$" => "")
    return replace(base, "_data_clean" => "")
end

# Build full path to the DDA binary
function dda_exe_path()
    if is_win()
        if !isfile("run_DDA_AsciiEdf.exe") && isfile("run_DDA_AsciiEdf")
            run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`)
        end
        return ".\\run_DDA_AsciiEdf.exe"
    else
        return "./run_DDA_AsciiEdf"
    end
end

# Robust loader for FieldTrip time vector (handles Vector or 1xN / Nx1 Matrix)
function load_time_vec(mat_path::AbstractString)::Vector{Float64}
    M = matread(mat_path)
    tcell = M["data_clean"]["time"][1]
    if tcell isa AbstractVector
        return Float64.(tcell)
    elseif tcell isa AbstractMatrix
        return Float64.(vec(tcell))  # flatten 1xN or Nx1 into Vector
    else
        error("Unexpected type for data_clean.time[1]: $(typeof(tcell))")
    end
end

# Robust loader for labels (handles Vector or Matrix of strings)
function load_labels(Mmeta)::Vector{String}
    lab = Mmeta["data_clean"]["label"]
    if lab isa AbstractVector
        return String.(lab)
    elseif lab isa AbstractMatrix
        return String.(vec(lab))
    else
        error("Unexpected type for data_clean.label: $(typeof(lab))")
    end
end

# Write ASCII for DDA into data_ascii/<PATIENT>
# Returns (ascii_fn, base, X)
function write_ascii_for_dda(mat_path::AbstractString; patient::AbstractString)
    dir_ascii = ensure_dir(joinpath(ASCII_ROOT, patient))
    M = matread(mat_path)
    X = M["data_clean"]["trial"][1]  # nCh x time
    base = @sprintf("CD_DDA_%s__WL%d_WS%d", base_label(mat_path), WL, WS)
    ascii_file = joinpath(dir_ascii, base * ".ascii")
    writedlm(ascii_file, permutedims(X, (2, 1)))
    return ascii_file, base, X
end

# Run ST-DDA (whole trial). Returns nothing (creates <FN_DDA>_ST)
function run_dda_whole_trial(FN_DATA::AbstractString, base::AbstractString, out_dir::AbstractString,
                             MODEL, DDAorder::Int)
    ensure_dir(out_dir)
    exe = dda_exe_path()
    FN_DDA = joinpath(out_dir, base * ".DDA")

    if !isfile(FN_DDA * "_ST")
        args = String[
            exe,
            "-MODEL", string.(MODEL)...,
            "-TAU",   string.(TAU)...,
            "-dm",    string(dm),
            "-order", string(DDAorder),
            "-nr_tau",string(nr_delays),
            "-DATA_FN", FN_DATA,
            "-OUT_FN",  FN_DDA,
            "-WL", string(WL),
            "-WS", string(WS),
            "-SELECT","1","0","0","0"   # ST-DDA
        ]
        run(Cmd(args))
        rm(@sprintf("%s.info",FN_DDA); force=true)
    end
    return nothing
end

# Build pair list chunks
function make_pair_chunks(nch::Int; delta::Int=DELTA)
    ch = 1:nch
    LIST = reduce(hcat, collect(combinations(ch, 2)))'  # (#pairs) x 2
    N = Int(ceil(size(LIST, 1) / delta))
    return LIST, N
end

# Compute C and E from DDA outputs; returns (C, E, T)
function compute_connectivity_from_DDA(base::AbstractString, out_dir::AbstractString, L_AF::Int, nch::Int, FN_DATA::AbstractString)
    # --- ST (for T, rhoS) ---
    ST = readdlm(joinpath(out_dir, base * ".DDA_ST"))
    T  = ST[:, 1:2]                 # WN x 2 (start,end samples)
    WN = size(T, 1)
    rhoS = ST[:, L_AF:L_AF:end]     # WN x nch
    ST = nothing; GC.gc()

    # --- pairwise chunks ---
    LIST, N = make_pair_chunks(nch)
    C = fill(NaN, WN, nch, nch)
    E = fill(NaN, WN, nch, nch)

    @sync @threads for n_N in 1:N
        @printf("%3d ", n_N)
        rng = ((n_N-1)*DELTA+1) : min(n_N*DELTA, size(LIST,1))
        LL  = LIST[rng, :]

        chunkstem = @sprintf("%s__%03d.DDA", base, n_N)
        FN_chunk  = joinpath(out_dir, chunkstem)

        # compute CT/CD if missing
        if !isfile(FN_chunk * "_CT") || !isfile(FN_chunk * "_CD_DDA_ST")
            exe = dda_exe_path()
            (MODEL2, _, DDAorder2) = make_MODEL(DDAmodel)
            ch_flat = vec(permutedims(LL))  # flatten pairs

            args = String[
                exe,
                "-MODEL", string.(MODEL2)...,
                "-TAU",   string.(TAU)...,
                "-dm",    string(dm),
                "-order", string(DDAorder2),
                "-nr_tau",string(nr_delays),
                "-DATA_FN", FN_DATA,
                "-OUT_FN",  joinpath(out_dir, chunkstem),
                "-WL", string(WL),
                "-WS", string(WS),
                "-SELECT","0","1","1","0",     # CT-DDA and CD-DDA
                "-CH_list", string.(ch_flat)...,
                "-WL_CT","2","-WS_CT","2"
            ]
            run(Cmd(args))
            rm(FN_chunk * ".info"; force=true)
        end

        # load CT, CD
        CT = readdlm(FN_chunk * "_CT")[:, 3:end]
        CT = CT[:, L_AF:L_AF:end]                         # WN x (#pairs in chunk)

        CD = readdlm(FN_chunk * "_CD_DDA_ST")[:, 3:end]
        CD = reshape(CD, WN, 2, size(LL,1))               # (WN, 2, #pairs)

        @inbounds for l in 1:size(LL,1)
            ch1, ch2 = LL[l,1], LL[l,2]
            num = abs.(dropdims(mean(rhoS[:, [ch1, ch2]], dims=2), dims=2))  # WN
            E[:, ch1, ch2] = num ./ CT[:, l] .- 1
            E[:, ch2, ch1] = E[:, ch1, ch2]

            C[:, ch1, ch2] = CD[:, 2, l]
            C[:, ch2, ch1] = CD[:, 1, l]
        end
        CT = nothing; CD = nothing; GC.gc()
    end
    @printf("\n")
    return C, E, T
end

# Map absolute seconds (meta) to sample bounds relative to this segment using FieldTrip time
function segment_sample_bounds(mat_path::AbstractString; t0_abs::Real, t1_abs::Real)
    tvec = load_time_vec(mat_path)             # robust to Vector or Matrix
    t0_seg = first(tvec)
    lo_sec = min(t0_abs, t1_abs) - t0_seg
    hi_sec = max(t0_abs, t1_abs) - t0_seg
    lo_samp = round(Int, lo_sec * SR)
    hi_samp = round(Int, hi_sec * SR)
    return lo_samp, hi_samp
end

function window_indices_in_range(T::AbstractMatrix, lo_samp::Int, hi_samp::Int)
    s = T[:, 1]
    e = T[:, 2]
    return findall(i -> (s[i] >= lo_samp) && (e[i] <= hi_samp), axes(T, 1))
end

# ------------------------ CORE PIPELINE (Single .mat) ------------------------
function process_mat_file(mat_path::AbstractString)
    patient = patient_id_from_file(mat_path)
    dir_ascii = ensure_dir(joinpath(ASCII_ROOT, patient))
    dir_dda   = ensure_dir(joinpath(DDA_ROOT, patient))
    dir_sz    = ensure_dir(joinpath(RES_SZ,  patient))
    dir_bl    = ensure_dir(joinpath(RES_BL,  patient))

    # 1) ASCII
    FN_DATA, base, X = write_ascii_for_dda(mat_path; patient=patient)

    # 2) MODEL constants
    (MODEL, L_AF, DDAorder) = make_MODEL(DDAmodel)

    # 3) ST-DDA (whole trial) with safe Cmd([...])
    run_dda_whole_trial(FN_DATA, base, dir_dda, MODEL, DDAorder)

    # 4) Build E, C (pair-chunked)
    NrCH = size(X, 1)
    C, E, T = compute_connectivity_from_DDA(base, dir_dda, L_AF, NrCH, FN_DATA)

    # 5) CE and selection by meta times; save to .mat with CE, labels, meta
    CE = C .* E
    C = nothing; E = nothing; GC.gc()

    Mmeta  = matread(mat_path)
    labels = load_labels(Mmeta)       # robust labels
    meta   = Mmeta["meta"]
    t_on   = Float64(meta["startTime"])
    t_off  = Float64(meta["endTime"])

    lo_sz, hi_sz = segment_sample_bounds(mat_path; t0_abs=t_on, t1_abs=t_off)
    lo_bl, hi_bl = segment_sample_bounds(mat_path; t0_abs=t_on-BASELINE_SECS, t1_abs=t_on)

    win_sz = window_indices_in_range(T, lo_sz, hi_sz)
    win_bl = window_indices_in_range(T, lo_bl, hi_bl)

    if !isempty(win_sz)
        CE_sz = permutedims(CE[win_sz, :, :], (2, 3, 1))
        out_sz = joinpath(dir_sz, base * "_seizure.mat")
        MAT.matwrite(out_sz, Dict("CE"=>CE_sz, "labels"=>labels, "meta"=>meta))
    else
        @warn "No seizure windows for $(basename(mat_path))"
    end

    if !isempty(win_bl)
         CE_bl = permutedims(CE[win_bl, :, :], (2, 3, 1))
        out_bl = joinpath(dir_bl, base * "_baseline.mat")
        MAT.matwrite(out_bl, Dict("CE"=>CE_bl, "labels"=>labels, "meta"=>meta))
    else
        @warn "No baseline windows for $(basename(mat_path))"
    end

    CE = nothing; GC.gc()
    return nothing
end

# ------------------------ DRIVER ------------------------
function run_all()
    patients = filter(isdir, readdir(DATA_ROOT; join=true))
    for pdir in patients
        mats = filter(f -> occursin(r"_data_clean\.mat$", f), readdir(pdir; join=true))
        isempty(mats) && continue
        @info "Processing patient $(basename(pdir)) with $(length(mats)) file(s)"
        for m in mats
            @info "→ $(basename(m))"
            try
                process_mat_file(m)
            catch e
                @error "Failed on $(basename(m))" exception=(e, catch_backtrace())
            end
        end
    end
    @info "All done."
end

# ------------------------ ENTRY ------------------------
run_all()