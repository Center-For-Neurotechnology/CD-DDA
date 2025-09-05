using Printf, DelimitedFiles, Combinatorics, Statistics, LinearAlgebra
using Plots
gr()
import Plots: plot, heatmap!, savefig, display, font, cgrad
using Colors

include("DDAfunctions.jl")  # must provide dir_exist, make_MODEL, etc.

# ------------------------ USER SETTINGS ------------------------
const BASE_ID = "MG177Sz1__WL2048_WS1024_WN2048"
const DDA_DIR = "DDA"
const FIG_DIR = "FIG"
dir_exist(FIG_DIR)

# ------------------------ MODEL (seizure) ------------------------
const DDAmodel = [
    0 0 0 1;
    0 0 0 2;
    1 1 1 1
]
MODEL, L_AF, DDAorder = make_MODEL(DDAmodel)

# ------------------------ HELPERS ------------------------
re_escape(s::AbstractString) = replace(s, r"([\\.^$|?*+()\[\]{}])" => s"\\\1")

tofloat(x) = x isa Number ? float(x) :
             x isa AbstractString ? (let s = strip(x)
                 if isempty(s) || lowercase(s) == "nan"
                     NaN
                 else
                     y = tryparse(Float64, s)
                     y === nothing && error("Non-numeric token: '$s'")
                     y
                 end
             end) :
             float(x)

force_float(M) = Array{Float64}(tofloat.(M))

function norm01_vec(v::AbstractVector{<:Real})
    u = copy(v)
    vals = filter(!isnan, u)
    isempty(vals) && return u
    mn, mx = minimum(vals), maximum(vals)
    if mx == mn
        u[.!isnan.(u)] .= 0.0
    else
        nz = .!isnan.(u)
        u[nz] = (u[nz] .- mn) ./ (mx - mn)
    end
    u
end

# find chunked files: <BASE_ID>__NNN<suffix>, sorted by NNN
function find_chunk_files(base::AbstractString, suffix::AbstractString)
    pat = Regex("^" * re_escape(base) * "__(\\d{3})" * re_escape(suffix) * "\$")
    matched = Tuple{Int,String}[]
    for f in readdir(DDA_DIR)
        if (m = match(pat, f)) !== nothing
            push!(matched, (parse(Int, m.captures[1]), joinpath(DDA_DIR, f)))
        end
    end
    sort!(matched; by = x -> x[1])
    [x[2] for x in matched]
end

function read_ST_matrix(base::AbstractString)
    single = joinpath(DDA_DIR, @sprintf("%s.DDA_ST", base))
    if isfile(single)
        return force_float(readdlm(single))
    end
    chunk_STs = find_chunk_files(base, ".DDA_ST")
    isempty(chunk_STs) && error("No ST files found for base=$(base).")
    force_float(readdlm(chunk_STs[1]))
end

function _min_rows_CT(base, L_AF)
    paths = find_chunk_files(base, ".DDA_CT")
    isempty(paths) && error("No CT chunk files found.")
    minimum(begin
        CT = force_float(readdlm(p))[:, 3:end]
        size(CT[:, L_AF:L_AF:end], 1)
    end for p in paths)
end

function _min_rows_CD(base)
    paths = find_chunk_files(base, ".DDA_CD_DDA_ST")
    isempty(paths) && error("No CD chunk files found.")
    minimum(begin
        CD = force_float(readdlm(p))[:, 3:end]
        size(CD, 1)
    end for p in paths)
end

function normalize_rows(M::AbstractMatrix{<:Real}, WN_target::Int)
    r, c = size(M)
    if r == WN_target
        M
    elseif r > WN_target
        @view M[1:WN_target, :]
    else
        vcat(M, fill(NaN, WN_target - r, c))
    end
end

function normalize_rows(M::Array{<:Real,3}, WN_target::Int)
    r, c2, c3 = size(M)
    if r == WN_target
        M
    elseif r > WN_target
        @view M[1:WN_target, :, :]
    else
        vcat(M, fill(NaN, WN_target - r, c2, c3))
    end
end

function read_CT_chunks(base::AbstractString, L_AF::Int, WN_target::Int)
    paths = find_chunk_files(base, ".DDA_CT")
    isempty(paths) && error("No CT chunk files found for base=$(base).")
    CT_parts = Vector{Matrix{Float64}}()
    for p in paths
        CT = force_float(readdlm(p))
        CT = CT[:, 3:end]         # drop window start/end
        CT = CT[:, L_AF:L_AF:end] # one col per pair
        CT = normalize_rows(CT, WN_target)
        push!(CT_parts, CT)
    end
    hcat(CT_parts...)            # WN_target × totalPairs
end

# CD_DDA_ST: two columns per pair (two directions) -> reshape WN×2×k, then concat
function read_CD_chunks(base::AbstractString, WN_target::Int)
    paths = find_chunk_files(base, ".DDA_CD_DDA_ST")
    isempty(paths) && error("No CD_DDA_ST chunk files found for base=$(base).")
    CD_parts = Vector{Array{Float64,3}}()  # each: WN × 2 × k
    for p in paths
        CD = force_float(readdlm(p))
        CD = CD[:, 3:end]
        k  = size(CD, 2) ÷ 2
        CD = reshape(CD, size(CD,1), 2, k)  # WN × 2 × k
        CD = normalize_rows(CD, WN_target)
        push!(CD_parts, CD)
    end
    cat(CD_parts...; dims = 3)             # WN_target × 2 × totalPairs
end

# ------------------------ LOAD & BUILD MATRICES ------------------------
# ST (single-system): after L_AF stride => WN × NrSyst (single case)
ST_raw = read_ST_matrix(BASE_ID)          # WN × (2 + cols)
T      = ST_raw[:, 1:2]
WN_ST  = size(T, 1)
STblk  = ST_raw[:, 3:end][:, L_AF:L_AF:end]   # WN_ST × NrSyst
NrSyst = size(STblk, 2)

# choose unified window count using minimum across ST/CT/CD
WN_CTmin = _min_rows_CT(BASE_ID, L_AF)
WN_CDmin = _min_rows_CD(BASE_ID)
const WN_TARGET = min(WN_ST, WN_CTmin, WN_CDmin)

# crop ST to WN_TARGET if needed
if WN_ST != WN_TARGET
    T     = @view T[1:WN_TARGET, :]
    STblk = @view STblk[1:WN_TARGET, :]
end

# read CT and CD (aligned to WN_TARGET)
CT = read_CT_chunks(BASE_ID, L_AF, WN_TARGET)  # WN × nPairs
CD = read_CD_chunks(BASE_ID, WN_TARGET)        # WN × 2 × nPairs
WN = WN_TARGET

expected_pairs = length(collect(combinations(1:NrSyst, 2)))
nPairs = size(CT, 2)
expected_pairs == nPairs || @warn "Pair count mismatch. Expected $(expected_pairs), got $(nPairs). Assuming chunk order == combinations(CH,2)."

LIST_comb = reduce(hcat, collect(combinations(1:NrSyst, 2)))'   # nComb × 2, i<j
LIST_perm = reduce(hcat, collect(permutations(1:NrSyst, 2)))'   # nPerm × 2, i≠j
labels_comb = [@sprintf("%d %d", r[1], r[2]) for r in eachrow(LIST_comb)]
labels_perm = [@sprintf("%d→%d", r[1], r[2]) for r in eachrow(LIST_perm)]

E = fill(NaN, WN, NrSyst, NrSyst)   # ergodicity (symmetric)
C = fill(NaN, WN, NrSyst, NrSyst)   # causality (directed)

# map from LIST_comb index (column in CT) to unordered pair (i<j)
# assumes CT columns are ordered by combinations(1:NrSyst,2) in the same order as LIST_comb
for (l, (i, j)) in enumerate(eachrow(LIST_comb))
    st_mean = dropdims(mean(STblk[:, [i, j]]; dims = 2), dims = 2)  # WN
    E[:, i, j] = abs.(st_mean ./ CT[:, l] .- 1)
    E[:, j, i] = E[:, i, j]
end

# fill C from CD (dir 2: i->j; dir 1: j->i) in LIST_comb order per pair; need to scatter to full matrix
for (l, (i, j)) in enumerate(eachrow(LIST_comb))
    C[:, i, j] = CD[:, 2, l]
    C[:, j, i] = CD[:, 1, l]
end

# ------------------------ PER-WINDOW FIGURES (CE | E | C) ------------------------
# colors
CG_E  = cgrad([:white, RGB(1, 0.97, 0.86), RGB(0.55, 0.27, 0.07)], [0, 0.1], scale = :linear)
CG_CC = cgrad([RGB(0.9, 0.9, 0.9), RGB(0.3, .3, 0.3), :magenta, :cyan],
              [0.0, 0.25, 0.2501, 0.635, 1.0], scale = :linear)

nComb = size(LIST_comb, 1)
nPerm = size(LIST_perm, 1)
ndigits = length(string(WN))

# height scaled to the longest list of rows so labels remain readable
function _fig_size(nrows_max::Int)
    # ~18 px per row + base margin
    (1500, max(600, 18*nrows_max + 140))
end

for w in 1:WN
    # extract per-window vectors
    # CE and C use permutations (i→j)
    CE_vec = similar(Vector{Float64}(undef, nPerm))
    C_vec  = similar(Vector{Float64}(undef, nPerm))
    @inbounds for idx in 1:nPerm
        i, j = LIST_perm[idx, 1], LIST_perm[idx, 2]
        Cij  = C[w, i, j]
        Eij  = E[w, i, j]
        C_vec[idx]  = Cij
        CE_vec[idx] = Cij * Eij
    end
    # E uses combinations (i j, i<j)
    E_vec = similar(Vector{Float64}(undef, nComb))
    @inbounds for idx in 1:nComb
        i, j    = LIST_comb[idx, 1], LIST_comb[idx, 2]
        E_vec[idx] = E[w, i, j]
    end

    # normalize to [0,1] (per measure/per window) for clean colorbars
    Cn  = norm01_vec(C_vec)
    CEn = norm01_vec(CE_vec)
    En  = norm01_vec(E_vec)

    # reshape to column "images"
    Cimg  = reshape(Cn,  :, 1)   # nPerm × 1
    CEimg = reshape(CEn, :, 1)   # nPerm × 1
    Eimg  = reshape(En,  :, 1)   # nComb × 1

    l = @layout [a b c]
    sz = _fig_size(max(nPerm, nComb))
    SG = plot(layout = l, size = sz)

    # CE (permutations)
    heatmap!(SG, subplot = 1, CEimg,
             c = CG_CC, clim = (0,1), colorbar = true,
             yticks = (1:nPerm, labels_perm),
             xticks = (1, [""]), ytickfont = font(10), xtickfont = font(10),
             title = @sprintf("CE (window %d/%d)", w, WN))

    # E (combinations)
    heatmap!(SG, subplot = 2, Eimg,
             c = CG_E, clim = (0,1), colorbar = true,
             yticks = (1:nComb, labels_comb),
             xticks = (1, [""]), ytickfont = font(10), xtickfont = font(10),
             title = @sprintf("E (window %d/%d)", w, WN))

    # C (permutations)
    heatmap!(SG, subplot = 3, Cimg,
             c = CG_CC, clim = (0,1), colorbar = true,
             yticks = (1:nPerm, labels_perm),
             xticks = (1, [""]), ytickfont = font(10), xtickfont = font(10),
             title = @sprintf("C (window %d/%d)", w, WN))

    display(SG)
    out = joinpath(FIG_DIR, @sprintf("%s__W%0*d__CE_E_C.png", BASE_ID, ndigits, w))
    savefig(SG, out)
end
