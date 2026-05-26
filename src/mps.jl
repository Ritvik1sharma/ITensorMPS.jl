using Adapt: adapt
using LinearAlgebra: qr, ColumnNorm, Diagonal
import SparseBackends
using NDTensors: using_auto_fermion
using NDTensors.TypeParameterAccessors: unspecify_type_parameters
using Random: Random
using ITensors.SiteTypes: SiteTypes, siteind, siteinds, state

"""
    MPS

A finite size matrix product state type.
Keeps track of the orthogonality center.
"""
mutable struct MPS <: AbstractMPS
    data::Vector{ITensor}
    llim::Int
    rlim::Int
end

function MPS(A::Vector{<:ITensor}; ortho_lims::UnitRange = 1:length(A))
    return MPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

set_data(A::MPS, data::Vector{ITensor}) = MPS(data, A.llim, A.rlim)

@doc """
    MPS(v::Vector{<:ITensor})

Construct an MPS from a Vector of ITensors.
""" MPS(v::Vector{<:ITensor})

"""
    MPS()

Construct an empty MPS with zero sites.
"""
MPS() = MPS(ITensor[], 0, 0)

"""
    MPS(N::Int)

Construct an MPS with N sites with default constructed
ITensors.
"""
function MPS(N::Int; ortho_lims::UnitRange = 1:N)
    return MPS(Vector{ITensor}(undef, N); ortho_lims = ortho_lims)
end

"""
    MPS([::Type{ElT} = Float64, ]sites; linkdims=1)

Construct an MPS filled with Empty ITensors of type `ElT` from a collection of indices.

Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.

In the future we may generalize `linkdims` to allow specifying each individual link dimension as a vector,
and additionally allow specifying quantum numbers.
"""
function MPS(
        ::Type{T}, sites::Vector{<:Index}; linkdims::Union{Integer, Vector{<:Integer}} = 1
    ) where {T <: Number}
    _linkdims = _fill_linkdims(linkdims, sites)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 1
        v[1] = ITensor(T, sites[1])
        return MPS(v)
    end

    spaces = if hasqns(sites)
        [[QN() => _linkdims[j]] for j in 1:(N - 1)]
    else
        [_linkdims[j] for j in 1:(N - 1)]
    end

    l = [Index(spaces[ii], "Link,l=$ii") for ii in 1:(N - 1)]
    for ii in eachindex(sites)
        s = sites[ii]
        if ii == 1
            v[ii] = ITensor(T, l[ii], s)
        elseif ii == N
            v[ii] = ITensor(T, dag(l[ii - 1]), s)
        else
            v[ii] = ITensor(T, dag(l[ii - 1]), s, l[ii])
        end
    end
    return MPS(v)
end

MPS(sites::Vector{<:Index}, args...; kwargs...) = MPS(Float64, sites, args...; kwargs...)

function randomU(eltype::Type{<:Number}, s1::Index, s2::Index)
    return randomU(Random.default_rng(), eltype, s1, s2)
end

function randomU(rng::AbstractRNG, eltype::Type{<:Number}, s1::Index, s2::Index)
    if !hasqns(s1) && !hasqns(s2)
        mdim = dim(s1) * dim(s2)
        RM = randn(rng, eltype, mdim, mdim)
        Q, _ = NDTensors.qr_positive(RM)
        G = itensor(Q, dag(s1), dag(s2), s1', s2')
    else
        M = random_itensor(rng, eltype, QN(), s1', s2', dag(s1), dag(s2))
        U, S, V = svd(M, (s1', s2'))
        u = commonind(U, S)
        v = commonind(S, V)
        replaceind!(U, u, v)
        G = U * V
    end
    return G
end

function randomizeMPS!(eltype::Type{<:Number}, M::MPS, sites::Vector{<:Index}, linkdims = 1)
    return randomizeMPS!(Random.default_rng(), eltype, M, sites, linkdims)
end

function randomizeMPS!(
        rng::AbstractRNG, eltype::Type{<:Number}, M::MPS, sites::Vector{<:Index}, linkdims = 1
    )
    _linkdims = _fill_linkdims(linkdims, sites)
    if isone(length(sites))
        randn!(rng, M[1])
        normalize!(M)
        return M
    end
    N = length(sites)
    c = div(N, 2)
    max_pass = 100
    for pass in 1:max_pass, half in 1:2
        if half == 1
            (db, brange) = (+1, 1:1:(N - 1))
        else
            (db, brange) = (-1, N:-1:2)
        end
        for b in brange
            s1 = sites[b]
            s2 = sites[b + db]
            G = randomU(rng, eltype, s1, s2)
            T = noprime(G * M[b] * M[b + db])
            rinds = uniqueinds(M[b], M[b + db])

            b_dim = half == 1 ? b : b + db
            U, S, V = svd(T, rinds; maxdim = _linkdims[b_dim], utags = "Link,l=$(b - 1)")
            M[b] = U
            M[b + db] = S * V
            M[b + db] /= norm(M[b + db])
        end
        if half == 2 && dim(commonind(M[c], M[c + 1])) >= _linkdims[c]
            break
        end
    end
    setleftlim!(M, 0)
    setrightlim!(M, 2)
    return if dim(commonind(M[c], M[c + 1])) < _linkdims[c]
        @warn "MPS center bond dimension is less than requested (you requested $(_linkdims[c]), but in practice it is $(dim(commonind(M[c], M[c + 1]))). This is likely due to technicalities of truncating quantum number sectors."
    end
end

function randomCircuitMPS(
        eltype::Type{<:Number}, sites::Vector{<:Index}, linkdims::Vector{<:Integer}; kwargs...
    )
    return randomCircuitMPS(Random.default_rng(), eltype, sites, linkdims; kwargs...)
end

function randomCircuitMPS(
        rng::AbstractRNG,
        eltype::Type{<:Number},
        sites::Vector{<:Index},
        linkdims::Vector{<:Integer};
        kwargs...,
    )
    N = length(sites)
    M = MPS(N)

    if N == 1
        M[1] = ITensor(randn(rng, eltype, dim(sites[1])), sites[1])
        M[1] /= norm(M[1])
        return M
    end

    l = Vector{Index}(undef, N)

    d = dim(sites[N])
    chi = min(linkdims[N - 1], d)
    l[N - 1] = Index(chi, "Link,l=$(N - 1)")
    O = NDTensors.random_unitary(rng, eltype, chi, d)
    M[N] = itensor(O, l[N - 1], sites[N])

    for j in (N - 1):-1:2
        chi *= dim(sites[j])
        chi = min(linkdims[j - 1], chi)
        l[j - 1] = Index(chi, "Link,l=$(j - 1)")
        O = NDTensors.random_unitary(rng, eltype, chi, dim(sites[j]) * dim(l[j]))
        T = reshape(O, (chi, dim(sites[j]), dim(l[j])))
        M[j] = itensor(T, l[j - 1], sites[j], l[j])
    end

    O = NDTensors.random_unitary(rng, eltype, 1, dim(sites[1]) * dim(l[1]))
    l0 = Index(1, "Link,l=0")
    T = reshape(O, (1, dim(sites[1]), dim(l[1])))
    M[1] = itensor(T, l0, sites[1], l[1])
    M[1] *= onehot(eltype, l0 => 1)

    M.llim = 0
    M.rlim = 2

    return M
end

function randomCircuitMPS(sites::Vector{<:Index}, linkdims::Vector{<:Integer}; kwargs...)
    return randomCircuitMPS(Random.default_rng(), sites, linkdims; kwargs...)
end

function randomCircuitMPS(
        rng::AbstractRNG, sites::Vector{<:Index}, linkdims::Vector{<:Integer}; kwargs...
    )
    return randomCircuitMPS(rng, Float64, sites, linkdims; kwargs...)
end

function _fill_linkdims(linkdims::Vector{<:Integer}, sites::Vector{<:Index})
    @assert length(linkdims) == length(sites) - 1
    return linkdims
end

function _fill_linkdims(linkdims::Integer, sites::Vector{<:Index})
    return fill(linkdims, length(sites) - 1)
end

"""
    random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims=1)

Construct a random MPS with link dimension `linkdims` of
type `eltype`.

`linkdims` can also accept a `Vector{Int}` with
`length(linkdims) == length(sites) - 1` for constructing an
MPS with non-uniform bond dimension.
"""
function random_mps(
        ::Type{ElT}, sites::Vector{<:Index}; linkdims::Union{Integer, Vector{<:Integer}} = 1
    ) where {ElT <: Number}
    return random_mps(Random.default_rng(), ElT, sites; linkdims)
end

function random_mps(
        rng::AbstractRNG,
        ::Type{ElT},
        sites::Vector{<:Index};
        linkdims::Union{Integer, Vector{<:Integer}} = 1,
    ) where {ElT <: Number}
    _linkdims = _fill_linkdims(linkdims, sites)
    if any(hasqns, sites)
        error("initial state required to use random_mps with QNs")
    end

    # For non-QN-conserving MPS, instantiate
    # the random MPS directly as a circuit:
    return randomCircuitMPS(rng, ElT, sites, _linkdims)
end

"""
    random_mps(sites::Vector{<:Index}; linkdims=1)
    random_mps(eltype::Type{<:Number}, sites::Vector{<:Index}; linkdims=1)

Construct a random MPS with link dimension `linkdims` which by
default has element type `Float64`.

`linkdims` can also accept a `Vector{Int}` with
`length(linkdims) == length(sites) - 1` for constructing an
MPS with non-uniform bond dimension.
"""
function random_mps(sites::Vector{<:Index}; linkdims::Union{Integer, Vector{<:Integer}} = 1)
    return random_mps(Random.default_rng(), sites; linkdims)
end

function random_mps(
        rng::AbstractRNG, sites::Vector{<:Index}; linkdims::Union{Integer, Vector{<:Integer}} = 1
    )
    return random_mps(rng, Float64, sites; linkdims)
end

function random_mps(
        sites::Vector{<:Index}, state; linkdims::Union{Integer, Vector{<:Integer}} = 1
    )
    return random_mps(Random.default_rng(), sites, state; linkdims)
end

function random_mps(
        rng::AbstractRNG,
        sites::Vector{<:Index},
        state;
        linkdims::Union{Integer, Vector{<:Integer}} = 1,
    )
    return random_mps(rng, Float64, sites, state; linkdims)
end

function random_mps(
        eltype::Type{<:Number},
        sites::Vector{<:Index},
        state;
        linkdims::Union{Integer, Vector{<:Integer}} = 1,
    )
    return random_mps(Random.default_rng(), eltype, sites, state; linkdims)
end

function random_mps(
        rng::AbstractRNG,
        eltype::Type{<:Number},
        sites::Vector{<:Index},
        state;
        linkdims::Union{Integer, Vector{<:Integer}} = 1,
    )::MPS
    M = MPS(eltype, sites, state)
    if any(>(1), linkdims)
        randomizeMPS!(rng, eltype, M, sites, linkdims)
    end
    return M
end

@doc """
    random_mps(sites::Vector{<:Index}, state; linkdims=1)

Construct a real, random MPS with link dimension `linkdims`,
made by randomizing an initial product state specified by
`state`. This version of `random_mps` is necessary when creating
QN-conserving random MPS (consisting of QNITensors). The initial
`state` array provided determines the total QN of the resulting
random MPS.
""" random_mps(::Vector{<:Index}, ::Any)

"""
    MPS(::Type{T<:Number}, ivals::Vector{<:Pair{<:Index}})

Construct a product state MPS with element type `T` and
nonzero values determined from the input IndexVals.
"""
function MPS(::Type{T}, ivals::Vector{<:Pair{<:Index}}) where {T <: Number}
    N = length(ivals)
    M = MPS(N)

    if N == 1
        M[1] = ITensor(T, ind(ivals[1]))
        M[1][ivals[1]] = one(T)
        return M
    end

    if hasqns(ind(ivals[1]))
        lflux = QN()
        for j in 1:(N - 1)
            lflux += qn(ivals[j])
        end
        links = Vector{QNIndex}(undef, N - 1)
        for j in (N - 1):-1:1
            links[j] = dag(Index(lflux => 1; tags = "Link,l=$j"))
            lflux -= qn(ivals[j])
        end
    else
        links = [Index(1, "Link,l=$n") for n in 1:(N - 1)]
    end

    M[1] = ITensor(T, ind(ivals[1]), links[1])
    M[1][ivals[1], links[1] => 1] = one(T)
    for n in 2:(N - 1)
        s = ind(ivals[n])
        M[n] = ITensor(T, dag(links[n - 1]), s, links[n])
        M[n][links[n - 1] => 1, ivals[n], links[n] => 1] = one(T)
    end
    M[N] = ITensor(T, dag(links[N - 1]), ind(ivals[N]))
    M[N][links[N - 1] => 1, ivals[N]] = one(T)

    return M
end

# For backwards compatibility
const productMPS = MPS

"""
    MPS(ivals::Vector{<:Pair{<:Index}})

Construct a product state MPS with element type `Float64` and
nonzero values determined from the input IndexVals.
"""
MPS(ivals::Vector{<:Pair{<:Index}}) = MPS(Float64, ivals)

"""
    MPS(::Type{T},
        sites::Vector{<:Index},
        states::Union{Vector{String},
                      Vector{Int},
                      String,
                      Int})

Construct a product state MPS of element type `T`, having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The input `states` may
be an array of strings or an array of ints recognized by the
`state` function defined for the relevant Index tag type.
In addition, a single string or int can be input to create
a uniform state.

# Examples

```julia
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi = MPS(ComplexF64, sites, states)
phi = MPS(sites, "Up")
```
"""
function MPS(eltype::Type{<:Number}, sites::Vector{<:Index}, states_)
    if length(sites) != length(states_)
        throw(DimensionMismatch("Number of sites and and initial vals don't match"))
    end
    N = length(states_)
    M = MPS(N)

    if N == 1
        M[1] = state(sites[1], states_[1])
        return convert_leaf_eltype(eltype, M)
    end

    states = [state(sites[j], states_[j]) for j in 1:N]

    if hasqns(states[1])
        lflux = QN()
        for j in 1:(N - 1)
            lflux += flux(states[j])
        end
        links = Vector{QNIndex}(undef, N - 1)
        for j in (N - 1):-1:1
            links[j] = dag(Index(lflux => 1; tags = "Link,l=$j"))
            lflux -= flux(states[j])
        end
    else
        links = [Index(1; tags = "Link,l=$n") for n in 1:N]
    end

    M[1] = ITensor(sites[1], links[1])
    M[1] += states[1] * state(links[1], 1)
    for n in 2:(N - 1)
        M[n] = ITensor(dag(links[n - 1]), sites[n], links[n])
        M[n] += state(dag(links[n - 1]), 1) * states[n] * state(links[n], 1)
    end
    M[N] = ITensor(dag(links[N - 1]), sites[N])
    M[N] += state(dag(links[N - 1]), 1) * states[N]

    return convert_leaf_eltype(eltype, M)
end

function MPS(
        ::Type{T}, sites::Vector{<:Index}, state::Union{String, Integer}
    ) where {T <: Number}
    return MPS(T, sites, fill(state, length(sites)))
end

function MPS(::Type{T}, sites::Vector{<:Index}, states::Function) where {T <: Number}
    states_vec = [states(n) for n in 1:length(sites)]
    return MPS(T, sites, states_vec)
end

"""
    MPS(sites::Vector{<:Index},states)

Construct a product state MPS having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The `states` array may
consist of either an array of integers or strings, as
recognized by the `state` function defined for the relevant
Index tag type.

# Examples

```julia
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
psi = MPS(sites, states)
```
"""
MPS(sites::Vector{<:Index}, states) = MPS(Float64, sites, states)

"""
    siteind(M::MPS, j::Int; kwargs...)

Get the first site Index of the MPS. Return `nothing` if none is found.
"""
SiteTypes.siteind(M::MPS, j::Int; kwargs...) = siteind(first, M, j; kwargs...)

"""
    siteind(::typeof(only), M::MPS, j::Int; kwargs...)

Get the only site Index of the MPS. Return `nothing` if none is found.
"""
function SiteTypes.siteind(::typeof(only), M::MPS, j::Int; kwargs...)
    is = siteinds(M, j; kwargs...)
    if isempty(is)
        return nothing
    end
    return only(is)
end

"""
    siteinds(M::MPS)
    siteinds(::typeof(first), M::MPS)

Get a vector of the first site Index found on each tensor of the MPS.

    siteinds(::typeof(only), M::MPS)

Get a vector of the only site Index found on each tensor of the MPS. Errors if more than one is found.

    siteinds(::typeof(all), M::MPS)

Get a vector of the all site Indices found on each tensor of the MPS. Returns a Vector of IndexSets.
"""
SiteTypes.siteinds(M::MPS; kwargs...) = siteinds(first, M; kwargs...)

function replace_siteinds!(M::MPS, sites)
    for j in eachindex(M)
        sj = only(siteinds(M, j))
        M[j] = replaceinds(M[j], sj => sites[j])
    end
    return M
end

replace_siteinds(M::MPS, sites) = replace_siteinds!(copy(M), sites)


# Stable key for canonical tensor index ordering before factorization.
# We avoid using raw id because ids differ across equivalent runs.
canonical_ind_key(i) = (string(tags(i)), plev(i), dim(i), Int(dir(i)))

function canonicalize_phi_inds(phi::ITensor, indsMb)
    # println("Canonicalizing phi indices ---- ", indsMb)
    ord = sort(collect(inds(phi)); by = canonical_ind_key)
    # println("Canonical order: ", ord)
    if ITensors.has_external_storage(phi)
        phi_c = SparseBackends.permute(phi, ord...)
    else
        phi_c = permute(phi, ord...)
    end
    indsMb_c = [i for i in ord if i in indsMb]
    return phi_c, indsMb_c
end

# Put the shared bond in a fixed place:
#   left tensor : [other inds..., bond]
#   right tensor: [bond, other inds...]
function reorder_split_tensors(L::ITensor, R::ITensor)
    l = commonind(L, R)
    isnothing(l) && return L, R
    Lind = collect(inds(L))
    Rind = collect(inds(R))
    L_other = [i for i in Lind if i != l && i != dag(l)]
    R_other = [i for i in Rind if i != l && i != dag(l)]
    Lp = permute(L, L_other..., l)
    Rp = permute(R, l, R_other...)
    return Lp, Rp
end

"""
    stable_factorize(phi, indsMb; ortho, mindim, maxdim, cutoff, svd_alg, tags,
                     atol, rtol, null_atol, null_rtol, debug) → (L, R, spec)

Drop-in replacement for `factorize + canonicalize_split` in `replacebond!`.
Works directly on the raw SVD matrices (U, exact S diagonal, Vt) instead of
estimating singular values from row norms after the fact.

Steps:
  1. Call ITensors.svd to get U (n×D), S (exact diagonal), V (D×m).
  2. Fuse all non-bond indices via combiners to get plain Julia matrices.
  3. Sort channels by exact S values, then content-based fingerprints.
  4. Apply weighted-sum phase to each column of U.
  5. Reconstruct L and R ITensors from the canonical matrices.
"""
function stable_factorize(
    phi::ITensor,
    indsMb;           # already sorted by canonicalize_phi_inds
    level               = 1,
    ortho               = "left",
    mindim              = nothing,
    maxdim              = nothing,
    cutoff              = nothing,
    svd_alg             = nothing,
    use_absolute_cutoff = nothing,
    use_relative_cutoff = nothing,
    min_blockdim        = nothing,
    tags                = ITensors.ts"Link,l",
    atol                = 1e-12,
    rtol                = 1e-10,
    null_atol           = 1e-10,
    null_rtol           = 1e-6,
    target_link_sparse_dim::Int = -1,    # pass through to the BlockSparse SVD
                                          # so boundary bonds keep their original
                                          # sparse dim (replacebond! computes it
                                          # from M[b]/M[b+1] before contracting phi).
    M_b                 = nothing,        # OLD M[b], M[b+1] tensors. When both
    M_b1                = nothing,        # are sparse and provided, the sparse
                                           # SVD routes through the channel-aware
                                           # kernel so the new L, R preserve
                                           # image(P) globally (not just per-bond
                                           # block-key sets).
    debug               = false,
)
    # ── 0. Sparse-psi path ────────────────────────────────────────────────────
    # When M_b and M_b1 are both sparse, use the channel-aware SVD: templates
    # from M_b, M_b1 enforce that the new L, R block-key sets are subsets of
    # the OLD ones, which is needed to keep ψ in image(P) across multi-factor
    # bonds. relax_iso_cap=true: lets mult grow past the cross-channel iso cap
    # in bulk bonds — Path B's M^{-1/2} correction absorbs the non-iso slack
    # via the gram matrix during eigsolve. The iso cap is only geometrically
    # required when all channels share L-rows (the cap formula is conservative
    # otherwise), and DMRG sweeps don't need strict canonicality.
    if ITensors.has_external_storage(phi)
        if M_b !== nothing && M_b1 !== nothing &&
           ITensors.has_external_storage(M_b) && ITensors.has_external_storage(M_b1)
            # BMF_ISO_PATH=1: enforce strict iso → L^T L = I → no M correction
            # needed → DMRG runs standard Lanczos. Empirically (probe) cap_fired
            # is false for this projector at the operating Schmidt rank.
            iso_strict = get(ENV, "BMF_ISO_PATH", "0") == "1"
            return SparseBackends.itensor_blocksparse_svd_channel_aware(
                phi, M_b, M_b1;
                ortho, maxdim, mindim, cutoff,
                relax_iso_cap = !iso_strict)
        end
        # Fallback (no templates available — e.g. user-direct call): old path.
        return SparseBackends.itensor_blocksparse_svd(phi, indsMb;
            ortho, maxdim, mindim, cutoff,
            tags = ITensors.TagSet("Link,l=$level"),
            bin_by_right = true,
            target_n_new_sp = target_link_sparse_dim)

        # return SparseBackends.itensor_blocksparse_svd(phi, indsMb;
        #     ortho,
        #     maxdim = something(maxdim, typemax(Int)),
        #     mindim = something(mindim, 1),
        #     cutoff = Float64(something(cutoff, 0.0)),
        #     tags)


        # wrapped = ITensors.get_external_storage(phi)              # WrappedBlockSparse{T,N,N2,P}
        # bs      = wrapped.blocksparse                # NewBlockSparseSorted
        # P       = SparseBackends._P(bs)   # or pull P from the type — see below
        # sparse_inds = wrapped.inds[1:P]              # the sparse half of phi's legs
        # phi_inds    = wrapped.inds                     # all legs of phi
        # dense_inds  = wrapped.inds[P+1:end]
        # # nls = count(i -> i ∈ sparse_inds, indsMb)
        # # nld = length(indsMb) - nls

        # indsMb_in_phi = filter(i -> i ∈ phi_inds, indsMb)
        # nls = count(i -> i ∈ sparse_inds, indsMb_in_phi)
        # nld = count(i -> i ∈ dense_inds,  indsMb_in_phi)


        # # wrapped = ITensors.get_external_storage(phi)        
        # # P_phi   = _P(wrapped)
        # # sparse_inds = wrapped.inds[1:P_phi]

        # # @show wrapped.inds
        # # @show sparse_inds
        # # @show indsMb
        # # @show [i ∈ sparse_inds for i in indsMb]
        # # nls = count(i -> i ∈ sparse_inds, indsMb)
        # # nld = length(indsMb) - nls
        # # @show nls, nld

        # return SparseBackends.blocksparse_svd(bs;
        #     n_left_sparse = nls,
        #     n_left_dense  = nld,
        #     ortho,
        #     maxdim = something(maxdim, typemax(Int)),
        #     mindim = something(mindim, 1),
        #     cutoff = Float64(something(cutoff, 0.0)))

        # # return SparseBackends.itensor_blocksparse_svd(phi, indsMb;
        # #            ortho, maxdim=something(maxdim, typemax(Int)),
        # #            mindim=something(mindim, 1),
        # #            cutoff=Float64(something(cutoff, 0.0)), tags)
    end
    # ── 1. Direct SVD ─────────────────────────────────────────────────────────
    result = ITensors.svd(
        phi, indsMb;
        mindim,
        maxdim,
        cutoff,
        alg                 = something(svd_alg, "divide_and_conquer"),
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
        lefttags            = tags,
        righttags           = tags,
    )
    isnothing(result) && error("stable_factorize: SVD returned nothing")
    U_it, S_it, V_it, spec, u_idx, v_idx = result
    D = dim(u_idx)
    # ── 2. Exact singular values from S diagonal ───────────────────────────────
    svs = [S_it[u_idx => j, v_idx => j] for j in 1:D]
    # ── 3. Fuse non-bond indices → plain matrices ──────────────────────────────
    # U: (physical_left..., u_idx)  → n × D matrix
    # V: (v_idx, physical_right...) → D × m matrix (V† in the decomposition)
    U_phys = filter(i -> i ≠ u_idx && i ≠ dag(u_idx), collect(inds(U_it)))
    V_phys = filter(i -> i ≠ v_idx && i ≠ dag(v_idx), collect(inds(V_it)))
    CU = ITensors.combiner(U_phys...; tags="cU")
    CV = ITensors.combiner(V_phys...; tags="cV")
    cU = ITensors.combinedind(CU)
    cV = ITensors.combinedind(CV)
    Umat  = Array(U_it * CU, cU, u_idx)        # n × D
    Vmat  = Array(V_it * CV, v_idx, cV)         # D × m  (V† rows)
    # ── 4. Canonical channel ordering ─────────────────────────────────────────
    sv_scale  = max(maximum(svs), atol)
    group_tol = max(atol, rtol * sv_scale)
    null_tol  = max(null_atol, null_rtol * sv_scale)
    # Snap degenerate group representatives.
    sv_rep = copy(svs)
    for k in 1:D, j in k+1:D
        abs(svs[j] - sv_rep[k]) <= group_tol && (sv_rep[j] = sv_rep[k])
    end
    _TOPK = 8
    col_fp      = [(sum(abs(x)^3 for x in @view Umat[:, j]))^(1/3) for j in 1:D]
    row_fp      = [(sum(abs(x)^3 for x in @view Vmat[j, :]))^(1/3) for j in 1:D]
    col_profile = [ntuple(k -> k <= size(Umat,1) ? -sort(abs.(@view Umat[:,j]), rev=true)[k] : 0.0, _TOPK) for j in 1:D]
    row_profile = [ntuple(k -> k <= size(Vmat,2) ? -sort(abs.(@view Vmat[j,:]), rev=true)[k] : 0.0, _TOPK) for j in 1:D]
    perm = sort(collect(1:D); by = j -> (
        -sv_rep[j], -col_fp[j], -row_fp[j], col_profile[j], row_profile[j],
    ))
    Umat, Vmat, svs = Umat[:, perm], Vmat[perm, :], svs[perm]
    # ── 4.5. Degenerate-block canonicalization ────────────────────────────────
    # Within each group of equal singular values, LAPACK can produce an arbitrary
    # unitary rotation of the degenerate subspace. The fingerprint-based sort
    # above cannot resolve this — both runs get different rotated bases whose
    # fingerprints differ, so the sort order within the block still differs.
    # Replace the basis of each degenerate block with the deterministic QR-based
    # basis from canonicalize_degenerate_block, then rotate Vmat rows consistently.
    let k = 1
        while k <= D
            g_end = k
            while g_end < D && abs(svs[g_end + 1] - svs[k]) <= group_tol
                g_end += 1
            end
            if g_end > k && svs[k] >= null_tol   # degenerate, non-null block
                B = copy(Umat[:, k:g_end])
                # Q, U = canonicalize_degenerate_block_fixed(B; atol=atol)
                # Umat[:, k:g_end]  = Q
                # Vmat[k:g_end, :] = U * copy(Vmat[k:g_end, :])
                Q = canonicalize_degenerate_block(B; atol=atol)
                W = Q' * B                        # unitary rotation within subspace
                Umat[:, k:g_end] = Q
                Vmat[k:g_end, :] = W * Vmat[k:g_end, :]
            end
            k = g_end + 1
        end
    end
    # ── 5. Canonical phase per channel (weighted-sum) ──────────────────────────
    for j in 1:D
        if svs[j] < null_tol
            Umat[:, j] .= zero(eltype(Umat))
            Vmat[j, :] .= zero(eltype(Vmat))
            continue
        end
        col     = @view Umat[:, j]
        col_abs = abs.(col)
        col_max = isempty(col_abs) ? zero(real(eltype(col))) : maximum(col_abs)
        if col_max > 10 * atol
            z = zero(eltype(col))
            @inbounds for i in eachindex(col)
                ai = col_abs[i]; ai > atol && (z += ai * col[i])
            end
            if abs(z) > atol
                ph = conj(z) / abs(z)
                Umat[:, j] .*= ph
                Vmat[j, :] .*= conj(ph)
            end
        end
    end
    # ── 6. Reconstruct ITensors ────────────────────────────────────────────────
    # Create a fresh bond index with the canonical ordering.
    u_new = ITensors.Index(D, ITensors.tags(u_idx))
    if ortho == "left"
        # L = U_canonical (isometry), R = diag(S) * V†_canonical (singular tensor)
        L_mat = ITensor(Umat, cU, u_new)
        L_it  = L_mat * dag(CU)                        # unfuse physical inds
        SV    = Diagonal(svs) * Vmat                   # D × m
        R_mat = ITensor(SV, u_new, cV)
        R_it  = R_mat * dag(CV)                        # unfuse physical inds
    else   # ortho == "right"
        # L = U * diag(S) (singular tensor), R = V†_canonical (isometry)
        US    = Umat * Diagonal(svs)                   # n × D
        L_mat = ITensor(US, cU, u_new)
        L_it  = L_mat * dag(CU)
        R_mat = ITensor(Vmat, u_new, cV)
        R_it  = R_mat * dag(CV)
    end
    if debug
        println("stable_factorize: D=$D  svs=$(round.(svs, sigdigits=4))")
    end
    return L_it, R_it, spec
end

function stable_argmax(v; atol=1e-14)
    m = maximum(v)
    for i in eachindex(v)
        if abs(v[i] - m) <= atol
            return i
        end
    end
    return argmax(v)
end


# Deterministic QR-based basis choice for a degenerate subspace.
# B is n × m with orthonormal columns spanning the subspace.
function canonicalize_degenerate_block(B::AbstractMatrix; atol=1e-12)
    n, m = size(B)
    m == 0 && return copy(B)
    # Deterministic row ordering: largest row norm first.
    # Tie-break by sorted absolute-value profile of the row (not by row index,
    # which is LAPACK-order-dependent and can differ between sparse/dense runs).
    rowinfo = [(norm(@view B[i, :]), Tuple(-sort(abs.(@view B[i,:]), rev=true)), i) for i in 1:n]
    perm = [x[3] for x in sort(rowinfo; by = x -> (-x[1], x[2], x[3]))]
    Bp = B[perm, :]
    # QR gives a deterministic orthonormal basis once row order is fixed.
    F = qr(Bp)
    Qp = Matrix(F.Q)[:, 1:m]
    R  = Matrix(F.R)[1:m, 1:m]
    # Fix column phases/signs by making diag(R) positive-real where possible.
    for j in 1:m
        d = R[j, j]
        if abs(d) > atol
            phase = conj(d) / abs(d)
            Qp[:, j] .*= phase
        end
    end
    # Undo the row permutation.
    Q = similar(B, size(B))
    Q[perm, :] = Qp
    return Q
end

function reshape_split_for_canonicalization(Tiso::ITensor, Tsing::ITensor, l::Index)
    iso_inds  = collect(inds(Tiso))
    sing_inds = collect(inds(Tsing))
    l_pos_iso  = findfirst(i -> i == l || i == dag(l), iso_inds)
    l_pos_sing = findfirst(i -> i == l || i == dag(l), sing_inds)
    if isnothing(l_pos_iso) || isnothing(l_pos_sing)
        return nothing
    end
    iso_arr  = Array(Tiso,  iso_inds...)
    sing_arr = Array(Tsing, sing_inds...)
    D = dim(l)
    other_iso  = [i for i in 1:ndims(iso_arr)  if i != l_pos_iso]
    other_sing = [i for i in 1:ndims(sing_arr) if i != l_pos_sing]
    iso_perm  = [other_iso; l_pos_iso]
    sing_perm = [l_pos_sing; other_sing]
    iso_mat  = reshape(permutedims(iso_arr,  iso_perm), :, D)
    sing_mat = reshape(permutedims(sing_arr, sing_perm), D, :)
    return (
        iso_inds = iso_inds,
        sing_inds = sing_inds,
        iso_perm = iso_perm,
        sing_perm = sing_perm,
        iso_arr_size = size(iso_arr),
        sing_arr_size = size(sing_arr),
        iso_mat = iso_mat,
        sing_mat = sing_mat,
    )
end

function rebuild_split_from_mats(
    iso_mat,
    sing_mat,
    info,
    l::Index,
    Tiso_proto::ITensor,
    Tsing_proto::ITensor,
)
    iso_sizes_perm  = [info.iso_arr_size[i]  for i in info.iso_perm]
    sing_sizes_perm = [info.sing_arr_size[i] for i in info.sing_perm]
    iso_arr_perm  = reshape(iso_mat,  iso_sizes_perm...)
    sing_arr_perm = reshape(sing_mat, sing_sizes_perm...)
    iso_arr  = permutedims(iso_arr_perm,  invperm(info.iso_perm))
    sing_arr = permutedims(sing_arr_perm, invperm(info.sing_perm))
    Tiso_new  = ITensor(iso_arr,  info.iso_inds...)
    Tsing_new = ITensor(sing_arr, info.sing_inds...)
    return Tiso_new, Tsing_new
end

function canonicalize_phase_for_channel(col, row; atol=1e-12)
    col_abs = abs.(col)
    col_max = isempty(col_abs) ? zero(real(eltype(col))) : maximum(col_abs)

    # Weighted-sum phase: z = Σ_i |col[i]| * col[i].
    # This aggregates ALL significant elements instead of picking one by argmax.
    # A sign-flipped column produces z → -z → phase = -1, restoring it.
    # Small perturbations (O(ε)) cause only O(ε) changes in z, so phase is stable
    # even when the dominant entry shifts sign between runs.
    if col_max > 10 * atol
        z = zero(eltype(col))
        @inbounds for i in eachindex(col)
            ai = col_abs[i]
            ai > atol && (z += ai * col[i])
        end
        if abs(z) > atol
            return conj(z) / abs(z)
        end
    end

    # Fallback: weighted-sum phase over sing row.
    row_abs = abs.(row)
    row_max = isempty(row_abs) ? zero(real(eltype(row))) : maximum(row_abs)
    if row_max > 10 * atol
        z = zero(eltype(row))
        @inbounds for i in eachindex(row)
            ai = row_abs[i]
            ai > atol && (z += ai * row[i])
        end
        if abs(z) > atol
            return z / abs(z)
        end
    end

    return one(eltype(col))
end

function reorder_channels_deterministically!(iso_mat, sing_mat, svs; atol=1e-12, rtol=1e-10)
    D = length(svs)
    D == 0 && return collect(1:D)
    sv_scale = max(maximum(svs), atol)
    group_tol = max(atol, rtol * sv_scale)

    # Snap each channel's SV to the first (largest) member of its degenerate group.
    # Without this, two channels with SVs differing by ~1e-14 (within group_tol)
    # can swap sort order when phi differs by ~1e-14 across runs, causing the
    # degenerate-block QR to receive columns in different order and produce a
    # different canonical basis.
    sv_rep = copy(svs)
    for k in 1:D
        for j in k+1:D
            if abs(svs[j] - sv_rep[k]) <= group_tol
                sv_rep[j] = sv_rep[k]
            end
        end
    end

    # Compute a stable fingerprint per channel.
    # Primary scalar: L3 norm (sign-free, smooth under perturbations).
    # Tie-break: sorted absolute-value profile of the column (top-K entries).
    # This makes sorting purely content-based and independent of the order in
    # which LAPACK/KrylovKit returns degenerate channels — which can differ
    # between sparse and dense arithmetic paths for O(ε) different inputs.
    _TOPK = 8   # how many sorted magnitudes to use as tie-break
    col_fp = [begin
        col = @view iso_mat[:, j]
        isempty(col) ? 0.0 : (sum(abs(x)^3 for x in col))^(1/3)
    end for j in 1:D]
    row_fp = [begin
        row = @view sing_mat[j, :]
        isempty(row) ? 0.0 : (sum(abs(x)^3 for x in row))^(1/3)
    end for j in 1:D]
    # Sorted magnitude profiles for stable tie-breaking among near-identical channels.
    col_profile = [begin
        col = @view iso_mat[:, j]
        isempty(col) ? ntuple(_ -> 0.0, _TOPK) :
            ntuple(k -> k <= length(col) ? -sort(abs.(col), rev=true)[k] : 0.0, _TOPK)
    end for j in 1:D]
    row_profile = [begin
        row = @view sing_mat[j, :]
        isempty(row) ? ntuple(_ -> 0.0, _TOPK) :
            ntuple(k -> k <= length(row) ? -sort(abs.(row), rev=true)[k] : 0.0, _TOPK)
    end for j in 1:D]

    perm = sort(collect(1:D); by = j -> (
        -sv_rep[j],        # primary: descending SV (grouped)
        -col_fp[j],        # secondary: descending col L3 fingerprint (sign-free)
        -row_fp[j],        # tertiary: descending row L3 fingerprint
        col_profile[j],    # quaternary: sorted col magnitude profile (content-based)
        row_profile[j],    # quinary: sorted row magnitude profile
    ))

    iso_mat[:, :] = iso_mat[:, perm]
    sing_mat[:, :] = sing_mat[perm, :]
    svs[:] = svs[perm]
    return perm
end

function canonicalize_degenerate_block_fixed(
    B::AbstractMatrix; atol = 1e-12, max_attempts = 3,
)
    n, m = size(B)
    m == 0 && return copy(B), Matrix{eltype(B)}(I, 0, 0)
    if m == 1
        col = copy(B[:, 1])
        ph  = _wsum_phase(col; atol = atol)
        col .*= ph
        return reshape(col, n, 1), reshape([conj(ph)], 1, 1)
    end

    T = real(eltype(B))
    for attempt in 1:max_attempts
        w = _make_weight(T, n, attempt)
        M = B' * (w .* B); M = (M + M') / 2

        F  = eigen(ITensors.Hermitian(M))
        ev = F.values
        ev_scale = max(maximum(abs, ev), one(T))
        gap = m > 1 ? minimum(ev[j] - ev[j-1] for j in 2:m) : ev_scale

        if gap > 1e-10 * ev_scale || attempt == max_attempts
            V = F.vectors
            Q = B * V
            phases = Vector{eltype(B)}(undef, m)
            for j in 1:m
                phases[j] = _wsum_phase(@view(Q[:, j]); atol = atol)
                Q[:, j] .*= phases[j]
            end
            U = Diagonal(conj.(phases)) * V'
            return Matrix(Q), Matrix(U)
        end
    end
end

function _wsum_phase(col; atol = 1e-12)
    z = zero(eltype(col))
    @inbounds for x in col
        ax = abs(x)
        ax > atol && (z += ax * x)
    end
    return abs(z) > atol ? conj(z) / abs(z) : one(eltype(col))
end

function _make_weight(::Type{T}, n, attempt) where {T}
    if attempt == 1
        return T[T(i) + T(i)^2 / T(n + 1) for i in 1:n]
    elseif attempt == 2
        return T[sin(T(i) * T(0.7) + T(0.3)) + T(i) / T(n + 1) for i in 1:n]
    else
        return T[T(i)^T(1.3) + cos(T(i) * T(0.3)) for i in 1:n]
    end
end


function canonicalize_split(
    L::ITensor,
    R::ITensor;
    ortho="left",
    atol=1e-12,
    rtol=1e-10,
    null_atol=1e-10,
    null_rtol=1e-6,
    debug=false,
)
    # Only do the canonicalization for dense ITensors.
    if ITensors.has_external_storage(L) || ITensors.has_external_storage(R)
        return L, R
    end
    l = commonind(L, R)
    isnothing(l) && return L, R
    # Determine which tensor is the isometry side.
    iso_is_left = (ortho == "left")
    Tiso, Tsing = iso_is_left ? (L, R) : (R, L)
    info = reshape_split_for_canonicalization(Tiso, Tsing, l)
    isnothing(info) && return L, R
    iso_mat  = copy(info.iso_mat)
    sing_mat = copy(info.sing_mat)
    D = size(iso_mat, 2)
    if D == 0
        return L, R
    end
    # Estimate singular values from sing_mat row norms.
    svs = [norm(@view sing_mat[k, :]) for k in 1:D]
    sv_scale = max(maximum(svs), atol)
    # More aggressive null handling than before.
    group_tol = max(atol, rtol * sv_scale)
    null_tol  = max(null_atol, null_rtol * sv_scale)
    if debug
        println("Singular values before reorder = ", svs)
        println("group_tol = ", group_tol, "  null_tol = ", null_tol)
    end
    # Reorder channels deterministically before any grouping.
    perm = reorder_channels_deterministically!(iso_mat, sing_mat, svs; atol=atol, rtol=rtol)
    if debug
        println("channel perm = ", perm)
        println("Singular values after reorder = ", svs)
    end
    processed = falses(D)
    for k in 1:D
        processed[k] && continue
        group = [j for j in k:D if abs(svs[j] - svs[k]) <= group_tol]
        for j in group
            processed[j] = true
        end
        if debug
            println("\nProcessing group starting at k=$k")
            println("  group = ", group)
            println("  group svs = ", svs[group])
        end
        # Numerically null block: zero it on both sides.
        if svs[k] < null_tol
            if debug
                println("  -> null block, zeroing")
            end
            iso_mat[:, group] .= zero(eltype(iso_mat))
            sing_mat[group, :] .= zero(eltype(sing_mat))
            continue
        end
        if length(group) == 1
            # Nondegenerate: fix sign/phase robustly.
            j = group[1]
            col = @view iso_mat[:, j]
            row = @view sing_mat[j, :]
            phase = canonicalize_phase_for_channel(col, row; atol=atol)
            if debug
                col_abs = abs.(col)
                row_abs = abs.(row)
                ic = isempty(col_abs) ? 1 : stable_argmax(col_abs; atol=atol)
                ir = isempty(row_abs) ? 1 : stable_argmax(row_abs; atol=atol)
                col_wsum = isempty(col) ? zero(eltype(col)) : sum(col_abs[i] * col[i] for i in eachindex(col) if col_abs[i] > atol; init=zero(eltype(col)))
                println("  -> nondegenerate channel $j")
                println("     sv = ", svs[j])
                println("     col max @ ", ic, " = ", col[ic], "  col_wsum = ", col_wsum)
                println("     row max @ ", ir, " = ", row[ir])
                println("     phase = ", phase)
            end
            iso_mat[:, j] .*= phase
            sing_mat[j, :] .*= conj(phase)
        else
            # Degenerate block: replace basis by deterministic QR-based basis.
            B = iso_mat[:, group]
            Q, U = canonicalize_degenerate_block_fixed(B; atol=atol)
            # Rotate sing_mat consistently so product stays unchanged.
            iso_mat[:, group]  = Q
            sing_mat[group, :] = U * copy(sing_mat[group, :])
            # W = Q' * B
            # iso_mat[:, group]  = Q
            # sing_mat[group, :] = W * sing_mat[group, :]
            if debug
                println("  -> degenerate block")
                println("     size = ", length(group))
                println("     ‖B - Q‖ = ", norm(B - Q))
                println("     ‖W'W - I‖ = ", norm(W' * W - I))
            end
            # Extra per-channel phase fix inside the canonicalized block.
            for j in group
                col = @view iso_mat[:, j]
                row = @view sing_mat[j, :]
                phase = canonicalize_phase_for_channel(col, row; atol=atol)
                iso_mat[:, j] .*= phase
                sing_mat[j, :] .*= conj(phase)
            end
        end
    end
    Tiso_new, Tsing_new = rebuild_split_from_mats(iso_mat, sing_mat, info, l, Tiso, Tsing)
    if iso_is_left
        return Tiso_new, Tsing_new
    else
        return Tsing_new, Tiso_new
    end
end

# ------------------------------------------------------------------
# Replacement function
# ------------------------------------------------------------------

function replacebond!(
    M::MPS,
    b::Int,
    phi::ITensor;
    normalize = nothing,
    swapsites = nothing,
    ortho = nothing,
    # Decomposition kwargs
    which_decomp = nothing,
    mindim = nothing,
    maxdim = nothing,
    cutoff = nothing,
    eigen_perturbation = nothing,
    # svd kwargs
    svd_alg = nothing,
    use_absolute_cutoff = nothing,
    use_relative_cutoff = nothing,
    min_blockdim = nothing,
    # canonicalization tolerances
    canonical_atol = 1e-12,
    canonical_rtol = 1e-10,
    debug = false
)
    normalize = NDTensors.replace_nothing(normalize, false)
    swapsites = NDTensors.replace_nothing(swapsites, false)
    ortho = NDTensors.replace_nothing(ortho, "left")
    indsMb = inds(M[b])
    if swapsites
        sb = siteind(M, b)
        sbp1 = siteind(M, b + 1)
        indsMb = replaceind(indsMb, sb, sbp1)
    end

    # 1) Canonicalize phi before factorization. Skip for sparse phi:
    # canonicalize_phi_inds calls permute() which requires staying within
    # the sparse-prefix / dense-suffix split — but the canonical leg ordering
    # generally crosses that boundary. Sparse phi gets its own canonical
    # ordering inside itensor_blocksparse_svd via reorder_invariant.
    if !ITensors.has_external_storage(phi)
        phi, indsMb = canonicalize_phi_inds(phi, indsMb)
    end
    if debug
        println("check the inds of phi: ", inds(phi), " -- ", indsMb)
    end

    # 2) Factorize phi → L, R.
    # stable_factorize works on raw U, S, Vt (SVD only — exact singular values,
    # single-pass canonical phase/ordering).  Fall back to the old factorize path
    # for QR or eigen decompositions that don't produce singular values.
    use_stable = isnothing(which_decomp) || which_decomp == "svd"
    # Capture the original M[b]↔M[b+1] sparse-link dim BEFORE the factorization
    # rewrites those tensors. At a boundary the SVD's natural binning would
    # collapse the new bond's sparse dim to 1; passing this hint forces the
    # new bond to keep the same sparse dim as the old one (extra slots empty).
    target_link_sparse_dim = SparseBackends.shared_bond_sparse_dim(M[b], M[b + 1])
    if use_stable
        L, R, spec = stable_factorize(
            phi,
            indsMb;
            level = b,
            ortho,
            mindim,
            maxdim,
            cutoff,
            svd_alg,
            use_absolute_cutoff,
            use_relative_cutoff,
            min_blockdim,
            target_link_sparse_dim,
            M_b  = M[b],
            M_b1 = M[b + 1],
            tags      = tags(linkind(M, b)),
            atol      = canonical_atol,
            rtol      = canonical_rtol,
            debug     = debug,
        )
    else
        # QR / eigen path: no singular values available, use old pipeline.
        L, R, spec = factorize(
            phi, indsMb;
            mindim, maxdim, cutoff, ortho,
            which_decomp, eigen_perturbation, svd_alg,
            tags                = tags(linkind(M, b)),
            use_absolute_cutoff, use_relative_cutoff, min_blockdim,
        )
        L, R = reorder_split_tensors(L, R)
        L, R = canonicalize_split(L, R; ortho, atol=canonical_atol, rtol=canonical_rtol, debug)
        L, R = reorder_split_tensors(L, R)
    end

    # println(ITensors.has_external_storage(L) ? "L has external storage." : "L is dense.")
    # println(ITensors.has_external_storage(R) ? "R has external storage." : "R is dense.")

    if debug
        println("L, R inds are $(inds(L)), $(inds(R)), $spec")
    end
    # 3) Put shared bond in a fixed place.
    # Skip permutation for external storage (BlockSparse) tensors: index ordering is
    # bookkeeping-only for ITensor ops, and permuting would break the prefix/dense layout.
    if !ITensors.has_external_storage(L) && !ITensors.has_external_storage(R)
        L, R = reorder_split_tensors(L, R)
    end


    # println(ITensors.has_external_storage(L) ? "After reorder, L has external storage." : "After reorder, L is dense.")
    # println(ITensors.has_external_storage(R) ? "After reorder, R has external storage." : "After reorder, R is dense.")

    if debug
        println("L tensor ", L)
        println("R tensor ", R)
    end
    M[b]     = L
    M[b + 1] = R
    # 6) Restore orthogonality metadata and normalize.
    if ortho == "left"
        leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
        rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
        normalize && (M[b + 1] ./= norm(M[b + 1]))
    elseif ortho == "right"
        leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
        rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
        normalize && (M[b] ./= norm(M[b]))
    else
        error("In replacebond!, got ortho = $ortho, only supports `left` and `right`.")
    end
    # println(ITensors.has_external_storage(M[b]) ? "After replacebond!, M[$b] has external storage." : "After replacebond!, M[$b] is dense.")
    # println(ITensors.has_external_storage(M[b+1]) ? "After replacebond!, M[$b+1] has external storage." : "After replacebond!, M[$(b+1)] is dense.")
    return spec
end

# """
#     replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)

# Factorize the ITensor `phi` and replace the ITensors
# `b` and `b+1` of MPS `M` with the factors. Choose
# the orthogonality with `ortho="left"/"right"`.
# """
# function replacebond!(
#         M::MPS,
#         b::Int,
#         phi::ITensor;
#         normalize = nothing,
#         swapsites = nothing,
#         ortho = nothing,
#         # Decomposition kwargs
#         which_decomp = nothing,
#         mindim = nothing,
#         maxdim = nothing,
#         cutoff = nothing,
#         eigen_perturbation = nothing,
#         # svd kwargs
#         svd_alg = nothing,
#         use_absolute_cutoff = nothing,
#         use_relative_cutoff = nothing,
#         min_blockdim = nothing,
#     )
#     normalize = NDTensors.replace_nothing(normalize, false)
#     swapsites = NDTensors.replace_nothing(swapsites, false)
#     ortho = NDTensors.replace_nothing(ortho, "left")

#     indsMb = inds(M[b])
#     if swapsites
#         sb = siteind(M, b)
#         sbp1 = siteind(M, b + 1)
#         indsMb = replaceind(indsMb, sb, sbp1)
#     end
#     # # Sort phi's indices to a canonical order before factorize.
#     # # phi is formed by contracting psi[b]*psi[b+1], where the new link index from
#     # # the previous replacebond! has a freshly-created ID that differs between runs.
#     # # This causes phi's internal index ordering to differ, so factorize sees a
#     # # differently-permuted matrix and LAPACK produces different singular vectors.
#     # # Sorting by (tag-string, dim) makes the matrix layout identical across runs.
#     # phi = permute(phi, sort(collect(inds(phi)); by = i -> (string(tags(i)), dim(i)))...)
#     # indsMb = filter(i -> i ∈ inds(phi), collect(indsMb))
#     L, R, spec = factorize(
#         phi,
#         indsMb;
#         mindim,
#         maxdim,
#         cutoff,
#         ortho,
#         which_decomp,
#         eigen_perturbation,
#         svd_alg,
#         tags = tags(linkind(M, b)),
#         use_absolute_cutoff,
#         use_relative_cutoff,
#         min_blockdim,
#     )
#     # Store L and R with canonically sorted indices so that subsequent contractions
#     # (e.g. phi = psi[b]*psi[b+1]) produce consistently-ordered tensors across runs.
#     # The new link index gets a fresh ID each run, which otherwise causes its position
#     # in L/R to vary, leading to different phi index orderings and thus different SVDs.
#     canon_order(T) = permute(T, sort(collect(inds(T)); by = i -> (string(tags(i)), dim(i)))...)
#     M[b]     = canon_order(L)
#     M[b + 1] = canon_order(R)

#     # Canonicalize SVD gauge: fixes both ±1 sign freedom (non-degenerate singular
#     # values) and unitary rotation freedom (degenerate singular values) so that
#     # results are reproducible across runs with different-but-equivalent H formats.
#     #
#     # Non-degenerate: force dominant element of each singular vector to be positive-real.
#     # Degenerate block: apply QR with column pivoting to the block of singular vectors,
#     # then fix signs on the R diagonal. This gives a canonical basis for the subspace.
#     if !ITensors.has_external_storage(L) && !ITensors.has_external_storage(R)
#         l = commonind(L, R)
#         iso_slot, sing_slot = ortho == "left" ? (b, b + 1) : (b + 1, b)
#         iso_inds  = inds(M[iso_slot])
#         sing_inds = inds(M[sing_slot])
#         l_pos_iso  = findfirst(i -> i == l || i == dag(l), collect(iso_inds))
#         l_pos_sing = findfirst(i -> i == l || i == dag(l), collect(sing_inds))
#         if !isnothing(l_pos_iso) && !isnothing(l_pos_sing)
#             iso_arr  = Array(M[iso_slot],  iso_inds...)
#             sing_arr = Array(M[sing_slot], sing_inds...)
#             D = dim(l)

#             # Reshape to matrices: iso → (n, D),  sing → (D, p)
#             other_iso  = [i for i in 1:ndims(iso_arr)  if i != l_pos_iso]
#             other_sing = [i for i in 1:ndims(sing_arr) if i != l_pos_sing]
#             iso_perm  = [other_iso;  l_pos_iso]
#             sing_perm = [l_pos_sing; other_sing]
#             iso_mat  = reshape(permutedims(iso_arr,  iso_perm),  :, D)
#             sing_mat = reshape(permutedims(sing_arr, sing_perm), D, :)

#             # Singular values = row norms of sing_mat
#             svs = [norm(sing_mat[k, :]) for k in 1:D]
#             sv_scale = max(maximum(svs), 1e-300)

#             println("  [canon b=$b ortho=$ortho] svs=$(round.(svs; sigdigits=6))")

#             # Identify and canonicalize each degenerate group
#             processed = falses(D)
#             for k in 1:D
#                 processed[k] && continue
#                 group = [j for j in k:D if abs(svs[j] - svs[k]) ≤ 1e-10 * sv_scale]
#                 for j in group; processed[j] = true; end

#                 if length(group) > 1
#                     println("  [canon b=$b] degenerate SV group size=$(length(group)), sv≈$(round(svs[k], sigdigits=4))")
#                 end
#                 # Near-zero singular values: their iso columns span a numerical null
#                 # space. Any orthonormal basis is valid (physical contribution ≈ sv ≈ 0),
#                 # but different LAPACK runs produce different arbitrary bases.
#                 # Zero out those columns so both runs agree exactly, without affecting
#                 # the physical state beyond the sv ≈ 0 contribution.
#                 # Use max(relative, absolute) threshold: when ALL SVs are near-zero,
#                 # sv_scale is also near-zero so 1e-10*sv_scale is uselessly small —
#                 # the absolute floor 1e-12 catches that case.
#                 if svs[k] < max(1e-10 * sv_scale, 1e-12)
#                     for j in group
#                         iso_mat[:, j] .= zero(eltype(iso_mat))
#                         # sing_mat rows already near-zero; leave them as-is
#                     end
#                     continue
#                 end
#                 if length(group) == 1
#                     # Non-degenerate: phase fix. Multiply column by conj(dominant)/|dominant|
#                     # so dominant element becomes real and positive. For real tensors this
#                     # reduces to the sign flip; for complex tensors it fixes arbitrary phase.
#                     col = view(iso_mat, :, group[1])
#                     i_star = argmax(abs.(col))
#                     dom = col[i_star]
#                     absdom = abs(dom)
#                     if absdom > 1e-15
#                         phase = conj(dom) / absdom   # e^{-iθ}: makes dom → |dom| > 0
#                         if abs(real(phase) - 1) > 1e-12 || abs(imag(phase)) > 1e-12
#                             iso_mat[:, group[1]] .*= phase
#                             sing_mat[group[1], :] .*= conj(phase)
#                         end
#                     end
#                 else
#                     # Degenerate subspace: find a canonical basis via row-norm-guided
#                     # Gram-Schmidt on the iso columns.
#                     #
#                     # Key invariance: for B1 = B2*W (same subspace, different gauge),
#                     # row norms ‖B[i,:]‖ are identical (right-unitary invariant), so
#                     # argmax always picks the same pivot row in both DMRG runs.
#                     # The projected vector p = B_rem * B_rem[i*,:]' is also invariant,
#                     # giving a canonical orthonormal basis Q_c for the subspace.
#                     # This avoids the P[:,1:m] + QR column-pivot instability that caused
#                     # different pivot orderings when phi differs by ~1e-14.
#                     B = iso_mat[:, group]   # n_iso × m, orthonormal columns
#                     m_deg = length(group)
#                     n_iso = size(B, 1)
#                     Q_c = zeros(eltype(B), n_iso, m_deg)
#                     B_rem = copy(B)
#                     for t in 1:m_deg
#                         row_norms = [norm(B_rem[i, :]) for i in 1:n_iso]
#                         i_star = argmax(row_norms)
#                         # For complex B: projector P = B·B†, so P·e_{i*} = B·conj(B[i*,:])ᵀ
#                         # (need conjugate for Hermitian projector, reduces to plain transpose for real)
#                         p = B_rem * conj(vec(B_rem[i_star, :]))  # n_iso-vector
#                         p_norm = norm(p)
#                         p_norm < 1e-14 && break  # degenerate residual, stop early
#                         p ./= p_norm
#                         Q_c[:, t] = p
#                         B_rem = B_rem - p * (p' * B_rem)
#                     end
#                     # Phase fix: force dominant element of each canonical vector real and positive.
#                     # For real tensors this is a sign flip; for complex it rotates out the phase.
#                     for t in 1:m_deg
#                         col = view(Q_c, :, t)
#                         i_star = argmax(abs.(col))
#                         dom = col[i_star]
#                         absdom = abs(dom)
#                         if absdom > 1e-15
#                             phase = conj(dom) / absdom
#                             if abs(real(phase) - 1) > 1e-12 || abs(imag(phase)) > 1e-12
#                                 Q_c[:, t] .*= phase
#                             end
#                         end
#                     end
#                     # Find rotation W from B to Q_c: B = Q_c · W† → W = Q_c' · B
#                     W = Q_c' * B            # m_deg × m_deg unitary
#                     # Product preservation: Q_c · (W · sing_rows) = B · sing_rows
#                     sing_rows = sing_mat[group, :]
#                     iso_mat[:, group]  = Q_c
#                     sing_mat[group, :] = W * sing_rows
#                 end
#             end

#             # Reshape back and store
#             iso_sizes  = [size(iso_arr,  i) for i in iso_perm]
#             sing_sizes = [size(sing_arr, i) for i in sing_perm]
#             M[iso_slot]  = ITensor(permutedims(reshape(iso_mat,  iso_sizes...),  invperm(iso_perm)),  iso_inds...)
#             M[sing_slot] = ITensor(permutedims(reshape(sing_mat, sing_sizes...), invperm(sing_perm)), sing_inds...)
#         end
#     end

#     if ortho == "left"
#         leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
#         rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
#         normalize && (M[b + 1] ./= norm(M[b + 1]))
#     elseif ortho == "right"
#         leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
#         rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
#         normalize && (M[b] ./= norm(M[b]))
#     else
#         error(
#             "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
#         )
#     end
#     return spec
# end

"""
    replacebond(M::MPS, b::Int, phi::ITensor; kwargs...)

Like `replacebond!`, but returns the new MPS.
"""
function replacebond(M0::MPS, b::Int, phi::ITensor; kwargs...)
    M = copy(M0)
    replacebond!(M, b, phi; kwargs...)
    return M
end

"""
    replacebond_sparse!(M, b, phi; ortho, maxdim, mindim, cutoff, normalize)

Alternative `replacebond!` for an MPS whose tensors carry BlockSparse external
storage. Uses `itensor_blocksparse_svd_channel_aware` which factorizes phi back
into L, R that EXACTLY inherit M[b]'s and M[b+1]'s block-key sets — i.e. the
factorization "lives in" the same restricted block subspace defined by the
P_sparse channel structure. Cross-channel orthogonality follows from disjoint
lk-sets per channel, so the isometric factor is globally isometric.
"""
function replacebond_sparse!(
    M::MPS,
    b::Int,
    phi::ITensor;
    normalize  = false,
    ortho      = "left",
    maxdim     = nothing,
    mindim     = nothing,
    cutoff     = nothing,
    kwargs...
)
    @assert ITensors.has_external_storage(M[b])
    @assert ITensors.has_external_storage(M[b + 1])
    @assert ITensors.has_external_storage(phi) "phi must be block-sparse for replacebond_sparse!"

    # Unified SVD: route through `itensor_blocksparse_svd_channel_aware` (same
    # kernel that orthogonalize! and 3-arg replacebond!/stable_factorize use).
    # Passing M[b], M[b+1] as templates lets the channel-aware path inherit
    # their block-key sets and apply the per-channel mult cap (mult_cap =
    # fld(maxdim, n_active_chan)) so total effective bond ≤ maxdim. Strict iso
    # enforced when BMF_ISO_PATH=1 (default in tests).
    iso_strict = get(ENV, "BMF_ISO_PATH", "0") == "1"
    factor_fn = (get(ENV, "SB_USE_QR", "0") == "1") ?
        SparseBackends.itensor_blocksparse_qr_channel_aware :
        SparseBackends.itensor_blocksparse_svd_channel_aware
    L, R, spec = factor_fn(
        phi, M[b], M[b + 1];
        ortho  = ortho,
        maxdim = something(maxdim, typemax(Int)),
        mindim = something(mindim, 1),
        cutoff = Float64(something(cutoff, 0.0)),
        relax_iso_cap = !iso_strict,
    )

    M[b]     = L
    M[b + 1] = R

    if ortho == "left"
        leftlim(M)  == b - 1 && setleftlim!(M,  leftlim(M) + 1)
        rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
        normalize && (M[b + 1] ./= norm(M[b + 1]))
    elseif ortho == "right"
        leftlim(M)  == b     && setleftlim!(M,  leftlim(M) - 1)
        rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
        normalize && (M[b] ./= norm(M[b]))
    else
        error("replacebond_sparse!: unknown ortho=$ortho")
    end

    return spec
end

# Allows overloading `replacebond!` based on the projected MPO type.
# Routes sparse psi through the channel-aware factorization.
function replacebond!(PH, M::MPS, b::Int, phi::ITensor; kwargs...)
    if ITensors.has_external_storage(M[b]) && ITensors.has_external_storage(M[b + 1]) &&
       ITensors.has_external_storage(phi)
        return replacebond_sparse!(M, b, phi; kwargs...)
    end
    return replacebond!(M, b, phi; kwargs...)
end

"""
    sample!(m::MPS)

Given a normalized MPS m, returns a `Vector{Int}`
of `length(m)` corresponding to one sample
of the probability distribution defined by
squaring the components of the tensor
that the MPS represents. If the MPS does
not have an orthogonality center,
orthogonalize!(m,1) will be called before
computing the sample.
"""
function sample!(m::MPS)
    return sample!(Random.default_rng(), m)
end

function sample!(rng::AbstractRNG, m::MPS)
    orthogonalize!(m, 1)
    return sample(rng, m)
end

"""
    sample(m::MPS)

Given a normalized MPS m with `orthocenter(m)==1`,
returns a `Vector{Int}` of `length(m)`
corresponding to one sample of the
probability distribution defined by
squaring the components of the tensor
that the MPS represents
"""
function sample(m::MPS)
    return sample(Random.default_rng(), m)
end

function sample(rng::AbstractRNG, m::MPS)
    N = length(m)

    if orthocenter(m) != 1
        error("sample: MPS m must have orthocenter(m)==1")
    end
    if abs(1.0 - norm(m[1])) > 1.0e-8
        error("sample: MPS is not normalized, norm=$(norm(m[1]))")
    end

    ElT = scalartype(m)

    result = zeros(Int, N)
    A = m[1]

    for j in 1:N
        s = siteind(m, j)
        d = dim(s)
        # Compute the probability of each state
        # one-by-one and stop when the random
        # number r is below the total prob so far
        pdisc = zero(real(ElT))
        r = rand(rng)
        # Will need n,An, and pn below
        n = 1
        An = ITensor()
        pn = zero(real(ElT))
        while n <= d
            projn = ITensor(s)
            projn[s => n] = one(ElT)
            An = A * dag(adapt(datatype(A), projn))
            pn = real(scalar(dag(An) * An))
            pdisc += pn
            (r < pdisc) && break
            n += 1
        end
        result[j] = n
        if j < N
            A = m[j + 1] * An
            A *= (one(ElT) / sqrt(pn))
        end
    end
    return result
end

_op_prod(o1::AbstractString, o2::AbstractString) = "$o1 * $o2"
_op_prod(o1::Matrix{<:Number}, o2::Matrix{<:Number}) = o1 * o2

"""
    correlation_matrix(psi::MPS,
                       Op1::AbstractString,
                       Op2::AbstractString;
                       kwargs...)

    correlation_matrix(psi::MPS,
                       Op1::Matrix{<:Number},
                       Op2::Matrix{<:Number};
                       kwargs...)

Given an MPS psi and two strings denoting
operators (as recognized by the `op` function),
computes the two-point correlation function matrix
C[i,j] = <psi| Op1i Op2j |psi>
using efficient MPS techniques. Returns the matrix C.

# Optional Keyword Arguments

  - `sites = 1:length(psi)`: compute correlations only
     for sites in the given range
  - `ishermitian = false` : if `false`, force independent calculations of the
     matrix elements above and below the diagonal, while if `true` assume they are complex conjugates.

For a correlation matrix of size NxN and an MPS of typical
bond dimension m, the scaling of this algorithm is N^2*m^3.

# Examples

```julia
N = 30
m = 4

s = siteinds("S=1/2", N)
psi = random_mps(s; linkdims=m)
Czz = correlation_matrix(psi, "Sz", "Sz")
Czz = correlation_matrix(psi, [1/2 0; 0 -1/2], [1/2 0; 0 -1/2]) # same as above

s = siteinds("Electron", N; conserve_qns=true)
psi = random_mps(s, n -> isodd(n) ? "Up" : "Dn"; linkdims=m)
Cuu = correlation_matrix(psi, "Cdagup", "Cup"; sites=2:8)
```
"""
function correlation_matrix(
        psi::MPS, _Op1, _Op2; sites = 1:length(psi), site_range = nothing,
        ishermitian = nothing
    )
    if !isnothing(site_range)
        @warn "The `site_range` keyword arg. to `correlation_matrix` is deprecated: use the keyword `sites` instead"
        sites = site_range
    end
    if !(sites isa AbstractRange)
        sites = collect(sites)
    end

    start_site = first(sites)
    end_site = last(sites)

    N = length(psi)
    ElT = scalartype(psi)
    s = siteinds(psi)

    Op1 = _Op1 #make copies into which we can insert "F" string operators, and then restore.
    Op2 = _Op2
    onsiteOp = _op_prod(Op1, Op2)
    fermionic1 = has_fermion_string(Op1, s[start_site])
    fermionic2 = has_fermion_string(Op2, s[end_site])
    if fermionic1 != fermionic2
        error(
            "correlation_matrix: Mixed fermionic and bosonic operators are not supported yet."
        )
    end

    # Decide if we need to calculate a non-hermitian corr. matrix, which is roughly double the work.
    is_cm_hermitian = ishermitian
    if isnothing(is_cm_hermitian)
        # Assume correlation matrix is non-hermitian
        is_cm_hermitian = false
        O1 = op(Op1, s, start_site)
        O2 = op(Op2, s, start_site)
        O1 /= norm(O1)
        O2 /= norm(O2)
        #We need to decide if O1 ∝ O2 or O1 ∝ O2^dagger allowing for some round off errors.
        eps = 1.0e-10
        is_op_proportional = norm(O1 - O2) < eps
        is_op_hermitian = norm(O1 - dag(swapprime(O2, 0, 1))) < eps
        if is_op_proportional || is_op_hermitian
            is_cm_hermitian = true
        end
        # finally if they are both fermionic and proportional then the corr matrix will
        # be anti symmetric insterad of Hermitian. Handle things like <C_i*C_j>
        # at this point we know fermionic2=fermionic1, but we put them both in the if
        # to clarify the meaning of what we are doing.
        if is_op_proportional && fermionic1 && fermionic2
            is_cm_hermitian = false
        end
    end

    psi = orthogonalize(psi, start_site)
    norm2_psi = norm(psi[start_site])^2

    # Nb = size of block of correlation matrix
    Nb = length(sites)

    op1_start = op(_Op1, s[start_site])
    op2_start = op(_Op2, s[start_site])
    ElT1 = eltype(op1_start)
    ElT2 = eltype(op2_start)
    ElT′ = promote_type(ElT1, ElT2, ElT)
    C = zeros(ElT′, Nb, Nb)

    if start_site == 1
        L = ITensor(1.0)
    else
        lind = commonind(psi[start_site], psi[start_site - 1])
        L = delta(dag(lind), lind')
    end
    pL = start_site - 1

    for (ni, i) in enumerate(sites[1:(end - 1)])
        while pL < i - 1
            pL += 1
            sᵢ = siteind(psi, pL)
            L = (L * psi[pL]) * prime(dag(psi[pL]), !sᵢ)
        end

        Li = L * psi[i]

        # Get j == i diagonal correlations
        rind = commonind(psi[i], psi[i + 1])
        oᵢ = adapt(unspecify_type_parameters(datatype(Li)), op(onsiteOp, s, i))
        C[ni, ni] = ((Li * oᵢ) * prime(dag(psi[i]), !rind))[] / norm2_psi

        # Get j > i correlations
        if !using_auto_fermion() && fermionic2
            Op1 = "$Op1 * F"
        end

        oᵢ = adapt(unspecify_type_parameters(datatype(Li)), op(Op1, s, i))

        Li12 = (dag(psi[i])' * oᵢ) * Li
        pL12 = i

        for (n, j) in enumerate(sites[(ni + 1):end])
            nj = ni + n

            while pL12 < j - 1
                pL12 += 1
                if !using_auto_fermion() && fermionic2
                    dtype = unspecify_type_parameters(datatype(psi[pL12]))
                    oᵢ = adapt(dtype, op("F", s[pL12]))
                    Li12 *= (oᵢ * dag(psi[pL12])')
                else
                    sᵢ = siteind(psi, pL12)
                    Li12 *= prime(dag(psi[pL12]), !sᵢ)
                end
                Li12 *= psi[pL12]
            end

            lind = commonind(psi[j], Li12)
            Li12 *= psi[j]

            oⱼ = adapt(unspecify_type_parameters(datatype(Li12)), op(Op2, s, j))
            sⱼ = siteind(psi, j)
            val = (Li12 * oⱼ) * prime(dag(psi[j]), (sⱼ, lind))

            # XXX: This gives a different fermion sign with
            # ITensors.enable_auto_fermion()
            # val = prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li12)

            C[ni, nj] = scalar(val) / norm2_psi
            if is_cm_hermitian
                C[nj, ni] = conj(C[ni, nj])
            end

            pL12 += 1
            if !using_auto_fermion() && fermionic2
                dtype = unspecify_type_parameters(datatype(psi[pL12]))
                oᵢ = adapt(dtype, op("F", s[pL12]))
                Li12 *= (oᵢ * dag(psi[pL12])')
            else
                sᵢ = siteind(psi, pL12)
                Li12 *= prime(dag(psi[pL12]), !sᵢ)
            end
            @assert pL12 == j
        end #for j
        Op1 = _Op1 #"Restore Op1 with no Fs"

        if !is_cm_hermitian #If isHermitian=false the we must calculate the below diag elements explicitly.

            #  Get j < i correlations by swapping the operators
            if !using_auto_fermion() && fermionic1
                Op2 = "$Op2 * F"
            end
            oᵢ = adapt(unspecify_type_parameters(datatype(psi[i])), op(Op2, s, i))
            Li21 = (Li * oᵢ) * dag(psi[i])'
            pL21 = i
            if !using_auto_fermion() && fermionic1
                Li21 = -Li21 #Required because we swapped fermionic ops, instead of sweeping right to left.
            end

            for (n, j) in enumerate(sites[(ni + 1):end])
                nj = ni + n

                while pL21 < j - 1
                    pL21 += 1
                    if !using_auto_fermion() && fermionic1
                        dtype = unspecify_type_parameters(datatype(psi[pL21]))
                        oᵢ = adapt(dtype, op("F", s[pL21]))
                        Li21 *= oᵢ * dag(psi[pL21])'
                    else
                        sᵢ = siteind(psi, pL21)
                        Li21 *= prime(dag(si[pL21]), !sᵢ)
                    end
                    Li21 *= psi[pL21]
                end

                lind = commonind(psi[j], Li21)
                Li21 *= psi[j]

                oⱼ = adapt(unspecify_type_parameters(datatype(psi[j])), op(Op1, s, j))
                sⱼ = siteind(psi, j)
                val = (prime(dag(psi[j]), (sⱼ, lind)) * (oⱼ * Li21))[]
                C[nj, ni] = val / norm2_psi

                pL21 += 1
                if !using_auto_fermion() && fermionic1
                    dtype = unspecify_type_parameters(datatype(psi[pL21]))
                    oᵢ = adapt(dtype, op("F", s[pL21]))
                    Li21 *= (oᵢ * dag(psi[pL21])')
                else
                    sᵢ = siteind(psi, pL21)
                    Li21 *= prime(dag(psi[pL21]), !sᵢ)
                end
                @assert pL21 == j
            end #for j
            Op2 = _Op2 #"Restore Op2 with no Fs"
        end #if is_cm_hermitian

        pL += 1
        sᵢ = siteind(psi, i)
        L = Li * prime(dag(psi[i]), !sᵢ)
    end #for i

    # Get last diagonal element of C
    i = end_site
    while pL < i - 1
        pL += 1
        sᵢ = siteind(psi, pL)
        L = L * psi[pL] * prime(dag(psi[pL]), !sᵢ)
    end
    lind = commonind(psi[i], psi[i - 1])
    oᵢ = adapt(unspecify_type_parameters(datatype(psi[i])), op(onsiteOp, s, i))
    sᵢ = siteind(psi, i)
    val = (L * (oᵢ * psi[i]) * prime(dag(psi[i]), (sᵢ, lind)))[]
    C[Nb, Nb] = val / norm2_psi

    return C
end

"""
    expect(psi::MPS, op::AbstractString...; kwargs...)
    expect(psi::MPS, op::Matrix{<:Number}...; kwargs...)
    expect(psi::MPS, ops; kwargs...)

Given an MPS `psi` and a single operator name, returns
a vector of the expected value of the operator on
each site of the MPS.

If multiple operator names are provided, returns a tuple
of expectation value vectors.

If a container of operator names is provided, returns the
same type of container with names replaced by vectors
of expectation values.

# Optional Keyword Arguments

  - `sites = 1:length(psi)`: compute expected values only for sites in the given range

# Examples

```julia
N = 10

s = siteinds("S=1/2", N)
psi = random_mps(s; linkdims=8)
Z = expect(psi, "Sz") # compute for all sites
Z = expect(psi, "Sz"; sites=2:4) # compute for sites 2,3,4
Z3 = expect(psi, "Sz"; sites=3)  # compute for site 3 only (output will be a scalar)
XZ = expect(psi, ["Sx", "Sz"]) # compute Sx and Sz for all sites
Z = expect(psi, [1/2 0; 0 -1/2]) # same as expect(psi,"Sz")

s = siteinds("Electron", N)
psi = random_mps(s; linkdims=8)
dens = expect(psi, "Ntot")
updens, dndens = expect(psi, "Nup", "Ndn") # pass more than one operator
```
"""
function expect(psi::MPS, ops; sites = 1:length(psi), site_range = nothing)
    psi = copy(psi)
    N = length(psi)
    ElT = scalartype(psi)
    s = siteinds(psi)

    if !isnothing(site_range)
        @warn "The `site_range` keyword arg. to `expect` is deprecated: use the keyword `sites` instead"
        sites = site_range
    end

    site_range = (sites isa AbstractRange) ? sites : collect(sites)
    Ns = length(site_range)
    start_site = first(site_range)

    el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

    psi = orthogonalize(psi, start_site)
    norm2_psi = norm(psi)^2
    iszero(norm2_psi) && error("MPS has zero norm in function `expect`")

    ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
    for (entry, j) in enumerate(site_range)
        psi = orthogonalize(psi, j)
        for (n, opname) in enumerate(ops)
            oⱼ = adapt(unspecify_type_parameters(datatype(psi[j])), op(opname, s[j]))
            val = inner(psi[j], apply(oⱼ, psi[j])) / norm2_psi
            ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
        end
    end

    if sites isa Number
        return map(arr -> arr[1], ex)
    end
    return ex
end

function expect(psi::MPS, op::AbstractString; kwargs...)
    return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op::Matrix{<:Number}; kwargs...)
    return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op1::AbstractString, ops::AbstractString...; kwargs...)
    return expect(psi, (op1, ops...); kwargs...)
end

function expect(psi::MPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...)
    return expect(psi, (op1, ops...); kwargs...)
end
