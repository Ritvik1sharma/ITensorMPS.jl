# aliased_mpo_bonds.jl — bond layout of an aliased MPO produced by an MPO × MPO product.
#
# contract(A::MPO, B::MPO, Abackend, Bbackend; Cbackend) calls canonicalize_aliased_bonds!
# on an aliased result, the way a dense MPO product combines (or compresses) the links
# each bond carries:
#   1. key link strands: every bond's links that are in the key of BOTH neighbouring sites
#      and share a tag (e.g. the links of P and of P† in P†HP) are fused into one key
#      index. Only the key tuples change; templates are untouched, so aliasing is kept.
#   2. dense block order: each site's dense block is stored as [right link, left link]
#      (descending l=), the order the aliased matvec reads without permuting H.
# Moved here from SparseBackends/test_sparse_ham/aliased_helpers.jl (fuse_sparse_links!,
# prepermute_aliased_mpo!), which now call these.

_aliased_wrap(T::ITensor) = (ITensors.has_external_storage(T) &&
    ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse) ?
    ITensors.get_external_storage(T) : nothing

function _aliased_key_inds(T::ITensor)
    w = _aliased_wrap(T)
    w === nothing && return Set(inds(T))
    return Set(w.inds[1:SparseBackends._abs_head_len(w)])
end

# Fuse the key strands `strands` of an aliased site into the single key index `fused`
# (column-major in the strands' stored order), at the first strand's position.
function _fuse_aliased_key_strands(T::ITensor, strands::Vector{<:Index}, fused::Index)
    w  = _aliased_wrap(T)
    A  = w.aliased
    PA = SparseBackends._abs_head_len(w)
    old_inds = w.inds
    pos = Int[]
    for s in strands
        p = findfirst(==(s), old_inds)
        (p === nothing || p > PA) && error("fuse: strand $s is not in the key of this site")
        push!(pos, p)
    end
    sort!(pos)
    nstr   = length(pos)
    sdims  = [dim(old_inds[p]) for p in pos]
    stride = cumprod([1; sdims[1:end-1]])
    at     = pos[1]
    new_P  = PA - (nstr - 1)
    new_N  = length(old_inds) - (nstr - 1)
    N2     = new_N - new_P
    keep   = [p for p in 1:length(old_inds) if !(p in pos)]
    new_inds = Vector{Index}(undef, new_N)
    k = 0
    for q in 1:new_N
        q == at ? (new_inds[q] = fused) : (k += 1; new_inds[q] = old_inds[keep[k]])
    end
    keep_key = [p for p in 1:PA if !(p in pos)]
    Kt = eltype(eltype(A.keys))
    new_keys = Vector{NTuple{new_P,Kt}}(undef, length(A.keys))
    for i in eachindex(A.keys)
        old = A.keys[i]
        f = 1
        for t in 1:nstr
            f += (old[pos[t]] - 1) * stride[t]
        end
        new_keys[i] = ntuple(new_P) do j
            j == at ? Kt(f) : Kt(old[keep_key[j < at ? j : j - 1]])
        end
    end
    Tel = eltype(A.templates)
    ali = SparseBackends.AliasedBlockSparse{Tel,new_N,N2,new_P,Kt}(
        ntuple(i -> dim(new_inds[i]), new_N), A.blksize, copy(A.templates), A.n_templates,
        new_keys, copy(A.alias_ids), copy(A.scalars))
    return ITensors._itensor_from_external_storage(
        SparseBackends.WrappedAliasedBlockSparse{Tel,new_N,N2,new_P}(ali, Tuple(new_inds)))
end

_link_number(I) = (m = match(r"l=(\d+)", string(tags(I)))) === nothing ? -1 : parse(Int, m.captures[1])

"""Fuse each bond's same-tag key link strands (key on both neighbouring aliased sites)
into one key index. Returns H."""
function fuse_aliased_bond_strands!(H::MPO)
    for k in 1:(length(H) - 1)
        (_aliased_wrap(H[k]) === nothing || _aliased_wrap(H[k + 1]) === nothing) && continue
        kk, kk1 = _aliased_key_inds(H[k]), _aliased_key_inds(H[k + 1])
        by_tag = Dict{String,Vector{Index}}()
        for I in commoninds(H[k], H[k + 1])
            (I in kk && I in kk1) || continue
            push!(get!(by_tag, string(tags(I)), Index[]), I)
        end
        for (_, strands) in by_tag
            length(strands) <= 1 && continue
            fused  = combinedind(combiner(strands...; tags = "Link,FusedSparse,bond=$(k)"))
            H[k]   = _fuse_aliased_key_strands(H[k], strands, fused)
            H[k+1] = _fuse_aliased_key_strands(H[k+1], strands, fused)
        end
    end
    return H
end

"""Store every aliased site's dense block as [right link, left link] (descending l=).
Returns the number of sites permuted."""
function order_aliased_dense_blocks!(H::MPO)
    n = 0
    for k in 1:length(H)
        w = _aliased_wrap(H[k])
        w === nothing && continue
        PA = SparseBackends._abs_head_len(w)
        N  = ndims(w.aliased)
        N - PA < 2 && continue
        tail  = collect(w.inds[PA+1:N])
        order = sortperm([_link_number(I) for I in tail]; rev = true)   # [right, left]
        order == collect(1:N-PA) && continue
        perm  = vcat(collect(1:PA), order .+ PA)
        ali   = SparseBackends.permutedims(w.aliased, perm)
        H[k]  = ITensors._itensor_from_external_storage(
            SparseBackends.WrappedAliasedBlockSparse{eltype(ali),N,N-PA,PA}(ali, Tuple(w.inds[i] for i in perm)))
        n += 1
    end
    return n
end

"""
    canonicalize_aliased_bonds!(H::MPO) -> H

fuse_aliased_bond_strands!, then order_aliased_dense_blocks!. Non-aliased sites are left
alone. Called by contract(A::MPO, B::MPO, Abackend, Bbackend) on an aliased result.
"""
canonicalize_aliased_bonds!(H::MPO) = (fuse_aliased_bond_strands!(H); order_aliased_dense_blocks!(H); H)

"""Storage of an MPO / MPS: SparseBackends.footprint of each site, summed over sites."""
SparseBackends.footprint(M::AbstractMPS) = reduce((a, b) -> map(+, a, b), SparseBackends.footprint.(M))
