abstract type AbstractProjMPO end
import SparseBackends

copy(::AbstractProjMPO) = error("Not implemented")

"""
    nsite(P::ProjMPO)

Retrieve the number of unprojected (open)
site indices of the ProjMPO object `P`
"""
nsite(P::AbstractProjMPO) = P.nsite

set_nsite!(::AbstractProjMPO, nsite) = error("Not implemented")

# The range of center sites
site_range(P::AbstractProjMPO) = (P.lpos + 1):(P.rpos - 1)

"""
    length(P::ProjMPO)

The length of a ProjMPO is the same as
the length of the MPO used to construct it
"""
Base.length(P::AbstractProjMPO) = length(P.H)

function lproj(P::AbstractProjMPO)::Union{ITensor, OneITensor}
    (P.lpos <= 0) && return OneITensor()
    return P.LR[P.lpos]
end

function rproj(P::AbstractProjMPO)::Union{ITensor, OneITensor}
    (P.rpos >= length(P) + 1) && return OneITensor()
    return P.LR[P.rpos]
end

function ITensors.contract(P::AbstractProjMPO, v::ITensor)::ITensor
    itensor_map = Union{ITensor, OneITensor}[lproj(P)]
    
    append!(itensor_map, P.H[site_range(P)])
    # for j in site_range(P)
    #     if ITensors.has_external_storage(P.H[j])
    #         # println("starting and ending inds ", inds(P.H[j]), " and ", ITensors.has_external_storage(P.H[j]))
    #         tensor = SparseBackends.to_dense_itensors(P.H[j])
    #         # println("starting and ending inds ", inds(tensor), " and ", ITensors.has_external_storage(tensor))        
    #         push!(itensor_map, tensor)
    #     else
    #         push!(itensor_map, P.H[j])
    #     end
    #     # dense_h = SparseBackends.to_dense(P.H[j])
    #     # push!(itensor_map, dense_h)
    # end

    push!(itensor_map, rproj(P))

    # # Reverse the contraction order of the map if
    # # the first tensor is a scalar (for example we
    # # are at the left edge of the system)
    if dim(first(itensor_map)) == 1
        reverse!(itensor_map)
    end
    idx = 0

    # Apply the map
    Hv = v
    for it in itensor_map
        idx += 1
        # println("Contracting with itensor ", idx, " inds are ", inds(it), " and v inds are ", inds(Hv))
        # println("Contracting with itensor: ", it)
        # println("Hv before contraction: ", Hv)
        Hv *= it
    end
    return Hv
end

"""
    product(P::ProjMPO,v::ITensor)::ITensor

    (P::ProjMPO)(v::ITensor)

Efficiently multiply the ProjMPO `P`
by an ITensor `v` in the sense that the
ProjMPO is a generalized square matrix
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::AbstractProjMPO, v::ITensor)::ITensor
    Pv = contract(P, v)
    if order(Pv) != order(v)
        error(
            string(
                "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
                "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
                "(1) You are trying to multiply the ProjMPO with the $(nsite(P))-site wave-function at the wrong position.\n",
                "(2) `orthogonalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
                "P*v inds: $(inds(Pv)) \n\n",
                "v inds: $(inds(v))",
            ),
        )
    end
    return noprime(Pv)
end

(P::AbstractProjMPO)(v::ITensor) = product(P, v)

"""
    eltype(P::ProjMPO)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPO
`P`.
"""
function Base.eltype(P::AbstractProjMPO)::Type
    ElType = eltype(lproj(P))
    for j in site_range(P)
        ElType = promote_type(ElType, eltype(P.H[j]))
    end
    return promote_type(ElType, eltype(rproj(P)))
end

"""
    size(P::ProjMPO)

The size of a ProjMPO are its dimensions
`(d,d)` when viewed as a matrix or linear operator
acting on a space of dimension `d`.

For example, if a ProjMPO maps from a space with
indices `(a,s1,s2,b)` to the space `(a',s1',s2',b')`
then the size is `(d,d)` where
`d = dim(a)*dim(s1)*dim(s1)*dim(b)`
"""
function Base.size(P::AbstractProjMPO)::Tuple{Int, Int}
    d = 1
    for i in inds(lproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    for j in site_range(P)
        for i in inds(P.H[j])
            plev(i) > 0 && (d *= dim(i))
        end
    end
    for i in inds(rproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    return (d, d)
end

function _makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)::Union{ITensor, Nothing}
    # println("\t\tPositioning ProjMPO ", debug)
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = P.lpos
    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        P.lpos = k
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        if debug
            L = L * P.H[ll + 1]
            println("========= ", ll + 1, " after multiplying H: ", ITensors.has_external_storage(P.H[ll + 1]))
            L = L * dag(prime(psi[ll + 1]))
            println("========= ", ll + 1, " after multiplying dag prime psi: ", ITensors.has_external_storage(L))
            L = L * psi[ll + 1]
            println("========= ", ll + 1, " after multiplying psi: ", ITensors.has_external_storage(L))
        else
            H_site = P.H[ll + 1]
            # if ITensors.has_external_storage(H_site)
            #     H_site = SparseBackends.to_dense_itensors(H_site)
            # end
            L = L * H_site
            L = L * dag(prime(psi[ll + 1]))
            L = L * psi[ll + 1]
        end
        P.LR[ll + 1] = L
        ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return L
end

function makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)
    # println("\tPositioning ProjMPO ", debug)
    _makeL!(P, psi, k; debug=debug)
    return P
end

function _makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)::Union{ITensor, Nothing}
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    rl = P.rpos
    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        P.rpos = k
        return nothing
    end
    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
        if debug
            println("========= ", rl, N+1)
            time = @elapsed begin
                R = R * P.H[rl - 1]
            end
            println("========= ", " after multiplying H: ", R)
            time = @elapsed begin
                R =  dag(prime(psi[rl - 1])) * R
            end
            println("========= ", " after multiplying psi: ", R)
            time = @elapsed begin
                R = psi[rl - 1] * R
            end
            println("========= ", " after multiplying dag psi: ", R)
        else
            H_site = P.H[rl - 1]
            # if ITensors.has_external_storage(H_site)
            #     H_site = SparseBackends.to_dense_itensors(H_site)
            # end
            R = R * H_site
            R = dag(prime(psi[rl - 1])) * R
            R = psi[rl - 1] * R
        end
        P.LR[rl - 1] = R
        # println(" check rl - 1 information ", rl - 1)
        # println("TENSOR IS ", R)
        rl -= 1
    end
    P.rpos = k
    return R
end

function makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)
    _makeR!(P, psi, k; debug=debug)
    return P
end

"""
    position!(P::ProjMPO, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPO `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPO on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function position!(P::AbstractProjMPO, psi::MPS, pos::Int; debug=false)
    # println("Positioning ProjMPO ", debug)
    makeL!(P, psi, pos - 1; debug=debug)
    makeR!(P, psi, pos + nsite(P); debug=debug)
    return P
end

"""
    noiseterm(P::ProjMPO,
              phi::ITensor,
              ortho::String)

Return a "noise term" or density matrix perturbation
ITensor as proposed in Phys. Rev. B 72, 180403 for aiding
convergence of DMRG calculations. The ITensor `phi`
is the contracted product of MPS tensors acted on by the
ProjMPO `P`, and `ortho` is a String which can take
the values `"left"` or `"right"` depending on the
sweeping direction of the DMRG calculation.
"""
function noiseterm(P::AbstractProjMPO, phi::ITensor, ortho::String)::ITensor
    if nsite(P) != 2
        error("noise term only defined for 2-site ProjMPO")
    end

    site_range_P = site_range(P)
    if ortho == "left"
        AL = P.H[first(site_range_P)]
        AL = lproj(P) * AL
        nt = AL * phi
    elseif ortho == "right"
        AR = P.H[last(site_range_P)]
        AR = AR * rproj(P)
        nt = phi * AR
    else
        error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
    end
    nt = nt * dag(noprime(nt))

    return nt
end

function checkflux(P::AbstractProjMPO)
    checkflux(P.H)
    for n in length(P.LR)
        if isassigned(P.LR, n)
            checkflux(P.LR[n])
        end
    end
    return nothing
end
