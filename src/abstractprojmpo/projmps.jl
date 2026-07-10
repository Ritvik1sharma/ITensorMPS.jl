mutable struct ProjMPS
    lpos::Int
    rpos::Int
    nsite::Int
    M::MPS
    LR::Vector{ITensor}
end
ProjMPS(M::MPS) = ProjMPS(0, length(M) + 1, 2, M, Vector{ITensor}(undef, length(M)))

copy(P::ProjMPS) = ProjMPS(P.lpos, P.rpos, P.nsite, copy(P.M), copy(P.LR))

nsite(P::ProjMPS) = P.nsite

# The range of center sites
# TODO: Use the `AbstractProjMPO` version.
site_range(P::ProjMPS) = (P.lpos + 1):(P.rpos - 1)

function set_nsite!(P::ProjMPS, nsite)
    P.nsite = nsite
    return P
end

Base.length(P::ProjMPS) = length(P.M)

function lproj(P::ProjMPS)
    (P.lpos <= 0) && return nothing
    return P.LR[P.lpos]
end

function rproj(P::ProjMPS)
    (P.rpos >= length(P) + 1) && return nothing
    return P.LR[P.rpos]
end

function product(P::ProjMPS, v::ITensor)::ITensor
    if nsite(P) != 2
        error("Only two-site ProjMPS currently supported")
    end

    # preserve_bs_output=true keeps the projection aliased when the ground state M
    # (and the current ψ in its env) is aliased — a no-op for dense inputs. Without
    # it these contractions densify, and the excited matvec Pv += weight·(…) then
    # tries to add a dense term to the aliased H·v (no aliased+dense add method).
    Lpm = dag(prime(P.M[P.lpos + 1], "Link"))
    !isnothing(lproj(P)) && (Lpm = *(Lpm, lproj(P); preserve_bs_output=true))

    Rpm = dag(prime(P.M[P.rpos - 1], "Link"))
    !isnothing(rproj(P)) && (Rpm = *(Rpm, rproj(P); preserve_bs_output=true))

    pm = *(Lpm, Rpm; preserve_bs_output=true)

    pv = scalar(pm * v)

    Mv = pv * dag(pm)

    return noprime(Mv)
end

#function Base.eltype(P::ProjMPS)
#  elT = eltype(P.M[P.lpos+1])
#  for j = P.lpos+2:P.rpos-1
#    elT = promote_type(elT,eltype(P.M[j]))
#  end
#  if !isnull(lproj(P))
#    elT = promote_type(elT,eltype(lproj(P)))
#  end
#  if !isnull(rproj(P))
#    elT = promote_type(elT,eltype(rproj(P)))
#  end
#  return elT
#end

(P::ProjMPS)(v::ITensor) = product(P, v)

#function Base.size(P::ProjMPS)::Tuple{Int,Int}
#  d = 1
#  if P.lpos > 0
#    d *= dim(linkind(M,P.lpos))
#  end
#  for j=P.lpos+1:P.rpos-1
#    d *= dim(siteind(P.M,j))
#  end
#  if P.rpos-1 < N
#    d *= dim(linkind(M,P.rpos-1))
#  end
#  return (d,d)
#end

function makeL!(P::ProjMPS, psi::MPS, k::Int)
    while P.lpos < k
        ll = P.lpos
        if ll <= 0
            P.LR[1] = *(psi[1], dag(prime(P.M[1], "Link")); preserve_bs_output=true)
            P.lpos = 1
        else
            P.LR[ll + 1] = *(*(P.LR[ll], psi[ll + 1]; preserve_bs_output=true),
                             dag(prime(P.M[ll + 1], "Link")); preserve_bs_output=true)
            P.lpos += 1
        end
    end
    return
end

function makeR!(P::ProjMPS, psi::MPS, k::Int)
    N = length(P.M)
    while P.rpos > k
        rl = P.rpos
        if rl >= N + 1
            P.LR[N] = *(psi[N], dag(prime(P.M[N], "Link")); preserve_bs_output=true)
            P.rpos = N
        else
            P.LR[rl - 1] = *(*(P.LR[rl], psi[rl - 1]; preserve_bs_output=true),
                             dag(prime(P.M[rl - 1], "Link")); preserve_bs_output=true)
            P.rpos -= 1
        end
    end
    return
end

function position!(P::ProjMPS, psi::MPS, pos::Int)
    makeL!(P, psi, pos - 1)
    makeR!(P, psi, pos + nsite(P))

    #These next two lines are needed
    #when moving lproj and rproj backward
    P.lpos = pos - 1
    P.rpos = pos + nsite(P)
    return P
end

function checkflux(P::ProjMPS)
    checkflux(P.M)
    foreach(eachindex(P.LR)) do i
        isassigned(P.LR, i) && checkflux(P.LR[i])
    end
    return nothing
end
