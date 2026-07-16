# Build the OPEN-BC KL aliased PHP for higher-spin site types (dim 4,5,7,10 =
# S=3/2, S=2, S=3, S=9/2) at N=10 and report the Hamiltonian MPO bond dims and
# the aliased-PHP params (link dims, dense-mult dims, n_templates, blksize, n_m).
#
# ITensors ships only S=1/2 and S=1; higher spin is added here from the standard
# spin-S matrices (pattern per ITensors.jl/utils/site3_2.jl). exp(i*pi*Sy/Sx) is
# handled generically by ITensors' op-expression parser once Sx/Sy are defined.
#
#   N_PLAQ=10 julia --project=. \
#     ITensorMPS.jl/src/abstractprojmpo/test_kernels/measure_highspin_dims.jl

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
BLAS.set_num_threads(1)
# General spin-S site types (S=3/2, S=2, S=3, S=9/2, …) — reusable utility.
include("/home/ritvik/temp/temp/edited_packages/ITensors.jl/utils/general_spin.jl")
include("/home/ritvik/temp/temp/edited_packages/test_sparse_ham/aliased_helpers.jl")

const HIGHSPIN = (("S=3/2", 4), ("S=2", 5), ("S=3", 7), ("S=9/2", 10))

# ---- builders (same as measure_fused_dims.jl) ----
function sandwich_mpo(P::MPO, H::MPO; output_hint::Symbol=:default)
    H1    = contract(P'', H', :coo, :dense; Cbackend=output_hint)
    H_eff = contract(P, H1, :coo, output_hint; Cbackend=output_hint)
    return replaceprime(H_eff, 3 => 1)
end
mulMPO(A::MPO, B::MPO) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)
function multiplyVecMPOtoMPO(vec)
    r = vec[1]; for j in 2:length(vec); r = mulMPO(r, vec[j]); end; r
end
function clean!(op::MPO; tol=1e-12)
    for j in 1:length(op)
        T = op[j]; A = array(T)
        for i in eachindex(A)
            v = A[i]
            if     abs(v)         < tol; A[i] =  0.0
            elseif abs(v - 1.0)   < tol; A[i] =  1.0
            elseif abs(v + 1.0)   < tol; A[i] = -1.0
            elseif abs(v - 0.5)   < tol; A[i] =  0.5
            elseif abs(v + 0.5)   < tol; A[i] = -0.5
            elseif abs(v - 1.0im) < tol; A[i] =  1.0im
            elseif abs(v + 1.0im) < tol; A[i] = -1.0im
            elseif abs(v - 0.5im) < tol; A[i] =  0.5im
            elseif abs(v + 0.5im) < tol; A[i] = -0.5im
            elseif abs(v + 2.0)   < tol; A[i] = -2.0
            elseif abs(v - 2.0)   < tol; A[i] =  2.0
            end
        end
        op[j] = ITensor(A, inds(T)...)
    end
    return op
end

function build_and_report(tstr, d, N, ss)
    states = 2N + 2
    println("\n################ site type \"$tstr\" (dim=$d, N=$N, states=$states) ################")
    sites = siteinds(tstr, states)

    os = OpSum()
    for j in 1:N+1; os += "Sz", 2j-1, "Sz", 2j; end
    for j in 1:N;   os += "Sx", 2j-1, "Sx", 2j+2; os += "Sy", 2j, "Sy", 2j+1; end

    ConsOps1 = MPO[]
    for j in 1:N
        temp = OpSum()
        temp += 0.5,    "Id",2j-1,"Id",2j,"Id",2j+1,"Id",2j+2
        temp += ss*0.5, "exp(i*pi*Sy)",2j-1,"exp(i*pi*Sx)",2j,"exp(i*pi*Sx)",2j+1,"exp(i*pi*Sy)",2j+2
        op = MPO(temp, sites, [2j-1, 2j, 2j+1, 2j+2]); clean!(op)
        push!(ConsOps1, op)
    end
    P_sparse = multiplyVecMPOtoMPO(ConsOps1)
    H = MPO(os, sites)

    Hbond = [dim(commonind(H[i], H[i+1])) for i in 1:length(H)-1]
    Pbond = [dim(commonind(P_sparse[i], P_sparse[i+1])) for i in 1:length(P_sparse)-1]
    println("  bare H       MPO bond dims: ", Hbond, "   (max ", maximum(Hbond), ")")
    println("  projector P  MPO bond dims: ", Pbond, "   (max ", maximum(Pbond), ")")

    H_ali = sandwich_mpo(P_sparse, copy(H); output_hint=:aliased)
    fuse_sparse_links!(H_ali)
    prepermute_aliased_mpo!(H_ali)

    L = length(H_ali)
    println("  --- aliased PHP per-site (Link | Site | mult | n_tmpl blksize) ---")
    linkmax = 0; multmax = 0; tmplmax = 0
    for i in 1:L
        ITensors.has_external_storage(H_ali[i]) || (println("   $i  | (no aliased storage / boundary)"); continue)
        Hw = ITensors.get_external_storage(H_ali[i]); hinds = inds(Hw)
        length(hinds) < 6 && (println("   $i  | boundary (", length(hinds), " inds)"); continue)
        sp = hinds[1:4]; de = hinds[5:6]; A = Hw.aliased
        lk = [dim(I) for I in sp if hastags(I,"Link")]; mu = [dim(I) for I in de]
        linkmax = max(linkmax, maximum(lk)); multmax = max(multmax, maximum(mu)); tmplmax = max(tmplmax, A.n_templates)
        (3 <= i <= 8 || i > L-3) && println("   $i  | Link=", lk, " Site=",
                [dim(I) for I in sp if hastags(I,"Site")], " | mult=", mu, " | ", A.n_templates, " ", A.blksize)
    end
    # bulk-bond fused seam (mid channel / mult)
    for si in (3, 4)
        H1 = H_ali[si]; H2 = H_ali[si+1]
        (ITensors.has_external_storage(H1) && ITensors.has_external_storage(H2)) || continue
        H1w = ITensors.get_external_storage(H1); H2w = ITensors.get_external_storage(H2)
        (length(inds(H1w))>=6 && length(inds(H2w))>=6) || continue
        mids = commoninds(H1, H2)
        h1sp = Set(ITensors.id(i) for i in inds(H1w)[1:4])
        mc = only(filter(i ->   ITensors.id(i) in h1sp, mids))
        mm = only(filter(i -> !(ITensors.id(i) in h1sp), mids))
        println("    bond ($si,$(si+1)): n_m=$(dim(mc)) mid-mult dr=$(dim(mm))  | H1 nblk=$(length(H1w.aliased.keys)) ntmpl=$(H1w.aliased.n_templates)  H2 nblk=$(length(H2w.aliased.keys)) ntmpl=$(H2w.aliased.n_templates)")
    end
    println("  SUMMARY dim=$d: max aliased Link=$linkmax  max mult=$multmax  max n_templates=$tmplmax  bare-H maxbond=$(maximum(Hbond))  P maxbond=$(maximum(Pbond))")
end

let
    N  = parse(Int, get(ENV, "N_PLAQ", "10"))
    ss = parse(Float64, get(ENV, "BENCH_PSIGN", "1.0"))
    # sanity: confirm the generic exp op works on a fresh high-spin site
    for (tstr, d) in HIGHSPIN
        s = siteinds(tstr, 1)[1]
        e = op("exp(i*pi*Sy)", s)
        println("[sitetype \"$tstr\" dim=$d OK]  exp(i*pi*Sy) built, size=", size(array(e)))
    end
    for (tstr, d) in HIGHSPIN
        try
            build_and_report(tstr, d, N, ss)
        catch err
            println("\n!!!! site type \"$tstr\" (dim=$d) FAILED: ", err)
        end
    end
    println("\ndone.")
end
