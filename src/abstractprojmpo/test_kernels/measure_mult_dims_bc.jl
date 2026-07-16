# Measure the aliased KL PHP MPO's dense-multiplicity (mult) dimensions under
# OPEN vs CLOSED (periodic) boundary conditions — no DMRG, just MPO build +
# aliased sandwich (same construction as measure_fused_dims.jl).
#
# Model: ladder of N+1 rungs (sites (2j-1,2j)); OBC has N inter-rung plaquette
# couplings + N projector plaquettes. CLOSED rings the rung-chain: one extra
# inter-rung coupling wrapping rung N+1 -> rung 1, and one extra projector
# plaquette on the wrapping sites [2N+1, 2N+2, 1, 2].
#
#   BENCH_PSIGN=1.0 N_PLAQ=12 julia --project=. \
#     ITensorMPS.jl/src/abstractprojmpo/test_kernels/measure_mult_dims_bc.jl

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
include("/home/ritvik/temp/temp/edited_packages/test_sparse_ham/aliased_helpers.jl")

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
            if     abs(v)       < tol; A[i] =  0.0
            elseif abs(v - 1.0) < tol; A[i] =  1.0
            elseif abs(v + 1.0) < tol; A[i] = -1.0
            elseif abs(v - 0.5) < tol; A[i] =  0.5
            elseif abs(v + 0.5) < tol; A[i] = -0.5
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

# Build the aliased PHP for a given boundary condition and return it.
function build_php(N, spin_sector, closed::Bool)
    states = 2N + 2
    sites  = siteinds("S=1", states)

    os = OpSum()
    for j in 1:N+1; os += "Sz", 2j-1, "Sz", 2j; end
    for j in 1:N
        os += "Sx", 2j-1, "Sx", 2j+2
        os += "Sy", 2j,   "Sy", 2j+1
    end

    plaq_sites = Vector{Vector{Int}}()
    for j in 1:N; push!(plaq_sites, [2j-1, 2j, 2j+1, 2j+2]); end

    if closed
        # wrap rung (N+1) -> rung 1, by analogy to the bulk coupling
        # (rung a odd,rung a+1 even) / (rung a even,rung a+1 odd):
        os += "Sx", 2N+1, "Sx", 2       # odd of rung N+1, even of rung 1
        os += "Sy", 2N+2, "Sy", 1       # even of rung N+1, odd of rung 1
        push!(plaq_sites, [2N+1, 2N+2, 1, 2])
    end

    ConsOps1 = MPO[]
    for ps in plaq_sites
        temp = OpSum()
        temp += 0.5,             "Id",           ps[1], "Id",           ps[2], "Id",           ps[3], "Id",           ps[4]
        temp += spin_sector*0.5, "exp(i*pi*Sy)", ps[1], "exp(i*pi*Sx)", ps[2], "exp(i*pi*Sx)", ps[3], "exp(i*pi*Sy)", ps[4]
        op = MPO(temp, sites, ps); clean!(op)
        push!(ConsOps1, op)
    end
    P_sparse = multiplyVecMPOtoMPO(ConsOps1)
    H = MPO(os, sites)

    H_ali = sandwich_mpo(P_sparse, H; output_hint=:aliased)
    fuse_sparse_links!(H_ali)
    prepermute_aliased_mpo!(H_ali)
    return H_ali, P_sparse
end

function report(tag, H_ali, P_sparse)
    L = length(H_ali)
    println("\n================ $tag ================")
    println("  P (projector) MPO bond dims: ", [dim(commonind(P_sparse[i], P_sparse[i+1])) for i in 1:L-1])
    println("\n  site | sparse-Link dims | dense-mult dims | n_tmpl blksize")
    for i in 1:L
        ITensors.has_external_storage(H_ali[i]) || (println("   $i  | (no aliased storage)"); continue)
        Hw = ITensors.get_external_storage(H_ali[i]); hinds = inds(Hw)
        length(hinds) < 6 && (println("   $i  | boundary (", length(hinds), " inds)"); continue)
        sp = hinds[1:4]; de = hinds[5:6]; A = Hw.aliased
        println("   $i  | Link=", [dim(I) for I in sp if hastags(I,"Link")],
                " Site=", [dim(I) for I in sp if hastags(I,"Site")],
                " | mult=", [dim(I) for I in de], " | ", A.n_templates, " ", A.blksize)
    end
    println("\n  bulk-bond fused dims (mid channel n_m, mid mult dr):")
    for si in 3:min(9, L-1)
        H1 = H_ali[si]; H2 = H_ali[si+1]
        (ITensors.has_external_storage(H1) && ITensors.has_external_storage(H2)) || continue
        H1w = ITensors.get_external_storage(H1); H2w = ITensors.get_external_storage(H2)
        (length(inds(H1w))>=6 && length(inds(H2w))>=6) || continue
        mids = commoninds(H1, H2)
        h1sp = Set(ITensors.id(i) for i in inds(H1w)[1:4])
        mid_channel = only(filter(i ->   ITensors.id(i) in h1sp, mids))
        mid_mult    = only(filter(i -> !(ITensors.id(i) in h1sp), mids))
        h1l = [i for i in inds(H1w)[1:4] if hastags(i,"Link")]
        h2l = [i for i in inds(H2w)[1:4] if hastags(i,"Link")]
        n_l1 = dim(only(filter(i -> ITensors.id(i) != ITensors.id(mid_channel), h1l)))
        n_r2 = dim(only(filter(i -> ITensors.id(i) != ITensors.id(mid_channel), h2l)))
        println("    bond ($si,$(si+1)): n_m=$(dim(mid_channel))  mid-mult dr=$(dim(mid_mult))  | n_l1=$n_l1 n_r2=$n_r2",
                "  | H1 nblk=$(length(H1w.aliased.keys)) ntmpl=$(H1w.aliased.n_templates)",
                "  H2 nblk=$(length(H2w.aliased.keys)) ntmpl=$(H2w.aliased.n_templates)")
    end
end

let
    N = parse(Int, get(ENV, "N_PLAQ", "12"))
    ss = parse(Float64, get(ENV, "BENCH_PSIGN", "1.0"))
    println("[KL N=$N, states=$(2N+2), spin_sector=$ss]  comparing OPEN vs CLOSED BC")
    Ho, Po = build_php(N, ss, false); report("OPEN BC", Ho, Po)
    Hc, Pc = build_php(N, ss, true);  report("CLOSED BC (ring)", Hc, Pc)
    println("\ndone.")
end
