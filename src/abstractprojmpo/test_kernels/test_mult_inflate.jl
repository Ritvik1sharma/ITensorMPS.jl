# How does the fused matvec scale as the Hamiltonian's dense MULTIPLICITY grows?
# (mult = bare-H MPO bond dim; KL is stuck at 4-5, but complex H would be larger.)
#
# We inflate the mult from its real value to M (=40) by embedding the real dense
# blocks in the top-left corner of MxM and ZERO-PADDING the rest. Zero-pad is
# EXACT (padded region contributes nothing), so:
#   - the inflated kernel computes the IDENTICAL linear map (free correctness check),
#   - dense BLAS still does the full M-sized work (no skipping zeros), so timing the
#     padded operator faithfully proxies a real bond-M Hamiltonian.
# Baselines: (dense-ITensor big-K contraction) validated vs stock product at real
# mult, then re-timed at M. Reports fused/dense ratio + fused per-step profile at
# real mult vs M.

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra, Random, Printf
import ITensorMPS: lproj, rproj, product
import TimerOutputs: reset_timer!, print_timer

include("/home/ritvik/temp/temp/edited_packages/test_sparse_ham/aliased_helpers.jl")
include("/home/ritvik/temp/temp/edited_packages/ITensorMPS.jl/src/abstractprojmpo/fused_mpo_helpers_env.jl")
include("/home/ritvik/temp/temp/edited_packages/ITensorMPS.jl/src/abstractprojmpo/fused_mpo_helpers_matvec.jl")

const PEAK_GFPS = Ref(0.0)   # BLAS ComplexF64 peak (2MNK conv), set in main

# ---- builders (verbatim from test_fused_real.jl) ----
function sandwich_mpo(P::MPO, H::MPO; output_hint::Symbol=:default)
    H1    = contract(P'', H', :coo, :dense; Cbackend=output_hint)
    H_eff = contract(P, H1, :coo, output_hint; Cbackend=output_hint)
    return replaceprime(H_eff, 3 => 1)
end
mulMPO(A::MPO, B::MPO) = replaceprime(contract(A, prime(B, "Site"), :coo, :coo), 2 => 1)
multiplyVecMPOtoMPO(vec) = (r = vec[1]; for j in 2:length(vec); r = mulMPO(r, vec[j]); end; r)
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

# ---- logical (dl x dr) template matrix in native orientation ----
function template_matrix(templates, blksize, tid, dl, dr, dl_fast)
    flat = @view templates[(tid-1)*blksize+1 : tid*blksize]
    if dl_fast
        m = reshape(flat, dl, dr); return Matrix(m)                # [a=dl, b=dr]
    else
        m = reshape(flat, dr, dl); return permutedims(Matrix(m), (2,1))  # -> [dl, dr]
    end
end

# ---- assemble the dense operator (as ITensors) from ctx; mult dims = Ml/Mmid/Mr ----
# Real dims (ctx.dl1, ctx.dr1==ctx.dl2, ctx.dr2) embedded top-left; rest zero.
function build_dense_op(ctx::MatvecContext{T}, phiarr, Ml, Mmid, Mr) where {T}
    dl1r, dr1r, dl2r, dr2r = ctx.dl1, ctx.dr1, ctx.dl2, ctx.dr2
    @assert dr1r == dl2r "middle mult mismatch"
    n1 = length(ctx.templates1) ÷ ctx.blksize1
    n2 = length(ctx.templates2) ÷ ctx.blksize2
    # dense H tensors (mult-left, mult-right, channel-left, channel-right, s, s')
    H1 = zeros(T, Ml, Mmid, ctx.n_l1, ctx.n_m, ctx.d1, ctx.d1)
    H2 = zeros(T, Mmid, Mr, ctx.n_m, ctx.n_r2, ctx.d2, ctx.d2)
    for l in 1:ctx.n_l1, s1 in 1:ctx.d1, rec in ctx.step1[l, s1]
        M1 = template_matrix(ctx.templates1, ctx.blksize1, rec.tid, dl1r, dr1r, ctx.dl1_fast)
        @views H1[1:dl1r, 1:dr1r, l, rec.dst_pos, s1, rec.out_phys] .+= rec.c .* M1
    end
    for m in 1:ctx.n_m, s2 in 1:ctx.d2, rec in ctx.step2[m, s2]
        M2 = template_matrix(ctx.templates2, ctx.blksize2, rec.tid, dl2r, dr2r, ctx.dl2_fast)
        @views H2[1:dl2r, 1:dr2r, m, rec.dst_pos, s2, rec.out_phys] .+= rec.c .* M2
    end
    # env: pad the mult axis
    Lpad = zeros(T, Ml, ctx.chiL_bra, ctx.chiL_ket, ctx.n_l1); @views Lpad[1:dl1r,:,:,:] .= ctx.Larr
    Rpad = zeros(T, ctx.chiR_bra, ctx.chiR_ket, Mr, ctx.n_r2); @views Rpad[:,:,1:dr2r,:] .= ctx.Rarr
    # indices
    idl1 = Index(Ml, "mL"); imid = Index(Mmid, "mM"); idr2 = Index(Mr, "mR")
    il = Index(ctx.n_l1,"chL"); im = Index(ctx.n_m,"chM"); ir = Index(ctx.n_r2,"chR")
    is1 = Index(ctx.d1,"s1"); is1p = Index(ctx.d1,"s1p"); is2 = Index(ctx.d2,"s2"); is2p = Index(ctx.d2,"s2p")
    iLb = Index(ctx.chiL_bra,"Lb"); iLk = Index(ctx.chiL_ket,"Lk")
    iRb = Index(ctx.chiR_bra,"Rb"); iRk = Index(ctx.chiR_ket,"Rk")
    L_it  = ITensor(Lpad, idl1, iLb, iLk, il)
    R_it  = ITensor(Rpad, iRb, iRk, idr2, ir)
    H1_it = ITensor(H1, idl1, imid, il, im, is1, is1p)
    H2_it = ITensor(H2, imid, idr2, im, ir, is2, is2p)
    # v in (Lket,Rket,s2,s1) layout = phiarr
    v_it  = ITensor(phiarr, iLk, iRk, is2, is1)
    out_inds = (iLb, is1p, is2p, iRb)
    return (L=L_it, v=v_it, H1=H1_it, H2=H2_it, R=R_it, out=out_inds)
end
dense_matvec(op) = op.L * op.v * op.H1 * op.H2 * op.R
dense_array(op)  = array(permute(dense_matvec(op), op.out...))

# ---- inflate a MatvecContext: mult dims -> Ml/Mmid/Mr, real blocks top-left, zeros else ----
function inflate_ctx(ctx::MatvecContext{T}, Ml, Mmid, Mr) where {T}
    dl1r, dr1r, dl2r, dr2r = ctx.dl1, ctx.dr1, ctx.dl2, ctx.dr2
    n1 = length(ctx.templates1) ÷ ctx.blksize1
    n2 = length(ctx.templates2) ÷ ctx.blksize2
    # templates: embed logical (dl x dr) into native (M x M) with same orientation
    bs1 = Ml*Mmid; t1 = zeros(T, bs1*n1)
    for tid in 1:n1
        M1 = template_matrix(ctx.templates1, ctx.blksize1, tid, dl1r, dr1r, ctx.dl1_fast)  # [dl1,dr1]
        blk = reshape((@view t1[(tid-1)*bs1+1 : tid*bs1]), ctx.dl1_fast ? (Ml,Mmid) : (Mmid,Ml))
        if ctx.dl1_fast; @views blk[1:dl1r,1:dr1r] .= M1; else; @views blk[1:dr1r,1:dl1r] .= permutedims(M1,(2,1)); end
    end
    bs2 = Mmid*Mr; t2 = zeros(T, bs2*n2)
    for tid in 1:n2
        M2 = template_matrix(ctx.templates2, ctx.blksize2, tid, dl2r, dr2r, ctx.dl2_fast)  # [dl2,dr2]
        blk = reshape((@view t2[(tid-1)*bs2+1 : tid*bs2]), ctx.dl2_fast ? (Mmid,Mr) : (Mr,Mmid))
        if ctx.dl2_fast; @views blk[1:dl2r,1:dr2r] .= M2; else; @views blk[1:dr2r,1:dl2r] .= permutedims(M2,(2,1)); end
    end
    Larr = zeros(T, Ml, ctx.chiL_bra, ctx.chiL_ket, ctx.n_l1); @views Larr[1:dl1r,:,:,:] .= ctx.Larr
    Rarr = zeros(T, ctx.chiR_bra, ctx.chiR_ket, Mr, ctx.n_r2); @views Rarr[:,:,1:dr2r,:] .= ctx.Rarr
    cL,cLk,cRb,cRk = ctx.chiL_bra,ctx.chiL_ket,ctx.chiR_bra,ctx.chiR_ket
    x2        = Array{T,6}(undef, Mmid, cL, cRk, ctx.d1, ctx.d2, ctx.n_m)
    stepC_out = Matrix{T}(undef, Mr, cL*cRk*ctx.d1)
    Rbatch    = permutedims(Rarr, (1,3,2,4))
    stepD_in  = Array{T,4}(undef, cL, ctx.d1, Mr, cRk)
    populated = Vector{Bool}(undef, ctx.n_m)
    return MatvecContext{T}(
        ctx.d1, ctx.d2, Ml, Mmid, Mmid, Mr, ctx.dl1_fast, ctx.dl2_fast,
        cL, cLk, cRb, cRk, ctx.n_l1, ctx.n_m, ctx.n_r2,
        Larr, Rarr, t1, t2, bs1, bs2, ctx.step1, ctx.step2,
        x2, stepC_out, Rbatch, stepD_in, populated,
        ctx.Lbra, ctx.Lket, ctx.Rbra, ctx.Rket, ctx.s1_ind, ctx.s2_ind)
end

# ---- Zero-pad selected MPO link bonds to dim M (embed real tensors top-left) ----
# Exact: the padded rows/cols are zero so contribute nothing to any contraction →
# the MPO represents the IDENTICAL operator. Only listed links are grown (the rest
# stay at their real 4/5), so the re-sandwiched PHP has dense mult = M at exactly
# those bonds. Shared links are replaced in BOTH tensors that carry them.
function pad_mpo_bonds(H::MPO, M::Int, pad_set)
    N = length(H)
    links = [commonind(H[j], H[j+1]) for j in 1:N-1]
    repl = Dict{UInt64,ITensors.Index}()
    for k in pad_set
        repl[ITensors.id(links[k])] = Index(M, tags(links[k]))
    end
    tensors = ITensor[]
    for j in 1:N
        T = H[j]
        oldis = collect(inds(T))
        newis = map(i -> get(repl, ITensors.id(i), i), oldis)
        A = array(T)                                    # in inds(T) order
        B = zeros(eltype(A), ntuple(a -> dim(newis[a]), length(newis))...)
        ranges = ntuple(a -> 1:size(A, a), ndims(A))    # top-left embed
        @views B[ranges...] .= A
        push!(tensors, ITensor(B, newis...))
    end
    return MPO(tensors)
end

# ---- stock aliased ProjMPO with dense mult inflated to M around `mid` ----
# Pads the bare H's three mid links (mid-1,mid,mid+1) → sandwich with the SAME
# projector P → genuine aliased PHP whose mult is M on H1's left, the H1|H2 bond,
# and H2's right. Reuses the real sandwich + product path (no hand-built storage).
function inflated_aliased_PH(P_sparse::MPO, H_bare::MPO, psi::MPS, mid::Int, M::Int)
    Hpad     = pad_mpo_bonds(H_bare, M, Set([mid-1, mid, mid+1]))
    Hpad_ali = sandwich_mpo(P_sparse, Hpad; output_hint=:aliased)
    fuse_sparse_links!(Hpad_ali); prepermute_aliased_mpo!(Hpad_ali)
    PH = ITensorMPS.ProjMPO(Hpad_ali)
    ITensorMPS.position!(PH, psi, mid)
    return PH
end

function main()
    N = 12; spin_sector = 1.0; states = 2N + 2
    sites = siteinds("S=1", states)
    os = OpSum()
    for j in 1:N+1; os += "Sz", 2j-1, "Sz", 2j; end
    for j in 1:N; os += "Sx", 2j-1, "Sx", 2j+2; os += "Sy", 2j, "Sy", 2j+1; end
    os2 = OpSum[]
    for j in 1:N
        t = OpSum()
        t += 0.5, "Id", 2j-1, "Id", 2j, "Id", 2j+1, "Id", 2j+2
        t += spin_sector*0.5, "exp(i*pi*Sy)", 2j-1, "exp(i*pi*Sx)", 2j, "exp(i*pi*Sx)", 2j+1, "exp(i*pi*Sy)", 2j+2
        push!(os2, t)
    end
    ConsOps1 = MPO[]; ConsOps2 = MPO[]
    for j in 1:N
        op = MPO(os2[j], sites, [2j-1, 2j, 2j+1, 2j+2]); clean!(op); push!(ConsOps1, op)
        push!(ConsOps2, MPO(os2[j], sites))
    end
    P_sparse = multiplyVecMPOtoMPO(ConsOps1); H = MPO(os, sites)
    println("[build aliased PHP]")
    H_ali = sandwich_mpo(P_sparse, H; output_hint=:aliased)
    fuse_sparse_links!(H_ali); prepermute_aliased_mpo!(H_ali)
    Random.seed!(42); psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end
    println("[dmrg to reach bd=40]")
    res = dmrg(H_ali, psi0; nsweeps=4, maxdim=40, mindim=40, cutoff=1e-12, outputlevel=0, use_early_exit=false)
    psi = res[2]
    PH = ITensorMPS.ProjMPO(H_ali); mid = div(length(psi),2)
    ITensorMPS.position!(PH, psi, mid)

    v   = psi[mid] * psi[mid+1]
    ctx = build_matvec_context(PH, mid)
    phiarr = array(permute(v, ctx.Lket, ctx.Rket, ctx.s2_ind, ctx.s1_ind))
    @printf("\nreal mults: dl1=%d dr1=dl2=%d dr2=%d   (d1=%d d2=%d, chiL=%d chiR=%d, n_l1=%d n_m=%d n_r2=%d)\n",
            ctx.dl1, ctx.dr1, ctx.dr2, ctx.d1, ctx.d2, ctx.chiL_ket, ctx.chiR_ket, ctx.n_l1, ctx.n_m, ctx.n_r2)

    # ---- validate the dense baseline against stock, at real mult ----
    wr   = product(PH, v)
    wf   = fused_matvec_batched(ctx, v, build_batched_scratch(ctx))   # real-mult oracle
    opR  = build_dense_op(ctx, phiarr, ctx.dl1, ctx.dr1, ctx.dr2)
    dR   = dense_array(opR)
    wr_a = array(permute(wr, noprime(ctx.Lbra), noprime(ctx.s1_ind), noprime(ctx.s2_ind), noprime(ctx.Rbra)))
    @printf("dense-baseline vs stock  rel err = %.3e  %s\n", norm(dR.-wr_a)/norm(wr_a), norm(dR.-wr_a)/norm(wr_a)<1e-9 ? "PASS" : "FAIL")
    @printf("batched      vs stock    rel err = %.3e\n", norm(wf - wr)/norm(wr))

    # ---- storage-order check: does the fused output match v / stock's order? A mismatch
    # means KrylovKit will permute Hv when combining with v downstream (relocating, not
    # removing, the cost). id() so we compare the ACTUAL index objects, not just dims.
    ord(t) = [(ITensors.id(i)%1000, dim(i), string(tags(i)), plev(i)) for i in inds(t)]
    println("v       order = ", ord(v))
    println("batched order = ", ord(wf))
    println("stock   order = ", ord(wr))
    println("batched order == v order?     ", collect(inds(wf)) == collect(inds(v)))
    println("batched order == stock order? ", collect(inds(wf)) == collect(inds(wr)))
    println("stock order == v order?     ", collect(inds(wr)) == collect(inds(v)))

    perm4(t) = array(permute(t, noprime(ctx.Lbra), noprime(ctx.s1_ind), noprime(ctx.s2_ind), noprime(ctx.Rbra)))
    # 3-way at each mult: stock aliased product(PHm,v) vs dense PHP (build_dense_op)
    # vs fused batched. PHm is the RE-SANDWICHED aliased ProjMPO inflated to this mult
    # (identical operator by zero-pad), so all three compute the same map — checkable.
    function bench(label, Ml, Mmid, Mr, PHm; reps=200)
        ci = inflate_ctx(ctx, Ml, Mmid, Mr)
        sci = build_batched_scratch(ci)
        ssi = build_streaming_scratch(ci)
        op = build_dense_op(ctx, phiarr, Ml, Mmid, Mr)
        wfib = fused_matvec_batched(ci, v, sci)
        wfis = fused_matvec_streaming(ci, v, ssi)    # streaming variant
        wsa  = product(PHm, v)                       # stock aliased at this mult
        dai  = dense_array(op)   # already in (Lbra, s1', s2', Rbra) axis order
        rel_b  = norm(perm4(wfib) .- dai)/norm(dai)
        rel_s  = norm(perm4(wfis) .- dai)/norm(dai)  # streaming vs dense (must PASS)
        rel_sa = norm(perm4(wsa)  .- dai)/norm(dai)  # stock-aliased vs dense (must PASS)
        rel_real_b = norm(wfib - wf)/norm(wf)        # batched must match real result
        rel_sb     = norm(wfis - wfib)/norm(wfib)    # streaming must match batched exactly
        # order consistency of batched output vs input v (should be TRUE by construction)
        ord_ok = collect(inds(wfib)) == collect(inds(v))
        # deterministic GEMM/FLOP census (contention-immune) + intermediate sizes
        reset_gemm_count!(); _GEMM_COUNT[] = true; fused_matvec_batched(ci, v, sci); _GEMM_COUNT[] = false
        ngemm = _GEMM_N[]; flops = _GEMM_FLOPS[]
        reset_gemm_count!(); _GEMM_COUNT[] = true; fused_matvec_streaming(ci, v, ssi); _GEMM_COUNT[] = false
        ngemm_s = _GEMM_N[]; flops_s = _GEMM_FLOPS[]
        x2_mb  = length(ci.x2)  * sizeof(eltype(ci.x2))  / 1e6
        x2c_mb = length(ssi.x2c) * sizeof(eltype(ssi.x2c)) / 1e6
        for _ in 1:5; fused_matvec_batched(ci,v,sci); fused_matvec_streaming(ci,v,ssi); dense_matvec(op); product(PHm,v); end
        GC.gc(); tfb = @elapsed for _ in 1:reps; fused_matvec_batched(ci,v,sci); end
        GC.gc(); tfs = @elapsed for _ in 1:reps; fused_matvec_streaming(ci,v,ssi); end
        GC.gc(); td  = @elapsed for _ in 1:reps; dense_matvec(op); end
        GC.gc(); ts  = @elapsed for _ in 1:reps; product(PHm,v); end
        @printf("\n[%s] mult=(%d,%d,%d)  aliased-vs-dense=%.2e  batched-vs-dense=%.2e  streaming-vs-dense=%.2e  streaming-vs-batched=%.2e  order==v? %s\n",
                label,Ml,Mmid,Mr,rel_sa,rel_b,rel_s,rel_sb,ord_ok)
        @printf("[%s] stock-aliased = %.1f us   dense = %.1f us   fused(batched) = %.1f us   fused(STREAMING) = %.1f us\n",
                label, 1e6*ts/reps, 1e6*td/reps, 1e6*tfb/reps, 1e6*tfs/reps)
        @printf("[%s] RATIOS  streaming/aliased = %.2fx   streaming/dense = %.2fx   streaming/batched = %.2fx   batched/aliased = %.2fx\n",
                label, tfs/ts, tfs/td, tfs/tfb, tfb/ts)
        # deterministic: GEMM count, FLOPs (2MNK conv), achieved GFLOP/s vs BLAS peak ref
        gfps  = flops   / (1e9 * tfb / reps)
        gfps_s = flops_s / (1e9 * tfs / reps)
        @printf("[%s] batched   GEMMs=%d FLOPs=%.3e  x2 =%.1f MB  achieved=%.1f GFLOP/s (%.0f%% of peak %.1f)\n",
                label, ngemm, flops, x2_mb, gfps, 100*gfps/PEAK_GFPS[], PEAK_GFPS[])
        @printf("[%s] STREAMING GEMMs=%d FLOPs=%.3e  x2c=%.1f MB  achieved=%.1f GFLOP/s (%.0f%% of peak %.1f)\n",
                label, ngemm_s, flops_s, x2c_mb, gfps_s, 100*gfps_s/PEAK_GFPS[], PEAK_GFPS[])
        println("  [batched per-step]")
        _FUSED_PROF[] = true; reset_fused_prof!(); for _ in 1:reps; fused_matvec_batched(ci,v,sci); end; _FUSED_PROF[] = false
        report_fused_prof!(reps)
        println("  [STREAMING per-step]")
        _FUSED_PROF[] = true; reset_fused_prof!(); for _ in 1:reps; fused_matvec_streaming(ci,v,ssi); end; _FUSED_PROF[] = false
        report_fused_prof!(reps)
    end

    # BLAS peak reference (ComplexF64, same 2MNK FLOP convention as _cgemm!) so the
    # per-kernel "achieved GFLOP/s" is comparable: low % ⇒ memory/overhead-bound.
    let n = 2000
        A = rand(ComplexF64, n, n); B = rand(ComplexF64, n, n); C = similar(A)
        LinearAlgebra.mul!(C, A, B); GC.gc()
        t = @elapsed for _ in 1:5; LinearAlgebra.mul!(C, A, B); end
        PEAK_GFPS[] = 5 * 2 * n^3 / (1e9 * t)
        @printf("[BLAS peak ref] %dx%d cgemm (2MNK conv) = %.1f GFLOP/s\n", n, n, PEAK_GFPS[])
    end

    println("\n[build inflated aliased ProjMPOs (re-sandwich padded H)]")
    PH20 = inflated_aliased_PH(P_sparse, H, psi, mid, 20)
    PH40 = inflated_aliased_PH(P_sparse, H, psi, mid, 40)
    bench("real",  ctx.dl1, ctx.dr1, ctx.dr2, PH)
    bench("M=20",  20, 20, 20, PH20)
    bench("M=40",  40, 40, 40, PH40)

    # ---- stock aliased per-step @ M=40 (built-in TimerOutputs) so we can compare
    # its step breakdown against the fused kernel's [batched per-step] above. ----
    reps = 200
    println("\n[stock aliased per-step @ M=40]")
    reset_timer!(ITensorMPS.PROJMPO_TIMER); reset_timer!(SparseBackends.TIMER)
    for _ in 1:reps; product(PH40, v); end
    println("--- PROJMPO_TIMER (per tensor in L·H1·H2·R chain) ---")
    print_timer(ITensorMPS.PROJMPO_TIMER); println()
    println("--- SparseBackends.TIMER (within-contract: setup/labels/kernel/to_dense/recast) ---")
    print_timer(SparseBackends.TIMER); println()
    println("\ndone.")
end
main()
