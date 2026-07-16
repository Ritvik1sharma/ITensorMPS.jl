# Head-to-head for the PARTIAL fused matvec (fuse ONLY the two aliased H legs,
# stock steps 2+3; env legs 1 & 4 stay stock) vs the standard aliased PHP
# dense-ψ matvec (stock product(PH, v)) on the REAL aliased PHP for KL N=12,
# bd=40 — same setup as test_fused_real.jl / test_check_working_aliased.jl.
#
#   standard : product(PH, v)                                  (full 4-leg stock aliased matvec)
#   partial  : Renv · matvec_partial_fused(ctx, Lenv·v)        (stock env legs + fused H legs)
#
# matvec_partial_fused / build_partial_matvec_context / partial_matvec_out_inds
# now live in the ITensorMPS module (added to the include list), so we reference
# them qualified rather than include the file into Main.

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random, Printf
import ITensorMPS: lproj, rproj, product,
                   build_partial_matvec_context, matvec_partial_fused, matvec_partial_fused_full,
                   partial_matvec_out_inds, _PF_PROF, reset_partial_prof!, report_partial_prof!

BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_BLAS_THREADS", "1")))
println("[BLAS threads pinned to ", BLAS.get_num_threads(), "]")

include("/home/ritvik/temp/temp/edited_packages/test_sparse_ham/aliased_helpers.jl")

# ---- builders (copied verbatim from test_fused_real.jl) ----
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

function main()
    N = 12; spin_sector = 1.0
    states = 2N + 2
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
        op = MPO(os2[j], sites, [2j-1, 2j, 2j+1, 2j+2]); clean!(op)
        push!(ConsOps1, op); push!(ConsOps2, MPO(os2[j], sites))
    end
    P_sparse = multiplyVecMPOtoMPO(ConsOps1)
    H        = MPO(os, sites)

    println("[build aliased PHP]")
    H_ali = sandwich_mpo(P_sparse, H; output_hint=:aliased)
    fuse_sparse_links!(H_ali)
    prepermute_aliased_mpo!(H_ali)

    Random.seed!(42)
    psi0 = random_mps(sites)
    for j in 1:N; psi0 = replaceprime(ConsOps2[j] * psi0, 1 => 0); normalize!(psi0); end

    _md = parse(Int, get(ENV, "BENCH_MAXDIM", "40"))
    _ns = parse(Int, get(ENV, "BENCH_NSWEEPS", "4"))
    println("[dmrg to reach bd=$_md, nsweeps=$_ns]")
    res = dmrg(H_ali, psi0; nsweeps=_ns, maxdim=_md, mindim=_md, cutoff=1e-12,
               outputlevel=1, use_early_exit=false)
    psi = res[2]

    PH = ITensorMPS.ProjMPO(H_ali)
    Lsites = length(psi)
    mid = div(Lsites, 2)
    ITensorMPS.position!(PH, psi, mid)
    println("\nsystem: $Lsites sites, mid=$mid, bond dims around mid: ",
            [dim(commonind(psi[i], psi[i+1])) for i in mid-3:mid+2])

    println("\n== PARTIAL fused MATVEC (fuse stock steps 2+3; env legs stock) vs stock product(PH,v) ==")
    v   = psi[mid] * psi[mid+1]                    # 2-site wavefunction spanning sites mid,mid+1
    ctx = build_partial_matvec_context(PH, mid)
    oi  = partial_matvec_out_inds(ctx, :bulk)

    # ---- step-B/C template dedup reuse (same probe as test_fused_real) ----
    let
        function reuse(stepM, tag)
            cells=0; totrec=0; totuniq=0; maxrec=0
            for cell in stepM
                isempty(cell) && continue
                cells += 1; r = length(cell); u = length(unique(x->(x.src_pos, x.tid), cell))
                totrec += r; totuniq += u; maxrec = max(maxrec, r)
            end
            @printf("  [%s] nonempty cells=%d  records=%d  Σ distinct-(src,tid)-per-cell=%d  max recs/cell=%d  => %d redundant template-GEMMs (%.2fx)\n",
                    tag, cells, totrec, totuniq, maxrec, totrec-totuniq, totrec/max(totuniq,1))
        end
        reuse(ctx.step1c, "stepB (H1, by mid-ch)")
        reuse(ctx.step2, "stepC (H2)")
        @printf("  templates: H1 n_templates=%d (blksize %d)  H2 n_templates=%d (blksize %d)\n",
                length(ctx.templates1)÷ctx.blksize1, ctx.blksize1,
                length(ctx.templates2)÷ctx.blksize2, ctx.blksize2)
        @printf("  fused-seam dims:  n_ham_left_ch=%d  n_ham_mid_ch=%d  n_ham_right_ch=%d  |  d1=%d d2=%d  |  mid mult dr1=dl2=%d  dr2=%d  |  chiL_bra=%d chiR_ket=%d\n",
                ctx.n_ham_left_ch, ctx.n_ham_mid_ch, ctx.n_ham_right_ch, ctx.d1, ctx.d2, ctx.dr1, ctx.dr2, ctx.chiL_bra, ctx.chiR_ket)
        @printf("  GEMM K:  stepB K=dl1=%d   stepC K=dl2=%d  (skinny-K intrinsic to aliased factorization)\n",
                ctx.dl1, ctx.dl2)
    end

    L = lproj(PH); R = rproj(PH)

    # The kernel now ASSERTS T1 arrives pinned as (dl1, n_ham_left_ch, Lbra, Rket, s2, s1).
    # Simulate the pinned step 1 ('T'-flag Lenv·φ with φ pinned to (Lket,Rket,s2,s1))
    # by permuting L*v into that order BEFORE the kernel — this is the "transpose
    # the input" that a real pinned step 1 would produce natively (zero-copy).
    pin_T1(vv) = permute(L*vv, ctx.dl1_ind, ctx.l1_ind, ctx.Lbra, ctx.Rket, ctx.s2_ind, ctx.s1_ind)

    # ---- ENV-COMPOSITION diagnostic ----
    let
        _tg(t) = [(dim(I), replace(string(tags(I)),"\""=>""), plev(I)) for I in inds(t)]
        T1raw = L * v
        T1p   = pin_T1(v)
        want_T1 = (ctx.dl1_ind, ctx.l1_ind, ctx.Lbra, ctx.Rket, ctx.s2_ind, ctx.s1_ind)
        println("\n  == ENV COMPOSITION ==")
        println("  Lenv inds                 = ", _tg(L))
        println("  v    inds                 = ", _tg(v))
        println("  T1 = Lenv·v  (naive *)    = ", _tg(T1raw))
        println("  T1 PINNED (kernel input)  = ", _tg(T1p))
        println("  kernel WANTS              = ", [(dim(I), replace(string(tags(I)),"\""=>""), plev(I)) for I in want_T1])
        println("  pinned T1 read is NO-OP?    ", collect(inds(T1p)) == collect(want_T1))
        T3 = matvec_partial_fused(ctx, T1p; out_inds=oi)
        println("  T3 (kernel out)           = ", _tg(T3))
        println("  Renv inds                 = ", _tg(R))
        _sh = commoninds(T3, R)
        println("  T3·Renv shared legs       = ", [(dim(I), replace(string(tags(I)),"\""=>""), plev(I)) for I in _sh],
                "   (contract dim K = ", prod(dim(I) for I in _sh; init=1), ")")
    end

    # noprime to match `product` (= contract(P,v) THEN noprime): contract(P,v)
    # yields the fully-primed bra result (L/R give primed links, H's give primed
    # bra sites); product noprimes it back into v's space.
    #
    # ROUND-TRIP permute (the one irreducible reorder): the strided read forces
    # φ's right bond into an interior position, but step 4's GEMM can only place
    # the Renv leg at an end — so a single GEMM can't emit Hv in φ's order. We
    # therefore reorder Hv → v's index order once at the end. This is a χ²·9
    # tensor (no channel/mult legs), ~16× smaller than the T1 permute we killed,
    # and a bd-independent static pattern (the analog of static_output_perm_dense
    # not being all-identity). With it, order(Hv)==order(v) — Krylov-consistent.
    partial_full(vv) = permute(noprime(R * matvec_partial_fused(ctx, pin_T1(vv); out_inds=oi)),
                               inds(vv)...)

    wf = partial_full(v)
    wr = product(PH, v)
    rel = norm(wf - wr) / norm(wr)
    @printf("\n  full-matvec correctness rel err = %.3e  %s\n", rel, rel < 1e-9 ? "PASS" : "*** FAIL ***")

    # per-(s1,s2) physical block relative error (localize any breakage)
    let s1i = noprime(ctx.s1_ind), s2i = noprime(ctx.s2_ind),
        lk = filter(i -> hastags(i, "Link"), collect(inds(wr)))
        Lb, Rb = lk[1], lk[2]
        af = Array(wf, Lb, s1i, s2i, Rb); ar = Array(wr, Lb, s1i, s2i, Rb)
        println("  per-(s1,s2) block rel err:")
        for a in 1:dim(s1i)
            row = [ (n=norm(@view(ar[:,a,b,:])); n<1e-14 ? 0.0 :
                     norm(@view(af[:,a,b,:]) .- @view(ar[:,a,b,:]))/n) for b in 1:dim(s2i) ]
            println("    s1=$a: ", round.(row, sigdigits=3))
        end
    end

    # index-order consistency: after the round-trip permute, does the FULL partial
    # matvec output land in v's order? (This is the Krylov round-trip: input order
    # → identical output order, so no per-iteration permute in the eigensolver.)
    println("\n  index-order check (Krylov round-trip):")
    println("    v  inds = ", [(dim(I), string(tags(I)), plev(I)) for I in inds(v)])
    println("    wf inds = ", [(dim(I), string(tags(I)), plev(I)) for I in inds(wf)])
    _rt = collect(inds(wf)) == collect(inds(v))
    println("    ROUND TRIP  order(Hv)==order(v) ? ", _rt, _rt ? "  ✅" : "  ✗")

    # ---- timing ----
    T1 = pin_T1(v)                                    # precomputed PINNED step-1 output (kernel-isolated input)
    for _ in 1:5; partial_full(v); product(PH, v); matvec_partial_fused(ctx, T1; out_inds=oi); end
    reps = 200
    GC.gc(); tf_full = @elapsed for _ in 1:reps; partial_full(v); end
    GC.gc(); tf_ker  = @elapsed for _ in 1:reps; matvec_partial_fused(ctx, T1; out_inds=oi); end
    GC.gc(); tr      = @elapsed for _ in 1:reps; product(PH, v); end
    a_full = @allocated partial_full(v)
    a_ker  = @allocated matvec_partial_fused(ctx, T1; out_inds=oi)
    a_r    = @allocated product(PH, v)
    @printf("\n  standard  product(PH,v)      = %8.1f us  (%d B/call)\n", 1e6*tr/reps, a_r)
    @printf("  partial   full (env+fused H) = %8.1f us  (%d B/call)   partial/standard = %.2fx  (want < 1)\n",
            1e6*tf_full/reps, a_full, tf_full/tr)
    @printf("  partial   kernel-only (H2+3) = %8.1f us  (%d B/call)   [on PINNED T1; strided read, 0 permute]\n",
            1e6*tf_ker/reps, a_ker)

    # ---- per-step breakdown of the kernel (T1-read / stepB / stepC) ----
    _PF_PROF[] = true; reset_partial_prof!()
    for _ in 1:reps; matvec_partial_fused(ctx, T1; out_inds=oi); end
    _PF_PROF[] = false
    report_partial_prof!(reps)

    # ================================================================
    # FULL WRAPPER (owns all 4 legs via explicit BLAS — NO ITensors *):
    # step 1 gemm 'T' + kernel (loop-swap core) + step 4 gemm 'N' + round-trip
    # permute. φ arrives pinned (Lket,Rket,s2,s1) — the DMRG φ-pin; simulate by
    # permuting v. On battery read the RATIO, not absolute µs.
    # ================================================================
    println("\n  == FULL WRAPPER (explicit-BLAS steps 1&4, no *) ==")
    φp   = permute(v, ctx.Lket, ctx.Rket, ctx.s2_ind, ctx.s1_ind)   # pinned φ (done once = DMRG pin)
    wF   = matvec_partial_fused_full(ctx, φp)
    relF = norm(wF - wr) / norm(wr)
    @printf("  full-wrapper correctness rel err = %.3e  %s\n", relF, relF < 1e-9 ? "PASS" : "*** FAIL ***")
    _rtF = collect(inds(wF)) == collect(inds(φp))
    println("  ROUND TRIP  order(Hv)==order(φ_pinned) ? ", _rtF, _rtF ? "  ✅" : "  ✗")
    for _ in 1:5; matvec_partial_fused_full(ctx, φp); end
    GC.gc(); tF = @elapsed for _ in 1:reps; matvec_partial_fused_full(ctx, φp); end
    aF = @allocated matvec_partial_fused_full(ctx, φp)
    @printf("  full-wrapper (steps 1-4)     = %8.1f us  (%d B/call)   fullwrap/standard = %.2fx  (want < 1)\n",
            1e6*tF/reps, aF, tF/tr)
    _PF_PROF[] = true; reset_partial_prof!()
    for _ in 1:reps; matvec_partial_fused_full(ctx, φp); end
    _PF_PROF[] = false
    report_partial_prof!(reps)

    # ================================================================
    # 5× REPEAT (throttle-averaged): each run measures standard / kernel /
    # full-wrapper INTERLEAVED & back-to-back so all three see the same throttle
    # state; report per-run ratios + the mean-of-ratios (robust to per-run drift).
    # ================================================================
    println("\n  == 5× REPEAT (throttle-averaged; battery → trust the RATIO) ==")
    nrun = 5
    tr_v = Float64[]; tk_v = Float64[]; tF_v = Float64[]; rk_v = Float64[]; rF_v = Float64[]
    for run in 1:nrun
        GC.gc(); _tr = @elapsed for _ in 1:reps; product(PH, v); end
        GC.gc(); _tk = @elapsed for _ in 1:reps; matvec_partial_fused(ctx, T1; out_inds=oi); end
        GC.gc(); _tF = @elapsed for _ in 1:reps; matvec_partial_fused_full(ctx, φp); end
        push!(tr_v, _tr); push!(tk_v, _tk); push!(tF_v, _tF)
        push!(rk_v, _tk/_tr); push!(rF_v, _tF/_tr)
        @printf("  run %d:  standard %7.1f us | kernel %7.1f us (%.2fx) | fullwrap %7.1f us (%.2fx)\n",
                run, 1e6*_tr/reps, 1e6*_tk/reps, _tk/_tr, 1e6*_tF/reps, _tF/_tr)
    end
    _mean(x) = sum(x)/length(x)
    _std(x)  = (m=_mean(x); sqrt(sum((xi-m)^2 for xi in x)/length(x)))
    @printf("\n  AVG over %d runs:\n", nrun)
    @printf("    standard  = %7.1f us\n", 1e6*_mean(tr_v)/reps)
    @printf("    kernel    = %7.1f us   kernel/standard   = %.3f ± %.3f  (mean of per-run ratios)\n",
            1e6*_mean(tk_v)/reps, _mean(rk_v), _std(rk_v))
    @printf("    fullwrap  = %7.1f us   fullwrap/standard = %.3f ± %.3f  (mean of per-run ratios)\n",
            1e6*_mean(tF_v)/reps, _mean(rF_v), _std(rF_v))

    println("\ndone.")
end

main()
