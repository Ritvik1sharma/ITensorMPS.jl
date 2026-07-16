# ============================================================
# PARTIAL fused matvec: fuse ONLY the two aliased H legs (stock steps 2 & 3).
#
# The 2-site effective-H matvec is a 4-leg chain
#     step 1  T1 = Lenv·v        (dense env leg — kept STOCK, big-K GEMM)
#     step 2  T2 = T1·H1         ┐ these two aliased H legs are fused here:
#     step 3  T3 = T2·H2         ┘ the intermediate T2 NEVER materializes.
#     step 4  Hv = T3·Renv       (dense env leg — kept STOCK, big-K GEMM)
#
# `matvec_partial_fused(ctx, T1) -> T3` consumes the dense T1 that stock step 1
# already produced and returns the dense T3 that stock step 4 will consume. The
# mid bond (channel n_ham_mid_ch × mult dr1=dl2) is contracted immediately inside the
# compact middle accumulator `x2`, so the ~44 GB dense T2 (and its forced
# step-2 permute_back) are never formed. Template GEMMs and the mid-bond sum are
# identical to the stock aliased path — only the accumulation order differs
# (commutative) ⇒ energy bit-identical.
#
# This is the middle ground between the stock per-leg aliased path and the full
# 4-leg `fused_matvec` (fused_mpo_helpers_matvec.jl): the env legs stay as
# efficient stock big-K contractions, so there is NO new skinny-K env penalty —
# only the two H legs (whose K = mid mult dr ∈ {4,5} is already what the stock
# aliased kernel contracts per template) are fused.
#
# This file is a self-contained SIBLING of fused_mpo_helpers_matvec.jl. The
# reusable pieces (record struct, classifier, template view) are copied/adapted
# with `_pf`/`Partial` names so the full-fusion WIP file stays untouched and the
# two never collide if both are ever included.
# ============================================================
using LinearAlgebra: BLAS

# One scatter record: template `tid` (scaled by `c`) written from source sparse
# channel `src_pos` into destination sparse channel `dst_pos`, routing bra
# physical `out_phys`. (Same shape as StepRecord in the full-fusion file.)
struct PartialStepRecord{T}
    src_pos::Int
    dst_pos::Int
    out_phys::Int
    tid::Int
    c::T
end

struct PartialMatvecContext{T}
    d1::Int; d2::Int
    dl1::Int; dr1::Int; dl2::Int; dr2::Int
    dl1_fast::Bool; dl2_fast::Bool     # is the contracted (left) mult the FAST stored dense axis?
    chiL_bra::Int; chiR_ket::Int
    chiL_ket::Int; chiR_bra::Int       # env-contracted MPS bonds (step 1 / step 4)
    n_ham_left_ch::Int; n_ham_mid_ch::Int; n_ham_right_ch::Int

    templates1::Vector{T}
    templates2::Vector{T}
    blksize1::Int
    blksize2::Int

    step2::Matrix{Vector{PartialStepRecord{T}}}   # (n_ham_mid_ch, d2) -> H2 records (stepC)
    step1c::Matrix{Vector{PartialStepRecord{T}}}  # (n_ham_mid_ch, d1) -> H1 records bucketed by mid-CHANNEL,
                                                  #   sorted (src_pos=lpos, tid) so buf1 dedups per cell.
                                                  #   Drives the per-channel loop (cache-resident x2c). (stepB)

    # ---- preallocated per-call scratch (reused across Krylov iterations) ----
    # Single-threaded assumption (KrylovKit runs serial). Buffers' CONTENTS are
    # overwritten each call; nothing escapes except a freshly-allocated T3 buffer.
    buf1::Matrix{T}           # (dr1, chiL_bra*chiR_ket*d2)                        step B
    x2c::Array{T,5}           # (dr1, chiL_bra, chiR_ket, d1[s1'], d2[s2]) ONE-CHANNEL accumulator
                              #   — ~1/n_ham_mid_ch the size of the old full x2, stays L2/L3-resident.

    # ---- pre-arranged env matrices (v-independent → built once/bond) for the
    #      full-wrapper's explicit-BLAS step 1 & step 4 (no ITensors * permutes) ----
    Lenv_mat::Matrix{T}       # (chiL_ket, dl1·n_ham_left_ch·chiL_bra)  — step 1: gemm('T','N', Lenv_mat, φ_mat)
    Renv_mat::Matrix{T}       # (chiR_ket·dr2·n_ham_right_ch, chiR_bra)  — step 4: gemm('N','N', T3gather, Renv_mat)
    T3g_buf::Array{T,6}       # (chiL_bra, d1, d2, chiR_ket, dr2, n_ham_right_ch) — step-4 gather dest (permutedims! target)

    # ---- indices for the per-call T1 read and the T3 build ----
    dl1_ind; l1_ind; Lbra; Rket; s1_ind; s2_ind   # T1 read: (dl1, n_ham_left_ch, Lbra, Rket, s2, s1)
    dr2_ind; r2_ind                                 # T3 open (Renv-facing) legs
    Lket; Rbra                                      # env MPS bonds (step1 contracts Lket; step4 outputs Rbra)
end

# Classify H's 6 legs given the SHARED middle bond (channel + mult) from
# commoninds(H1,H2). Fixes left/right UNAMBIGUOUSLY: for H1 the shared bond is
# its RIGHT link; for H2 its LEFT link. (Adapted verbatim from the full-fusion
# _classify_mpo_tensor_static.)
function _pf_classify_mpo_tensor_static(H::ITensor, mid_channel, mid_mult, mid_is_right::Bool)
    Hw = ITensors.get_external_storage(H)
    hinds = inds(Hw)
    A = Hw.aliased
    sparse_inds = hinds[1:4]
    dense_inds  = hinds[5:6]

    s_pos  = findfirst(i -> hastags(i, "Site") && plev(i) == 0, sparse_inds)
    sp_pos = findfirst(i -> hastags(i, "Site") && plev(i) != 0, sparse_inds)
    link_positions = [i for i in 1:4 if hastags(sparse_inds[i], "Link")]
    @assert length(link_positions) == 2 "expected exactly 2 sparse Link axes, got $(length(link_positions))"
    mid_lp = findfirst(i -> ITensors.id(sparse_inds[i]) == ITensors.id(mid_channel), link_positions)
    @assert mid_lp !== nothing "shared middle channel not found among H's sparse links"
    mid_pos   = link_positions[mid_lp]
    other_pos = only(setdiff(link_positions, (mid_pos,)))
    l_pos, r_pos = mid_is_right ? (other_pos, mid_pos) : (mid_pos, other_pos)

    mid_di   = ITensors.id(dense_inds[1]) == ITensors.id(mid_mult) ? 1 : 2
    other_di = 3 - mid_di
    dl_ind, dr_ind = mid_is_right ? (dense_inds[other_di], dense_inds[mid_di]) :
                                     (dense_inds[mid_di],   dense_inds[other_di])

    return (A=A, l_pos=l_pos, s_pos=s_pos, sp_pos=sp_pos, r_pos=r_pos,
            l_ind=sparse_inds[l_pos], s_ind=sparse_inds[s_pos],
            sp_ind=sparse_inds[sp_pos], r_ind=sparse_inds[r_pos],
            dl_ind=dl_ind, dr_ind=dr_ind, dl=dim(dl_ind), dr=dim(dr_ind))
end

@inline _pf_template(templates, blksize, tid, dl, dr) =
    reshape(view(templates, (tid-1)*blksize+1 : tid*blksize), dl, dr)

# ============================================================
# Build the v-independent context for the bond (si, si+1). Trimmed vs
# build_matvec_context: NO Larr/Rarr/Rbatch/Abuf/Dout — the env legs stay stock,
# so we never touch lproj/rproj DATA (only their MPS-bond index objects/dims).
# ============================================================
function build_partial_matvec_context(P, si::Int)
    H1, H2 = P.H[si], P.H[si+1]

    # Shared middle bond = commoninds(H1,H2) = one sparse channel + one dense mult.
    mids = commoninds(H1, H2)
    H1w  = ITensors.get_external_storage(H1)
    h1_sparse_ids = Set(ITensors.id(i) for i in inds(H1w)[1:4])
    mid_channel = only(filter(i ->   ITensors.id(i) in h1_sparse_ids, mids))
    mid_mult    = only(filter(i -> !(ITensors.id(i) in h1_sparse_ids), mids))

    cls1 = _pf_classify_mpo_tensor_static(H1, mid_channel, mid_mult, true)   # shared = H1's RIGHT
    cls2 = _pf_classify_mpo_tensor_static(H2, mid_channel, mid_mult, false)  # shared = H2's LEFT
    s1_ind, s2_ind = cls1.s_ind, cls2.s_ind
    d1, d2 = dim(s1_ind), dim(s2_ind)
    @assert cls1.r_ind === cls2.l_ind "middle channel not consistent between H1(right)/H2(left)"
    @assert cls1.dr_ind === cls2.dl_ind "middle mult not consistent between H1(right)/H2(left)"
    @assert cls1.dr == cls2.dl "middle-bond mult mismatch: H1.dr=$(cls1.dr) vs H2.dl=$(cls2.dl)"

    # Env MPS-bond index objects (dims + Index handles only — env DATA stays stock).
    # T1 = Lenv·v carries H1's left channel (l_ind) + left mult (dl_ind) + the bra
    # MPS bond (Lbra) + v's right ket MPS bond (Rket). We only need Lbra and Rket.
    L, R = lproj(P), rproj(P)
    h1ids = Set(ITensors.id(i) for i in inds(H1))
    h2ids = Set(ITensors.id(i) for i in inds(H2))
    L_mps = filter(i -> !(ITensors.id(i) in h1ids), collect(inds(L)))
    R_mps = filter(i -> !(ITensors.id(i) in h2ids), collect(inds(R)))
    Lbra = only(filter(i -> plev(i) != 0, L_mps))   # bra MPS bond (kept in T1 & T3)
    Rket = only(filter(i -> plev(i) == 0, R_mps))   # ket MPS bond (kept in T1 & T3; Renv contracts it)
    Lket = only(filter(i -> plev(i) == 0, L_mps))   # ket MPS bond (step 1 contracts it with φ)
    Rbra = only(filter(i -> plev(i) != 0, R_mps))   # bra MPS bond (surviving output of step 4)

    T = promote_type(eltype(cls1.A.templates), eltype(cls2.A.templates), eltype(L), eltype(R))

    n_ham_left_ch = dim(cls1.l_ind)      # H1 left channel  (T1's sparse-link axis)
    n_ham_mid_ch  = dim(mid_channel)     # shared middle channel
    n_ham_right_ch = dim(cls2.r_ind)      # H2 right channel (T3's Renv-facing sparse link)
    chiL_bra = dim(Lbra)
    chiR_ket = dim(Rket)
    chiL_ket = dim(Lket)
    chiR_bra = dim(Rbra)
    dl1, dr1, dl2, dr2 = cls1.dl, cls1.dr, cls2.dl, cls2.dr

    # Pre-arranged env matrices (one-time, v-independent) for explicit-BLAS steps
    # 1 & 4 in the full wrapper. Lenv → (Lket, dl1, n_ham_left_ch, Lbra) so step 1's 'T'-GEMM
    # emits T1 in the strided-readable pinned order; Renv → (Rket, dr2, n_ham_right_ch, Rbra)
    # so its contracted legs are leading (they already are). These permutes are
    # amortized across all Krylov matvecs.
    Lenv_mat = reshape(T.(array(permute(L, Lket, cls1.dl_ind, cls1.l_ind, Lbra))),
                       chiL_ket, dl1*n_ham_left_ch*chiL_bra)
    Renv_mat = reshape(T.(array(permute(R, Rket, cls2.dr_ind, cls2.r_ind, Rbra))),
                       chiR_ket*dr2*n_ham_right_ch, chiR_bra)

    # Step records indexed by RAW channel/site values (no compaction → no pos/axis mismatch).
    # stepC (H2): step2[m, s2] holds the H2 records for mid-channel m and ket s2.
    step2 = [PartialStepRecord{T}[] for _ in 1:n_ham_mid_ch, _ in 1:d2]
    for (i, key) in enumerate(cls2.A.keys)
        mval, sval, spval, rval = key[cls2.l_pos], key[cls2.s_pos], key[cls2.sp_pos], key[cls2.r_pos]
        push!(step2[mval, sval], PartialStepRecord{T}(mval, rval, Int(spval), cls2.A.alias_ids[i], T(cls2.A.scalars[i])))
    end

    # Channel-bucketed records for the per-channel kernel loop: step1c[m, s1] holds
    # the H1 records routing INTO mid-channel m. Sorted by (src_pos=lpos, tid) so
    # buf1 = templateᵀ·M1mat_{lpos} is recomputed only when (lpos, tid) changes
    # within a channel cell (per-channel dedup; the cross-channel dedup is traded
    # away in exchange for a cache-resident one-channel accumulator).
    step1c = [PartialStepRecord{T}[] for _ in 1:n_ham_mid_ch, _ in 1:d1]
    for (i, key) in enumerate(cls1.A.keys)
        lval, sval, spval, mval = key[cls1.l_pos], key[cls1.s_pos], key[cls1.sp_pos], key[cls1.r_pos]
        push!(step1c[mval, sval], PartialStepRecord{T}(lval, mval, Int(spval), cls1.A.alias_ids[i], T(cls1.A.scalars[i])))
    end
    for cell in step1c
        sort!(cell, by = r -> (r.src_pos, r.tid))
    end

    # Is the CONTRACTED (left) mult the FAST stored dense axis? Templates are
    # stored column-major (dense_inds[1] fast); reshape(dl,dr) is only valid when
    # dl===dense_inds[1], else the gemm must transpose (same fix as the env kernel).
    H2w = ITensors.get_external_storage(H2)
    dl1_fast = cls1.dl_ind === inds(H1w)[5]
    dl2_fast = cls2.dl_ind === inds(H2w)[5]

    # ---- preallocate all per-call scratch once (reused across Krylov iterations) ----
    buf1 = Matrix{T}(undef, dr1, chiL_bra*chiR_ket*d2)
    x2c  = Array{T,5}(undef, dr1, chiL_bra, chiR_ket, d1, d2)   # ONE channel — cache-resident
    T3g_buf = Array{T,6}(undef, chiL_bra, d1, d2, chiR_ket, dr2, n_ham_right_ch)  # step-4 gather dest

    # Templates are READ-ONLY GEMM operands → alias H's banks directly when the
    # eltype already matches (no copy); only convert when a promotion is needed.
    tmpl1 = cls1.A.templates isa Vector{T} ? cls1.A.templates : T.(cls1.A.templates)
    tmpl2 = cls2.A.templates isa Vector{T} ? cls2.A.templates : T.(cls2.A.templates)

    return PartialMatvecContext{T}(
        d1, d2, dl1, dr1, dl2, dr2, dl1_fast, dl2_fast,
        chiL_bra, chiR_ket, chiL_ket, chiR_bra, n_ham_left_ch, n_ham_mid_ch, n_ham_right_ch,
        tmpl1, tmpl2, cls1.A.blksize, cls2.A.blksize,
        step2, step1c,
        buf1, x2c,
        Lenv_mat, Renv_mat, T3g_buf,
        cls1.dl_ind, cls1.l_ind, Lbra, Rket, s1_ind, s2_ind,
        cls2.dr_ind, cls2.r_ind,
        Lket, Rbra,
    )
end

# ============================================================
# Per-step profiling (debug only). Toggle _PF_PROF[]=true from the harness, reset
# before the timed loop, report after. When OFF each probe is a single Ref read +
# short-circuit; the GEMMs are untouched. Slots: 1 step1/T1-read | 2 stepB |
# 3 stepC | 4 step4 (full wrapper only). (Independent of _FUSED_PROF.)
# ============================================================
const _PF_PROF   = Ref(false)
const _PF_TIMES  = zeros(Float64, 4)
const _PF_LABELS = ("step1/T1-read", "stepB(H1)", "stepC(H2)", "step4(Renv)")
reset_partial_prof!() = (fill!(_PF_TIMES, 0.0); nothing)
function report_partial_prof!(reps::Int)
    tot = sum(_PF_TIMES)
    tot == 0 && (println("    (no profiling data — was _PF_PROF[] set?)"); return)
    println("    partial-fused per-step breakdown (us/call, % of instrumented time):")
    for i in 1:4
        us  = round(1e6*_PF_TIMES[i]/reps, digits=1)
        pct = round(100*_PF_TIMES[i]/tot, digits=1)
        println("      ", rpad(_PF_LABELS[i], 20), " = ", lpad(string(us), 8), " us  (", pct, "%)")
    end
    println("      ", rpad("Σ instrumented", 20), " = ", lpad(string(round(1e6*tot/reps, digits=1)), 8), " us")
end
@inline _pf_now() = _PF_PROF[] ? Base.time_ns() : UInt64(0)
@inline _pf_add!(slot, t0) = (_PF_PROF[] && (@inbounds _PF_TIMES[slot] += (Base.time_ns()-t0)*1e-9); nothing)

# ============================================================
# Kernel core (steps 2+3): PER-CHANNEL fused H1·H2 on the strided-readable T1
# tensor `T1r` (= reshape of pinned T1 to (dl1, n_ham_left_ch, Lbra·Rket·s2, s1)). Returns
# a fresh T3buf = (dr2, chiL_bra, chiR_ket, d1[s1'], d2[s2'], n_ham_right_ch). Shared by
# both wrappers so the two never diverge.
#
# One mid-channel m at a time so the accumulator x2c (one channel ≈ 1/n_ham_mid_ch the
# full size) stays L2/L3-resident: stepB scatters H1 into x2c, stepC drains it
# into T3 — x2c never spills to RAM. buf1 dedups per (lpos,tid) WITHIN a channel
# cell (cross-channel dedup traded for the cache win; buf1 GEMMs are cheap,
# K=dl1). stepC accumulates straight into T3's contiguous (sp2,r) slice
# (α=rec.c, β=1) — no buf2, no scatter.
#
# BATCH-s1 COLLAPSE: the input ket-s1 is a CONTRACTED index and H2 (stepC) treats
# the bra s1' purely as a spectator (it is folded into stepC's N axis). stepC is
# linear in x2c, so Σ_{s1} stepC(x2c_{s1}) == stepC(Σ_{s1} x2c_{s1}). We therefore
# accumulate ALL input s1 into x2c FIRST (inner s1 loop, no reset between them),
# then run stepC ONCE per channel — instead of once per (s1,m). This removes the
# d1× re-run of stepC (step2[m,s2] is s1-INDEPENDENT: its 144 records were being
# executed d1=3× with ⅔ of each GEMM's N zero). Bit-identical; ~3× fewer stepC
# GEMMs. Same collapse the full-fusion `batched` kernel got ("step C/D ~3×").
# ============================================================
function _pf_core!(ctx::PartialMatvecContext{T}, T1r) where {T}
    T3buf = zeros(T, ctx.dr2, ctx.chiL_bra, ctx.chiR_ket, ctx.d1, ctx.d2, ctx.n_ham_right_ch)
    T3mat = reshape(T3buf, ctx.dr2, ctx.chiL_bra*ctx.chiR_ket*ctx.d1, ctx.d2, ctx.n_ham_right_ch)
    buf1_4 = reshape(ctx.buf1, ctx.dr1, ctx.chiL_bra, ctx.chiR_ket, ctx.d2)
    for m in 1:ctx.n_ham_mid_ch                                         # m = mid channel (processed in isolation)
        # ---- stepB: accumulate EVERY input ket-s1 into x2c (batch-s1 collapse) ----
        _tb = _pf_now()
        fill!(ctx.x2c, zero(T))                                # small ⇒ cache write, not RAM
        any_mass = false
        for s1 in 1:ctx.d1                                     # s1 = H1 KET physical (CONTRACTED)
            recs1 = ctx.step1c[m, s1]
            isempty(recs1) && continue
            any_mass = true
            cur_lpos = 0; cur_tid = 0
            for rec in recs1                                   # sorted by (src_pos=lpos, tid)
                if rec.src_pos != cur_lpos || rec.tid != cur_tid
                    M1mat = @view T1r[:, rec.src_pos, :, s1]   # (dl1, Lbra·Rket·s2) StridedMatrix (LDA=dl1·n_ham_left_ch)
                    blk   = view(ctx.templates1, (rec.tid-1)*ctx.blksize1+1 : rec.tid*ctx.blksize1)
                    if ctx.dl1_fast
                        BLAS.gemm!('T', 'N', one(T), reshape(blk, ctx.dl1, ctx.dr1), M1mat, zero(T), ctx.buf1)
                    else
                        BLAS.gemm!('N', 'N', one(T), reshape(blk, ctx.dr1, ctx.dl1), M1mat, zero(T), ctx.buf1)
                    end
                    cur_lpos = rec.src_pos; cur_tid = rec.tid
                end
                sp1 = rec.out_phys
                @views ctx.x2c[:, :, :, sp1, :] .+= rec.c .* buf1_4   # scale+route into (…, s1'=sp1, …)
            end
        end
        _pf_add!(2, _tb)
        any_mass || continue                                   # no H1 mass into m ⇒ x2c=0 ⇒ stepC=0, skip

        # ---- stepC: drain the FULLY-accumulated x2c into T3 ONCE (s1' folded into N) ----
        _tc = _pf_now()
        for s2 in 1:ctx.d2                                     # s2 = H2 KET physical
            recs2 = ctx.step2[m, s2]
            isempty(recs2) && continue
            x2slice = @view ctx.x2c[:, :, :, :, s2]            # (dr1, chiL_bra, chiR_ket, d1) contiguous
            x2mat   = reshape(x2slice, ctx.dl2, ctx.chiL_bra*ctx.chiR_ket*ctx.d1)   # dl2===dr1
            for rec in recs2
                blk2 = view(ctx.templates2, (rec.tid-1)*ctx.blksize2+1 : rec.tid*ctx.blksize2)
                r, sp2 = rec.dst_pos, rec.out_phys             # r = H2 right channel, sp2 = H2 bra s2'
                Dslice = @view T3mat[:, :, sp2, r]
                if ctx.dl2_fast
                    BLAS.gemm!('T', 'N', rec.c, reshape(blk2, ctx.dl2, ctx.dr2), x2mat, one(T), Dslice)
                else
                    BLAS.gemm!('N', 'N', rec.c, reshape(blk2, ctx.dr2, ctx.dl2), x2mat, one(T), Dslice)
                end
            end
        end
        _pf_add!(3, _tc)
    end
    return T3buf
end

# ============================================================
# FULL wrapper: owns ALL FOUR legs with explicit BLAS (no ITensors *). Takes the
# pinned 2-site φ, does step 1 (Lenv·φ, gemm 'T', permute-free), the fused kernel
# (steps 2+3), step 4 (T3·Renv, gemm 'N'), and the one irreducible round-trip
# reorder → Hv in φ's index order. Returns Hv (noprime'd, order == φ).
#
# ASSERTS φ pinned to (Lket, Rket, s2, s1) (see matvec_partial_fused header for why).
# STEP 4 CAVEAT: the kernel's T1-side ordering forces chiL_bra ahead of chiR_ket,
# but step 4 needs (chiR_ket,dr2,n_ham_right_ch) contiguous — so a single GEMM can't avoid
# gathering T3 (one permutedims). This gather is STRUCTURAL and is exactly what
# stock's step 4 / ITensors * also pays; it is NOT an extra cost vs stock.
# ============================================================
function matvec_partial_fused_full(ctx::PartialMatvecContext{T}, φ::ITensor) where {T}
    # ---- step 1: T1 = Lenv·φ via gemm('T','N'), contract Lket (leading in both) ----
    _tr = _pf_now()
    _wantφ = (ctx.Lket, ctx.Rket, ctx.s2_ind, ctx.s1_ind)
    @assert collect(inds(φ)) == collect(_wantφ) "matvec_partial_fused_full: φ must be pinned as (Lket,Rket,s2,s1); got $(collect(inds(φ)))"
    φraw  = array(φ)
    φarr  = eltype(φraw) === T ? φraw : T.(φraw)
    φ_mat = reshape(φarr, ctx.chiL_ket, ctx.chiR_ket*ctx.d2*ctx.d1)
    # T1flat rows = Lenv_mat cols (dl1,n_ham_left_ch,Lbra), cols = φ cols (Rket,s2,s1)
    T1flat = BLAS.gemm('T', 'N', one(T), ctx.Lenv_mat, φ_mat)   # (dl1·n_ham_left_ch·chiL_bra, chiR_ket·d2·d1)
    T1r = reshape(T1flat, ctx.dl1, ctx.n_ham_left_ch, ctx.chiL_bra*ctx.chiR_ket*ctx.d2, ctx.d1)
    _pf_add!(1, _tr)

    T3buf = _pf_core!(ctx, T1r)   # steps 2+3 → (dr2, chiL_bra, chiR_ket, d1, d2, n_ham_right_ch)

    # ---- step 4: gather T3 so (chiR_ket,dr2,n_ham_right_ch) are contiguous, then gemm('N','N') ----
    _t4 = _pf_now()
    # (dr2, chiL_bra, chiR_ket, d1, d2, n_ham_right_ch) → (chiL_bra, d1, d2, chiR_ket, dr2, n_ham_right_ch)
    permutedims!(ctx.T3g_buf, T3buf, (2, 4, 5, 3, 1, 6))        # in-place gather into prealloc buf (0 alloc)
    T3g_mat = reshape(ctx.T3g_buf, ctx.chiL_bra*ctx.d1*ctx.d2, ctx.chiR_ket*ctx.dr2*ctx.n_ham_right_ch)
    Hvflat  = BLAS.gemm('N', 'N', one(T), T3g_mat, ctx.Renv_mat)  # (chiL_bra·d1·d2, chiR_bra)
    Hvarr   = reshape(Hvflat, ctx.chiL_bra, ctx.d1, ctx.d2, ctx.chiR_bra)  # (Lbra, s1', s2', Rbra)
    Hv = noprime(itensor(Hvarr, ctx.Lbra, ctx.s1_ind', ctx.s2_ind', ctx.Rbra))
    _pf_add!(4, _t4)
    # round-trip: reorder Hv → φ's index order (the one irreducible χ²·9 permute)
    return permute(Hv, ctx.Lket, ctx.Rket, ctx.s2_ind, ctx.s1_ind)
end

# ============================================================
# Per-call fused H1·H2 leg. `T1` = dense ITensor from stock step 1 (= Lenv·v, or
# Renv·v at the left edge). Returns the dense T3 = (T1·H1)·H2 for stock step 4.
#
# OUTPUT ORDER MATTERS. Stock step 4 (T3·Renv) is left UNCHANGED, and its output
# order is a deterministic function of T3's order. For the FULL matvec to return
# Hv in the SAME index order as v — so the wrap-around φ-pinning holds and Krylov
# iterations don't pay a per-iteration permute — T3 must be emitted in the SAME
# order the stock step-3 leg produced for this (bondtype, step). Pass that order
# as `out_inds` (the dispatch reads it from the same static_output_perm_dense
# machinery the rest of the chain uses); the natural build order is then permuted
# to it ONCE (a small T3 copy — no 44 GB T2, so still a net win). `out_inds ===
# nothing` returns the natural build order (correct index SET; use only when the
# caller does its own downstream reorder).
#
# Site bra indices are left PRIMED (s1_ind', s2_ind'), matching the stock step-3
# output exactly (the stock chain applies any final noprime downstream).
# ============================================================
function matvec_partial_fused(ctx::PartialMatvecContext{T}, T1::ITensor;
                              out_inds::Union{Nothing,AbstractVector}=nothing) where {T}
    # ---- ASSERT T1 arrives PINNED in the strided-readable order ----
    # Contract: the caller pins φ to (Lket,Rket,s2,s1) and runs step 1 as the
    # 'T'-flag GEMM (Lenv·φ, contract Lket), which delivers
    #     T1 = (dl1, n_ham_left_ch, Lbra, Rket, s2, s1)   [column-major, dl1 fastest]
    # with n_ham_left_ch IMMEDIATELY after dl1. Then a fixed (n_ham_left_ch,s1) slice is a strided
    # BLAS operand (M1mat = (dl1, Lbra·Rket·s2), LDA=dl1·n_ham_left_ch) — NO permute here.
    _tr = _pf_now()
    _want = (ctx.dl1_ind, ctx.l1_ind, ctx.Lbra, ctx.Rket, ctx.s2_ind, ctx.s1_ind)
    @assert collect(inds(T1)) == collect(_want) "matvec_partial_fused: T1 must arrive pinned as (dl1,n_ham_left_ch,Lbra,Rket,s2,s1); got $(collect(inds(T1)))"
    T1raw = array(T1)                                     # native order == _want (asserted) — no permute
    T1arr = eltype(T1raw) === T ? T1raw : T.(T1raw)
    # Collapse (Lbra,Rket,s2) into one axis so a fixed (n_ham_left_ch,s1) slice is a strided
    # StridedMatrix with LDA=dl1·n_ham_left_ch (the n_ham_left_ch gap folds into LDA).
    T1r = reshape(T1arr, ctx.dl1, ctx.n_ham_left_ch, ctx.chiL_bra*ctx.chiR_ket*ctx.d2, ctx.d1)

    _pf_add!(1, _tr)

    T3buf = _pf_core!(ctx, T1r)   # steps 2+3 (per-channel fused kernel)

    # T3 axes = (dr2, chiL_bra, chiR_ket, s1'[bra], s2'[bra], n_ham_right_ch). Bra sites stay
    # PRIMED (match stock step-3 output); dr2 & n_ham_right_ch are the open Renv-facing legs.
    result = itensor(T3buf, ctx.dr2_ind, ctx.Lbra, ctx.Rket, ctx.s1_ind', ctx.s2_ind', ctx.r2_ind)
    # Reorder ONCE to the stock step-3 output order so the unchanged step 4
    # reproduces v's index order (Krylov consistency; see header).
    return out_inds === nothing ? result : permute(result, out_inds...)
end

# ============================================================
# Target T3 output order = the SAME order the stock step-3 leg emits, read off
# the STATIC_OUTPUT_PERM_DENSE table's documented (bondtype, 3) layouts (in
# SparseBackends/tensor_wrappers_aliased.jl). We use the DECODED role order from
# that table's `-> T3(...)` comment, mapped onto ctx's index handles — NOT the
# raw numeric perm (which is relative to the stock kernel's internal labelsC_vec,
# a different natural order than this kernel's build order). Feeding this as
# `out_inds` makes the unchanged stock step 4 reproduce v's index order, so
# Krylov stays consistent (no per-iteration permute).
#
#   (:bulk, 3)  -> T3(s2¹, l3⁰, l1³¹, l3⁴¹, s3¹, F3)
#                 = (s1',  Rket, Lbra, dr2,  s2', n_ham_right_ch)
#
# Edges (:left/:right) reverse the chain and drop the right-env dense legs
# (T3 has only 3 inds there); pass 1 is bulk-only, so edges return `nothing`
# (natural order) and are expected to fall back to the stock per-leg path.
# ============================================================
function partial_matvec_out_inds(ctx::PartialMatvecContext, bondtype::Symbol)
    bondtype === :bulk || return nothing
    return [ctx.s1_ind', ctx.Rket, ctx.Lbra, ctx.dr2_ind, ctx.s2_ind', ctx.r2_ind]
end

# ============================================================
# Per-bond context cache (v-independent → build once per bond, reuse across the
# ~5–20 Krylov matvecs). Keyed by (P, si); rebuilt when si changes. No public API.
# ============================================================
const _PARTIAL_CTX_CACHE = IdDict{Any,Tuple{Int,Any}}()

function partial_matvec_context!(P, si::Int)
    hit = get(_PARTIAL_CTX_CACHE, P, nothing)
    if hit !== nothing && hit[1] == si
        return hit[2]
    end
    ctx = build_partial_matvec_context(P, si)
    _PARTIAL_CTX_CACHE[P] = (si, ctx)
    return ctx
end
