using LinearAlgebra: BLAS

# ============================================================
# Context: everything independent of v. Built once per bond position.
# ============================================================
struct StepRecord{T}
    src_pos::Int
    dst_pos::Int
    out_phys::Int
    tid::Int
    c::T
end

struct MatvecContext{T}
    d1::Int; d2::Int
    dl1::Int; dr1::Int; dl2::Int; dr2::Int
    dl1_fast::Bool; dl2_fast::Bool     # is the contracted (left) mult the FAST stored dense axis?
    chiL_bra::Int; chiL_ket::Int
    chiR_bra::Int; chiR_ket::Int
    n_l1::Int; n_m::Int; n_r2::Int

    Larr::Array{T,4}         # (dl1, chiL_bra, chiL_ket, n_l1)   -- dl1 leading
    Rarr::Array{T,4}         # (chiR_bra, chiR_ket, dr2, n_r2)   -- dr2 SECOND-TO-LAST (step D needs it trailing-sliceable)
    templates1::Vector{T}
    templates2::Vector{T}
    blksize1::Int
    blksize2::Int

    step1::Matrix{Vector{StepRecord{T}}}   # (n_l1, d1) -> records writing into hsm-position
    step2::Matrix{Vector{StepRecord{T}}}   # (n_m,  d2) -> records writing into hsr-position

    # ---- preallocated per-call scratch (reused across Krylov iterations) ----
    # NOTE: single-threaded assumption (KrylovKit runs serial here). The struct is
    # immutable but the buffers are mutable arrays whose CONTENTS are overwritten
    # each call; nothing escapes except a freshly-allocated `vout`.
    x2::Array{T,6}            # (dr1, chiL_bra, chiR_ket, d1[s1'], d2[s2], n_m)  middle accumulator
    stepC_out::Matrix{T}      # (dr2, chiL_bra*chiR_ket*d1)      step C (H2 GEMM output)
    Rbatch::Array{T,4}        # (chiR_bra, dr2, chiR_ket, n_r2)  Renv permuted for step D
    stepD_in::Array{T,4}      # (chiL_bra, d1, dr2, chiR_ket)    step-D input scratch (per H2 record)
    populated::Vector{Bool}   # (n_m,)  which middle-channel blocks were touched this s1

    Lbra; Lket; Rbra; Rket; s1_ind; s2_ind
end

# Classify H's 6 legs given the SHARED middle bond (channel + mult) from
# commoninds(H1,H2). This fixes left/right UNAMBIGUOUSLY (no tag-level guessing):
# for H1 the shared bond is its RIGHT link; for H2 its LEFT link.
function _classify_mpo_tensor_static(H::ITensor, mid_channel, mid_mult, mid_is_right::Bool)
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

function build_matvec_context(P, si::Int)
    L, R = lproj(P), rproj(P)
    H1, H2 = P.H[si], P.H[si+1]

    # Shared middle bond = commoninds(H1,H2) = one sparse channel + one dense mult.
    mids = commoninds(H1, H2)
    H1w  = ITensors.get_external_storage(H1)
    h1_sparse_ids = Set(ITensors.id(i) for i in inds(H1w)[1:4])
    mid_channel = only(filter(i ->   ITensors.id(i) in h1_sparse_ids, mids))
    mid_mult    = only(filter(i -> !(ITensors.id(i) in h1_sparse_ids), mids))

    cls1 = _classify_mpo_tensor_static(H1, mid_channel, mid_mult, true)   # shared = H1's RIGHT
    cls2 = _classify_mpo_tensor_static(H2, mid_channel, mid_mult, false)  # shared = H2's LEFT
    s1_ind, s2_ind = cls1.s_ind, cls2.s_ind
    d1, d2 = dim(s1_ind), dim(s2_ind)
    @assert cls1.r_ind === cls2.l_ind "middle channel not consistent between H1(right)/H2(left)"
    @assert cls1.dr_ind === cls2.dl_ind "middle mult not consistent between H1(right)/H2(left)"

    # MPS bond legs = env legs NOT shared with the MPO tensor (channel/mult ARE shared by id).
    h1ids = Set(ITensors.id(i) for i in inds(H1))
    h2ids = Set(ITensors.id(i) for i in inds(H2))
    L_mps = filter(i -> !(ITensors.id(i) in h1ids), collect(inds(L)))
    R_mps = filter(i -> !(ITensors.id(i) in h2ids), collect(inds(R)))
    Lket = only(filter(i -> plev(i) == 0, L_mps)); Lbra = only(filter(i -> plev(i) != 0, L_mps))
    Rket = only(filter(i -> plev(i) == 0, R_mps)); Rbra = only(filter(i -> plev(i) != 0, R_mps))

    # ---- common element type (use eltype(L)/eltype(R) — do NOT materialize dense) ----
    T = promote_type(eltype(cls1.A.templates), eltype(cls2.A.templates), eltype(L), eltype(R))

    # ---- one-time dense copies (Larr 4th axis = H1 left channel, indexed by RAW value) ----
    Larr = T.(array(permute(L, cls1.dl_ind, Lbra, Lket, cls1.l_ind)))       # (dl1,chiL_bra,chiL_ket,n_l1)
    Rarr = T.(array(permute(R, Rbra, Rket, cls2.dr_ind, cls2.r_ind)))       # (chiR_bra,chiR_ket,dr2,n_r2)

    n_l1 = dim(cls1.l_ind)      # H1 left channel  (Larr 4th axis)
    n_m  = dim(mid_channel)     # shared middle channel
    n_r2 = dim(cls2.r_ind)      # H2 right channel (Rarr 4th axis)
    @assert size(Larr,4) == n_l1 && size(Rarr,4) == n_r2

    # Step records indexed by RAW channel/site values (no compaction → no pos/axis mismatch).
    step1 = [StepRecord{T}[] for _ in 1:n_l1, _ in 1:d1]
    for (i, key) in enumerate(cls1.A.keys)
        lval, sval, spval, mval = key[cls1.l_pos], key[cls1.s_pos], key[cls1.sp_pos], key[cls1.r_pos]
        push!(step1[lval, sval], StepRecord{T}(lval, mval, Int(spval), cls1.A.alias_ids[i], T(cls1.A.scalars[i])))
    end
    step2 = [StepRecord{T}[] for _ in 1:n_m, _ in 1:d2]
    for (i, key) in enumerate(cls2.A.keys)
        mval, sval, spval, rval = key[cls2.l_pos], key[cls2.s_pos], key[cls2.sp_pos], key[cls2.r_pos]
        push!(step2[mval, sval], StepRecord{T}(mval, rval, Int(spval), cls2.A.alias_ids[i], T(cls2.A.scalars[i])))
    end
    # Step-B template dedup: within a cell (fixed lpos,s1) M1mat is fixed, so the
    # template-GEMM templateᵀ·M1mat depends only on tid. Sort each cell by tid so
    # records sharing a template are contiguous → compute the GEMM once per tid,
    # then scale+route per record (measured ~2.4× fewer step-B GEMMs). Accumulate
    # is commutative so reordering is exact.
    for cell in step1
        sort!(cell, by = r -> r.tid)
    end

    # dr1 (H1 right mult) is the SAME middle-bond mult as dl2 (H2 left mult).
    @assert cls1.dr == cls2.dl "middle-bond mult mismatch: H1.dr=$(cls1.dr) vs H2.dl=$(cls2.dl)"
    # Is the CONTRACTED (left) mult the FAST stored dense axis? The template is stored
    # column-major (dense_inds[1] fast); a reshape to (dl,dr) is only valid when
    # dl===dense_inds[1]. Otherwise the gemm must transpose (same fix as the env kernel).
    H2w = ITensors.get_external_storage(H2)
    dl1_fast = cls1.dl_ind === inds(H1w)[5]
    dl2_fast = cls2.dl_ind === inds(H2w)[5]
    chiL_bra, chiL_ket = size(Larr,2), size(Larr,3)
    chiR_bra, chiR_ket = size(Rarr,1), size(Rarr,2)
    dl1, dr1, dl2, dr2 = cls1.dl, cls1.dr, cls2.dl, cls2.dr

    # ---- preallocate all per-call scratch once (reused across Krylov iterations) ----
    x2        = Array{T,6}(undef, dr1, chiL_bra, chiR_ket, d1, d2, n_m)
    stepC_out = Matrix{T}(undef, dr2, chiL_bra*chiR_ket*d1)
    Rbatch    = permutedims(Rarr, (1, 3, 2, 4))          # (chiR_bra, dr2, chiR_ket, n_r2) — once
    stepD_in  = Array{T,4}(undef, chiL_bra, d1, dr2, chiR_ket)
    populated = Vector{Bool}(undef, n_m)

    return MatvecContext{T}(
        d1, d2, dl1, dr1, dl2, dr2, dl1_fast, dl2_fast,
        chiL_bra, chiL_ket, chiR_bra, chiR_ket,
        n_l1, n_m, n_r2,
        Larr, Rarr, T.(cls1.A.templates), T.(cls2.A.templates), cls1.A.blksize, cls2.A.blksize,
        step1, step2,
        x2, stepC_out, Rbatch, stepD_in, populated,
        Lbra, Lket, Rbra, Rket, s1_ind, s2_ind,
    )
end

# ============================================================
# DESIRED INDEX ORDERS (column-major; leftmost axis is fastest/contiguous)
# ------------------------------------------------------------
# Notation: s1,s2 = ket physical (H1,H2); s1',s2' = bra physical; L,R = MPS bonds
# (primed = dagger/bra side); dl*,dr* = dense multiplicities; n_l1/n_m/n_r2 = sparse
# channels; each per-step GEMM is a reshape of a CONTIGUOUS view (no hidden permute).
#
#   Larr      (dl1, chiL_bra, chiL_ket, n_l1)               step-A Lmat rows = dl1*chiL_bra
#   Rarr      (chiR_bra, chiR_ket, dr2, n_r2)               env layout as delivered
#   Rbatch    (chiR_bra, dr2, chiR_ket, n_r2)              Rarr pre-permuted ONCE (build time) for step D
#   stepA_out (dl1*chiL_bra, d1*chiR_ket*d2)               step-A output = M1mat rows dl1  [in BatchedScratch]
#   stepB_out (dr1, chiL_bra*d1*chiR_ket*d2)               step-B template-GEMM output     [in BatchedScratch]
#   x2        (dr1, chiL_bra, chiR_ket, d1[s1'], d2[s2], n_m)   middle accumulator
#   stepC_out (dr2, chiL_bra*chiR_ket*d1)                  step-C output (dr2 leading, forced)
#   stepD_in  (chiL_bra, d1, dr2, chiR_ket)                step-D input (co-locates dr2,chiR_ket)
#   vout      (chiL_bra, d1[s1'], chiR_bra, d2[s2'])       result (step-D writes here directly, β=1)
#
# Per-call permutes — only two, and they are DIFFERENT in nature:
#   v -> (Lket,Rket,s2,s1) [see `phiperm` below]: pure INPUT-ORDER artifact. If the
#         eigensolver handed v already in this order, array(permute(...)) would be a
#         no-op (zero copy). Unavoidable HERE only because KrylovKit dictates v's
#         layout upstream.
#   stepC_out -> stepD_in via permutedims! [step D]: NOT an input-order artifact. stepC_out is a GEMM
#         output whose leading axis is forced to dr2; step D wants chiL_bra leading.
#         BLAS can fix only one leading axis, so chaining C->D needs this move. It can
#         only be removed by restructuring the C/D contraction, not by re-ordering data.
# All context-build permutes (Larr/Rarr/Rbatch) are one-time and amortized.
# ============================================================

# ============================================================
# Per-step profiling (debug only). Toggle _FUSED_PROF[]=true from the harness,
# reset before the timed loop, report after. When OFF each probe is a single
# Ref read + short-circuit; the GEMMs are untouched, so the production path is
# unaffected. Slots: 1 phiperm | 2 stepA | 3 stepB | 4 stepC | 5 stepD-permute
# | 6 stepD-gemm(accumulate).
# ============================================================
const _FUSED_PROF   = Ref(false)
const _FUSED_TIMES  = zeros(Float64, 6)
const _FUSED_LABELS = ("phiperm", "stepA", "stepB", "stepC", "stepD-permute", "stepD-gemm")
reset_fused_prof!() = (fill!(_FUSED_TIMES, 0.0); nothing)
function report_fused_prof!(reps::Int)
    tot = sum(_FUSED_TIMES)
    tot == 0 && (println("    (no profiling data — was _FUSED_PROF[] set?)"); return)
    println("    per-step breakdown (us/call, % of instrumented time):")
    for i in 1:length(_FUSED_LABELS)
        us  = round(1e6*_FUSED_TIMES[i]/reps, digits=1)
        pct = round(100*_FUSED_TIMES[i]/tot, digits=1)
        println("      ", rpad(_FUSED_LABELS[i], 20), " = ", lpad(string(us), 8), " us  (", pct, "%)")
    end
    println("      ", rpad("Σ instrumented", 20), " = ", lpad(string(round(1e6*tot/reps, digits=1)), 8), " us")
end
@inline _prof_now() = _FUSED_PROF[] ? Base.time_ns() : UInt64(0)
@inline _prof_add!(slot, t0) = (_FUSED_PROF[] && (@inbounds _FUSED_TIMES[slot] += (Base.time_ns()-t0)*1e-9); nothing)

# ---- GEMM/FLOP counter (debug). When _GEMM_COUNT[] on, every _cgemm! tallies a call
# and 2·M·N·K flops (M,N = size(C); K = inner dim). Off ⇒ plain BLAS.gemm!. ----
const _GEMM_COUNT = Ref(false)
const _GEMM_N     = Ref(0)
const _GEMM_FLOPS = Ref(0.0)
reset_gemm_count!() = (_GEMM_N[] = 0; _GEMM_FLOPS[] = 0.0; nothing)
@inline function _cgemm!(ta::Char, tb::Char, alpha, A, B, beta, C)
    if _GEMM_COUNT[]
        M, N = size(C, 1), size(C, 2)
        K = ta == 'N' ? size(A, 2) : size(A, 1)
        _GEMM_N[] += 1
        _GEMM_FLOPS[] += 2.0 * M * N * K
    end
    BLAS.gemm!(ta, tb, alpha, A, B, beta, C)
end

# ============================================================
# Per-call matvec — BATCHED-s1 variant (zero-boundary-permute fixed point).
# ------------------------------------------------------------
# Consumes v in D=(Lket, s1, Rket, s2) and produces Hv in D=(Lbra, s1', Rbra, s2')
# (same axis roles) → when psi is created in D, BOTH the input repack and the output
# labeling are no-ops. s1 is BATCHED (not looped) into steps A/B, and step C/D runs
# ONCE over the fully-summed x2 (the s1-loop version re-runs C/D per ket-s1).
#
# Correctness identity: stepCD is linear, and batched-x2 = Σ_{s1} (per-s1 x2), so
# stepCD(Σ x2_s1) = Σ stepCD(x2_s1) = the same vout.
#
# Scratch (stepA_out/stepB_out + the per-lpos record grouping) is HOISTED into
# BatchedScratch, built once per bond (v-independent) and reused across all Krylov matvecs,
# so per-call churn is just phiperm + vout. Build via build_batched_scratch(ctx).
# ============================================================
struct BatchedScratch{T}
    stepA_out::Matrix{T}   # (dl1*chiL_bra, d1*chiR_ket*d2)    step A: L·φ
    stepB_out::Matrix{T}   # (dr1, chiL_bra*d1*chiR_ket*d2)    step B: H1 template GEMM
    recs_lpos::Vector{Vector{Tuple{Int,Int,Int,Int,T}}}  # per-lpos (tid,s1_ket,m,sp1,c) tid-sorted
end

function build_batched_scratch(ctx::MatvecContext{T}) where {T}
    BigN   = ctx.chiL_bra*ctx.d1*ctx.chiR_ket*ctx.d2
    stepA_out = Matrix{T}(undef, ctx.dl1*ctx.chiL_bra, ctx.d1*ctx.chiR_ket*ctx.d2)
    stepB_out    = Matrix{T}(undef, ctx.dr1, BigN)
    recs_lpos = [Tuple{Int,Int,Int,Int,T}[] for _ in 1:ctx.n_l1]
    for lpos in 1:ctx.n_l1
        for s1 in 1:ctx.d1, rec in ctx.step1[lpos, s1]
            push!(recs_lpos[lpos], (rec.tid, s1, rec.dst_pos, rec.out_phys, rec.c))
        end
        sort!(recs_lpos[lpos], by = first)                       # tid-sorted (dedup grouping)
    end
    return BatchedScratch{T}(stepA_out, stepB_out, recs_lpos)
end

function fused_matvec_batched(ctx::MatvecContext{T}, v::ITensor, scr::BatchedScratch{T}) where {T}
    _tp = _prof_now()
    # input consumed in D_in = (Lket, s1, Rket, s2); no-op (zero copy) if v already in D.
    phiperm = array(permute(v, ctx.Lket, ctx.s1_ind, ctx.Rket, ctx.s2_ind; allow_alias=true))
    phiarr  = eltype(phiperm) === T ? phiperm : T.(phiperm)     # (chiL_ket, d1, chiR_ket, d2)
    phi_mat = reshape(phiarr, ctx.chiL_ket, ctx.d1*ctx.chiR_ket*ctx.d2)
    # output produced in D_out = (Lbra, s1', Rbra, s2') → same axis roles as D_in.
    vout    = zeros(T, ctx.chiL_bra, ctx.d1, ctx.chiR_bra, ctx.d2)
    voutD   = reshape(vout, ctx.chiL_bra*ctx.d1, ctx.chiR_bra, ctx.d2)
    stepB_out_5d  = reshape(scr.stepB_out, ctx.dr1, ctx.chiL_bra, ctx.d1, ctx.chiR_ket, ctx.d2)
    fill!(ctx.populated, false)
    _prof_add!(1, _tp)

    for lpos in 1:ctx.n_l1
        recs = scr.recs_lpos[lpos]                               # prebuilt, tid-sorted
        isempty(recs) && continue

        # ---- Step A: x1 = L·phi, batched over (s1, chiR_ket, s2) ----
        _ta = _prof_now()
        Lslice = @view ctx.Larr[:, :, :, lpos]
        Lmat = reshape(Lslice, ctx.dl1*ctx.chiL_bra, ctx.chiL_ket)
        _cgemm!('N', 'N', one(T), Lmat, phi_mat, zero(T), scr.stepA_out)
        M1mat = reshape(scr.stepA_out, ctx.dl1, ctx.chiL_bra*ctx.d1*ctx.chiR_ket*ctx.d2)
        _prof_add!(2, _ta)

        # ---- Step B: one template-GEMM per (lpos, tid); s1_ket rides in the batch,
        #      routing picks the ket-s1 slice into x2's bra-s1' slot. ----
        _tb = _prof_now()
        cur_tid = 0
        for (tid, s1k, m, sp1, c) in recs
            if tid != cur_tid
                blk = view(ctx.templates1, (tid-1)*ctx.blksize1+1 : tid*ctx.blksize1)
                if ctx.dl1_fast
                    _cgemm!('T', 'N', one(T), reshape(blk, ctx.dl1, ctx.dr1), M1mat, zero(T), scr.stepB_out)
                else
                    _cgemm!('N', 'N', one(T), reshape(blk, ctx.dr1, ctx.dl1), M1mat, zero(T), scr.stepB_out)
                end
                cur_tid = tid
            end
            if !ctx.populated[m]
                fill!((@view ctx.x2[:, :, :, :, :, m]), zero(T))
                ctx.populated[m] = true
            end
            # route ket-s1=s1k → bra-s1'=sp1 (strided slice; the price of s2-trailing D)
            @views ctx.x2[:, :, :, sp1, :, m] .+= c .* stepB_out_5d[:, :, s1k, :, :]
        end
        _prof_add!(3, _tb)
    end

    # ---- Step C + D: ONCE over the fully-summed x2 (was per ket-s1) ----
    for mpos in 1:ctx.n_m
        ctx.populated[mpos] || continue
        for s2 in 1:ctx.d2
            recs2 = ctx.step2[mpos, s2]
            isempty(recs2) && continue
            x2slice = @view ctx.x2[:, :, :, :, s2, mpos]
            x2mat = reshape(x2slice, ctx.dl2, ctx.chiL_bra*ctx.chiR_ket*ctx.d1)
            for rec in recs2
                _tc = _prof_now()
                blk2 = view(ctx.templates2, (rec.tid-1)*ctx.blksize2+1 : rec.tid*ctx.blksize2)
                if ctx.dl2_fast
                    _cgemm!('T', 'N', rec.c, reshape(blk2, ctx.dl2, ctx.dr2), x2mat, zero(T), ctx.stepC_out)
                else
                    _cgemm!('N', 'N', rec.c, reshape(blk2, ctx.dr2, ctx.dl2), x2mat, zero(T), ctx.stepC_out)
                end
                stepC_out_4d = reshape(ctx.stepC_out, ctx.dr2, ctx.chiL_bra, ctx.chiR_ket, ctx.d1)
                r, sp2 = rec.dst_pos, rec.out_phys
                _prof_add!(4, _tc)
                _td = _prof_now()
                permutedims!(ctx.stepD_in, stepC_out_4d, (2, 4, 1, 3))
                _prof_add!(5, _td)
                _te = _prof_now()
                Amat = reshape(ctx.stepD_in, ctx.chiL_bra*ctx.d1, ctx.dr2*ctx.chiR_ket)
                Rmat = reshape((@view ctx.Rbatch[:, :, :, r]), ctx.chiR_bra, ctx.dr2*ctx.chiR_ket)
                Dslice = @view voutD[:, :, sp2]
                _cgemm!('N', 'T', one(T), Amat, Rmat, one(T), Dslice)
                _prof_add!(6, _te)
            end
        end
    end

    # output in D_out = (Lbra, s1', Rbra, s2') — same axis roles as D_in ⇒ no output permute.
    return noprime(itensor(vout, ctx.Lbra, ctx.s1_ind', ctx.Rbra, ctx.s2_ind'))
end

# ============================================================
# STREAMING variant — eliminates the resident 147MB x2 accumulator by exploiting
# the measured channel structure: FAN-IN = 1, i.e. each middle
# channel m is fed by exactly ONE left channel lpos. So once lpos finishes, every m
# it feeds is COMPLETE (no other lpos contributes) → we flush stepC/D for those m
# immediately, while their data is cache-hot, instead of writing all n_m into a big
# buffer and re-reading it cold. Buffer shrinks from (…, n_m) [147MB @ M=40] to a
# compact (…, F) where F = max fan-out (# m per lpos) [~37MB], reused across lpos.
# Keeps the stepC/D batching over d1 (fan-in=1 ⇒ no GEMM-count blowup). REQUIRES
# fan-in=1 (asserted at build); general fan-in would need the full accumulator.
# ============================================================
struct StreamingScratch{T}
    stepA_out::Matrix{T}   # (dl1*chiL_bra, d1*chiR_ket*d2)    step A: L·φ
    stepB_out::Matrix{T}   # (dr1, chiL_bra*d1*chiR_ket*d2)    step B: H1 template GEMM
    x2c::Array{T,6}        # (dr1, chiL_bra, chiR_ket, d1, d2, F)  compact per-lpos accumulator
    # per-lpos records with m REPLACED by a local slot (1..F): (tid, s1_ket, slot, sp1, c), tid-sorted
    recs_lpos::Vector{Vector{Tuple{Int,Int,Int,Int,T}}}
    m_of_lpos::Vector{Vector{Int}}   # slot → real middle-channel m (for step2 lookup)
end

function build_streaming_scratch(ctx::MatvecContext{T}) where {T}
    # unique m per lpos + fan-in=1 check (each m owned by ≤1 lpos)
    m_of_lpos = [Int[] for _ in 1:ctx.n_l1]
    owner = fill(0, ctx.n_m)
    for lpos in 1:ctx.n_l1
        seen = Int[]
        for s1 in 1:ctx.d1, rec in ctx.step1[lpos, s1]
            m = rec.dst_pos
            if !(m in seen); push!(seen, m); end
            if owner[m] == 0; owner[m] = lpos
            elseif owner[m] != lpos
                error("streaming kernel requires fan-in=1; middle channel $m fed by lpos $(owner[m]) and $lpos")
            end
        end
        m_of_lpos[lpos] = sort!(seen)
    end
    F = maximum(length.(m_of_lpos); init=0)
    # records with local slot in place of m
    recs_lpos = [Tuple{Int,Int,Int,Int,T}[] for _ in 1:ctx.n_l1]
    for lpos in 1:ctx.n_l1
        slot_of_m = Dict(m => i for (i, m) in enumerate(m_of_lpos[lpos]))
        for s1 in 1:ctx.d1, rec in ctx.step1[lpos, s1]
            push!(recs_lpos[lpos], (rec.tid, s1, slot_of_m[rec.dst_pos], rec.out_phys, rec.c))
        end
        sort!(recs_lpos[lpos], by = first)                       # tid-sorted (dedup grouping)
    end
    stepA_out = Matrix{T}(undef, ctx.dl1*ctx.chiL_bra, ctx.d1*ctx.chiR_ket*ctx.d2)
    stepB_out    = Matrix{T}(undef, ctx.dr1, ctx.chiL_bra*ctx.d1*ctx.chiR_ket*ctx.d2)
    x2c       = Array{T,6}(undef, ctx.dr1, ctx.chiL_bra, ctx.chiR_ket, ctx.d1, ctx.d2, F)
    return StreamingScratch{T}(stepA_out, stepB_out, x2c, recs_lpos, m_of_lpos)
end

function fused_matvec_streaming(ctx::MatvecContext{T}, v::ITensor, scr::StreamingScratch{T}) where {T}
    _tp = _prof_now()
    phiperm = array(permute(v, ctx.Lket, ctx.s1_ind, ctx.Rket, ctx.s2_ind; allow_alias=true))
    phiarr  = eltype(phiperm) === T ? phiperm : T.(phiperm)
    phi_mat = reshape(phiarr, ctx.chiL_ket, ctx.d1*ctx.chiR_ket*ctx.d2)
    vout    = zeros(T, ctx.chiL_bra, ctx.d1, ctx.chiR_bra, ctx.d2)
    voutD   = reshape(vout, ctx.chiL_bra*ctx.d1, ctx.chiR_bra, ctx.d2)
    stepB_out_5d  = reshape(scr.stepB_out, ctx.dr1, ctx.chiL_bra, ctx.d1, ctx.chiR_ket, ctx.d2)
    _prof_add!(1, _tp)

    for lpos in 1:ctx.n_l1
        recs = scr.recs_lpos[lpos]
        isempty(recs) && continue
        ms = scr.m_of_lpos[lpos]

        # ---- Step A: x1 = L·phi ----
        _ta = _prof_now()
        Lslice = @view ctx.Larr[:, :, :, lpos]
        Lmat = reshape(Lslice, ctx.dl1*ctx.chiL_bra, ctx.chiL_ket)
        _cgemm!('N', 'N', one(T), Lmat, phi_mat, zero(T), scr.stepA_out)
        M1mat = reshape(scr.stepA_out, ctx.dl1, ctx.chiL_bra*ctx.d1*ctx.chiR_ket*ctx.d2)
        _prof_add!(2, _ta)

        # ---- Step B: template-GEMM per tid; route ket-s1 → bra-s1' into the compact
        #      per-lpos slot (fan-in=1 ⇒ these slots see only this lpos). ----
        _tb = _prof_now()
        for i in 1:length(ms); fill!((@view scr.x2c[:, :, :, :, :, i]), zero(T)); end
        cur_tid = 0
        for (tid, s1k, slot, sp1, c) in recs
            if tid != cur_tid
                blk = view(ctx.templates1, (tid-1)*ctx.blksize1+1 : tid*ctx.blksize1)
                if ctx.dl1_fast
                    _cgemm!('T', 'N', one(T), reshape(blk, ctx.dl1, ctx.dr1), M1mat, zero(T), scr.stepB_out)
                else
                    _cgemm!('N', 'N', one(T), reshape(blk, ctx.dr1, ctx.dl1), M1mat, zero(T), scr.stepB_out)
                end
                cur_tid = tid
            end
            @views scr.x2c[:, :, :, sp1, :, slot] .+= c .* stepB_out_5d[:, :, s1k, :, :]
        end
        _prof_add!(3, _tb)

        # ---- Step C + D: flush each complete m WHILE HOT (fan-in=1) ----
        for (slot, m) in enumerate(ms)
            for s2 in 1:ctx.d2
                recs2 = ctx.step2[m, s2]
                isempty(recs2) && continue
                x2slice = @view scr.x2c[:, :, :, :, s2, slot]
                x2mat = reshape(x2slice, ctx.dl2, ctx.chiL_bra*ctx.chiR_ket*ctx.d1)
                for rec in recs2
                    _tc = _prof_now()
                    blk2 = view(ctx.templates2, (rec.tid-1)*ctx.blksize2+1 : rec.tid*ctx.blksize2)
                    if ctx.dl2_fast
                        _cgemm!('T', 'N', rec.c, reshape(blk2, ctx.dl2, ctx.dr2), x2mat, zero(T), ctx.stepC_out)
                    else
                        _cgemm!('N', 'N', rec.c, reshape(blk2, ctx.dr2, ctx.dl2), x2mat, zero(T), ctx.stepC_out)
                    end
                    stepC_out_4d = reshape(ctx.stepC_out, ctx.dr2, ctx.chiL_bra, ctx.chiR_ket, ctx.d1)
                    r, sp2 = rec.dst_pos, rec.out_phys
                    _prof_add!(4, _tc)
                    _td = _prof_now()
                    permutedims!(ctx.stepD_in, stepC_out_4d, (2, 4, 1, 3))
                    _prof_add!(5, _td)
                    _te = _prof_now()
                    Amat = reshape(ctx.stepD_in, ctx.chiL_bra*ctx.d1, ctx.dr2*ctx.chiR_ket)
                    Rmat = reshape((@view ctx.Rbatch[:, :, :, r]), ctx.chiR_bra, ctx.dr2*ctx.chiR_ket)
                    Dslice = @view voutD[:, :, sp2]
                    _cgemm!('N', 'T', one(T), Amat, Rmat, one(T), Dslice)
                    _prof_add!(6, _te)
                end
            end
        end
    end

    return noprime(itensor(vout, ctx.Lbra, ctx.s1_ind', ctx.Rbra, ctx.s2_ind'))
end
