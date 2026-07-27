using SparseBackends
using ITensors
using LinearAlgebra: mul!, transpose

# ============================================================
# Shared core. E = environment tensor (4 legs: bra, ket, static-dense, static-sparse).
# psi_site = the MPS tensor being absorbed this step.
# neighbor_site = the adjacent MPS tensor, used only to identify psi_site's "new" bond
#                 (the link NOT shared with E).
# ============================================================
function _fused_sparse_env_contract(E::ITensor, H::ITensor, psi_site::ITensor,
                                    neighbor_site::ITensor; batch_step3::Bool=true)
    Hw = ITensors.get_external_storage(H)
    hinds = inds(Hw)
    A = Hw.aliased

    sparse_inds = hinds[1:4]
    dense_inds  = hinds[5:6]              # dense_inds[1] = col-major-fast reshape dim

    psidag_site = dag(prime(psi_site))

    # ---- role detection (works identically for L or R direction) ----
    stat_pos = findfirst(i -> hasind(E, sparse_inds[i]), 1:4)
    s_pos    = findfirst(i -> i != stat_pos && plev(sparse_inds[i]) == 0 &&
                              hasind(psi_site, sparse_inds[i]), 1:4)
    sp_pos   = findfirst(i -> i != stat_pos && i != s_pos && plev(sparse_inds[i]) != 0, 1:4)
    new_pos  = only(setdiff(1:4, (stat_pos, s_pos, sp_pos)))

    stat_ind, s_ind, sp_ind, new_ind = sparse_inds[stat_pos], sparse_inds[s_pos],
                                        sparse_inds[sp_pos], sparse_inds[new_pos]

    stat_dense_ind = hasind(E, dense_inds[1]) ? dense_inds[1] : dense_inds[2]
    new_dense_ind  = stat_dense_ind === dense_inds[1] ? dense_inds[2] : dense_inds[1]
    dl, dr = dim(stat_dense_ind), dim(new_dense_ind)
    @assert dl * dr == A.blksize
    # The template block is ALWAYS stored column-major as (left_mult, right_mult) =
    # (dense_inds[1] fast, dense_inds[2] slow). `stat_is_left` tells us whether the
    # E-shared (contracted) dense axis is the LEFT mult — true on makeL (E = left env,
    # contracted axis is the fast dense_inds[1]), false on makeR (E = right env,
    # contracted axis is the slow dense_inds[2]). Constant for every block in this
    # call, so we decide it once and switch a transpose flag on the tiny T operand in
    # step 2 — no template permute.
    stat_is_left = stat_dense_ind === dense_inds[1]
    d1, d2 = dim(dense_inds[1]), dim(dense_inds[2])

    Eket = commonind(E, psi_site)       # contracted in step 1
    Ebra = commonind(E, psidag_site)    # contracted in step 3

    kNew  = uniqueind(psi_site, neighbor_site, s_ind)   # freshly produced bond
    kNewp = kNew'

    # Layout: dl LEADING (so it's isolable after step1), Ebra next (rides through
    # steps 1&2 untouched), Eket trailing (contracted in step1).
    Earr      = array(permute(E, stat_dense_ind, Ebra, Eket, stat_ind))   # (dl,chiB,chiK,n_stat)
    psiarr    = array(permute(psi_site, Eket, s_ind, kNew))               # (chiK,d,chiNew)
    psidagarr = array(permute(psidag_site, Ebra, sp_ind, kNewp))          # (chiB,d,chiNewp)

    d = dim(s_ind)
    psi_slices    = [@view psiarr[:, s, :]    for s  in 1:d]
    psidag_slices = [@view psidagarr[:, sp,:] for sp in 1:d]

    chiB, chiK      = size(Earr, 2), size(Earr, 3)
    chiNew, chiNewp = dim(kNew), dim(kNewp)
    n_new = dim(new_ind)

    # Common element type: keys may be a narrow integer, but VALUES can be complex,
    # so scalars/α/β must carry the promoted type (mirrors contract_aliased_dense_to_dense.jl).
    TC = promote_type(eltype(Earr), eltype(A.templates), eltype(psidagarr))

    Enewarr = zeros(TC, chiNewp, chiNew, dr, n_new)

    M2buf = Matrix{TC}(undef, chiB*chiNew, dr)
    # Step-1 cache: M1all[:, :, s] = E_channel · ψ_s  holds one result per physical
    # ket value s. Reused within a channel group (see below); reset per group.
    M1all = Array{TC}(undef, dl*chiB, chiNew, d)
    s_present = falses(d)

    groups = Dict{eltype(A.keys[1]), Vector{Int}}()
    for (i, key) in enumerate(A.keys)
        push!(get!(groups, key[stat_pos], Int[]), i)
    end

    for (statval, idxs) in groups
        Eslice = @view Earr[:, :, :, statval]           # (dl,chiB,chiK), sliced once/group
        Emat   = reshape(Eslice, dl*chiB, chiK)          # merge leading two — valid (contiguous)

        # PRECOMPUTE step 1 ONCE per physical ket value s that actually occurs in this
        # channel group (scan the group's keys → sparsity filter). M1 = E_channel·ψ_s
        # depends only on (channel, s), NOT on (s', new_channel, template), so hoisting
        # it here removes the per-block recompute (Redundancy #1: block-fanout per
        # (channel,s), measured 3× on makeL / 12× on makeR for the KL-N12 bd40 PHP).
        fill!(s_present, false)
        for i in idxs
            s = A.keys[i][s_pos]
            if !s_present[s]
                s_present[s] = true
                mul!(@view(M1all[:, :, s]), Emat, psi_slices[s])
            end
        end

        # STEP 2 + STEP 3, batched over contiguous runs of constant (s', new_channel).
        # Keys are column-major sorted with s FASTEST and s' next, so every maximal run
        # of consecutive blocks sharing (s', new_channel) has the SAME ψ†-slice AND the
        # SAME output slice. Hence Σ_run ψ†[s']·M2_block = ψ†[s']·(Σ_run M2_block): we
        # accumulate step 2 into M2buf (β=1) across the run and do ONE step-3 GEMM per
        # run. Contiguous ⇒ no gather/bucket pass. (batch_step3=false → per-block, i.e.
        # flush every block: the original behaviour, for A/B.)
        M2mat  = reshape(M2buf, chiB, chiNew*dr)               # view over M2buf (chiB leading)
        cur_sp = 0; cur_nv = 0; active = false
        for i in idxs
            key = A.keys[i]
            s, sp, newval = key[s_pos], key[sp_pos], key[new_pos]
            tid, c = A.alias_ids[i], A.scalars[i]

            # Flush the current run before starting a new (s', new_channel) — or every
            # block when batching is off.
            if active && (!batch_step3 || sp != cur_sp || newval != cur_nv)
                Enewmat = reshape(@view(Enewarr[:, :, :, cur_nv]), chiNewp, chiNew*dr)
                mul!(Enewmat, transpose(psidag_slices[cur_sp]), M2mat, one(TC), one(TC))
                active = false
            end

            # Reshape the template in its NATIVE stored order (d1 fast, d2 slow) — never permuted.
            Tstored = reshape(view(A.templates, (tid-1)*A.blksize+1 : tid*A.blksize), d1, d2)
            # step 1 result: reuse the cached E·ψ_s (computed once above).
            M1mat = reshape(view(M1all, :, :, s), dl, chiB*chiNew)

            # step 2: * H (contract stat_dense via template). TRANSPOSE TRICK: M1matᵀ·T
            # puts chiB leading for free (BᵀAᵀ), no permutedims. β=1 within an open run
            # accumulates Σ M2_block; β=0 starts a fresh run.
            α = convert(TC, c)
            β = active ? one(TC) : zero(TC)
            if stat_is_left
                mul!(M2buf, transpose(M1mat), Tstored,            α, β)  # makeL ('T','N')
            else
                mul!(M2buf, transpose(M1mat), transpose(Tstored), α, β)  # makeR ('T','T')
            end
            cur_sp = sp; cur_nv = newval; active = true
        end
        # step 3 for the final open run.
        if active
            Enewmat = reshape(@view(Enewarr[:, :, :, cur_nv]), chiNewp, chiNew*dr)
            mul!(Enewmat, transpose(psidag_slices[cur_sp]), M2mat, one(TC), one(TC))
        end
    end

    return itensor(Enewarr, kNewp, kNew, new_dense_ind, new_ind)
end

# ============================================================
# Live-DMRG wiring: fused env stepping + position!, used by run_mode=:fused.
#
# INDEX-ORDER AUDIT (per-seam, for "no redundant re-permute"):
#  • env → env (chained fused steps): the kernel reads E via
#      array(permute(E, stat_dense_ind, Ebra, Eket, stat_ind))  [line ~54]
#    and OUTPUTS  itensor(Enewarr, kNewp, kNew, new_dense_ind, new_ind)
#    = (bra, ket, dense-mult, sparse-chan). The next step wants
#    (dense, bra, ket, sparse), so it DOES permute E once. This permute is
#    INTRINSIC to the kernel (it is present and measured in test_fused_real's
#    profile_kernel as part of the FULL-kernel time) — the wiring adds none.
#    ψ and ψ† are likewise permuted to (Eket,s,kNew)/(Ebra,s',kNew') once per
#    step; template blocks are read in native stored order (never permuted, via
#    the step-2 transpose flag). So per env step the ONLY data moves are the 3
#    structural operand permutes + the 3 BLAS gemms — same set the stock
#    _env_mul chain pays, minus the intermediate materializations.
#  • env → matvec: build_partial_matvec_context re-derives the env legs by Index
#    id and permutes lproj/rproj into Lenv_mat/Renv_mat ONCE per bond (v-independent,
#    amortized over every Krylov matvec). So the fused env's stored order does not
#    force any per-matvec permute regardless of what order it emits.
#  ⇒ We deliberately do NOT append _reorder_env_for_aliased to the fused output:
#    every consumer re-derives by id/role, and adding it would cost one extra
#    permute on every bulk step for no correctness or matvec benefit. (Edge-bond
#    stock `product` matvecs consume these envs by id too; without the canonical
#    reorder they take their permute_B fallback — correct, and only on 2 bonds.)
# ============================================================

# Eligible for the fused kernel: the incoming env E is a real 4-leg tensor (NOT the
# boundary OneITensor scalar) and the site H tensor is a 6-leg aliased bulk tensor.
# Boundary steps and any non-aliased/dense H fall back to the stock _env_mul chain.
_fused_env_eligible(E, H_site) =
    (E isa ITensor) && _is_aliased_itensor(H_site) && length(inds(H_site)) == 6

# One LEFT env step, LR[ll] → LR[ll+1] (mirrors _makeL!'s stock body; swaps the
# aliased H·dag(ψ')·ψ chain for the fused kernel when eligible).
function _fused_makeL!(P::ProjMPO, psi::MPS, k::Int; roofline::Bool=false, run_label::String="?")
    ll = P.lpos
    if ll ≥ k
        P.lpos = k; return nothing
    end
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        H_site = P.H[ll + 1]
        if _fused_env_eligible(L, H_site)
            # E=LR[ll]=L, H[ll+1], psi_site=psi[ll+1], neighbor=psi[ll] (identifies psi's new bond)
            L = _fused_sparse_env_contract(L, H_site, psi[ll + 1], psi[ll])
        else
            _keep = _is_aliased_itensor(H_site) && _is_aliased_itensor(psi[ll + 1])
            L = _env_mul(L, H_site, _keep)
            L = _env_mul(L, dag(prime(psi[ll + 1])), _keep)
            L = _env_mul(L, psi[ll + 1], _keep)
            L = _reorder_env_for_aliased(L, _keep, _is_aliased_itensor(H_site))
        end
        P.LR[ll + 1] = L
        roofline && _record_env_footprint(ll + 1, L, run_label)
        ll += 1
    end
    P.lpos = k
    return L
end

# One RIGHT env step, LR[rl] → LR[rl-1] (mirrors _makeR!'s stock body).
function _fused_makeR!(P::ProjMPO, psi::MPS, k::Int; roofline::Bool=false, run_label::String="?")
    rl = P.rpos
    if rl ≤ k
        P.rpos = k; return nothing
    end
    N = length(P.H)
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
        H_site = P.H[rl - 1]
        if _fused_env_eligible(R, H_site)
            # E=LR[rl]=R, H[rl-1], psi_site=psi[rl-1], neighbor=psi[rl]
            R = _fused_sparse_env_contract(R, H_site, psi[rl - 1], psi[rl])
        else
            _keep = _is_aliased_itensor(H_site) && _is_aliased_itensor(psi[rl - 1])
            R = _env_mul(R, H_site, _keep)
            R = _env_mul(dag(prime(psi[rl - 1])), R, _keep)
            R = _env_mul(psi[rl - 1], R, _keep)
            R = _reorder_env_for_aliased(R, _keep, _is_aliased_itensor(H_site))
        end
        P.LR[rl - 1] = R
        roofline && _record_env_footprint(rl - 1, R, run_label)
        rl -= 1
    end
    P.rpos = k
    return R
end

# NOTE: env dispatch is unified under `position!(…; run_mode=:fused)` (abstractprojmpo.jl /
# projmpo_mps.jl / projmposum.jl), which calls the `_fused_makeL!`/`_fused_makeR!` steppers
# above for a concrete ProjMPO. There is no separate `fused_position!` entry point.