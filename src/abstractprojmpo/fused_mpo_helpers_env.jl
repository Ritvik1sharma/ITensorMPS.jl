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
# Thin direction-specific wrappers
# ============================================================
function fused_sparse_makeL!(P, psi, ll)
    Enew = _fused_sparse_env_contract(P.LR[ll], P.H[ll+1], psi[ll+1], psi[ll])
    P.LR[ll+1] = Enew
    return Enew
end

function fused_sparse_makeR!(P, psi, ll)
    Enew = _fused_sparse_env_contract(P.LR[ll], P.H[ll], psi[ll], psi[ll+1])
    P.LR[ll-1] = Enew
    return Enew
end