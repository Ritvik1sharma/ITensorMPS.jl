# Accurate test of the fused makeL/makeR env kernel on the REAL aliased PHP H
# for KL N=12, bd=40 (the aliased-Hamiltonian + dense-psi regime, matching
# test_sparse_ham/test_check_working_aliased.jl).
#
# Builds the aliased PHP, runs a few DMRG sweeps to reach bd=40, then position!s
# a fresh ProjMPO at the middle bond and calls _fused_sparse_env_contract on the
# ACTUAL Lenv / H[site] / psi[site] tensors — comparing correctness vs the
# densified reference and timing vs the default (aliased-*) env step.

using SparseBackends, ITensors, ITensorMPS
using LinearAlgebra
using Random, Printf
# matvec file uses lproj/rproj (ITensorMPS internals) unqualified; bring into Main scope
import ITensorMPS: lproj, rproj, product

include("/home/ritvik/temp/temp/edited_packages/test_sparse_ham/aliased_helpers.jl")
include("/home/ritvik/temp/temp/edited_packages/ITensorMPS.jl/src/abstractprojmpo/fused_mpo_helpers_env.jl")
include("/home/ritvik/temp/temp/edited_packages/ITensorMPS.jl/src/abstractprojmpo/fused_mpo_helpers_matvec.jl")

# ---- builders (copied verbatim from test_check_working_aliased.jl) ----
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

# ---- setup: KL N=12, spin=1 (matches the runner defaults) ----
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

println("[dmrg to reach bd=40]")
res = dmrg(H_ali, psi0; nsweeps=4, maxdim=40, mindim=40, cutoff=1e-12,
           outputlevel=1, use_early_exit=false)
psi = res[2]

# ---- position a fresh ProjMPO at the middle bond ----
PH = ITensorMPS.ProjMPO(H_ali)
L  = length(psi)
mid = div(L, 2)
ITensorMPS.position!(PH, psi, mid)
println("\nsystem: $L sites, mid=$mid, bond dims around mid: ",
        [dim(commonind(psi[i], psi[i+1])) for i in mid-3:mid+2])

# ---- inspect the real aliased H site structure + Redundancy-#1 fanout ----
function inspect_H(H_site, tag)
    Hw = ITensors.get_external_storage(H_site)
    A  = Hw.aliased
    Pp = SparseBackends._abs_head_len(Hw); N2 = SparseBackends._abs_tail_len(Hw)
    println("  [$tag] N=$(ndims(A)) P=$Pp N2=$N2  dims=$(A.dims)  blksize=$(A.blksize)")
    println("         n_keys=$(length(A.keys))  n_templates=$(A.n_templates)  dedup=$(round(length(A.keys)/max(A.n_templates,1),digits=2))")
    # Key/block layout: prefix axes (keys are column-major sorted, axis1 fastest),
    # role of each axis, and how s' (bra Site) is distributed → tells us whether
    # step-3 can batch by contiguous s'-runs without a gather.
    sinds = inds(Hw)[1:4]
    println("         prefix axis order (axis1 fastest): ", [(dim(I), string(tags(I)), plev(I)) for I in sinds])
    sp_pos = findfirst(i -> hastags(sinds[i], "Site") && plev(sinds[i]) == 1, 1:4)
    s_pos  = findfirst(i -> hastags(sinds[i], "Site") && plev(sinds[i]) == 0, 1:4)
    sdim = dim(sinds[s_pos])
    sps = [k[sp_pos] for k in A.keys]
    runs = 1; for t in 2:length(sps); sps[t] != sps[t-1] && (runs += 1); end
    println("         s' is prefix axis $sp_pos; contiguous s'-runs across all $(length(A.keys)) keys = $runs  (run length ≈ dim(s)=$sdim)")
    return Hw, A, Pp, N2
end

# For a makeL step (E=LR[ll], H=H[ll+1], psi[ll+1], psi[ll]) figure out the
# channel/s fanout as the kernel's role detection would.
function fanout_report(E, H_site, psi_site, tag)
    Hw = ITensors.get_external_storage(H_site); A = Hw.aliased
    hinds = inds(Hw); sinds = hinds[1:4]
    stat_pos = findfirst(i -> hasind(E, sinds[i]), 1:4)
    s_pos    = findfirst(i -> i != stat_pos && plev(sinds[i]) == 0 && hasind(psi_site, sinds[i]), 1:4)
    pairs = Set{Tuple{Int,Int}}()
    for k in A.keys; push!(pairs, (k[stat_pos], k[s_pos])); end
    nb = length(A.keys); np = length(pairs)
    @printf("  [%s] n_blocks=%d  distinct(channel,s)=%d  =>  step1 fanout (Redundancy #1) = %.2fx\n",
            tag, nb, np, nb/np)
end

println("\n== real aliased H structure (bulk sites near mid) ==")
inspect_H(PH.H[mid-1], "H[mid-1]")
inspect_H(PH.H[mid+1], "H[mid+1]")

# ---- correctness + timing for a makeL and a makeR step ----
densify(T) = SparseBackends.to_dense_itensors(T)

function test_step(E, H_site, psi_site, neighbor, tag; reps=200)
    fanout_report(E, H_site, psi_site, tag)
    Hd  = densify(H_site)
    ref = E * Hd * dag(prime(psi_site)) * psi_site
    # correctness for BOTH batched and per-block step3
    for bt in (false, true)
        fused = _fused_sparse_env_contract(E, H_site, psi_site, neighbor; batch_step3=bt)
        rel   = norm(fused - ref) / norm(ref)
        @printf("  [%s] batch_step3=%-5s correctness rel err = %.3e  %s\n",
                tag, bt, rel, rel < 1e-9 ? "PASS" : "*** FAIL ***")
    end
    for _ in 1:5
        _fused_sparse_env_contract(E, H_site, psi_site, neighbor; batch_step3=false)
        _fused_sparse_env_contract(E, H_site, psi_site, neighbor; batch_step3=true)
        E * H_site * dag(prime(psi_site)) * psi_site
        E * Hd * dag(prime(psi_site)) * psi_site
    end
    GC.gc(); t_perblk = @elapsed for _ in 1:reps; _fused_sparse_env_contract(E, H_site, psi_site, neighbor; batch_step3=false); end
    GC.gc(); t_batch  = @elapsed for _ in 1:reps; _fused_sparse_env_contract(E, H_site, psi_site, neighbor; batch_step3=true); end
    GC.gc(); t_alias  = @elapsed for _ in 1:reps; E * H_site * dag(prime(psi_site)) * psi_site; end
    GC.gc(); t_dense  = @elapsed for _ in 1:reps; E * Hd * dag(prime(psi_site)) * psi_site; end
    @printf("  [%s] fused(per-block)=%.1f us  fused(batched)=%.1f us   batch speedup=%.2fx\n",
            tag, 1e6*t_perblk/reps, 1e6*t_batch/reps, t_perblk/t_batch)
    @printf("  [%s] aliased-*(default)=%.1f us  dense-*=%.1f us   batched/default=%.2fx  batched/dense=%.2fx\n",
            tag, 1e6*t_alias/reps, 1e6*t_dense/reps, t_batch/t_alias, t_batch/t_dense)
end

# GEMM-efficiency probe: confirm each mul! operand is a StridedMatrix / Transpose
# thereof (routes to BLAS gemm!, no generic fallback) and that a warmed mul! is
# allocation-free (no hidden permute/copy). Reproduces the kernel's 3 GEMM shapes.
function gemm_check(E, H_site, psi_site, neighbor, tag)
    Hw = ITensors.get_external_storage(H_site); A = Hw.aliased
    hinds = inds(Hw); sinds = hinds[1:4]; dinds = hinds[5:6]
    stat_pos = findfirst(i -> hasind(E, sinds[i]), 1:4)
    s_pos    = findfirst(i -> i != stat_pos && plev(sinds[i]) == 0 && hasind(psi_site, sinds[i]), 1:4)
    sp_pos   = findfirst(i -> i != stat_pos && i != s_pos && plev(sinds[i]) != 0, 1:4)
    stat_ind, s_ind, sp_ind = sinds[stat_pos], sinds[s_pos], sinds[sp_pos]
    stat_dense_ind = hasind(E, dinds[1]) ? dinds[1] : dinds[2]
    new_dense_ind  = stat_dense_ind === dinds[1] ? dinds[2] : dinds[1]
    dl, dr = dim(stat_dense_ind), dim(new_dense_ind)
    psidag = dag(prime(psi_site))
    Eket = commonind(E, psi_site); Ebra = commonind(E, psidag)
    kNew = uniqueind(psi_site, neighbor, s_ind); kNewp = kNew'
    Earr = array(permute(E, stat_dense_ind, Ebra, Eket, stat_ind))
    psiarr = array(permute(psi_site, Eket, s_ind, kNew))
    psidagarr = array(permute(psidag, Ebra, sp_ind, kNewp))
    chiB, chiK = size(Earr,2), size(Earr,3); chiNew = dim(kNew); chiNewp = dim(kNewp)
    d1, d2 = dim(dinds[1]), dim(dinds[2]); TC = ComplexF64
    Emat  = reshape((@view Earr[:,:,:,1]), dl*chiB, chiK)
    psisl = @view psiarr[:,1,:]; pdsl = @view psidagarr[:,1,:]
    M1all = Array{TC}(undef, dl*chiB, chiNew); M2buf = Matrix{TC}(undef, chiB*chiNew, dr)
    Tst   = reshape((@view A.templates[1:A.blksize]), d1, d2)
    M1mat = reshape((@view M1all[:,:]), dl, chiB*chiNew)
    M2mat = reshape(M2buf, chiB, chiNew*dr)
    En    = Array{TC}(undef, chiNewp, chiNew*dr)
    _sm(x) = x isa StridedMatrix
    # Print each GEMM as: stored-A(dims) [flag] · stored-B(dims) [flag] -> C(dims),
    # reduction dim K, and WHERE K sits in the stored A (leading=needs 'T', trailing='N').
    # stride1==1 on every operand ⇒ contiguous view, BLAS takes it with no copy.
    println("  [$tag] GEMM index orders (reduction K; 'N'=K trailing in stored A, 'T'=K leading):")
    @printf("    step1  A=Emat%s['N']  · B=ψ_slice%s['N'] -> M1%s ;  K=chiK=%d  (Emat: K is TRAILING → in-order 'N')   strided?%s/%s stride1=%d/%d\n",
            string(size(Emat)), string(size(psisl)), string((dl*chiB,chiNew)), chiK, _sm(Emat), _sm(psisl), strides(Emat)[1], strides(psisl)[1])
    @printf("    step2  A=M1mat%s['T'] · B=Tstored%s['N'] -> M2%s ;  K=dl=%d   (M1mat: K is LEADING → 'T', BLAS contracts contiguous cols, no permute)   strided?%s/%s stride1=%d/%d\n",
            string(size(M1mat)), string(size(Tst)), string((chiB*chiNew,dr)), dl, _sm(M1mat), _sm(Tst), strides(M1mat)[1], strides(Tst)[1])
    @printf("    step3  A=ψ†_slice%s['T'] · B=M2mat%s['N'] -> Enew%s ;  K=chiB=%d (ψ†: K is LEADING → 'T')   strided?%s/%s stride1=%d/%d\n",
            string(size(pdsl)), string(size(M2mat)), string((chiNewp,chiNew*dr)), chiB, _sm(pdsl), _sm(M2mat), strides(pdsl)[1], strides(M2mat)[1])
    stat_is_left = stat_dense_ind === dinds[1]
    Top = stat_is_left ? Tst : transpose(Tst)   # replicate the kernel's step-2 orientation
    mul!(M1all, Emat, psisl); mul!(M2buf, transpose(M1mat), Top, one(TC), zero(TC)); mul!(En, transpose(pdsl), M2mat, one(TC), one(TC))
    a1 = @allocated mul!(M1all, Emat, psisl)
    a2 = @allocated mul!(M2buf, transpose(M1mat), Top, one(TC), zero(TC))
    a3 = @allocated mul!(En, transpose(pdsl), M2mat, one(TC), one(TC))
    @printf("    warmed @allocated: step1=%d B  step2=%d B  step3=%d B  (≈0 ⇒ BLAS gemm!, no hidden permute/copy)\n", a1, a2, a3)
end

_tg(I) = (dim(I), string(tags(I)), plev(I))
function profile_kernel(E, H, psi_site, neighbor, tag; reps=100)
    Hw = ITensors.get_external_storage(H); A = Hw.aliased
    hinds = inds(Hw); sinds = hinds[1:4]; dinds = hinds[5:6]
    stat_pos = findfirst(i -> hasind(E, sinds[i]), 1:4)
    s_pos    = findfirst(i -> i != stat_pos && plev(sinds[i]) == 0 && hasind(psi_site, sinds[i]), 1:4)
    sp_pos   = findfirst(i -> i != stat_pos && i != s_pos && plev(sinds[i]) != 0, 1:4)
    new_pos  = only(setdiff(1:4, (stat_pos, s_pos, sp_pos)))
    stat_ind, s_ind, sp_ind, new_ind = sinds[stat_pos], sinds[s_pos], sinds[sp_pos], sinds[new_pos]
    stat_dense_ind = hasind(E, dinds[1]) ? dinds[1] : dinds[2]
    new_dense_ind  = stat_dense_ind === dinds[1] ? dinds[2] : dinds[1]
    psidag = dag(prime(psi_site))
    Eket = commonind(E, psi_site); Ebra = commonind(E, psidag)
    kNew = uniqueind(psi_site, neighbor, s_ind); kNewp = kNew'

    Etgt   = (stat_dense_ind, Ebra, Eket, stat_ind)
    psitgt = (Eket, s_ind, kNew)
    pdtgt  = (Ebra, sp_ind, kNewp)
    println("  [$tag] E    inds = ", _tg.(collect(inds(E))))
    println("        E    want = ", _tg.(collect(Etgt)), "   identity? ", collect(inds(E)) == collect(Etgt))
    println("        psi  inds = ", _tg.(collect(inds(psi_site))), "   want ", _tg.(collect(psitgt)), "   identity? ", collect(inds(psi_site)) == collect(psitgt))
    println("        psidag inds=", _tg.(collect(inds(psidag))), "   want ", _tg.(collect(pdtgt)), "   identity? ", collect(inds(psidag)) == collect(pdtgt))

    tperm = 0.0; tperm_alias = 0.0
    for _ in 1:reps
        tperm += @elapsed (array(permute(E, Etgt...)); array(permute(psi_site, psitgt...)); array(permute(psidag, pdtgt...)))
        tperm_alias += @elapsed (array(permute(E, Etgt...; allow_alias=true)); array(permute(psi_site, psitgt...; allow_alias=true)); array(permute(psidag, pdtgt...; allow_alias=true)))
    end
    tfull = 0.0
    for _ in 1:reps; tfull += @elapsed _fused_sparse_env_contract(E, H, psi_site, neighbor); end
    @printf("        permute(3x, copy)=%.1f us   permute(allow_alias)=%.1f us   FULL kernel=%.1f us   => permute is %.0f%% of full\n",
            1e6*tperm/reps, 1e6*tperm_alias/reps, 1e6*tfull/reps, 100*tperm/tfull)
end

println("\n== index layout + permute cost ==")
profile_kernel(PH.LR[mid-2], PH.H[mid-1], psi[mid-1], psi[mid-2], "makeL")
profile_kernel(PH.LR[mid+2], PH.H[mid+1], psi[mid+1], psi[mid+2], "makeR")

println("\n== GEMM operand / allocation check ==")
gemm_check(PH.LR[mid-2], PH.H[mid-1], psi[mid-1], psi[mid-2], "makeL")
gemm_check(PH.LR[mid+2], PH.H[mid+1], psi[mid+1], psi[mid+2], "makeR")

println("\n== makeL step (ll=mid-2): E=LR[mid-2], H=H[mid-1], psi[mid-1], psi[mid-2] ==")
test_step(PH.LR[mid-2], PH.H[mid-1], psi[mid-1], psi[mid-2], "makeL")

println("\n== makeR step (rl=mid+2): E=LR[mid+2], H=H[mid+1], psi[mid+1], psi[mid+2] ==")
test_step(PH.LR[mid+2], PH.H[mid+1], psi[mid+1], psi[mid+2], "makeR")

println("\n== fused 2-site MATVEC (Heff·v) vs stock product(PH,v) ==")
let
    v   = psi[mid] * psi[mid+1]                    # 2-site wavefunction spanning sites mid,mid+1
    ctx = build_matvec_context(PH, mid)
    # ---- hidden-reuse report: within a (lpos,s1)/(mpos,s2) cell, records share the SAME
    # operand (M1mat / x2mat); if they also share templates (tid), step B/C recompute the
    # identical template·operand GEMM. Count that redundancy. ----
    let
        function reuse(stepM, tag)
            cells=0; totrec=0; totuniq=0; maxrec=0
            for cell in stepM
                isempty(cell) && continue
                cells += 1; r = length(cell); u = length(unique(x->x.tid, cell))
                totrec += r; totuniq += u; maxrec = max(maxrec, r)
            end
            @printf("  [%s] nonempty cells=%d  records=%d  Σ distinct-tid-per-cell=%d  max recs/cell=%d  => %d redundant template-GEMMs (%.2fx)\n",
                    tag, cells, totrec, totuniq, maxrec, totrec-totuniq, totrec/max(totuniq,1))
        end
        reuse(ctx.step1, "step1/B (H1)")
        reuse(ctx.step2, "step2/C (H2)")
        nt1 = length(ctx.templates1)÷ctx.blksize1; nt2 = length(ctx.templates2)÷ctx.blksize2
        @printf("  templates: H1 n_templates=%d (blksize %d)  H2 n_templates=%d (blksize %d)\n", nt1, ctx.blksize1, nt2, ctx.blksize2)
        @printf("  GEMM contraction dims K:  stepA K=chiL_ket=%d   stepB K=dl1=%d   stepC K=dl2=%d   stepD K=dr2*chiR_ket=%d  (small K ⇒ skinny/mem-bound)\n",
                ctx.chiL_ket, ctx.dl1, ctx.dl2, ctx.dr2*ctx.chiR_ket)
    end
    scr = build_batched_scratch(ctx)                  # hoisted batched scratch, reused across matvecs
    ssr = build_streaming_scratch(ctx)                # streaming scratch (fan-in=1)
    wr  = product(PH, v)
    wf  = fused_matvec_batched(ctx, v, scr)           # batched = the correctness oracle
    rel = norm(wf - wr) / norm(wr)
    @printf("  BATCHED matvec  vs stock rel err = %.3e  %s\n", rel, rel < 1e-9 ? "PASS" : "*** FAIL ***")

    # ---- streaming variant (fan-in=1) on the REAL Hamiltonian tensor ----
    wfs   = fused_matvec_streaming(ctx, v, ssr)
    rels  = norm(wfs - wr) / norm(wr)            # vs stock product
    relsb = norm(wfs - wf) / norm(wf)            # vs batched (must be ~machine-eps)
    ords  = collect(inds(wfs)) == collect(inds(v))    # streaming output order == v?
    @printf("  STREAMING matvec vs stock rel = %.3e  %s   vs batched rel = %.3e   out-order==v? %s\n",
            rels, rels < 1e-9 ? "PASS" : "*** FAIL ***", relsb, ords)
    # localize: per-(s1,s2) physical block relative error
    let s1i = noprime(ctx.s1_ind), s2i = noprime(ctx.s2_ind), Lb = noprime(ctx.Lbra), Rb = noprime(ctx.Rbra)
        af = Array(wf, Lb, s1i, s2i, Rb); ar = Array(wr, Lb, s1i, s2i, Rb)
        println("  per-(s1,s2) block rel err:")
        for a in 1:dim(s1i)
            row = [ (n=norm(@view(ar[:,a,b,:])); n<1e-14 ? 0.0 : norm(@view(af[:,a,b,:]) .- @view(ar[:,a,b,:]))/n) for b in 1:dim(s2i) ]
            println("    s1=$a: ", round.(row, sigdigits=3))
        end
    end
    for _ in 1:5; fused_matvec_batched(ctx, v, scr); fused_matvec_streaming(ctx, v, ssr); product(PH, v); end
    reps = 200
    GC.gc(); tfb = @elapsed for _ in 1:reps; fused_matvec_batched(ctx, v, scr); end
    GC.gc(); tfs = @elapsed for _ in 1:reps; fused_matvec_streaming(ctx, v, ssr); end
    GC.gc(); tr  = @elapsed for _ in 1:reps; product(PH, v); end
    afb = @allocated fused_matvec_batched(ctx, v, scr)
    afs = @allocated fused_matvec_streaming(ctx, v, ssr)
    ar  = @allocated product(PH, v)
    @printf("  fused_matvec_batched   = %.1f us  (%d B/call)   batched/stock = %.2fx\n", 1e6*tfb/reps, afb, tfb/tr)
    @printf("  fused_matvec_streaming = %.1f us  (%d B/call)   streaming/stock = %.2fx   streaming/batched = %.2fx\n",
            1e6*tfs/reps, afs, tfs/tr, tfs/tfb)
    @printf("  stock product          = %.1f us  (%d B/call)\n", 1e6*tr/reps, ar)
    # ---- GEMM count + FLOPs per call (deterministic) ----
    _GEMM_COUNT[] = true
    reset_gemm_count!(); fused_matvec_batched(ctx, v, scr);   gb_n = _GEMM_N[]; gb_f = _GEMM_FLOPS[]
    reset_gemm_count!(); fused_matvec_streaming(ctx, v, ssr); gs_n = _GEMM_N[]; gs_f = _GEMM_FLOPS[]
    _GEMM_COUNT[] = false
    @printf("  GEMM/call: batched   = %d gemms, %.3e flops\n", gb_n, gb_f)
    @printf("  GEMM/call: streaming = %d gemms, %.3e flops\n", gs_n, gs_f)
    # ---- per-step breakdown (separate loop: time_ns probes add overhead) ----
    _FUSED_PROF[] = true; reset_fused_prof!()
    for _ in 1:reps; fused_matvec_batched(ctx, v, scr); end
    _FUSED_PROF[] = false
    println("  [batched per-step]"); report_fused_prof!(reps)
    _FUSED_PROF[] = true; reset_fused_prof!()
    for _ in 1:reps; fused_matvec_streaming(ctx, v, ssr); end
    _FUSED_PROF[] = false
    println("  [streaming per-step]"); report_fused_prof!(reps)
end

println("\ndone.")
end # main

main()
