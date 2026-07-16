# factor_core_dmrg.jl — self-contained core-mode DMRG (run_mode = :core_php).
#
# Separated from dmrg.jl (per design): the factor-core scheme keeps the stored state
# as the aliased ψ = P·core, optimizes the dense `core` with local metric I (NO M,
# no M-inversion), and writes the optimized core back into ψ each bond. Supports PXP
# (diagonal P) AND KL (flip P). Target: PXP N=12 → E = −14.77 (beats Path-B/RR −14.42).
#
# Local update at a bond (b, b+1) — the Krylov vector is the CHANNEL-FREE 2-site core:
#   1. read the dense 2-site core  c = read_core(ψ[b])·read_core(ψ[b+1]).
#   2. build the aliased window φ = ψ[b]·ψ[b+1] ONCE (fixed P structure for the joined
#      (s_b,s_{b+1}) space); eigsolve f(c) over dense cores (metric I) where each matvec
#      RE-ATTACHES the ket-P by 2-site write_core into φ (write_core_window!), then
#      applies product(CoreProjMPO, φ) = Lenv·φ·PH·Renv → new channel-free core. The
#      env's ket-channels close against φ's re-attached ket-P; PH's bra-P closes the
#      bra-channels ⇒ channel-free output.
#   3. svd(ground core) + two per-site write_core! → ψ[b], ψ[b+1] (P fixed).
# Envs come from CoreProjMPO (bare H + aliased ψ → P†…P); the matvec operator is
# PH = build_ph_output(P,H) (single P on H's output). See core_projmpo.jl.

using KrylovKit: eigsolve

_core_ext(t) = ITensors.get_external_storage(t)
_core_readcore(t) = SparseBackends.read_core(_core_ext(t))
function _core_densify(t::ITensor)
    ITensors.has_external_storage(t) || return t
    s = ITensors.get_external_storage(t)
    s isa SparseBackends.WrappedAliasedBlockSparse &&
        return ITensors.itensor(SparseBackends.to_dense(s.aliased), s.inds...)
    s isa SparseBackends.WrappedBlockSparse &&
        return ITensors.itensor(SparseBackends.to_dense(s.blocksparse), s.inds...)
    return t
end

# The ket-side P is the aliased STRUCTURE of ψ, not a contraction (plan §5 step 0:
# "route core → aliased φ — not a contraction (template placement)"). φ = ψ[b]·ψ[b+1]
# already IS P·core (P baked into the aliased site tensors), so forming φ for the
# EVOLVING Krylov core is template placement into ψ's FIXED window structure — done
# by the existing SparseBackends.write_core! (generalized to the 2-Site window),
# writing the dense core into the aliased φ buffer IN PLACE. No P contraction.

# Dense window core = ∏ read_core(ψ[j]) over `sites`.
function _core_read_window(psi::MPS, sites)
    c = _core_readcore(psi[first(sites)])
    for j in sites[2:end]; c = c * _core_readcore(psi[j]); end
    return c
end

# Resize-capable write_core!: drop a dense single-site `core` into ψ[j]'s aliased
# template slots. Keeps the P structure (keys/alias_ids/scalars, P-FSM channels)
# FIXED; the core-link Index/dim/blksize come from `core` (so the bond may grow or
# shrink). Returns a fresh aliased ITensor (the core-link Index changes).
function _core_rebuild(t::ITensor, core::ITensor)
    w = _core_ext(t); a = w.aliased
    Pn = SparseBackends._abs_head_len(w)
    prefix_inds = w.inds[1:Pn]
    phys_pos = findfirst(i -> ITensors.hastags(w.inds[i], "Site"), 1:Pn)
    phys_ind = w.inds[phys_pos]
    new_dense_inds = [i for i in inds(core) if !ITensors.hastags(i, "Site")]
    new_dense_dims = Tuple(ITensors.dim(i) for i in new_dense_inds)
    core_arr = Array(core, phys_ind, new_dense_inds...)
    new_bs = prod(new_dense_dims)
    s2t = SparseBackends.slice_to_template(w)
    new_templates = Vector{eltype(a.templates)}(undef, a.n_templates * new_bs)
    tail_ci = CartesianIndices(new_dense_dims)
    seen = falses(a.n_templates)
    @inbounds for s in 1:a.dims[phys_pos]
        tid = s2t[s]; tid == 0 && continue
        seen[tid] && error("factor-core: template $tid shared by 2 slices"); seen[tid] = true
        off = (tid - 1) * new_bs
        for (lin, ci) in enumerate(tail_ci)
            new_templates[off + lin] = core_arr[s, Tuple(ci)...]
        end
    end
    new_dims = (ntuple(i -> a.dims[i], Pn)..., new_dense_dims...)
    new_ali  = typeof(a)(new_dims, new_bs, new_templates, a.n_templates,
                         copy(a.keys), copy(a.alias_ids), copy(a.scalars))
    # preserve the pre-P routing map (P structure unchanged) so read_core stays general
    # (off-diagonal P) across the sweep instead of re-deriving by output-site grouping.
    isempty(a.slice_to_template) || (new_ali.slice_to_template = copy(a.slice_to_template))
    return ITensors._itensor_from_external_storage(typeof(w)(new_ali, (prefix_inds..., new_dense_inds...)))
end

# Write the ground 2-site core `cg` back into ψ[b], ψ[b+1].
# ha=1 (L→R): left-iso at b;  ha=2 (R→L): right-iso at b+1. Truncates on the core bond.
function _core_writeback!(psi::MPS, b::Int, cg::ITensor, ha::Int; maxdim::Int, cutoff::Real, mindim::Int=1)
    left_inds = commoninds(cg, _core_readcore(psi[b]))         # (s_b, left core-link)
    F = svd(cg, left_inds...; lefttags="Link,l=$b", maxdim=maxdim, mindim=mindim, cutoff=cutoff)
    U, S, V = F.U, F.S, F.V
    cb, cb1 = ha == 1 ? (U, S * V) : (U * S, V)
    psi[b]   = _core_rebuild(psi[b],   cb)
    psi[b+1] = _core_rebuild(psi[b+1], cb1)
    return psi
end

# Initial core RIGHT-canonicalization (mixed-canonical center → bond 1), no eigensolve
# and no truncation, so the per-bond eigenproblem is genuinely metric I from the start.
function _core_canonicalize!(psi::MPS)
    N = length(psi)
    for b in (N-1):-1:1
        _core_writeback!(psi, b, _core_read_window(psi, b:b+1), 2; maxdim=typemax(Int), cutoff=0.0)
    end
    return psi
end

# Physical energy of the aliased state: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ (dense contraction; the
# metric-I eigenvalue is c·E so we read the true energy off ψ directly — no M).
function core_php_energy(psi::MPS, H::MPO)
    pd = MPS([_core_densify(psi[j]) for j in 1:length(psi)])
    return real(inner(pd', H, pd) / inner(pd, pd))
end

"""
    dmrg_core_php(H::MPO, P::MPO, psi0::MPS; nsweeps, maxdim, cutoff=1e-12,
                  eigsolve_krylovdim=30, eigsolve_tol=1e-12, outputlevel=1)
        -> (energy, psi)

Core-mode DMRG (factor-core, diagonal P). `psi0` is the aliased ψ = P·core. Optimizes
the core with metric I (no M), keeping ψ aliased. Returns the physical ground energy
and the optimized aliased ψ. Delegated to from `dmrg(...; run_mode=:core_php, P=P)`.
"""
function dmrg_core_php(H::MPO, P::MPO, psi0::MPS; nsweeps::Int, maxdim::Int, mindim::Int=1,
                       cutoff::Real=1e-12, eigsolve_krylovdim::Int=30,
                       eigsolve_tol::Real=1e-12, outputlevel::Int=1)
    N = length(psi0); psi = copy(psi0)
    cpm = CoreProjMPO(H, P; nsite=2)
    _core_canonicalize!(psi)
    local E = core_php_energy(psi, H)
    outputlevel > 0 && @printf("[core_php] init  E=%.10f\n", E)
    for sw in 1:nsweeps
        t_sweep = @elapsed for (b, ha) in sweepnext(N)
            position!(cpm, psi, b)
            # eigsolve vector = the 2-site CHANNEL-FREE core. Build the aliased window
            # φ = ψ[b]·ψ[b+1] ONCE (fixed P structure for the joined (s_b,s_{b+1}) space);
            # reuse it as the matvec buffer. Each matvec RE-ATTACHES the evolving core's
            # ket-P by 2-site write_core into φ (so the env's ket-channels close), then
            # applies the operator → new channel-free core. In-place is safe: product
            # fully contracts φ each call.
            core0   = _core_read_window(psi, b:b+1)
            # Build the aliased window with emit_window_map=true so the AliasedBS×AliasedBS
            # contract EMITS the joined pre-P (rv_b,rv_{b+1})→template map (the routing that
            # can't be recovered post-hoc for flip P). Correct for diagonal AND flip P.
            phi_buf = SparseBackends.contract(psi[b], psi[b+1], :aliased, :aliased, :aliased;
                                              preserve_bs_output=true, emit_window_map=true)
            w_buf   = ITensors.get_external_storage(phi_buf)
            wmap    = SparseBackends.window_write_map(w_buf,
                          ITensors.get_external_storage(psi[b]),
                          ITensors.get_external_storage(psi[b+1]))
            f(core) = (SparseBackends.write_core_window!(w_buf, core, wmap); product(cpm, phi_buf))
            _, vecs = eigsolve(f, core0, 1, :SR; ishermitian=true,
                               krylovdim=eigsolve_krylovdim, tol=eigsolve_tol)
            _core_writeback!(psi, b, vecs[1], ha; maxdim=maxdim, mindim=mindim, cutoff=cutoff)
        end
        E = core_php_energy(psi, H)
        mbd = maximum(j -> maximum(ITensors.dim, ITensors.get_external_storage(psi[j]).inds), 1:N)
        outputlevel > 0 && @printf("[core_php] sweep %2d  E=%.10f  t_sweep=%.3fs  maxbond=%d\n",
                                   sw, E, t_sweep, mbd)
    end
    return E, psi
end
