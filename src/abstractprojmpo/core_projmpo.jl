# abstractprojmpo/core_projmpo.jl — factor-core DMRG operator + (later) CoreProjMPO.
#
# Factor-core scheme (diagonal-P, e.g. PXP): the stored state is the aliased
# ψ = P·core; the variable is `core` (the aliased templates); the local metric is I.
# DMRG uses a DUAL operator:
#   • bare H          — for position!/env growth (aliased ψ bra+ket supply the two
#                       P's for off-window sites, so the env carries P†…P with open
#                       P-FSM "channel" indices).
#   • PH = build_ph_output(P,H) — the matvec operator: P on H's OUTPUT indices only
#                       (single P; the ket P comes from φ=ψ[b]ψ[b+1]). Its P-FSM
#                       bonds are at plev 1 and close the env's (primed) bra-channels,
#                       so the matvec output is a channel-free bare core.
#
# build_ph_output validated (test_factor_core/ph_matvec_oneshot_test.jl, PXP N=12):
# single- AND two-site v_o' channel-free, ⟨c|v_o'⟩ == dense ⟨φ|P†HP|φ⟩ to 3e-16.

"""
    build_ph_output(P::MPO, H::MPO) -> MPO

Factor-core matvec operator: `P` applied to `H`'s **output (bra)** indices only.

For each site: raise `H`'s output site index onto `P`'s input, leaving site indices
`ket@0 / bra@1`, `H`'s MPO bonds at plev 0, and `P`'s FSM (channel) bonds at plev 1.
The plev-1 FSM bonds match the env's primed bra-channel (built from aliased ψ), so
`Lenv·φ·PH[b]·PH[b+1]·Renv` contracts every channel and returns a channel-free core.

Assumes `P` and `H` share site indices and `P` is physically diagonal (PXP).
`P` must carry `"Link"`-tagged FSM bonds (as produced by the constraint MPO).
"""
function build_ph_output(P::MPO, H::MPO)
    length(P) == length(H) || error("build_ph_output: length(P)=$(length(P)) != length(H)=$(length(H))")
    Ovec = Vector{ITensor}(undef, length(H))
    for j in eachindex(Ovec)
        # Prime by EXPLICIT index objects, NOT tag filters. `prime(P; tags="Link")`
        # silently primes EVERY index for a COO-built P (e.g. KL's mulMPO-folded P) —
        # its tag matching doesn't survive the COO contraction — so the site ended up 2
        # levels too high and never met H's bra (→ un-contracted, doubled-site PH). H is
        # a clean hand-built MPO, so identify P's legs relative to H by id:
        #   • FSM (Link) bonds = P legs NOT shared with H  → prime to plev 1 (meet env
        #     bra-channel);
        #   • physical site legs = P legs shared with H    → prime +1 so P's ket (0->1)
        #     meets H's bra (plev 1) and P's bra (1->2) becomes the new output.
        Pj = P[j]
        Pj = prime(Pj, ITensors.uniqueinds(Pj, H[j]))     # FSM bonds: 0 -> 1
        Pj = prime(Pj, ITensors.commoninds(Pj, H[j]))     # sites: ket 0->1, bra 1->2
        # Single-P P·H (P on H's output) via COO × DENSE → aliased (the proven KL
        # sandwich_mpo form). H[j] is a plain dense tensor ⇒ :dense; the shared site
        # (P's raised ket == H's bra, both plev 1) contracts, giving a clean ket/bra PH.
        Oj  = SparseBackends.contract(Pj, H[j], :coo, :dense, :aliased)
        Ovec[j] = replaceprime(Oj, 2 => 1)                # P's bra (only plev-2 leg) -> 1
    end
    return MPO(Ovec)
end

"""
    CoreProjMPO(H::MPO, P::MPO; nsite=2)

Dual-operator projected MPO for the factor-core matvec. Holds a plain `ProjMPO(H)`
for **env growth** (`position!` with bare H — aliased ψ supplies the two P's for
off-window sites) and the matvec operator `PH = build_ph_output(P,H)`.

`product(cpm, φ)` runs `Lenv·φ·PH[sites]·Renv` and `noprime`s the result. Unlike the
stock ProjMPO product, the output space DIFFERS from the input: the aliased φ (with
P-FSM channel legs) maps to a **channel-free bare core** `v_o' = P†HP·core`. The
Krylov wrapper (aliased-φ + multiplicity inner) re-routes that core back to an
aliased φ between matvecs; `product` itself just returns the core.
"""
struct CoreProjMPO
    Hbare::ProjMPO      # env growth with bare H (mutated in place by position!)
    PH::MPO             # matvec operator: P on H's output only
end

function CoreProjMPO(H::MPO, P::MPO; nsite::Int=2)
    pm = ProjMPO(H)
    set_nsite!(pm, nsite)
    return CoreProjMPO(pm, build_ph_output(P, H))
end

nsite(cpm::CoreProjMPO)      = nsite(cpm.Hbare)
lproj(cpm::CoreProjMPO)      = lproj(cpm.Hbare)
rproj(cpm::CoreProjMPO)      = rproj(cpm.Hbare)
site_range(cpm::CoreProjMPO) = site_range(cpm.Hbare)
set_nsite!(cpm::CoreProjMPO, n::Int) = (set_nsite!(cpm.Hbare, n); cpm)
position!(cpm::CoreProjMPO, psi::MPS, b::Int) = (position!(cpm.Hbare, psi, b); cpm)

"""
    product(cpm::CoreProjMPO, φ::ITensor) -> ITensor  (channel-free core)

`Lenv·φ·PH[sites]·Renv`, `noprime`d. Contraction order is Lenv·φ first (matches the
validated matvec order). Output carries no P-FSM channel index.
"""
function product(cpm::CoreProjMPO, phi::ITensor)
    L = lproj(cpm.Hbare); R = rproj(cpm.Hbare)
    v = phi
    L !== nothing && (v = L * v)
    @inbounds for j in site_range(cpm.Hbare)
        v = v * cpm.PH[j]
    end
    R !== nothing && (v = R * v)
    return noprime(v)
end

(cpm::CoreProjMPO)(phi::ITensor) = product(cpm, phi)
