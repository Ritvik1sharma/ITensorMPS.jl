# dmrg_php.jl — fused-kernel dispatch helpers for the aliased-PHP + dense-ψ case.
#
# The generic `dmrg` loop stays the SINGLE code path (ground + excited + observer /
# early-exit / truncation all shared). When `run_mode == :fused` and the operator is an
# aliased P†HP MPO, the loop swaps the matvec kernel at its eigsolve dispatch point via
# `_dmrg_fused_matvec` below (φ is pinned to (Lket,Rket,s2,s1) there). No forked driver,
# no global flag — `run_mode` is a plain argument threaded to the loop.
#
# [Follow-up: a matching `position!` dispatch will select the fused env
#  (fused_sparse_makeL!/makeR!); the env is currently built by the stock position!.]

# Inner H-operator ProjMPO of a ProjMPO (ground) / ProjMPO_MPS (excited); else nothing
# (e.g. ProjMPOSum → no fused path, falls back to the stock matvec).
_dmrg_inner_projmpo(PH::ProjMPO)     = PH
_dmrg_inner_projmpo(PH::ProjMPO_MPS) = PH.PH
_dmrg_inner_projmpo(PH)              = nothing

# Is `H` an aliased P†HP MPO (any site tensor is WrappedAliasedBlockSparse)?
# (`_is_aliased_itensor` from abstractprojmpo.jl.)
_is_aliased_mpo(H::MPO) = any(j -> _is_aliased_itensor(H[j]), 1:length(H))

# Does PH's operator MPO qualify for the fused matvec?
function _dmrg_op_is_aliased(PH)
  ph = _dmrg_inner_projmpo(PH)
  ph === nothing && return false
  return _is_aliased_mpo(ph.H)
end

# Build the partial-fused matvec operator for bond b (caller guarantees a bulk,
# aliased-PHP bond, PH positioned at b). The full-wrapper `matvec_partial_fused_full`
# owns all 4 legs (step-1 Lenv·φ, fused H1·H2, step-4 ·Renv) and returns Hv in φ's
# pinned (Lket,Rket,s2,s1) order (round-trip TRUE ⇒ Krylov-consistent).
#   ProjMPO (ground)     → pure fused H·v.
#   ProjMPO_MPS (excited) → fused H·v PLUS the stock weight·Σ|Mᵢ⟩⟨Mᵢ| projector terms
#                           (ITensor `+` aligns by index; the small projector matvecs are
#                           left stock — the fused kernel is single-H).
_dmrg_fused_matvec(PH::ProjMPO, b::Int) =
  (ctx = build_partial_matvec_context(PH, b); v -> matvec_partial_fused_full(ctx, v))

function _dmrg_fused_matvec(PH::ProjMPO_MPS, b::Int)
  ctx = build_partial_matvec_context(PH.PH, b)
  return function (v)
    Hv = matvec_partial_fused_full(ctx, v)
    @inbounds for p in PH.pm
      Hv += PH.weight * product(p, v)
    end
    return Hv
  end
end
