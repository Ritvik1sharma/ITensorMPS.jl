using Adapt: adapt
using KrylovKit: eigsolve, InnerProductVec
import VectorInterface
using NDTensors: scalartype, timer
using Printf: @printf
using TupleTools: TupleTools
import SparseBackends

# Tell KrylovKit/VectorInterface what scalar type an ITensor uses when it's
# wrapped in InnerProductVec for the M-inner-product Lanczos path. Defined at
# module scope so it's resolved once, not in the inner dmrg loop.
VectorInterface.scalartype(::Type{ITensors.ITensor}) = ComplexF64

# In-place aliased Lanczos add!!. The ITensorsVectorInterfaceExt extension runs `a + b*α`
# for external storage (2 allocs: the b*α copy + the Base.:+ plus_merge — the dominant
# aliased-Krylov add cost). For key-aligned dedup-1 aliased operands we do a true in-place
# axpby (zero alloc, no merge). The ::Real/::Complex methods below are strictly MORE SPECIFIC
# than the extension's add!!(…,::Number) (so no precompile "method overwriting" error) and
# cover every concrete α Lanczos produces. Falls back to the extension's exact behavior when
# operands aren't key-aligned aliased storage ⇒ byte-identical there.
function _aliased_addbang!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number, β::Number)
  SparseBackends._ADD_INPLACE_TRY[] += 1
  awa = SparseBackends._alias_storage(a); bwa = SparseBackends._alias_storage(b)
  if awa !== nothing && bwa !== nothing &&
     SparseBackends._alias_inplace_axpby!(awa, bwa, α, β)
    return a
  end
  SparseBackends._ADD_INPLACE_FAIL[] += 1
  if ITensors.has_external_storage(a)
    result = a * β + b * α
    a.tensor = result.tensor
    return a
  end
  if promote_type(eltype(a), eltype(b), typeof(α), typeof(β)) <: eltype(a)
    return VectorInterface.add!(a, b, α, β)
  end
  return a * β + b * α
end
VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Real) =
  _aliased_addbang!(a, b, α, one(α))
VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Complex) =
  _aliased_addbang!(a, b, α, one(α))
VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Real, β::Real) =
  _aliased_addbang!(a, b, α, β)
VectorInterface.add!!(a::ITensors.ITensor, b::ITensors.ITensor, α::Complex, β::Complex) =
  _aliased_addbang!(a, b, α, β)
# NOTE: the analogous Lanczos `inner` can't be overridden the same way — inner(::ITensor,
# ::ITensor) matches the extension exactly (no more-specific variant) and an __init__/@eval
# install breaks the ChainRulesCore extension's precompile. So `inner` keeps the extension's
# path; the dot speedup will instead happen at the SparseBackends contraction level.

function naive_overlap(psi1::MPS, psi2::MPS)
  psi1dag = dag(psi1)
  # ITensors.replaceprime!(psi1dag, 0 => 1)  # match primes
  return ITensors.inner(psi1', psi2)
end

function permute(
  M::AbstractMPS, ::Tuple{typeof(linkind),typeof(siteinds),typeof(linkind)}
)::typeof(M)
  M̃ = typeof(M)(length(M))
  for n in 1:length(M)
    if ITensors.has_external_storage(M[n])
      lₙ₋₁ = n > 1 ? commonind(M[n], M[n-1]) : nothing
      lₙ   = n < length(M) ? commonind(M[n], M[n+1]) : nothing
      s⃗ₙ   = TupleTools.sort(Tuple(siteinds(M, n)); by=plev)
      expected = filter(!isnothing, (lₙ₋₁, s⃗ₙ..., lₙ))
      @assert all(i -> i ∈ inds(M[n]), expected) "Site $n missing expected MPS indices: $expected"
      M̃[n] = M[n]
      continue
    end
    lₙ₋₁ = linkind(M, n - 1)
    lₙ = linkind(M, n)
    s⃗ₙ = TupleTools.sort(Tuple(siteinds(M, n)); by=plev)
    M̃[n] = ITensors.permute(M[n], filter(!isnothing, (lₙ₋₁, s⃗ₙ..., lₙ)))
  end
  set_ortho_lims!(M̃, ortho_lims(M))
  return M̃
end

function dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; P::Union{Nothing,MPO}=nothing,
             run_mode::Symbol=:standard,
             target_energy=nothing, use_early_exit=true, last_sweep_energy=nothing, kwargs...)
  run_mode in (:standard, :fused) ||
    error("dmrg: unknown run_mode=$(run_mode); expected :standard or :fused")
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # A plain block-sparse ψ (WrappedBlockSparse, not aliased) is not supported by this
  # front-end: the only sparse-ψ path is factor-core, which expects an aliased ψ = P·core.
  # Error out explicitly rather than misrouting a bare BS ψ into dmrg_core_php.
  !SparseBackends.is_blocksparse_mps(psi0) ||
    error("dmrg: a block-sparse (non-aliased) ψ is not supported. The only sparse-ψ path is " *
          "the factor-core method (aliased ψ = P·core, via `P=`); use a dense ψ, or an aliased " *
          "ψ with the constraint MPO `P=`.")
  # Aliased-ψ + dense-H is served ONLY by the factor-core method (ψ = P·core, metric I).
  # The old Path-B aliased run modes have been removed. Delegate to dmrg_core_php, which
  # needs the constraint MPO P explicitly (unlike Path-B, which derived M from the H-env).
  if P !== nothing || SparseBackends.is_sparse_mps(psi0)
    P !== nothing ||
      error("dmrg: an aliased ψ (is_sparse_mps) requires the constraint MPO `P=` for the " *
            "factor-core method; the Path-B aliased-ψ path has been removed.")
    n = nsweep(sweeps)
    # factor-core takes scalar dims (no per-sweep schedule yet) → use the final sweep's.
    return dmrg_core_php(H, P, psi0;
      nsweeps=n, maxdim=maxdim(sweeps, n), mindim=mindim(sweeps, n), cutoff=cutoff(sweeps, n),
      eigsolve_krylovdim=get(kwargs, :eigsolve_krylovdim, 30),
      eigsolve_tol=get(kwargs, :eigsolve_tol, 1e-12),
      outputlevel=get(kwargs, :outputlevel, 1))
  end
  # run_mode threads into the single generic loop below. For an aliased-PHP operator +
  # dense ψ it selects the matvec (and, later, env) kernel at the dispatch points; for a
  # plain dense-H operator it is inert (the fused branches gate on an aliased H). Excited
  # / Vector{MPO} runs receive run_mode via kwargs → the same inner loop.
  PH = ProjMPO(H)
  return dmrg(PH, psi0, sweeps; run_mode, target_energy, use_early_exit, kwargs...)
end

function constrained_dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; parMPO=nothing, kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  if parMPO === nothing
    println("Type  --  4")
    PH = ProjMPO(H)
    return dmrg(PH, psi0, sweeps; target_energy, kwargs...)  
  else
    println("Type  --  3")
    return constrained_dmrg2(H, psi0, sweeps; parMPO=parMPO, kwargs...)
  end
end

function dmrg(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)
  !SparseBackends.is_sparse_mps(psi0) ||
    error("dmrg(Vector{MPO}, …): aliased ψ is not supported here — factor-core covers only " *
          "the single-H case (use dmrg(H, psi0; P=P)); the Path-B aliased path was removed.")
  for H in Hs
    check_hascommoninds(siteinds, H, psi0)
    check_hascommoninds(siteinds, H, psi0')
  end
  Hs .= permute.(Hs, Ref((linkind, siteinds, linkind)))
  PHS = ProjMPOSum(Hs)
  println("Type5")
  return dmrg(PHS, psi0, sweeps; kwargs...)
end

function dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight=true, kwargs...)
  !SparseBackends.is_sparse_mps(psi0) ||
    error("dmrg (excited-state): aliased ψ is not supported — factor-core covers only the " *
          "ground-state single-H case (dmrg(H, psi0; P=P)); the Path-B aliased path was removed.")
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  for M in Ms
    check_hascommoninds(siteinds, M, psi0)
  end
  H = permute(H, (linkind, siteinds, linkind))
  Ms .= permute.(Ms, Ref((linkind, siteinds, linkind)))
  if weight <= 0
    error(
      "weight parameter should be > 0.0 in call to excited-state dmrg (value passed was weight=$weight)",
    )
  end
  PMM = ProjMPO_MPS(H, Ms; weight)
  println("Type5")
  return dmrg(PMM, psi0, sweeps; kwargs...)
end

# Align link indices of t2 to match t1 by tag name.
# to_dense_itensors already fuses doubled links, so just match by tag and replace ids.
# Returns nothing on dim mismatch.
function align_links(t1::ITensor, t2::ITensor, label::String; debug=false)
  links1 = filter(i -> hastags(i, "Link"), inds(t1))
  links2 = filter(i -> hastags(i, "Link"), inds(t2))
  old_inds = Index{Int64}[]
  new_inds = Index{Int64}[]
  for l2 in links2
    base_tag = tags(l2)
    # Match by tag and dim; skip already-consumed indices so that two old
    # indices sharing the same (tag, dim) get paired to distinct current indices.
    matches = filter(l1 -> tags(l1) == base_tag && dim(l1) == dim(l2) && l1 ∉ new_inds, links1)
    if isempty(matches)
      println("  [$label] no matching link for tag $base_tag (dim=$(dim(l2)))")
      return nothing
    end
    l1 = first(matches)
    push!(old_inds, l2)
    push!(new_inds, l1)
  end
  return replaceinds(t2, old_inds, new_inds)
end

function merge_t2_to_match_t1(T1::ITensor, T2::ITensor)::ITensor
    idx1 = collect(inds(T1))
    result = T2
    exact_key(idx) = (string(tags(idx)), plev(idx), dim(idx))
    tag_key(idx) = string(tags(idx))
    # Step 1: mark exact matches already present in T2
    cur = collect(inds(result))
    used = falses(length(cur))
    for t1 in idx1
        for j in eachindex(cur)
            if !used[j] && exact_key(cur[j]) == exact_key(t1)
                used[j] = true
                break
            end
        end
    end
    # Step 2: for each unmatched T1 index, merge leftover T2 indices
    # with same tag string whose dims multiply to target dim
    for t1 in idx1
        cur = collect(inds(result))
        already_present = any(i -> exact_key(cur[i]) == exact_key(t1), eachindex(cur))
        already_present && continue
        cands = Int[]
        for j in eachindex(cur)
            if tag_key(cur[j]) == tag_key(t1)
                push!(cands, j)
            end
        end
        # Keep only those not already exact-matched to some T1 index
        filtered = Int[]
        for j in cands
            is_exact_match = any(x -> exact_key(cur[j]) == exact_key(x), idx1)
            !is_exact_match && push!(filtered, j)
        end
        merged = false
        # try pairs
        for a in 1:length(filtered)-1, b in a+1:length(filtered)
            j1, j2 = filtered[a], filtered[b]
            if dim(cur[j1]) * dim(cur[j2]) == dim(t1)
                C = ITensors.combiner(cur[j1], cur[j2]; tags=tags(t1))
                result = result * C
                merged = true
                break
            end
        end
        merged && continue

        # try triples if needed
        for a in 1:length(filtered)-2, b in a+1:length(filtered)-1, c in b+1:length(filtered)
            j1, j2, j3 = filtered[a], filtered[b], filtered[c]
            if dim(cur[j1]) * dim(cur[j2]) * dim(cur[j3]) == dim(t1)
                C = ITensors.combiner(cur[j1], cur[j2], cur[j3]; tags=tags(t1))
                result = result * C
                merged = true
                break
            end
        end
    end
    # println("final merged idx ", inds(result))
    return result
end

merge_t1_to_match_t2(T1::ITensor, T2::ITensor)::ITensor = merge_t2_to_match_t1(T2, T1)

# # Merge a 6-index projection tensor by positional pairs: (1,4), (2,5), (3,6).
# # Each pair is fused into a single combined index via ITensors.combiner.
# # Expected layout: (psi_left, psi'_left, H_left, psi_right, psi'_right, H_right)
# function merge_paired_inds(R::ITensor)::ITensor
#   idx = inds(R)
#   if length(idx) === 4
#     println("original idx 1: ", idx)
#     C1 = ITensors.combiner(idx[3], idx[4]; tags=tags(idx[1]))
#     # C2 = ITensors.combiner(idx[2], idx[4]; tags=tags(idx[2]))
#     # C3 = ITensors.combiner(idx[3], idx[6]; tags=tags(idx[3]))
#     result = R * C1
#     println("merged idx ", inds(result))
#     return result
#   elseif length(idx) === 5
#     println("original idx 2: ", idx)
#     C1 = ITensors.combiner(idx[1], idx[3]; tags=tags(idx[1]))
#     C2 = ITensors.combiner(idx[2], idx[4]; tags=tags(idx[2]))
#     # C3 = ITensors.combiner(idx[3], idx[6]; tags=tags(idx[3]))
#     result = R * C1 * C2
#     println("merged idx ", inds(result))
#     return result
#   else
#     println("Unexpected number of indices in projection tensor: ", length(idx))
#     return R
#   end
# end

function canonicalize_phi_phase(phi::ITensor; atol=1e-12)
    arr     = array(phi)
    arr_abs = abs.(arr)
    amax    = maximum(arr_abs)
    amax < atol && return phi
    # Weighted-sum phase: aggregates all significant elements so a single
    # near-zero dominant entry cannot flip the sign.
    z = zero(eltype(arr))
    @inbounds for i in eachindex(arr)
        ai = arr_abs[i]
        ai > atol && (z += ai * arr[i])
    end
    abs(z) < atol && return phi

    return (conj(z) / abs(z)) * phi
end


using NDTensors.TypeParameterAccessors: unwrap_array_type
"""
    dmrg(H::MPO, psi0::MPS; nsweeps, kwargs...)
    dmrg(Hs::Vector{MPO}, psi0::MPS; nsweeps, kwargs...)
    dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS; nsweeps, weight=1.0, kwargs...)
    dmrg(…, sweeps::Sweeps; kwargs...)          # Sweeps-object form (no longer preferred)

Optimize an MPS `psi0` via DMRG toward the lowest eigenvalue of a Hermitian `H`.
Returns `(energy, psi)`.

- `Hs::Vector{MPO}` represents `H = H1 + H2 + …` (looped over, never actually summed).
- `Ms::Vector{MPS}` runs excited-state DMRG: the state is kept orthogonal to each `Mᵢ`
  by adding `w·Σ|Mᵢ⟩⟨Mᵢ|` (weight `w`). NOTE the reported energy is of `H + w·Σ|Mᵢ⟩⟨Mᵢ|`,
  not `H` — for `⟨H⟩` use `inner(psi', H, psi)` afterward.

Keyword arguments:
  - `nsweeps::Int` (required) - number of DMRG sweeps.
  - `maxdim`, `mindim`, `cutoff`, `noise` - scalar or per-sweep array; bond-dim/accuracy control.
  - `outputlevel::Int = 1` - 0 = silent, ≥2 = per-bond info.
  - `observer` - Observer for measurements / early stop.
  - `write_when_maxdim_exceeds::Int`, `write_path=tempdir()` - spill env tensors to disk above
     the given maxdim (large runs).
  - `eigsolve_tol=1e-14`, `eigsolve_krylovdim=3`, `eigsolve_maxiter=1`, `eigsolve_verbosity=0`,
     `ishermitian=true` - local KrylovKit `eigsolve` controls.
"""
function dmrg(
  PH,
  psi0::MPS,
  sweeps::Sweeps;
  target_energy=nothing,
  use_early_exit=true,
  last_sweep_energy=nothing,
  which_decomp=nothing,
  svd_alg=nothing,
  observer=NoObserver(),
  outputlevel=1,
  write_when_maxdim_exceeds=nothing,
  write_path=tempdir(),
  # eigsolve kwargs
  eigsolve_tol=1e-14,
  eigsolve_krylovdim=3,
  eigsolve_maxiter=1,
  eigsolve_verbosity=0,
  eigsolve_which_eigenvalue=:SR,
  ishermitian=true,
  tensor_tracker=nothing,
  only_store=false,
  debug=false,
  # Enable roofline/flop/footprint/permute-profile accumulators for this run (does not
  # reset them — call SparseBackends.reset_roofline!/reset_flops! first for a fresh loop).
  roofline::Bool=false,
  # Tags env-footprint/permute-profile records with the run name ("DENSE"/"ALIASED"/…).
  run_label::String="?",
  # Kernel selector for the aliased-PHP + dense-ψ case (inert otherwise): :standard =
  # stock per-leg aliased matvec (byte-identical to before); :fused = partial-fused
  # matvec (matvec_partial_fused_full) on bulk bonds. See dispatch points below.
  run_mode::Symbol=:standard
)
  run_mode in (:standard, :fused) ||
    error("dmrg: unknown run_mode=$(run_mode); expected :standard or :fused")
  SparseBackends.ALIASED_TRACE[] = debug   # route this call's debug flag to the aliased trace prints
  SparseBackends.set_roofline!(roofline)
  println("Use early exit is set to ", use_early_exit, " num sweeeps are ", nsweep(sweeps))
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  psi = copy(psi0)
  N = length(psi)

  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end

  # Is the operator an aliased P†HP MPO? Gates the φ-pin (below) and the fused matvec.
  # Env fusion is handled inside position! via run_mode (self-guarded there).
  _op_aliased_mpo = _dmrg_op_is_aliased(PH)
  _op_is_aliased  = run_mode === :fused && _op_aliased_mpo

  t = @elapsed begin
    PH = position!(PH, psi, 1; roofline=roofline, run_label=run_label, run_mode=run_mode)
  end

  println("Time to position at start of DMRG: ", t, " seconds")
  

  energy = 0.0
  # THis is a check to take last energy as a specific value
  if last_sweep_energy != nothing
    last_energy = last_sweep_energy
    if use_early_exit != true
      error("use_early_exit must be set to true when last_sweep_energy is specified")
    end
  else
    last_energy = 1e10
  end

  early_exit_sweep = -1
  total_truncation_err = 0
  pos_time = 0.0
  opt_time = 0.0

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      time_sweep = @elapsed begin
        for (b, ha) in sweepnext(N)
          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          timer = @elapsed begin
            _dmrg_set_bond_context!(b, sw)   # permute-profile probe context (see dmrg_debug.jl)
            @timeit PROJMPO_TIMER "dmrg.position!" begin
              _IN_POSITION[] = true
              try
                PH = position!(PH, psi, b; roofline=roofline, run_label=run_label, run_mode=run_mode)
              finally
                _IN_POSITION[] = false
              end
            end
          end
          pos_time += timer

          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          @timeit PROJMPO_TIMER "dmrg.phi_build" begin
            # preserve_bs_output=true keeps φ aliased/dedup when both site tensors are
            # aliased blocksparse (no-op for dense inputs).
            phi = *(psi[b], psi[b + 1]; preserve_bs_output=true)
          end
          # Fuse only on BULK bonds (partial-fused context needs both L and R envs); edge
          # bonds fall back to stock. Use the INNER ProjMPO's envs (ProjMPO_MPS has none).
          use_fused = false
          if _op_is_aliased
            _iph = _dmrg_inner_projmpo(PH)
            use_fused = !(lproj(_iph) isa OneITensor) && !(rproj(_iph) isa OneITensor)
          end
          # Pin dense φ to the aliased kernel's strided-read layout (only when the operator
          # is aliased — a dense-H matvec is order-agnostic, so no wasted permute there).
          # Layout-only ⇒ E unchanged. :standard → [:l,:s2,:r,:s]; :fused → [:l,:r,:s2,:s].
          if _op_aliased_mpo && !SparseBackends.is_sparse_mps(psi)
            @timeit PROJMPO_TIMER "dmrg.phi_reorder_dense" begin
              phi = SparseBackends.reorder_to_roles(phi, use_fused ? [:l, :r, :s2, :s] :
                                                              [:l, :s2, :r, :s])
            end
          end

          _dmrg_probe_phi_in(phi, b, sw, ha)   # schema + keytrace dumps (see dmrg_debug.jl)

          time = @elapsed begin
            @timeit PROJMPO_TIMER "dmrg.eigsolve" begin
              # Bulk aliased-PHP bond under :fused → partial-fused matvec closure (+ stock
              # projector terms for excited ProjMPO_MPS); otherwise the stock operator PH.
              _mv_op = use_fused ? _dmrg_fused_matvec(PH, b) : PH
              vals, vecs = eigsolve(
                _mv_op,
                phi,
                1,
                eigsolve_which_eigenvalue;
                ishermitian,
                tol=eigsolve_tol,
                krylovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
                verbosity=eigsolve_verbosity,
              )
            end
          end
          opt_time += time

          energy = vals[1]
          ## Right now there is a conversion problem in CUDA.jl where `UnifiedMemory` Arrays are being converted
          ## into `DeviceMemory`. This conversion line is here temporarily to fix that problem when it arises
          ## Adapt is only called when using CUDA backend. CPU will work as implemented previously.
          ## TODO this might be the only place we really need iscu if its not fixed.
          phi = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
            adapt(ITensors.set_eltype(unwrap_array_type(phi), eltype(vecs[1])), vecs[1])
          else
            vecs[1]
          end
          _dmrg_probe_phi_out(phi, b)

          ortho = ha == 1 ? "left" : "right"

          drho = nothing
          if noise(sweeps, sw) > 0
            @timeit_debug timer "dmrg: noiseterm" begin
              # Use noise term when determining new MPS basis.
              # This is used to preserve the element type of the MPS.
              elt = real(scalartype(psi))
              drho = elt(noise(sweeps, sw)) * noiseterm(PH, phi, ortho)
            end
          end

          @debug_check begin
            checkflux(phi)
          end

          t = @elapsed begin
            @timeit PROJMPO_TIMER "dmrg.replacebond!" begin
              spec = replacebond!(
                PH,
                psi,
                b,
                phi;
                maxdim=maxdim(sweeps, sw),
                mindim=mindim(sweeps, sw),
                cutoff=cutoff(sweeps, sw),
                eigen_perturbation=drho,
                ortho,
                normalize=true,
                which_decomp,
                svd_alg
              )
            end
          end
          _dmrg_probe_bond_out(psi, b)

          maxtruncerr = max(maxtruncerr, spec.truncerr)

          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          if outputlevel >= 2
            @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
            @printf(
              "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
              cutoff(sweeps, sw),
              maxdim(sweeps, sw),
              mindim(sweeps, sw)
            )
            @printf(
              "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
            )
            flush(stdout)
          end

          sweep_is_done = (b == 1 && ha == 2)
          measure!(
            observer;
            energy,
            psi,
            projected_operator=PH,
            bond=b,
            sweep=sw,
            half_sweep=ha,
            spec,
            outputlevel,
            sweep_is_done,
          )
        end
      end
      println(" Check time breakdown ", "Time spent in position! ", pos_time, " seconds. Time spent in optimization step (eigsolve + replacebond!) ", opt_time, " seconds.")
      pos_time, opt_time = 0.0, 0.0
    end
    println("=====================================")

    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    total_truncation_err = max(total_truncation_err, maxtruncerr)

    # if use_early_exit
    #   println("diff vbetween last energy and energy is ", last_energy - energy)
    # end
    early_exit_sweep = sw
    if use_early_exit && (last_energy - energy < 1e-8)
      if outputlevel >= 1
        println("Energy change less than 1e-8, stopping DMRG")
      end
      break
    end

    last_energy = energy
    isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    isdone && break
    # if target_energy != nothing
    #   println("diff vbetween energy abnd target energy is ", energy - target_energy)
    # end
    if target_energy != nothing && (energy - target_energy) < 1e-8
      println("Target energy reached in sweeps ", sw)
      break
    end
    # isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    # isdone && break
  end
  # println(" Check time breakdown ", "Time spent in position! ", pos_time, " seconds. Time spent in optimization step (eigsolve + replacebond!) ", opt_time, " seconds.")

  return (energy, psi, early_exit_sweep, total_truncation_err)
end

function constrained_dmrg2(
  H,
  psi0::MPS,
  sweeps::Sweeps;
  target_energy=nothing,
  parMPO::Union{MPO, Nothing}=nothing,
  which_decomp=nothing,
  svd_alg=nothing,
  observer=NoObserver(),
  outputlevel=1,
  write_when_maxdim_exceeds=nothing,
  write_path=tempdir(),
  # eigsolve kwargs
  eigsolve_tol=1e-14,
  eigsolve_krylovdim=3,
  eigsolve_maxiter=1,
  eigsolve_verbosity=0,
  eigsolve_which_eigenvalue=:SR,
  ishermitian=true,
)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  H = complex(H)
  Pop = complex(parMPO)
  psi0 = complex(psi0)
  PH = ConstrainedProjMPO(H, Pop)

  @debug_check begin
    # Debug level checks
    # Enable with ITensors.enable_debug_checks()
    checkflux(psi0)
    checkflux(PH)
  end

  psi = copy(psi0)
  N = length(psi)
  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  # Canonicalize index ordering of all psi tensors so phi = psi[b]*psi[b+1]
  # always has the same index layout across runs (run 1 and run 2 create link
  # indices with different session-global IDs, causing different natural orderings).
  let canon_order_tensor(T) = permute(T, sort(collect(inds(T)); by = i -> (string(tags(i)), dim(i)))...)
    for i in 1:length(psi)
      psi[i] = canon_order_tensor(psi[i])
    end
  end

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end
  PH = position!(PH, psi, 1; roofline=roofline, run_label=run_label)
  energy = 0.0


  for sw in 1:nsweep(sweeps)
    ovlp_error = 0.0
    center_nums = 0
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N)
        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        @timeit_debug timer "dmrg: position!" begin
          PH = position!(PH, psi, b; roofline=roofline, run_label=run_label)
        end

        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        
        @timeit_debug timer "dmrg: phi = psi[b]*psi[b+1]" begin
          phi = psi[b] * psi[b + 1]
        end

        @timeit_debug timer "dmrg: eigsolve" begin
          vals, vecs = eigsolve(
            PH,
            phi,
            1,
            eigsolve_which_eigenvalue;
            ishermitian,
            tol=eigsolve_tol,
            krylovdim=eigsolve_krylovdim,
            maxiter=eigsolve_maxiter,
            verbosity=eigsolve_verbosity,
          )
        end

        energy = vals[1]
        ## Right now there is a conversion problem in CUDA.jl where `UnifiedMemory` Arrays are being converted
        ## into `DeviceMemory`. This conversion line is here temporarily to fix that problem when it arises
        ## Adapt is only called when using CUDA backend. CPU will work as implemented previously.
        ## TODO this might be the only place we really need iscu if its not fixed.
        phi = if NDTensors.iscu(phi) && NDTensors.iscu(vecs[1])
          adapt(ITensors.set_eltype(unwrap_array_type(phi), eltype(vecs[1])), vecs[1])
        else
          vecs[1]
        end

        ortho = ha == 1 ? "left" : "right"

        drho = nothing
        if noise(sweeps, sw) > 0
          @timeit_debug timer "dmrg: noiseterm" begin
            # Use noise term when determining new MPS basis.
            # This is used to preserve the element type of the MPS.
            elt = real(scalartype(psi))
            drho = elt(noise(sweeps, sw)) * noiseterm(PH, phi, ortho)
          end
        end

        @debug_check begin
          checkflux(phi)
        end

        @timeit_debug timer "dmrg: replacebond!" begin
          spec = replacebond!(
            PH,
            psi,
            b,
            phi;
            maxdim=maxdim(sweeps, sw),
            mindim=mindim(sweeps, sw),
            cutoff=cutoff(sweeps, sw),
            eigen_perturbation=drho,
            ortho,
            normalize=true,
            which_decomp,
            svd_alg,
          )
        end

        maxtruncerr = max(maxtruncerr, spec.truncerr)

        @debug_check begin
          checkflux(psi)
          checkflux(PH)
        end

        if outputlevel >= 2
          @printf("Sweep %d, half %d, bond (%d,%d) energy=%s\n", sw, ha, b, b + 1, energy)
          @printf(
            "  Truncated using cutoff=%.1E maxdim=%d mindim=%d\n",
            cutoff(sweeps, sw),
            maxdim(sweeps, sw),
            mindim(sweeps, sw)
          )
          @printf(
            "  Trunc. err=%.2E, bond dimension %d\n", spec.truncerr, dim(linkind(psi, b))
          )
          flush(stdout)
        end

        sweep_is_done = (b == 1 && ha == 2)
        measure!(
          observer;
          energy,
          psi,
          projected_operator=PH,
          bond=b,
          sweep=sw,
          half_sweep=ha,
          spec,
          outputlevel,
          sweep_is_done,
        )
      end
      # if parMPO != nothing
      #   println("  Ovlp error with previous state = ", ovlp_error, center_nums)
      # end
    end
    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      flush(stdout)
    end
    isdone = checkdone!(observer; energy, psi, sweep=sw, outputlevel)
    isdone && break
    if target_energy != nothing && (energy - target_energy) < 1e-6
      break
    end
  end
  return (energy, psi)
end

function _dmrg_sweeps(;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
)
  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, maxdim...)
  setmindim!(sweeps, mindim...)
  setcutoff!(sweeps, cutoff...)
  setnoise!(sweeps, noise...)
  return sweeps
end

function dmrg(
  x1,
  x2,
  psi0::MPS;
  nsweeps,
  maxdim = default_maxdim(),
  mindim = default_mindim(),
  cutoff = default_cutoff(Float64),
  noise = default_noise(),
  kwargs...,
)
return dmrg(
  x1, x2, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); kwargs...
)
end

function dmrg(
  x1,
  psi0::MPS;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  target_energy=nothing,
  use_early_exit=true,
  last_sweep_energy=nothing,
  kwargs...,
)
  # println("Type1 is here -- :: ", cutoff, noise, use_early_exit)

  return dmrg(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); target_energy, use_early_exit, last_sweep_energy, kwargs...)
end

function constrained_dmrg(
  x1,
  psi0::MPS,
  parMPO::MPO;
  nsweeps,
  maxdim=default_maxdim(),
  mindim=default_mindim(),
  cutoff=default_cutoff(Float64),
  noise=default_noise(),
  target_energy=nothing,
  kwargs...,
)
  println("Type2 is here -- :: ", cutoff, noise)
  return constrained_dmrg(x1, psi0, _dmrg_sweeps(; nsweeps, maxdim, mindim, cutoff, noise); parMPO, kwargs...)
end