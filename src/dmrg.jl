using Adapt: adapt
using KrylovKit: eigsolve, InnerProductVec
const _EIGSOLVE_PHI_TRACE_COUNT = Ref(0)
import VectorInterface
using NDTensors: scalartype, timer
using Printf: @printf
using TupleTools: TupleTools
import SparseBackends

# Tell KrylovKit/VectorInterface what scalar type an ITensor uses when it's
# wrapped in InnerProductVec for the M-inner-product Lanczos path. Defined at
# module scope so it's resolved once, not in the inner dmrg loop.
VectorInterface.scalartype(::Type{ITensors.ITensor}) = ComplexF64

# In-place aliased Lanczos add!! — the ITensorsVectorInterfaceExt extension runs
# `a + b*α` directly for external storage (two allocations: the b*α template copy
# + the Base.:+ `plus_merge` result — the dominant aliased-Krylov add cost). For
# key-aligned dedup-1 aliased operands we do a truly in-place axpby (zero alloc,
# no merge). To OVERRIDE the extension WITHOUT a precompile "method overwriting"
# error, the methods below are strictly MORE SPECIFIC than the extension's
# add!!(…, ::Number): ::Real and ::Complex cover every concrete α Lanczos produces
# (the complex one is the 2-arg add!!'s one(ComplexF64)). HARDENED: the in-place
# path is unconditional; it falls back to the extension's exact behavior only when
# operands aren't key-aligned aliased storage ⇒ byte-identical in that case.
function _aliased_addbang!(a::ITensors.ITensor, b::ITensors.ITensor, α::Number, β::Number)
  # HARDENED: the in-place aliased axpby is always attempted (it's bit-identical
  # and strictly faster); it falls through below only when the operands aren't
  # key-aligned aliased storage — a correctness condition, not a toggle.
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
# NOTE: the analogous Lanczos `inner` (the `wrapped×wrapped` dots) cannot be
# overridden the same way — its signature inner(::ITensor,::ITensor) is identical
# to the extension's with no more-specific variant, and an __init__/@eval install
# breaks the ChainRulesCore extension's precompile. The dot speedup will instead
# be done at the SparseBackends contraction level (full aliased×aliased reduction
# → scalar). For now `inner` keeps the extension's contraction path.

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
             target_energy=nothing, use_early_exit=true, last_sweep_energy=nothing, kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
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
  PH = ProjMPO(H)
  return dmrg(PH, psi0, sweeps; target_energy, use_early_exit, kwargs...)
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
    dmrg(H::MPO, psi0::MPS; kwargs...)
    dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`,
represented as a matrix product operator (MPO).

    dmrg(Hs::Vector{MPO}, psi0::MPS; kwargs...)
    dmrg(Hs::Vector{MPO}, psi0::MPS, sweeps::Sweeps; kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`.
This version of `dmrg` accepts a representation of H as a
Vector of MPOs, `Hs = [H1, H2, H3, ...]` such that `H` is defined
`as H = H1 + H2 + H3 + ...`
Note that this sum of MPOs is not actually computed; rather
the set of MPOs `[H1,H2,H3,..]` is efficiently looped over at
each step of the DMRG algorithm when optimizing the MPS.

    dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS; weight=1.0, kwargs...)
    dmrg(H::MPO, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight=1.0, kwargs...)

Use the density matrix renormalization group (DMRG) algorithm
to optimize a matrix product state (MPS) such that it is the
eigenvector of lowest eigenvalue of a Hermitian matrix `H`,
subject to the constraint that the MPS is orthogonal to each
of the MPS provided in the Vector `Ms`. The orthogonality
constraint is approximately enforced by adding to `H` terms of
the form `w|M1><M1| + w|M2><M2| + ...` where `Ms=[M1, M2, ...]` and
`w` is the "weight" parameter, which can be adjusted through the
optional `weight` keyword argument.

!!! note
    `dmrg` will report the energy of the operator
    `H + w|M1><M1| + w|M2><M2| + ...`, not the operator `H`.
    If you want the expectation value of the MPS eigenstate
    with respect to just `H`, you can compute it yourself with
    an observer or after DMRG is run with `inner(psi', H, psi)`.

The MPS `psi0` is used to initialize the MPS to be optimized.

The number of sweeps of thd DMRG algorithm is controlled by
passing the `nsweeps` keyword argument. The keyword arguments
`maxdim`, `cutoff`, `noise`, and `mindim` can also be passed
to control the cost versus accuracy of the algorithm - see below
for details.

Alternatively the number of sweeps and accuracy parameters can
be passed through a `Sweeps` object, though this interface is
no longer preferred.

Returns:

  - `energy::Number` - eigenvalue of the optimized MPS
  - `psi::MPS` - optimized MPS

Keyword arguments:

  - `nsweeps::Int` - number of "sweeps" of DMRG to perform

Optional keyword arguments:

  - `maxdim` - integer or array of integers specifying the maximum size
     allowed for the bond dimension or rank of the MPS being optimized.
  - `cutoff` - float or array of floats specifying the truncation error cutoff
     or threshold to use for truncating the bond dimension or rank of the MPS.
  - `eigsolve_krylovdim::Int = 3` - maximum dimension of Krylov space used to
     locally solve the eigenvalue problem. Try setting to a higher value if
     convergence is slow or the Hamiltonian is close to a critical point. [^krylovkit]
  - `eigsolve_tol::Number = 1e-14` - Krylov eigensolver tolerance. [^krylovkit]
  - `eigsolve_maxiter::Int = 1` - number of times the Krylov subspace can be
     rebuilt. [^krylovkit]
  - `eigsolve_verbosity::Int = 0` - verbosity level of the Krylov solver.
     Warning: enabling this will lead to a lot of outputs to the terminal. [^krylovkit]
  - `ishermitian=true` - boolean specifying if dmrg should assume the MPO (or more
     general linear operator) represents a Hermitian matrix. [^krylovkit]
  - `noise` - float or array of floats specifying strength of the "noise term"
     to use to aid convergence.
  - `mindim` - integer or array of integers specifying the minimum size of the
     bond dimension or rank, if possible.
  - `outputlevel::Int = 1` - larger outputlevel values make DMRG print more
     information and 0 means no output.
  - `observer` - object implementing the [Observer](@ref observer) interface
     which can perform measurements and stop DMRG early.
  - `write_when_maxdim_exceeds::Int` - when the allowed maxdim exceeds this
     value, begin saving tensors to disk to free RAM memory in large calculations
  - `write_path::String = tempdir()` - path to use to save files to disk
     (to save RAM) when maxdim exceeds the `write_when_maxdim_exceeds` option, if set

[^krylovkit]:

    The `dmrg` function in `ITensorMPS.jl` currently uses the `eigsolve`
    function in `KrylovKit.jl` as the internal the eigensolver.
    See the `KrylovKit.jl` documention on the `eigsolve` function for more details:
    [KrylovKit.eigsolve](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).
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
  # Consolidates the old SB_ROOFLINE / SB_FLOP_COUNT / SB_ENV_FOOTPRINT /
  # SB_PERM_CAPTURE / GEMM_DIMS_HIST env vars (+ SB_PERMUTE_PROFILE's
  # SparseBackends/ITensorMPS writers) into one switch. Only flips the
  # enabled flag on every call (via set_roofline!) — does NOT reset the
  # accumulators, so a per-sweep loop of dmrg(...) calls still accumulates
  # stats across the whole run. Call SparseBackends.reset_roofline!(true)/
  # reset_flops!(true) once yourself before such a loop to zero them first.
  roofline::Bool=false,
  # Was SB_RUN_LABEL env var — a real threaded argument now (was a functional
  # argument all along conceptually; test scripts pass it directly instead of
  # setting ENV), tagging env-footprint/permute-profile records with which
  # named run ("DENSE"/"ALIASED"/etc.) produced them. No correctness effect.
  run_label::String="?"
)
  # Was SB_ALIASED_TRACE — thread this call's debug flag into SparseBackends'
  # trace Ref (not an ENV var), consulted by trace prints throughout the
  # aliased pipeline. Each print site's own fire-count budget is a hardcoded
  # literal, not separately configurable.
  SparseBackends.ALIASED_TRACE[] = debug
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

  t = @elapsed begin
    PH = position!(PH, psi, 1; debug=false, roofline=roofline, run_label=run_label)
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
  only_idx = 1

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
            # Per-matvec bond-position context for permute-profile probes.
            ITensorMPS._BOND_POSITION[] = b
            ITensorMPS._SWEEP_NUM[] = sw
            ENV["SB_BOND"] = string(b)
            @timeit PROJMPO_TIMER "dmrg.position!" begin
              ITensorMPS._IN_POSITION[] = true
              try
                PH = position!(PH, psi, b; roofline=roofline, run_label=run_label)
              finally
                ITensorMPS._IN_POSITION[] = false
              end
            end
          end
          pos_time += timer

          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          @timeit PROJMPO_TIMER "dmrg.phi_build" begin
            # `preserve_bs_output=true` routes through ITensors' `*` to the aliased-
            # preserving contraction when both site tensors are aliased blocksparse
            # (and is a no-op for dense inputs), so φ keeps its aliased/dedup schema.
            phi = *(psi[b], psi[b + 1]; preserve_bs_output=true)
          end
          # Dense-ψ: pin φ to a canonical (link,site,site,link) order every bond so the
          # matvec sees a stable input layout each Krylov iteration (replacebond!'s SVD
          # otherwise emits φ in drifting orders → the step-output perm varies, defeating
          # a static table). Layout-only (a permute of φ); psi/H storage untouched, E
          # convergence-equivalent. Applies to GROUND (ProjMPO) AND EXCITED
          # (ProjMPOSum): reorder_to_roles only touches φ (no PH accessors needed), and
          # the excited H-term is a ProjMPO whose contract runs the same step-1 swap +
          # env canonicalization, so pinning φ makes the strided read fire for excited too.
          if !SparseBackends.is_sparse_mps(psi)
            @timeit PROJMPO_TIMER "dmrg.phi_reorder_dense" begin
              # Layout-C target (s2⁰ = role :s LAST): with L reordered so step 1
              # emits T1 = [red(l1⁴¹) | F1 | keepB(l1³¹,s3⁰,l3⁰) | s2⁰], step-2's B
              # arrives red-leading + keepB-contiguous → the kernel's strided-read
              # (K2) fires and permute_B@2 is skipped. E is label-based, so this
              # order change is convergence-equivalent.
              phi = SparseBackends.reorder_to_roles(phi, [:l, :s2, :r, :s])
            end
          end

          SparseBackends.schema_dbg("eigsolve-OPERAND phi b=$b", phi)

          # Dumps psi[b]/psi[b+1]/phi's prefix/dense axis classification before
          # the eigsolve. Debug-only, disabled; flip to `true` (and restore the
          # body below) to re-enable.
          if false
          end
          # if get(ENV, "SB_FACT_DIAG", "0") == "1"
          #   _clsdump(lbl, T) = begin
          #     if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
          #       w = T.tensor.data; P = SparseBackends._abs_head_len(w); N = ndims(w.aliased)
          #       println("   [", lbl, "] P=$P  prefix=",
          #               [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in 1:P],
          #               "  dense=",
          #               [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in P+1:N])
          #     else
          #       println("   [", lbl, "] storage=", ITensors.has_external_storage(T) ? string(typeof(T.tensor.data)) : "dense")
          #     end
          #   end
          #   println("[FACT_DIAG phi @ b=$b ha=$ha sw=$sw]")
          #   _clsdump("psi[b]",   psi[b])
          #   _clsdump("psi[b+1]", psi[b+1])
          #   _clsdump("phi",      phi)
          #   flush(stdout)
          # end

          # phi = canonicalize_phi_phase(phi)

          # TRACE_IMAGE: capture the TRUE φ's constraint image (before M^{±1/2}) so the
          # per-step matvec trace can flag intermediate keys outside image(φ).
          SparseBackends.capture_phi_image!(phi)

          # ── KEYTRACE (SB_KEYTRACE=1): trace the prefix-key set + index order at each
          # stage of one bond (default b=2, sw=1, ha=1) to follow the M^{±1/2} key flow.
          _keytrace = function(lbl, T)
            _kt_bonds = Set(parse.(Int, split(get(ENV, "SB_KEYTRACE_BOND", "2"), ",")))
            (get(ENV, "SB_KEYTRACE", "0") == "1" &&
             (b in _kt_bonds) && sw == 1 && ha == 1) || return nothing
            if T isa ITensors.ITensor && ITensors.has_external_storage(T) &&
               ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse
              w = ITensors.get_external_storage(T); a = w.aliased
              P = SparseBackends._abs_head_len(w); Nn = length(w.inds)
              _ord = [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in 1:Nn]
              _dedup = round(length(a.keys) / max(a.n_templates, 1); digits=3)
              # key → alias_id (template index) pairs, so the actual dedup GROUPING is visible:
              _keymap = [(a.keys[i], Int(a.alias_ids[i])) for i in eachindex(a.keys)]
              # group keys by template id to show which keys SHARE each template
              _bytmpl = Dict{Int,Vector{eltype(a.keys)}}()
              for i in eachindex(a.keys); push!(get!(_bytmpl, Int(a.alias_ids[i]), eltype(a.keys)[]), a.keys[i]); end
              println("[KEYTRACE b=", b, " ", lbl, "] P=", P, " nb=", length(a.keys), " nt=", a.n_templates,
                      " dedup=", _dedup, "x",
                      "\n   inds(order)= ", _ord,
                      "\n   sparse_prefix(1:P)= ", _ord[1:P],
                      "\n   key=>template_id= ", _keymap,
                      "\n   keys_sharing_each_template= ", sort(collect(_bytmpl)))
            else
              _st = (T isa ITensors.ITensor && ITensors.has_external_storage(T)) ?
                    string(typeof(ITensors.get_external_storage(T))) : "dense/plain"
              println("[KEYTRACE ", lbl, "] (", _st, ")")
            end
            flush(stdout); return nothing
          end
          # _keytrace("1.phi_original (template)", phi)  # disabled — uncomment to re-enable

          # println("eigsolve at sweep $sw, half $ha, bond ($b, $(b+1))")
          time = @elapsed begin
            @timeit PROJMPO_TIMER "dmrg.eigsolve" begin
              # Standard Lanczos local eigensolve. ψ here is always dense (an aliased
              # ψ = P·core is routed to factor-core `dmrg_core_php` at the MPO front
              # end). The operator PH may still be an aliased P†HP (dense-ψ+aliased-PHP
              # fused path) — that is handled inside `product`/`position!`, not here.
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
          SparseBackends.schema_dbg("eigsolve-RESULT phi b=$b", phi)
          # _keytrace("5.phi before factorize (= vecs[1], post any drop)", phi)  # disabled — uncomment to re-enable
          if SparseBackends.ALIASED_TRACE[] && _EIGSOLVE_PHI_TRACE_COUNT[] < 5
            _EIGSOLVE_PHI_TRACE_COUNT[] += 1
            phi_st = ITensors.has_external_storage(phi) ? typeof(phi.tensor.data) : "dense"
            println("[SB_ALIASED_TRACE eigsolve returned #$(_EIGSOLVE_PHI_TRACE_COUNT[])]  phi storage=$phi_st  b=$b")
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

          replace_debug = false
          if b == 3
            replace_debug = true
          end

          # ── FACT_KEY (before/after factorize key diff) ──────────────────────
          # Capture φ's key set going INTO replacebond!, then reconstitute the
          # post-SVD two-site tensor and diff: keys in φ but gone after = what the
          # factorize truncation actually discards (the provably-safe-to-drop set).
          # Debug-only, disabled; flip to `true` (and restore the two bodies —
          # here and further below at the matching `if _factkey` block) to
          # re-enable.
          _factkey = false
          _fk_in = nothing
          # if _factkey && ITensors.has_external_storage(phi) &&
          #    ITensors.get_external_storage(phi) isa SparseBackends.WrappedAliasedBlockSparse
          #   _fk_in = Set(ITensors.get_external_storage(phi).aliased.keys)
          # end
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
          # if _factkey && _fk_in !== nothing
          #   try
          #     _phiw = ITensors.get_external_storage(phi)
          #     _aw = ITensors.get_external_storage(psi[b]); _bw = ITensors.get_external_storage(psi[b+1])
          #     _rw = SparseBackends.wrapped_contract_aliased(_aw, _bw; preserve_bs_output=true)
          #     _rw = _rw isa ITensors.ITensor ? ITensors.get_external_storage(_rw) : _rw
          #     if _rw isa SparseBackends.WrappedAliasedBlockSparse
          #       _rw = SparseBackends.align_aliased_axes(_rw, _phiw)   # align axes to φ
          #       _rw = _rw isa ITensors.ITensor ? ITensors.get_external_storage(_rw) : _rw
          #     end
          #     _outkeys = (_rw isa SparseBackends.WrappedAliasedBlockSparse) ? Set(_rw.aliased.keys) : Set(eltype(_fk_in)[])
          #     _dropped = setdiff(_fk_in, _outkeys); _added = setdiff(_outkeys, _fk_in)
          #     println("[FACT_KEY b=", b, " ha=", ha, " sw=", sw, "] phi_in=", length(_fk_in),
          #             " recon_out=", length(_outkeys), " dropped(in∖out)=", length(_dropped),
          #             " added(out∖in)=", length(_added))
          #     flush(stdout)
          #   catch e
          #     println("[FACT_KEY b=", b, "] recon failed: ", sprint(showerror, e)); flush(stdout)
          #   end
          # end
          SparseBackends.schema_dbg("replacebond-OUT psi[$b]", psi[b])
          SparseBackends.schema_dbg("replacebond-OUT psi[$(b+1)]", psi[b+1])

          # println("================ print here ================ ", ITensors.has_external_storage(phi))

          maxtruncerr = max(maxtruncerr, spec.truncerr)

          # Path-B: no gram-cache update needed — the gram is sliced fresh from
          # the H-env each bond (Stage 1), and the H-env is maintained by
          # position!/makeL!/makeR!. (Superseded update_left!/update_right!.)

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

          only_idx += 1
        end
      end
      println(" Check time breakdown ", "Time spent in position! ", pos_time, " seconds. Time spent in optimization step (eigsolve + replacebond!) ", opt_time, " seconds.")
      pos_time, opt_time = 0.0, 0.0
    end
    #   only_idx += 1
    #   check_equality!(tensor_tracker, PH, psi; only_store=only_store, only_idx=only_idx)
    # end
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