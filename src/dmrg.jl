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

function dmrg(H::MPO, psi0::MPS, sweeps::Sweeps; target_energy=nothing, use_early_exit=true, last_sweep_energy=nothing, kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  # H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  # println("Type  --  3")
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
  # Eigensolve PATHWAY selector (replaces the old BMF_ISO_PATH / BMF_BOP_PROJECT /
  # BMF_APPLY_MINV / BMF_ARNOLDI env flags + the implicit _env_dress densify gate).
  # Applies only when is_sparse_mps(psi); dense ψ skips Path-B structurally.
  #   :iso         — standard Lanczos, M=I (sparse-but-isometric ψ; e.g. BS strict-cap SVD)
  #   :bop_aliased — B = M^{-1/2} H M^{-1/2}, null-projected, seed/Krylov stay ALIASED (no densify)
  #   :bop_densify — same B_op but env-dressed + DENSIFIED seed (needs dense L/R envs)
  #   :minner      — A = M^{-1} H_eff, M-inner-product Lanczos
  run_mode::Symbol=:bop_aliased,
  # Step 2b (from-P M^{±1/2}): when true, build M^{±1/2}=c^{∓...}·G from the geometric
  # constant c=2^⌈env/2⌉ (no eigen), instead of eigendecomposing the gram. Default nothing
  # → eigen path unchanged. Optional arg, no env var.
  minv_from_p::Union{Nothing,Bool}=nothing,
  debug=false
)
  run_mode in (:iso, :bop_aliased, :bop_densify, :minner) ||
    error("dmrg: unknown run_mode=$(run_mode); expected one of :iso, :bop_aliased, :bop_densify, :minner")
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
    PH = position!(PH, psi, 1; debug=false)
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

  # Path-B gram-env cache: incremental L[i], R[i] for sparse psi.
  # Built once here, then updated per replacebond! step (one contraction
  # instead of O(N) rebuild). When psi is dense, gram_cache stays `nothing`
  # and the dense path is unaffected.
  gram_cache = SparseBackends.is_sparse_mps(psi) ?
               SparseBackends.init_gram_cache(psi) :
               nothing

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
              ENV["SB_IN_POSITION"] = "1"
              try
                PH = position!(PH, psi, b)
              finally
                ITensorMPS._IN_POSITION[] = false
                ENV["SB_IN_POSITION"] = "0"
              end
            end
          end
          pos_time += timer

          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
            # Manually preserve the aliased output if both inputs are aliased blocksparse 
            phi = if ITensors.has_external_storage(psi[b]) &&
                     ITensors.has_external_storage(psi[b + 1]) &&
                     psi[b].tensor.data isa SparseBackends.WrappedAliasedBlockSparse &&
                     psi[b + 1].tensor.data isa SparseBackends.WrappedAliasedBlockSparse
              Aw = ITensors.get_external_storage(psi[b])
              Bw = ITensors.get_external_storage(psi[b + 1])
              Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw; preserve_bs_output=true)
              Cw isa ITensors.ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
            else
              psi[b] * psi[b + 1]
            end
          end
          # WRAP-AROUND (Part B): put φ (the eigensolver seed + recast template)
          # into the order the matvec's FIRST step wants, so step-1's permA is the
          # identity every Krylov iteration. The first operator is lproj (Lenv) in
          # the bulk/right-edge chain, or rproj (Renv) at the left edge where the
          # chain reverses. Layout-only — contraction math unchanged.
          if SparseBackends.is_sparse_mps(psi) && run_mode !== :iso
            # φ (eigensolver seed + recast template) reordered ONCE per bond by the
            # matvec-chain reduction rank, so step-1's permA is the identity every
            # Krylov iteration. φ's raw `psi[b]*psi[b+1]` shape is bond-specific (so
            # this can't be a single hard-coded vector like the per-step output
            # table), but it's a once-per-bond template setup, not per-matvec. Build
            # the chain via the same reverse gate the matvec uses.
            _ops = Union{ITensor,OneITensor}[lproj(PH)]
            for s in site_range(PH); push!(_ops, PH.H[s]); end
            push!(_ops, rproj(PH))
            (first(_ops) isa OneITensor) && reverse!(_ops)
            phi = SparseBackends.reorder_aliased_by_rank(phi, _ops)
          end

          SparseBackends.schema_dbg("eigsolve-OPERAND phi b=$b", phi)

          if get(ENV, "SB_FACT_DIAG", "0") == "1"
            _clsdump(lbl, T) = begin
              if ITensors.has_external_storage(T) && T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
                w = T.tensor.data; P = SparseBackends._abs_head_len(w); N = ndims(w.aliased)
                println("   [", lbl, "] P=$P  prefix=",
                        [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in 1:P],
                        "  dense=",
                        [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in P+1:N])
              else
                println("   [", lbl, "] storage=", ITensors.has_external_storage(T) ? string(typeof(T.tensor.data)) : "dense")
              end
            end
            println("[FACT_DIAG phi @ b=$b ha=$ha sw=$sw]")
            _clsdump("psi[b]",   psi[b])
            _clsdump("psi[b+1]", psi[b+1])
            _clsdump("phi",      phi)
            flush(stdout)
          end

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
          _keytrace("1.phi_original (template)", phi)

          # println("eigsolve at sweep $sw, half $ha, bond ($b, $(b+1))")
          time = @elapsed begin
            @timeit PROJMPO_TIMER "dmrg.eigsolve" begin
              # run_mode === :iso: ψ is treated as isometric (strict-iso SVD in
              # stable_factorize → L^T L = I → M = I → no M correction) → just use
              # standard Lanczos. Dense ψ also lands in the standard-eigsolve fall-
              # through structurally (is_sparse_mps == false), regardless of run_mode.
              if SparseBackends.is_sparse_mps(psi) && run_mode !== :iso
                # Path B (M-inner-product Lanczos): redefine the Krylov inner
                # product to <x,y>_M = <x, M·y>. The Lanczos basis is then
                # M-orthonormal; the eigenvalue problem in this frame is
                # equivalent to the generalized H v = λ M v. Needs MORE Krylov
                # vectors than standard Lanczos because the M-inner-product
                # frame is spectrally harder (M may be near-singular).
                @timeit PROJMPO_TIMER "dmrg.gram_envs" begin
                  Lgram = SparseBackends.get_left_gram(gram_cache, b)
                  Rgram = SparseBackends.get_right_gram(gram_cache, b)
                  SparseBackends.schema_dbg("GRAM Lgram b=$b", Lgram)
                  SparseBackends.schema_dbg("GRAM Rgram b=$b", Rgram)
                  # Gram-structure diagnostic (eigenvalues, rank, separability, block-diagonality; proved M=c·Π). Manually enable: change `if false` → `if true`.
                  if false
                    _gdump = function(lbl, G)
                      gi = collect(ITensors.inds(G))
                      if isempty(gi); println("[GRAM_DUMP ", lbl, "] scalar/empty"); flush(stdout); return; end
                      unp = filter(I->ITensors.plev(I)==0, gi); prm = filter(I->ITensors.plev(I)==1, gi)
                      Gd = ITensors.has_external_storage(G) ? SparseBackends.to_dense_itensors_unfused(G) : G
                      d = prod(I->ITensors.dim(I), unp; init=1)
                      arr = reshape(Array(Gd, unp..., prm...), d, d)
                      # eigenvalue spectrum (compact rank-deficiency readout — works for large d)
                      ev = sort(real.(LinearAlgebra.eigvals(LinearAlgebra.Hermitian((arr+arr')/2))); rev=true)
                      mx = isempty(ev) ? 0.0 : maximum(ev)
                      rnk = count(>(1e-8*max(mx,eps())), ev)
                      println("[GRAM_DUMP ", lbl, "]  dim=", d, "x", d,
                              "  unp_inds= ", [(ITensors.dim(I), string(ITensors.tags(I))) for I in unp],
                              "\n   eigenvalues(desc)= ", [round(e; sigdigits=3) for e in ev],
                              "\n   numerical_rank(>1e-8·max)= ", rnk, " / ", d,
                              "   (null dim = ", d-rnk, ")")
                      # SEPARABILITY: is G[a1,a2,a1',a2'] ≈ A[a1,a1'] ⊗ B[a2,a2'] for the two
                      # link axes (one is the sparse CHANNEL, one the dense MULT)? Reshape to
                      # (a1,a1')×(a2,a2') and SVD: top-singular-value² fraction = how separable.
                      # Then test whether either factor ≈ proportional to identity (= "contributes
                      # little" / trivial). unp order is [first, second]; we label by dim.
                      if length(unp) == 2
                        d1 = ITensors.dim(unp[1]); d2 = ITensors.dim(unp[2])
                        G4 = reshape(arr, d1, d2, d1, d2)             # [a1,a2,a1',a2']
                        Msep = reshape(permutedims(G4, (1,3,2,4)), d1*d1, d2*d2)  # [(a1,a1'),(a2,a2')]
                        Fsep = LinearAlgebra.svd(Msep)
                        sv = Fsep.S
                        sep_frac = sv[1]^2 / max(sum(abs2, sv), eps())
                        # extract the two factors from the leading rank-1 term
                        A1 = reshape(Fsep.U[:,1], d1, d1) .* sqrt(sv[1])
                        B1 = reshape(Fsep.V[:,1], d2, d2) .* sqrt(sv[1])
                        _iddist(X,n) = (dg = [real(X[i,i]) for i in 1:n];
                          offnrm = sqrt(max(sum(abs2,X) - sum(abs2,dg), 0.0));
                          dgcv = (isempty(dg)||abs(sum(dg))<eps()) ? 1.0 : LinearAlgebra.norm(dg .- sum(dg)/n)/(abs(sum(dg))/sqrt(n));
                          (round(offnrm/max(sqrt(sum(abs2,X)),eps());sigdigits=3), round(dgcv;sigdigits=3)))
                        oa,va = _iddist(A1,d1); ob,vb = _iddist(B1,d2)
                        println("   SEPARABILITY  top-SV²frac= ", round(sep_frac;sigdigits=4),
                                " (1.0=separable A⊗B)  | factor1 dim", d1, " offdiag_frac=", oa, " diag_CV=", va,
                                " | factor2 dim", d2, " offdiag_frac=", ob, " diag_CV=", vb,
                                "   (offdiag_frac≈0 & diag_CV≈0 ⇒ that factor ≈ scalar·I)")
                        # BLOCK-DIAGONALITY: is M[a1,a2,a1',a2'] block-diagonal in axis1
                        # (a1=a1') or axis2 (a2=a2')? off-axisX frac ≈ 0 ⇒ M does NOT mix
                        # that axis's values. Tests M = Σ_k |k⟩⟨k| ⊗ (m-block) form.
                        _tot = sum(abs2, G4)
                        _offk = 0.0
                        for i in 1:d1, ip in 1:d1
                          i == ip && continue
                          _offk += sum(abs2, @view G4[i,:,ip,:])
                        end
                        _offm = 0.0
                        for j in 1:d2, jp in 1:d2
                          j == jp && continue
                          _offm += sum(abs2, @view G4[:,j,:,jp])
                        end
                        println("   BLOCKDIAG  off-axis1(dim", d1, ") frac=",
                                round(sqrt(_offk/max(_tot,eps()));sigdigits=3),
                                "  off-axis2(dim", d2, ") frac=",
                                round(sqrt(_offm/max(_tot,eps()));sigdigits=3),
                                "   (≈0 ⇒ M block-diagonal in that axis)")
                      end
                      if d <= 16
                        for i in 1:d
                          println("   row ", i, ": ", [round(real(arr[i,j]); sigdigits=3) for j in 1:d])
                        end
                      else
                        println("   (matrix ", d, "x", d, " too large to print; spectrum above shows rank structure)")
                      end
                      flush(stdout)
                    end
                    _gdump("Lgram b=$b", Lgram); _gdump("Rgram b=$b", Rgram)
                  end
                  # Case 4 (aliased ψ × aliased PHP) signal — drives the aliased
                  # relayout of the M^{−1/2} factors (canonical pre-H apply, no
                  # forced fission). Case 2 (dense H) ⇒ false ⇒ factors stay dense
                  # ⇒ byte-identical to before.
                  # M is built ENTIRELY from ψ (Lgram/Rgram = Σψ·dag(ψ')), so the
                  # M^{−1/2} factor format should depend only on φ being aliased —
                  # NOT on H's format. The `any(H[j] aliased)` term was a conservative
                  # "byte-identical for dense-H" gate, but it leaves the M^{−1/2} apply
                  # densifying the aliased operand (P=3/dedup→P=2/dense) for dense-H.
                  _minv_both_aliased = _is_aliased_itensor(phi)
                  # Step 2b: geometric c = 2^⌈env/2⌉ per side (env = #sites contracted into
                  # that gram). Lgram at bond b = gram of sites 1..b-1; Rgram = sites b+2..N.
                  _p_c = if minv_from_p === true
                    _N = length(psi)
                    (2.0^cld(b - 1, 2), 2.0^cld(_N - b - 1, 2))
                  else
                    nothing
                  end
                  Mhalf_L, Linv_L, Mhalf_R, Linv_R =
                    SparseBackends.build_minv_half_pair_factored(Lgram, Rgram; phi_template=phi,
                                                                 both_aliased=_minv_both_aliased,
                                                                 p_c=_p_c)
                end

                # Recast H_eff output back to phi's aliased/BS classification
                # (schema-preserving alignment), shared by both eigsolve paths.
                # SB_RECAST_DEDUP_CHK=1: after each recast, compare the OUTPUT's dedup
                # (nb/nt) + sparsity (nb, key-set) to φ's — to confirm the recast restores
                # φ's compression/sparsity (if not, Krylov vectors stay de-deduped and the
                # adds churn). Per-bond latch, capped at SB_RECAST_DEDUP_CHK_MAX (default 8).
                _recast_chk = Ref(0)
                recast_to_phi = function(Hv)
                  if ITensors.has_external_storage(Hv) && ITensors.has_external_storage(phi)
                    Tw = ITensors.get_external_storage(phi)
                    Cw = ITensors.get_external_storage(Hv)
                    if Cw isa SparseBackends.WrappedBlockSparse && Tw isa SparseBackends.WrappedBlockSparse
                      return ITensors._itensor_from_external_storage(
                          SparseBackends.recast_bs_to_template(Cw, Tw))
                    elseif Cw isa SparseBackends.WrappedAliasedBlockSparse &&
                           Tw isa SparseBackends.WrappedAliasedBlockSparse
                      # SB_SNAP_PHI=1 (experiment): snap the matvec output to φ's exact
                      # 36-key/dedup schema — DROPS the M^½-inflation keys (out∉φ) and
                      # restores dedup, instead of the permute-only recast. Tests whether
                      # those full-norm-but-factorize-discarded keys affect the eigenvalue.
                      out = if get(ENV, "SB_SNAP_PHI", "0") == "1"
                          # recast first (permute to φ axis order), THEN snap (filter to φ keys)
                          _rc = SparseBackends.recast_aliased_to_template(Cw, Tw)
                          ITensors._itensor_from_external_storage(
                              _rc isa SparseBackends.WrappedAliasedBlockSparse ?
                                  SparseBackends._snap_to_schema(_rc, Tw) : _rc)
                      elseif get(ENV, "SB_DROP_KRYLOV", "0") == "1"
                          # DISCRIMINATOR: drop the same out-of-φ keys as SNAP, but keep the
                          # matvec output's OWN dedup (templates unchanged) → the downstream
                          # add!! stays on the in-place path. If this is exact at md40 while
                          # SNAP (dedup-restoring) drifted 1.7e-3, the drift was the merge-add
                          # path, not an M^½-frame matrix-element loss → matvec pruning is safe.
                          _rc = SparseBackends.recast_aliased_to_template(Cw, Tw)
                          if _rc isa SparseBackends.WrappedAliasedBlockSparse
                              _allowed = Set(Tw.aliased.keys)
                              ITensors._itensor_from_external_storage(
                                  SparseBackends.filter_keys_keepdedup(_rc, _allowed))
                          else
                              ITensors._itensor_from_external_storage(_rc)
                          end
                      else
                          ITensors._itensor_from_external_storage(
                              SparseBackends.recast_aliased_to_template(Cw, Tw))
                      end
                      if get(ENV, "SB_RECAST_DEDUP_CHK", "0") == "1" &&
                         _recast_chk[] < parse(Int, get(ENV, "SB_RECAST_DEDUP_CHK_MAX", "8")) &&
                         ITensors.has_external_storage(out)
                        _recast_chk[] += 1
                        Ow = ITensors.get_external_storage(out).aliased; Pw = Tw.aliased
                        onb = length(Ow.keys); ont = Ow.n_templates
                        pnb = length(Pw.keys); pnt = Pw.n_templates
                        # WASTED-BLOCK CHECK: keys present in the temporary (matvec output, now
                        # in phi's axis order) but ABSENT from phi → blocks we instantiated /
                        # GEMM'd that are NOT in the final psi-space and get filtered out here.
                        # (recast only permutes axes, so Ow.keys is the temporary's key-set.)
                        _pset = Set(Pw.keys)
                        wasted = [k for k in Ow.keys if !(k in _pset)]   # in temporary, not in phi
                        _oset = Set(Ow.keys)
                        n_zero = count(k -> !(k in _oset), Pw.keys)       # in phi, not produced
                        # WASTED-BLOCK NORMS: is each discarded block ~zero (⇒ filtering exact)?
                        let bs = Ow.blksize, T = eltype(Ow.templates)
                            _bn(i) = (off=(Int(Ow.alias_ids[i])-1)*bs;
                                      abs(Ow.scalars[i]) * sqrt(sum(abs2, @view Ow.templates[off+1:off+bs])))
                            _allmax = maximum((_bn(i) for i in 1:onb); init=0.0)
                            _widx = [i for i in 1:onb if !(Ow.keys[i] in _pset)]
                            _wmax = isempty(_widx) ? 0.0 : maximum(_bn(i) for i in _widx)
                            _wsum = isempty(_widx) ? 0.0 : sqrt(sum(_bn(i)^2 for i in _widx))
                            _ksum = sqrt(sum((_bn(i)^2 for i in 1:onb if Ow.keys[i] in _pset); init=0.0))
                            println("[WASTED_NORM #", _recast_chk[], " b=", b, "]  max_block=", round(_allmax,sigdigits=3),
                                    "  WASTED max=", round(_wmax,sigdigits=3), " (rel ", round(_wmax/max(_allmax,eps()),sigdigits=3),
                                    ")  ||wasted||/||kept||=", round(_wsum/max(_ksum,eps()),sigdigits=3))
                        end
                        println("[RECAST_CHK #", _recast_chk[], " b=", b, " sw=", sw, "]  out_nb=", onb,
                                " nt=", ont, " dedup=", round(onb/max(ont,1), digits=2),
                                "x | phi_nb=", pnb, " nt=", pnt, " dedup=", round(pnb/max(pnt,1), digits=2),
                                "x | WASTED(out∉phi)=", length(wasted), " (",
                                round(100*length(wasted)/max(onb,1), digits=1), "% of out)",
                                "  zero_in_phi(phi∉out)=", n_zero,
                                "  keyset_match=", isempty(wasted) && n_zero == 0)
                        for k in wasted[1:min(end, 12)]
                            println("    wasted key (instantiated, discarded on recast): ", k)
                        end
                        length(wasted) > 12 && println("    … (", length(wasted)-12, " more wasted keys)")
                      end
                      return out
                    end
                  end
                  return Hv
                end

                if run_mode === :bop_aliased || run_mode === :bop_densify
                  # B_op family: null-projected symmetric eigensolve (was BMF_BOP_PROJECT=1)
                  # ── Option A: null-space-projected symmetric eigensolve ────────
                  # B = M^{−1/2} · H_eff · M^{−1/2}, solved with the STANDARD inner
                  # product (plain Lanczos). Linv = M^{−1/2} has null(M) directions
                  # zeroed (build_half_pair_single rtol cut), so B maps null(M)→0 on
                  # BOTH sides ⇒ the eigensolve lives entirely on range(M) — the
                  # genuine independent DOF of the aliased ψ (template sharing makes
                  # M structurally rank-deficient; iso is unreachable without
                  # un-deduplicating). This removes the ghost modes and the 1/λ
                  # blowup that the A=M⁻¹·H_eff + M-inner-product path suffers when M
                  # is near-singular at large bond dim. Generalized eigenpair recovered
                  # via φ = M^{−1/2}·y (eigenvalue of B IS the generalized eigenvalue).
                  # All M^{±1/2} applies go through apply_minv_preserve_bs, which is
                  # schema-preserving, so (keys, alias_ids, scalars) — i.e. P's action
                  # — are untouched; only the template values transform.
                  # SB_STEP_IDX_DBG=1: dump the M^{±1/2}-apply index order (the "Minv" steps of
                  # the B = M^{-1/2}·H_eff·M^{-1/2} chain). Combine with SB_KRYLOV_IDX_DBG (Lenv/
                  # H1/H2/Renv operators + B(y) IN/OUT) and TRACE_BOND (per-step intermediates)
                  # for the full Minv→Lenv→H1→H2→Renv→Minv index trace. Per-bond latch.
                  _step_idx_n = Ref(0)
                  apply_half = function(opL, opR, z; fission::Bool=true)
                    _sidbg = get(ENV, "SB_STEP_IDX_DBG", "0") == "1" &&
                             _step_idx_n[] < parse(Int, get(ENV, "SB_STEP_IDX_DBG_MAX", "12"))
                    if _sidbg
                      _step_idx_n[] += 1
                      println("[STEP_IDX #", _step_idx_n[], " b=", b, " sw=", sw,
                              "] Minv-apply IN  inds=", collect(ITensors.inds(z)))
                    end
                    z = SparseBackends.apply_minv_preserve_bs(opL, z, phi; fission)
                    z = SparseBackends.apply_minv_preserve_bs(opR, z, phi; fission)
                    if _sidbg
                      println("[STEP_IDX #", _step_idx_n[], " b=", b, " sw=", sw,
                              "] Minv-apply OUT inds=", collect(ITensors.inds(z)))
                    end
                    return z
                  end
                  # Deferred-fission  onthe PRE-H apply_half (HARDENED 2026-06): its output
                  # is consumed only by product(PH,·), which re-establishes φ's schema
                  # internally (matvec hint-lastonly), so the M^{−1/2}y intermediate need
                  # not be fissioned to φ's classification. Both sides skip fission together
                  # (uniform big-block — no mixed L/R classification, unlike the earlier
                  # opL-only attempt that errored, see comment above). The POST-H apply_half
                  # keeps fission=true: its result is a Krylov vector and must match y0's (φ)
                  # schema for eigsolve. Validated bit-identical E (-17.16104, N=12 md=40)
                  # with a 2.06× steady-state per-sweep speedup (4.58s→2.22s): big-block x
                  # also lets the matvec itself run big-block GEMMs (matvec cpb 35.0→24.9s).
                  # NOTE (case 4, aliased ψ × aliased PHP): the fission=false big-block x
                  # is now safe because the M^{−1/2} factors are relayout-wrapped aliased
                  # (SB_ALIASED_MINV_WRAP, see build_minv_half_pair_factored) so x stays
                  # canonical (channel in prefix) and matches the aliased env. Case 2
                  # (dense H) keeps dense factors. No special-casing needed here.
                  # ── ENV-DRESSING (Path-B, dense-env case) ──────────────────────
                  # Fold M^{−1/2}=Linv into BOTH link legs of the (dense) L/R envs ONCE
                  # per bond, so B = M^{−1/2} H M^{−1/2} becomes a SINGLE matvec with no
                  # per-iteration apply_half. M=Lgram⊗Rgram is an outer product → Linv_L
                  # hits only the left link (= Lenv's ket/bra legs), Linv_R only the right
                  # (= Renv's legs). Lenv=[l(0),h,l(1)]; dressing the ket-link (l plev0) is
                  # the input-side M^{−1/2}, dressing the bra-link (l plev1, the output leg)
                  # is the output-side M^{−1/2}. plev-2 temporaries avoid colliding with the
                  # env's own bra-link (same bond Index at plev 0/1). Exact re-association of
                  # the current B_op; PH.LR is saved/restored (the env cache is assumed
                  # immutable by _makeL!/_makeR!). Gated on the EXISTING classification: this
                  # plain dense `env * Linv` dressing is valid only when the envs are dense
                  # (aliased ψ on bare/dense H, i.e. !_minv_both_aliased). Both-aliased
                  # (aliased PHP) has wrapped envs needing contract_preserve_bs → deferred;
                  # it falls through to the apply_half path below. y0 / recovery still pay
                  # one M^{1/2} / M^{−1/2} per eigsolve.
                  _lpos, _rpos = PH.lpos, PH.rpos
                  _L0 = (1 <= _lpos <= length(PH.LR) && isassigned(PH.LR, _lpos)) ? PH.LR[_lpos] : nothing
                  _R0 = (1 <= _rpos <= length(PH.LR) && isassigned(PH.LR, _rpos)) ? PH.LR[_rpos] : nothing
                  # Env-dressing is a function of ENV STORAGE, not run_mode: it's valid
                  # whenever the L/R envs are dense (⇔ H is dense, e.g. aliased ψ on bare
                  # H). Both B_op modes use it when available; the densify run_mode decision
                  # does NOT gate dressing. Aliased envs (both-aliased ψ×PHP) would need a
                  # contract_preserve_bs fold (deferred) → no dressing, per-vector apply_half.
                  _envs_dense = (_L0 === nothing || !ITensors.has_external_storage(_L0)) &&
                                (_R0 === nothing || !ITensors.has_external_storage(_R0))
                  if run_mode === :bop_densify && !_envs_dense
                    error("run_mode=:bop_densify requires dense L/R environments; got aliased/external-storage envs — use run_mode=:bop_aliased")
                  end
                  if _envs_dense
                    # ── ENV-DRESSING (dense-env case, INDEPENDENT of run_mode) ─────────
                    # Fold M^{−1/2}=Linv into BOTH link legs of the dense L/R envs ONCE per
                    # bond, so B = M^{−1/2} H M^{−1/2} becomes a SINGLE matvec with no
                    # per-iteration apply_half. M=Lgram⊗Rgram is an outer product → Linv_L
                    # hits only the left link (= Lenv's ket/bra legs), Linv_R only the right.
                    # plev-2 temporaries avoid colliding with the env's own bra-link.
                    # PH.LR is saved/restored (env cache assumed immutable by _makeL!/_makeR!).
                    # NOTE: `env(dense) * Linv(aliased)` routes through
                    # contract_aliased_dense_to_dense ⇒ a DENSE dressed env. The env was
                    # already dense before dressing (matvec was denseH_*), so dressing does
                    # NOT change the OPERAND's storage: dense env × aliased V still yields an
                    # aliased V (denseH_wrapV), exactly as the old per-vector path. The
                    # run_mode therefore still selects the operand form: :bop_aliased keeps V
                    # aliased through the solve; :bop_densify densifies the seed so the local
                    # Lanczos runs in dense BLAS (and re-aliases to φ's keys at recovery).
                    _dress_env = function(env, Linv)
                      Lk  = ITensors.replaceprime(Linv, 1 => 2)            # [l(0), l(2)]
                      env = ITensors.replaceprime(env * Lk, 2 => 0; tags = "Link")  # contract l(0); l(2)→l(0)
                      Lb  = ITensors.replaceprime(Linv, 0 => 2)            # [l(2), l(1)]
                      env = ITensors.replaceprime(env * Lb, 2 => 1; tags = "Link")  # contract l(1); l(2)→l(1)
                      return env
                    end
                    try
                      (_L0 isa ITensor) && (PH.LR[_lpos] = _dress_env(_L0, Linv_L))
                      (_R0 isa ITensor) && (PH.LR[_rpos] = _dress_env(_R0, Linv_R))
                      # SB_KRYLOV_IDX_DBG: dump the ITensor index orders of the tensors in
                      # the Krylov matvec (operator pieces once; operand y IN / B(y) OUT per
                      # matvec) for selected bonds — to see exactly which leg order the kernel
                      # receives vs emits (the leg order printed IS the kernel's label order).
                      # SB_KRYLOV_IDX_BONDS (default "1,3,5") picks bonds (1 = left edge, last
                      # = right edge for a 6-site N=2 chain); SB_KRYLOV_IDX_SWEEP picks the sweep.
                      _kdbg_bonds = Set(parse.(Int, split(get(ENV,"SB_KRYLOV_IDX_BONDS","1,3,5"), ",")))
                      _kdbg = get(ENV,"SB_KRYLOV_IDX_DBG","0")=="1" && (b in _kdbg_bonds) &&
                              ha == 1 && sw == parse(Int, get(ENV,"SB_KRYLOV_IDX_SWEEP","2"))
                      _kn = Ref(0)
                      if _kdbg
                        println("\n===== [KRYLOV-IDX  bond b=$b (sites $b,$(b+1))  ha=$ha sw=$sw  run_mode=$run_mode] =====")
                        println("  OPERATOR pieces (dressed env folds in M^{-1/2}):")
                        (_L0 isa ITensor) ? println("    Lenv(dressed) inds = ", ITensors.inds(PH.LR[_lpos])) :
                                            println("    Lenv          = (none — LEFT EDGE)")
                        println("    H[$b]         inds = ", ITensors.inds(PH.H[b]))
                        println("    H[$(b+1)]         inds = ", ITensors.inds(PH.H[b+1]))
                        (_R0 isa ITensor) ? println("    Renv(dressed) inds = ", ITensors.inds(PH.LR[_rpos])) :
                                            println("    Renv          = (none — RIGHT EDGE)")
                        println("    phi(template) inds = ", ITensors.inds(phi))
                      end
                      B_op = function(y)
                        out = recast_to_phi(product(PH, y))  # folded M^{−1/2}HM^{−1/2}
                        if _kdbg && _kn[] < 4
                          _kn[] += 1
                          println("\n  [matvec #$(_kn[])]  (index print order == kernel label order)")
                          println("    y    IN  inds = ", ITensors.inds(y))
                          println("    B(y) OUT inds = ", ITensors.inds(out))
                          println("    IN order == OUT order? ", collect(ITensors.inds(y)) == collect(ITensors.inds(out)))
                          SparseBackends.schema_dbg("eigstep B(y)#$(_kn[]) b=$b", out)
                        end
                        return out
                      end
                      _densify_solve = run_mode === :bop_densify
                      # SB_KRYLOV_IDX_DBG: trace sparse-vs-dense classification + fill +
                      # dedup along the Path-B pipeline at this bond — φ → M^{1/2}·φ (y0) →
                      # the eigsolve Krylov steps (in B_op above) → recovered φ. (schema_dbg
                      # self-gates on SB_SCHEMA_DBG, so enable both.)
                      if _kdbg; SparseBackends.schema_dbg("phi (input)      b=$b", phi); end
                      # y0 = M^{1/2} φ; aliased for :bop_aliased (V stays wrapped through the
                      # solve), densified for :bop_densify (dense BLAS local solve).
                      y0 = apply_half(Mhalf_L, Mhalf_R, phi)
                      _keytrace("2.y0 = M^{1/2}*phi (seed, raw)", y0)
                      if get(ENV, "SB_SNAP_PHI", "0") == "1" &&
                         ITensors.has_external_storage(y0) && ITensors.has_external_storage(phi)
                        _y0w = ITensors.get_external_storage(y0); _pw = ITensors.get_external_storage(phi)
                        if _y0w isa SparseBackends.WrappedAliasedBlockSparse &&
                           _pw isa SparseBackends.WrappedAliasedBlockSparse
                          y0 = ITensors._itensor_from_external_storage(SparseBackends._snap_to_schema(_y0w, _pw))
                        end
                      end
                      # SB_DROP_KRYLOV: filter the SEED to φ's key set too (dedup-1 kept),
                      # so v₀ and every matvec output w share the SAME 36-key set → the
                      # orthogonalization w−α·v₀ can't reintroduce dropped keys. Without
                      # this the seed stays 72-key and −α·v₀ merges the dropped keys back.
                      if get(ENV, "SB_DROP_KRYLOV", "0") == "1" &&
                         ITensors.has_external_storage(y0) && ITensors.has_external_storage(phi)
                        _y0w = ITensors.get_external_storage(y0); _pw = ITensors.get_external_storage(phi)
                        if _y0w isa SparseBackends.WrappedAliasedBlockSparse &&
                           _pw isa SparseBackends.WrappedAliasedBlockSparse
                          if get(ENV, "SB_DROP_KRYLOV_DBG", "0") == "1"
                            _seedset = Set(_pw.aliased.keys)
                            _seed_out = count(k -> !(k in _seedset), _y0w.aliased.keys)
                            println("[SEED_DBG b=", b, " ha=", ha, " sw=", sw,
                                    "] y0=M^{1/2}phi nb=", length(_y0w.aliased.keys),
                                    " nt=", _y0w.aliased.n_templates,
                                    " | phi nb=", length(_pw.aliased.keys),
                                    " nt=", _pw.aliased.n_templates,
                                    " | y0 keys NOT in phi=", _seed_out); flush(stdout)
                          end
                          y0 = ITensors._itensor_from_external_storage(
                                   SparseBackends.filter_keys_keepdedup(_y0w, Set(_pw.aliased.keys)))
                        end
                      end
                      if _kdbg; SparseBackends.schema_dbg("y0 = M^{1/2}*phi b=$b", y0); end
                      _densify_solve && (y0 = SparseBackends.to_dense_itensors_unfused(y0))
                      vals, yvecs = eigsolve(
                          B_op, y0, 1, eigsolve_which_eigenvalue;
                          ishermitian = true,
                          tol         = eigsolve_tol,
                          krylovdim   = eigsolve_krylovdim,
                          maxiter     = eigsolve_maxiter,
                          verbosity   = eigsolve_verbosity,
                      )
                      _keytrace("3.y = eigsolve result (Krylov eigenvector)", yvecs[1])
                      # φ = M^{−1/2} y. :bop_aliased keeps it aliased; :bop_densify re-imposes
                      # φ's keys (operand was dense) so replacebond!'s factorize rebuilds dedup.
                      vecs = [begin
                                _phi = apply_half(Linv_L, Linv_R, y)
                                _densify_solve ?
                                    SparseBackends.wrap_dense_as_aliased_via_template(
                                        ITensors.has_external_storage(_phi) ?
                                            SparseBackends.to_dense_itensors_unfused(_phi) : _phi,
                                        phi) :
                                    _phi
                              end for y in yvecs]
                      _keytrace("4.recovered phi = M^{-1/2}*y", vecs[1])
                      if _kdbg; SparseBackends.schema_dbg("recovered phi = M^{-1/2}*y b=$b", vecs[1]); end
                    finally
                      (_L0 isa ITensor) && (PH.LR[_lpos] = _L0)
                      (_R0 isa ITensor) && (PH.LR[_rpos] = _R0)
                    end
                  else
                  # No dressing (aliased L/R envs): per-vector apply_half keeps the operand
                  # aliased through the solve. (run_mode=:bop_densify already errored above.)
                  B_op = function(y)
                    x  = apply_half(Linv_L, Linv_R, y; fission = false)  # M^{−1/2} y (big-block)
                    Hx = recast_to_phi(product(PH, x))         # H_eff M^{−1/2} y
                    return apply_half(Linv_L, Linv_R, Hx)      # M^{−1/2} H_eff M^{−1/2} y
                  end
                  y0 = apply_half(Mhalf_L, Mhalf_R, phi)       # y0 = M^{1/2} φ
                  if get(ENV, "SB_PHI_Y0_DEDUP", "0") == "1"
                    println("\n===== [SB_PHI_Y0_DEDUP b=", b, " ha=", ha, " sw=", sw,
                            "] does M^{1/2} preserve φ's dedup? =====")
                    _dump_aliased_schema("phi          ", phi; max_blocks=0)
                    _dump_aliased_schema("y0=M^1/2 phi ", y0;  max_blocks=0)
                    flush(stdout)
                  end
                  vals, yvecs = eigsolve(
                      B_op, y0, 1, eigsolve_which_eigenvalue;
                      ishermitian = true,
                      tol         = eigsolve_tol,
                      krylovdim   = eigsolve_krylovdim,
                      maxiter     = eigsolve_maxiter,
                      verbosity   = eigsolve_verbosity,
                  )
                  vecs = [apply_half(Linv_L, Linv_R, y) for y in yvecs]   # φ = M^{−1/2} y
                  end
                  # ── PREFILTER-A (drop-at-factorize) ────────────────────────────
                  # DROP-AT-FACTORIZE (default, hardened): snap the FINAL recovered φ back
                  # to φ's key/dedup schema ONCE here (after eigsolve, before replacebond!),
                  # leaving every Krylov vector untouched. M^{-1/2} de-duplicates the recovered
                  # φ (dedup-4 → dedup-1) and at even bonds populates QN-forbidden channel
                  # combos (36→72 keys); replacebond!'s factorize would project those out
                  # anyway, so re-imposing φ's schema here is EXACT — validated BIT-IDENTICAL
                  # across all 10 sweeps at N=12 md=40 (and N=2 md=10) — while restoring ψ's
                  # dedup-4 compression on the stored tensor and shrinking the factorize input.
                  # NOT an env-gated experiment: this is the schema-preserving contract of the
                  # aliased Path-B (the stored ψ must live in φ's schema). It is a no-op on
                  # non-aliased φ (structural guard below). Distinct from SB_SNAP_PHI, which
                  # filtered EVERY Krylov vector and drifted 1.7e-3 (the keys carry real weight
                  # DURING the iteration; only the final stored vector is free to snap).
                  # SB_DROP_AT_FACT_DBG=1 prints per-bond key counts.
                  if !isempty(vecs) &&
                     ITensors.has_external_storage(vecs[1]) && ITensors.has_external_storage(phi)
                    _vw0 = ITensors.get_external_storage(vecs[1])
                    _pw0 = ITensors.get_external_storage(phi)
                    if _vw0 isa SparseBackends.WrappedAliasedBlockSparse &&
                       _pw0 isa SparseBackends.WrappedAliasedBlockSparse
                      _rc0 = SparseBackends.recast_aliased_to_template(_vw0, _pw0)
                      if _rc0 isa SparseBackends.WrappedAliasedBlockSparse
                        if get(ENV, "SB_DROP_AT_FACT_DBG", "0") == "1"
                          println("[DROP_AT_FACT b=", b, " ha=", ha, " sw=", sw,
                                  "] phi_recovered nb=", length(_rc0.aliased.keys),
                                  " nt=", _rc0.aliased.n_templates,
                                  " | template nb=", length(_pw0.aliased.keys),
                                  " nt=", _pw0.aliased.n_templates,
                                  " | dropping=", length(_rc0.aliased.keys) - length(_pw0.aliased.keys))
                          flush(stdout)
                        end
                        vecs[1] = ITensors._itensor_from_external_storage(
                                      SparseBackends._snap_to_schema(_rc0, _pw0))
                      end
                    end
                  end
                  if debug
                    println("[BOND_EIG b=", b, " ha=", ha, " sw=", sw, "] B_op smallest=", real(vals[1]),
                            "  Mrank-ok? Linv built; <phi|phi>=", round(real(inner(phi, phi)); digits=6))
                    flush(stdout)
                  end
                else  # run_mode === :minner
                # Path B (M-inner-product Lanczos): redefine the Krylov inner
                # product to <x,y>_M = <x, M·y>.
                M_dot = (x, y) -> begin
                    # M · y via factored: M = (Mhalf_L · Mhalf_L) · (Mhalf_R · Mhalf_R)
                    My = SparseBackends.apply_minv_preserve_bs(Mhalf_L, y,  y)
                    My = SparseBackends.apply_minv_preserve_bs(Mhalf_L, My, y)
                    My = SparseBackends.apply_minv_preserve_bs(Mhalf_R, My, y)
                    My = SparseBackends.apply_minv_preserve_bs(Mhalf_R, My, y)
                    return inner(x, My)
                end
                phi_wrapped = InnerProductVec(phi, M_dot)
                # Apply M⁻¹ to z (= Linv_L² ⊗ Linv_R², since Linv = M^(−1/2)),
                # preserving z's sparse storage / classification against phi.
                apply_Minv = z -> begin
                    z = SparseBackends.apply_minv_preserve_bs(Linv_L, z, phi)
                    z = SparseBackends.apply_minv_preserve_bs(Linv_L, z, phi)
                    z = SparseBackends.apply_minv_preserve_bs(Linv_R, z, phi)
                    z = SparseBackends.apply_minv_preserve_bs(Linv_R, z, phi)
                    return z
                end
                # A = M⁻¹·H_eff is the M-self-adjoint operator whose M-inner-product
                # Lanczos returns the generalized eigenvalues. Fragile when M is
                # near-singular (use run_mode=:bop_aliased for the robust variant above).
                # :minner is the collapsed A_op mode: M⁻¹ correction always applied,
                # Hermitian Lanczos (the old BMF_APPLY_MINV=0 / BMF_ARNOLDI=1 corners
                # are dropped — no test used them).
                H_op = function(v)
                  Hv = recast_to_phi(product(PH, v[]))
                  Av = apply_Minv(Hv)
                  return InnerProductVec(Av, M_dot)
                end
                vals, vecs = eigsolve(
                    H_op,
                    phi_wrapped,
                    1,
                    eigsolve_which_eigenvalue;
                    ishermitian = true,
                    tol         = eigsolve_tol,
                    krylovdim   = eigsolve_krylovdim,
                    maxiter     = eigsolve_maxiter,
                    verbosity   = eigsolve_verbosity,
                )
                vecs = [v[] for v in vecs]
                end
              # elseif get(ENV, "BMF_RAYLEIGH_RITZ", "0") == "1"
                #   # ⚠ ARCHIVED / PARKED (2026-06) — default OFF (BMF_RAYLEIGH_RITZ unset).
                #   # Validated correct (matches dense oracle b=1/b=3 ~1e-11; matches B_op
                #   # energy end-to-end at N=4) and stays aliased (plus_dense=0), but ~2.5×
                #   # slower than the default B_op path (extra H-matvecs + fissioned matvec
                #   # input). Parked for later perf work; do NOT enable in benchmarks. See
                #   # README "ARCHIVED: Rayleigh-Ritz". Resume by setting BMF_RAYLEIGH_RITZ=1.
                #   # ── Generalized Rayleigh-Ritz (no M^{±1/2} as a vector op) ────
                #   # Solves H_eff·φ = E·M·φ by projecting onto a small aliased
                #   # Krylov subspace built only from H·v (stays aliased via
                #   # recast_to_phi), forming k×k H_small/M_small with SCALAR inner
                #   # products (M applied with RAW Lgram/Rgram — no square root, no
                #   # BMF_MINV_RTOL), and solving the tiny null-projected generalized
                #   # eig. Ritz vector φ_new = Σ cᵢ vᵢ is an aliased combo of
                #   # φ-schema vectors → stays aliased. Mhalf_*/Linv_* are unused on
                #   # this path (build_minv_half_pair_factored above is wasted work
                #   # here; left running so this branch is a pure add — gate it for
                #   # perf once correctness is proven). 
                #   vals, vecs = SparseBackends.rayleigh_ritz_local_eigsolve(
                #       Hop_rr, phi, Lgram, Rgram;
                #       which     = eigsolve_which_eigenvalue,
                #       tol       = eigsolve_tol,
                #       krylovdim = eigsolve_krylovdim,
                #       maxiter   = eigsolve_maxiter,
                #       rtol      = 1e-8, # Hard coded to the defualt to remove old flag
                #       b = b, ha = ha, sw = sw,
                #   )
                #   if debug
                #       println("[BOND_EIG b=", b, " ha=", ha, " sw=", sw, "] RR smallest=", real(vals[1]))
                #       flush(stdout)
                #   end
              else
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
          end
          opt_time += time

          # Path-B uses the symmetric Cholesky/whitening transform so the
          # eigsolve runs with ishermitian=true (Lanczos). vals[1] is real
          # already in that case; the explicit `real()` is just defensive.
          energy = SparseBackends.is_sparse_mps(psi) ? real(vals[1]) : vals[1]
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
          _keytrace("5.phi before factorize (= vecs[1], post any drop)", phi)
          if get(ENV, "SB_ALIASED_TRACE", "0") == "1" && _EIGSOLVE_PHI_TRACE_COUNT[] < 5
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

          # ── FACT_KEY (before/after factorize key diff, SB_FACT_KEY_DBG=1) ──────
          # Capture φ's key set going INTO replacebond!, then reconstitute the
          # post-SVD two-site tensor and diff: keys in φ but gone after = what the
          # factorize truncation actually discards (the provably-safe-to-drop set).
          _factkey = get(ENV, "SB_FACT_KEY_DBG", "0") == "1"
          _fk_in = nothing
          if _factkey && ITensors.has_external_storage(phi) &&
             ITensors.get_external_storage(phi) isa SparseBackends.WrappedAliasedBlockSparse
            _fk_in = Set(ITensors.get_external_storage(phi).aliased.keys)
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
          if _factkey && _fk_in !== nothing
            try
              _phiw = ITensors.get_external_storage(phi)
              _aw = ITensors.get_external_storage(psi[b]); _bw = ITensors.get_external_storage(psi[b+1])
              _rw = SparseBackends.wrapped_contract_aliased(_aw, _bw; preserve_bs_output=true)
              _rw = _rw isa ITensors.ITensor ? ITensors.get_external_storage(_rw) : _rw
              if _rw isa SparseBackends.WrappedAliasedBlockSparse
                _rw = SparseBackends.recast_aliased_to_template(_rw, _phiw)   # align axes to φ
                _rw = _rw isa ITensors.ITensor ? ITensors.get_external_storage(_rw) : _rw
              end
              _outkeys = (_rw isa SparseBackends.WrappedAliasedBlockSparse) ? Set(_rw.aliased.keys) : Set(eltype(_fk_in)[])
              _dropped = setdiff(_fk_in, _outkeys); _added = setdiff(_outkeys, _fk_in)
              println("[FACT_KEY b=", b, " ha=", ha, " sw=", sw, "] phi_in=", length(_fk_in),
                      " recon_out=", length(_outkeys), " dropped(in∖out)=", length(_dropped),
                      " added(out∖in)=", length(_added))
              flush(stdout)
            catch e
              println("[FACT_KEY b=", b, "] recon failed: ", sprint(showerror, e)); flush(stdout)
            end
          end
          SparseBackends.schema_dbg("replacebond-OUT psi[$b]", psi[b])
          SparseBackends.schema_dbg("replacebond-OUT psi[$(b+1)]", psi[b+1])

          # println("================ print here ================ ", ITensors.has_external_storage(phi))

          maxtruncerr = max(maxtruncerr, spec.truncerr)

          # Path-B: update incremental gram-env cache after replacebond!.
          # Forward sweep (ha=1, ortho="left"): psi[b] is now the new left-
          # ortho tensor → L[b+1] must be refreshed.
          # Backward sweep (ha=2, ortho="right"): psi[b+1] is the new right-
          # ortho tensor → R[b+1] must be refreshed.
          if gram_cache !== nothing
            if ha == 1
              SparseBackends.update_left!(gram_cache, psi, b)
            else
              SparseBackends.update_right!(gram_cache, psi, b + 1)
            end
          end

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
  PH = position!(PH, psi, 1)
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
          PH = position!(PH, psi, b)
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