using Adapt: adapt
using KrylovKit: eigsolve
using NDTensors: scalartype, timer
using Printf: @printf
using TupleTools: TupleTools
import SparseBackends

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

debug_tracking() = get(ENV, "DEBUG_TRACKING", "") ∉ ("", "0", "false")

phi_tracker = Dict{Int,ITensor}()  # position → stored phi from run 1
phi_tracker2 = Dict{Int,ITensor}()
hphi_tracker = Dict{Int,ITensor}()  # position → stored PH*phi from run 1
bond2_tracker = Dict{String,ITensor}()  # sw==1,b==2,ha==1 intermediates from run 1

function compare_data(phi, old_phi, b, sw, string)
  t1 = ITensors.has_external_storage(phi)     ? SparseBackends.to_dense_itensors(phi)     : phi
  t2 = ITensors.has_external_storage(old_phi) ? SparseBackends.to_dense_itensors(old_phi) : old_phi
  i1_list = collect(inds(t1))
  i2_list = collect(inds(t2))
  ids1 = Set(ITensors.id(i) for i in i1_list)
  ids2 = Set(ITensors.id(i) for i in i2_list)
  if ids1 == ids2
    t2_aligned = t2
  else
    # Some link indices have different IDs across runs — remap by matching tag+dim,
    # but only for indices in t2 that have no ID match in t1
    idx_map = Index[]
    idx_replacement = Index[]
    used = falses(length(i1_list))
    for i2 in i2_list
      ITensors.id(i2) in ids1 && continue  # already exists in t1, no remap needed
      matched = findfirst(j -> !used[j] && dim(i1_list[j]) == dim(i2) && hastags(i1_list[j], tags(i2)), 1:length(i1_list))
      if !isnothing(matched)
        used[matched] = true
        push!(idx_map, i2)
        push!(idx_replacement, i1_list[matched])
      end
    end
    t2_aligned = isempty(idx_map) ? t2 : replaceinds(t2, idx_map, idx_replacement)
  end
  a1 = Array(t1, i1_list...)
  a2 = Array(t2_aligned, i1_list...)
  n1 = sqrt(sum(abs2, a1))
  n2 = sqrt(sum(abs2, a2))
  ovlp = abs(dot(vec(a1), vec(a2)))
  fidelity = ovlp / (n1 * n2)
  diff = a1 .- a2
  diff_norm = sqrt(sum(abs2, diff))
  diff_max = maximum(abs, diff)
  diff_sum = sum(abs, diff)
  println("   Sweep [$sw], comp. $string  phi[$b] ‖Δ‖=$(round(diff_norm, sigdigits=4))  max|Δ|=$(round(diff_max, sigdigits=4))  Σ|Δ|=$(round(diff_sum, sigdigits=4))  fidelity=$(round(fidelity, sigdigits=8))")
end


function check_phi!(phi, b, sw, ha; only_store=false)
  key = (sw - 1) * 1000 + ha * 100 + b
  if only_store
    haskey(phi_tracker, key) && println("WARNING replacing tracker data")
    phi_tracker[key] = deepcopy(phi)
  elseif haskey(phi_tracker, key)
    compare_data(phi, phi_tracker[key], b, sw, "phi")
  end
end

# Compare a tensor against the run-1 stored copy (in `bond2_tracker`) using
# the same align-by-(tag,dim) + dense-array diff as `compare_data`.
function compare_bond2(t::ITensor, name::String)
  haskey(bond2_tracker, name) || return
  old = bond2_tracker[name]
  t1 = ITensors.has_external_storage(t)   ? SparseBackends.to_dense_itensors(t)   : t
  t2 = ITensors.has_external_storage(old) ? SparseBackends.to_dense_itensors(old) : old
  t2_aligned = align_links(t1, t2, name)
  if isnothing(t2_aligned)
    println("  bond2[$name]: align_links failed; inds(t1)=$(inds(t1))  inds(t2)=$(inds(t2))")
    return
  end
  i1 = collect(inds(t1))
  a1 = Array(t1, i1...)
  a2 = Array(t2_aligned, i1...)
  d  = a1 .- a2
  println("  bond2[$name]  ‖Δ‖=", round(norm(d); sigdigits=4),
                          "  max|Δ|=", round(maximum(abs, d); sigdigits=4),
                          "  Σ|Δ|=", round(sum(abs, d); sigdigits=4))
end

# Dump environment tensors and product()-step intermediates at sweep 1 bond 2,
# half-sweep 1. On only_store=true (run 1, dense H) it stashes into bond2_tracker;
# on only_store=false (run 2, sparse H) it diffs against the stored copies.
# Called immediately after `PH = position!(PH, psi, b)`.
function bond2_diag!(PH, psi, b, sw, ha; only_store=false)
  (sw == 1 && b == 2 && ha == 1) || return
  println("===== bond2 diagnostic dump (sw=$sw, b=$b, ha=$ha) =====")

  function record!(name::String, T::ITensor)
    if only_store
      bond2_tracker[name] = deepcopy(T)
    else
      compare_bond2(T, name)
    end
  end

  # 1) cached environments built up to bond 2
  isassigned(PH.LR, 1) && record!("LR[1]", PH.LR[1])
  N = length(PH.H)
  isassigned(PH.LR, 3) && 3 <= N && record!("LR[3]", PH.LR[3])
  isassigned(PH.LR, 4) && 4 <= N && record!("LR[4]", PH.LR[4])

  # 2) Step-by-step build of `lproj * H[b] * H[b+1] * rproj`,
  # mirroring `contract(P::AbstractProjMPO, v::ITensor)` so we can locate
  # which sequential `*` first introduces the e-14 noise.
  Lp = ITensorMPS.lproj(PH)
  Rp = ITensorMPS.rproj(PH)
  if !(Lp isa ITensorMPS.OneITensor)
    record!("lproj_input", Lp)
  end
  if !(Rp isa ITensorMPS.OneITensor)
    record!("rproj_input", Rp)
  end

  step = Lp
  Hsites = collect(ITensorMPS.site_range(PH))
  for (i, s) in enumerate(Hsites)
    record!("H[$s]_raw", PH.H[s])
    step = step * PH.H[s]
    record!("after_L*H[$s]", step)
  end
  step = step * Rp
  record!("after_*R", step)

  # 3) Apply phi to compare the final tensor before noprime
  phi = psi[b] * psi[b+1]
  record!("phi", phi)
  hphi_raw = step * phi
  record!("hphi_pre_noprime", hphi_raw)
  hphi = noprime(hphi_raw)
  record!("hphi_final", hphi)

  println("===== end bond2 diagnostic dump =====")
end

function check_equality!(tensor_tracker, PH, psi; only_store=false, only_idx=nothing)
  if only_store
    push!(tensor_tracker, (deepcopy(PH), deepcopy(psi)))
    @assert length(tensor_tracker) === only_idx
  elseif only_idx <= length(tensor_tracker)
    # compare current and old projection MPOs
    PH_old = tensor_tracker[only_idx][1]
    println("Comparing current and old projection MPOs...")
    stop = false
    psi_old = tensor_tracker[only_idx][2]
    total_sq = 0.0
    for i in 1:length(psi)
      t1 = ITensors.has_external_storage(psi[i])     ? SparseBackends.to_dense_itensors(psi[i])     : psi[i]
      t2 = ITensors.has_external_storage(psi_old[i]) ? SparseBackends.to_dense_itensors(psi_old[i]) : psi_old[i]
      t2_aligned = align_links(t1, t2, "psi[$i]")
      if !isnothing(t2_aligned)
        diff = t1 - t2_aligned
        diff_norm = norm(diff)
        total_sq += diff_norm^2
        if !isapprox(t1, t2_aligned)
          println("  psi[$i] ✗ ‖Δ‖ = $diff_norm")
          stop = true
        end
        psi[i] = t2_aligned
      end
    end
    direct_norm = sqrt(total_sq)
    println("    at step $only_idx -> ‖psi - psi_old‖ (element-wise) = $direct_norm")
  end
end

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
  only_store=false
)
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

  if !ITensors.has_external_storage(psi[1])
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
  if debug_tracking()
    check_equality!(tensor_tracker, PH, psi; only_store=only_store, only_idx=only_idx)
  end

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
            PH = position!(PH, psi, b)
          end
          pos_time += timer

          # Bond-2 sweep-1 diagnostic dump (gated inside bond2_diag!)
          if debug_tracking()
            bond2_diag!(PH, psi, b, sw, ha; only_store=only_store)
          end

          @debug_check begin
            checkflux(psi)
            checkflux(PH)
          end

          @timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
            phi = psi[b] * psi[b + 1]
          end

          # phi = canonicalize_phi_phase(phi)

          if debug_tracking()
            # if sw > 2
            check_phi!(phi, b, sw, ha; only_store=only_store)
            # end
            if !only_store
              key = (sw - 1) * 1000 + ha * 100 + b
              old_phi = phi_tracker[key]
              # println("Aligning old phi inds to new phi inds: ", inds(old_phi), " -> ", inds(phi))
              phi_aligned = align_links(phi, old_phi, "phi_tracker")
              @assert !isnothing(phi_aligned) "align_links failed for phi_tracker at b=$b sw=$sw ha=$ha: inds(phi)=$(inds(phi)) inds(old_phi)=$(inds(old_phi))"
              # phi = phi_aligned
              compare_data(phi, old_phi, b, sw, "phi--")
            end
            # check_phi!(phi, b, sw, ha; only_store=only_store)

            # Single PH*phi application to isolate contraction bug from Krylov accumulation
            let hphi = product(PH, phi), key = (sw - 1) * 1000 + ha * 100 + b
              if only_store
                hphi_tracker[key] = deepcopy(hphi)
              elseif haskey(hphi_tracker, key)
                stored = hphi_tracker[key]
                stored_aligned = align_links(hphi, stored, "hphi")
                if !isnothing(stored_aligned)
                  compare_data(hphi, stored_aligned, b, sw, "hphi")
                end
              end
            end
          end

          # println("Starting eigsolve at sweep $sw, half $ha, bond ($b, $(b+1))")
          # if ITensors.has_external_storage(phi)
          #   println("Phi has inds ", inds(phi))
          #   println("external storage information: has_external_storage(phi)=$(ITensors.has_external_storage(phi))")
          #   println("  storage type: ", typeof(phi), phi)
          #   println("  unwrapped storage type: ", psi[b])
          #   println(" unwrapped storage type: ", psi[b+1])
          # end

          # println("eigsolve at sweep $sw, half $ha, bond ($b, $(b+1))")
          time = @elapsed begin
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

          if debug_tracking()
            key = (sw - 1) * 1000 + ha * 100 + b
            if only_store
              phi_tracker2[key] = phi
            else
              compare_data(phi, phi_tracker2[key], b, sw, "phi_post")
              phi_aligned2 = align_links(phi, phi_tracker2[key], "phi_tracker2")
              @assert !isnothing(phi_aligned2) "align_links failed for phi_tracker2 at b=$b sw=$sw ha=$ha: inds(phi)=$(inds(phi)) inds(stored)=$(inds(phi_tracker2[key]))"
              # phi = phi_aligned2
            end
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

          t = @elapsed begin
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
                svd_alg
              )
            end
          end

          # println("================ print here ================ ", ITensors.has_external_storage(phi))

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

          only_idx += 1
          if debug_tracking()
            check_equality!(tensor_tracker, PH, psi; only_store=only_store, only_idx=only_idx)
          end
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