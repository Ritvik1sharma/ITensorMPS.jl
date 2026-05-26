abstract type AbstractProjMPO end
import SparseBackends
import Serialization
using TimerOutputs: TimerOutput, @timeit, reset_timer!, print_timer

# Always-on instrumentation timer for the ProjMPO matvec hot path.
# Reset and print from user code via `ITensorMPS.PROJMPO_TIMER`.
const PROJMPO_TIMER = TimerOutput()

# Dumpers for matvec branches: serializes (it, Hv) pairs to a file gated
# by SB_DENSEDENSE_DUMP / SB_SPARSEDENSE_DUMP. Cap with *_MAX (default 100).
const _DENSEDENSE_DUMP_COUNT = Ref{Int}(0)
const _SPARSEDENSE_DUMP_COUNT = Ref{Int}(0)
const _DENSE_INDS_BUDGET = Ref{Int}(20)
# Single-matvec trace: when TRACE_BOND env is set (e.g., "3"), dump full
# pre/post indices for EVERY contract step of the FIRST product() call at
# that bond. After firing once, this latch becomes false to avoid floods.
const _TRACE_BOND_FIRED = Ref{Bool}(false)

# Permute profile logger gated by SB_PERMUTE_PROFILE=<path>. Captures, for every
# matvec/position! kernel call, the input/output structure plus the time spent
# in permutes vs GEMM. Tagged with call-site (matvec | position).
const _IN_POSITION = Ref{Bool}(false)
const _BOND_POSITION = Ref{Int}(-1)   # set by dmrg.jl before each matvec
const _SWEEP_NUM = Ref{Int}(-1)
const _PERMUTE_PROFILE_IO = Ref{Union{Nothing,IO}}(nothing)
const _PERMUTE_PROFILE_INIT = Ref{Bool}(false)
const _RUN_LABEL = Ref{String}("?")   # set by dmrg.jl: "DENSE" or "SPARSE"
function _permute_profile_io()
    if !_PERMUTE_PROFILE_INIT[]
        _PERMUTE_PROFILE_INIT[] = true
        path = get(ENV, "SB_PERMUTE_PROFILE", "")
        if !isempty(path)
            # Append mode so SparseBackends's parallel writer can also write
            # to the same path without coordination. Columns:
            # site \t kernel \t rank_A \t rank_B \t rank_out \t shared \t
            # permute_A_s \t permute_B_s \t gemm_s \t total_s \t
            # shared_pos_A \t shared_pos_B \t shared_pos_out \t inds_in \t inds_out
            io = open(path, "a")
            _PERMUTE_PROFILE_IO[] = io
            atexit() do
                io2 = _PERMUTE_PROFILE_IO[]
                if io2 !== nothing
                    close(io2)
                    _PERMUTE_PROFILE_IO[] = nothing
                end
            end
        end
    end
    return _PERMUTE_PROFILE_IO[]
end
function permute_profile_site()
    base = _IN_POSITION[] ? "position" : "matvec"
    run  = get(ENV, "SB_RUN_LABEL", "?")
    bond = get(ENV, "SB_BOND", "-1")
    step = get(ENV, "SB_STEP", "-1")
    return string(base, "|", run, "|", bond, "|", step)
end

DEBUG_FLAG = get(ENV, "INDEX_DEBUG", "0") == "1"

copy(::AbstractProjMPO) = error("Not implemented")

"""
    nsite(P::ProjMPO)

Retrieve the number of unprojected (open)
site indices of the ProjMPO object `P`
"""
nsite(P::AbstractProjMPO) = P.nsite

set_nsite!(::AbstractProjMPO, nsite) = error("Not implemented")

# The range of center sites
site_range(P::AbstractProjMPO) = (P.lpos + 1):(P.rpos - 1)

"""
    length(P::ProjMPO)

The length of a ProjMPO is the same as
the length of the MPO used to construct it
"""
Base.length(P::AbstractProjMPO) = length(P.H)

function lproj(P::AbstractProjMPO)::Union{ITensor, OneITensor}
    (P.lpos <= 0) && return OneITensor()
    return P.LR[P.lpos]
end

function rproj(P::AbstractProjMPO)::Union{ITensor, OneITensor}
    (P.rpos >= length(P) + 1) && return OneITensor()
    return P.LR[P.rpos]
end

function ITensors.contract(P::AbstractProjMPO, v::ITensor)::ITensor    
    global DEBUG_FLAG
    itensor_map = Union{ITensor, OneITensor}[lproj(P)]
    
    # push!(itensor_map, reduce(*, P.H[site_range(P)]))
    append!(itensor_map, P.H[site_range(P)])

    push!(itensor_map, rproj(P))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    first_t = first(itensor_map)
    comp = if first_t isa OneITensor
        1
    elseif ITensors.has_external_storage(first_t)
        prod(SparseBackends._dims(ITensors.get_external_storage(first_t)))
    else
        dim(first_t)
    end

    if comp == 1
        reverse!(itensor_map)
    end
    idx = 0

    # Apply the map
    Hv = v
    # Decide ONCE whether to preserve BS output through this contract. If v
    # entered as BS, route every step through contract_preserve_bs so the
    # output stays BS (avoids dense fallthrough in the H × phi loop, which
    # would later break KrylovKit's mixed-storage add!).
    preserve_bs_v = ITensors.has_external_storage(v)
    # When v is BS AND BMF_USE_HINT=1, derive an output classification hint
    # from v's dense_inds. Gated because in-kernel hint support isn't done yet.
    v_dense_hint = if preserve_bs_v && get(ENV, "BMF_USE_HINT", "0") == "1"
      vw = ITensors.get_external_storage(v)
      vw isa SparseBackends.WrappedBlockSparse ? SparseBackends.dense_inds(vw) : nothing
    else
      nothing
    end
    index = 0
    position = site_range(P).start
    debug = false
    if DEBUG_FLAG && position == 3
        debug = true
    end
    if get(ENV, "INDEX_DEBUG", "0") == "1"
        println("Contracting ProjMPO at position ", position, " ", debug, " ", DEBUG_FLAG)
    end

    # TRACE_BOND: one-shot full-step dump of a single product(P, v) call at
    # the chosen bond. Captures inputs (it, Hv), shared inds, output Hv per step.
    trace_bond_target = get(ENV, "TRACE_BOND", "")
    _trace_label = get(ENV, "TRACE_LABEL", "")
    do_trace = !isempty(trace_bond_target) && !_TRACE_BOND_FIRED[] &&
               get(ENV, "SB_BOND", "") == trace_bond_target &&
               (isempty(_trace_label) || get(ENV, "SB_RUN_LABEL", "") == _trace_label)
    if do_trace
      _TRACE_BOND_FIRED[] = true
      _show_inds(t) = [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(t)]
      _v_storage = ITensors.has_external_storage(v) ?
                   string(typeof(ITensors.get_external_storage(v))) : "dense"
      println("\n========== [TRACE_BOND=$trace_bond_target] product(P, v) ==========")
      println("v storage = $_v_storage")
      println("v.inds = ", _show_inds(v))
    end

    @timeit PROJMPO_TIMER "ProjMPO.contract" begin
      for it in itensor_map
          idx += 1
          # Step context for permute-profile probes (read by SparseBackends kernel too)
          ENV["SB_STEP"] = string(idx)
          if do_trace && !(it isa OneITensor)
            _it_storage = ITensors.has_external_storage(it) ?
                          string(typeof(ITensors.get_external_storage(it))) : "dense"
            _hv_storage = ITensors.has_external_storage(Hv) ?
                          string(typeof(ITensors.get_external_storage(Hv))) : "dense"
            _show_inds_local(t) = [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(t)]
            _shared = ITensors.commoninds(it, Hv)
            println("\n--- step idx=$idx ---")
            println("  it storage = $_it_storage")
            println("  Hv storage = $_hv_storage")
            println("  it.inds = ", _show_inds_local(it))
            println("  Hv.inds (BEFORE) = ", _show_inds_local(Hv))
            println("  shared = ", _show_inds_local(_shared))
          end
          if it isa OneITensor
              @timeit PROJMPO_TIMER "matvec.OneITensor_$(index)" begin
                  Hv *= it
              end
          else
              # 4-way split: tag by (H[j] storage) × (Hv storage). The mixed
              # cells (dense H × wrapped Hv, or sparse H × dense Hv) tell us
              # whether the "mixed-storage tax" is real.
              it_wrap = ITensors.has_external_storage(it)
              v_wrap  = ITensors.has_external_storage(Hv)
              if it_wrap
                  if v_wrap
                      if debug
                          println("Before multiplying sparse H and wrapped V at position ", position, " and index ", idx)
                          println("inds(it) = ", inds(it))
                          println("inds(Hv) = ", inds(Hv))
                      end
                      @timeit PROJMPO_TIMER "matvec.sparseH_wrapV_$(idx)" begin
                          Hv = preserve_bs_v ?
                              SparseBackends.contract_preserve_bs(Hv, it; template=nothing,
                                  output_inds_hint=v_dense_hint) :
                              Hv * it
                      end
                      if debug
                          println("After multiplying sparse H and wrapped V at position ", position, " and index ", idx)
                          println("inds(Hv) = ", inds(Hv))
                      end
                  else
                      _sdd_path = get(ENV, "SB_SPARSEDENSE_DUMP", "")
                      if !isempty(_sdd_path)
                          _sdd_max = parse(Int, get(ENV, "SB_SPARSEDENSE_DUMP_MAX", "100"))
                          if _SPARSEDENSE_DUMP_COUNT[] < _sdd_max
                              open(_sdd_path, "a") do io
                                  Serialization.serialize(io, (it = deepcopy(it), Hv = deepcopy(Hv)))
                              end
                              _SPARSEDENSE_DUMP_COUNT[] += 1
                          end
                      end
                      if debug
                          println("Before multiplying sparse H and dense V at position ", position, " and index ", idx)
                          println("inds(it) = ", inds(it))
                          println("inds(Hv) = ", inds(Hv))
                      end
                      @timeit PROJMPO_TIMER "matvec.sparseH_denseV_$(idx)" begin
                          # Plan B: put `it` on the left so output starts with
                          # uncontr(it). Default behavior (off) keeps `Hv * it`.
                          Hv = get(ENV, "SB_PLAN_B", "0") == "1" ? it * Hv : Hv * it
                      end
                      if debug
                          println("After multiplying sparse H and dense V at position ", position, " and index ", idx)
                          println("inds(Hv) = ", inds(Hv))
                      end
                  end
              else
                  if v_wrap
                      if debug
                          println("Before multiplying dense H and wrapped V at position ", position, " and index ", idx)
                          println("inds(it) = ", inds(it))
                          println("inds(Hv) = ", inds(Hv))
                      end
                      @timeit PROJMPO_TIMER "matvec.denseH_wrapV_$(idx)" begin
                          Hv = preserve_bs_v ?
                              SparseBackends.contract_preserve_bs(Hv, it; template=nothing,
                                  output_inds_hint=v_dense_hint) :
                              Hv * it
                      end
                      if debug
                          println("After multiplying dense H and wrapped V at position ", position, " and index ", idx)
                          println("inds(Hv) = ", inds(Hv))
                      end
                  else
                      # DENSE_INDS_DEBUG=1: budgeted (idx, inds(it), inds(Hv), inds(Hv*it))
                      # dump so we can compare what natural-order ITensor produces
                      # for dense H·v contracts vs what the sparse hint kernel does.
                      if get(ENV, "DENSE_INDS_DEBUG", "0") == "1" && _DENSE_INDS_BUDGET[] > 0
                        _DENSE_INDS_BUDGET[] -= 1
                        _di_shared = ITensors.commoninds(it, Hv)
                        println("[DENSE_INDS] idx=$idx  shared=$(length(_di_shared))")
                        println("  inds(it) = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in inds(it)])
                        println("  inds(Hv) = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in inds(Hv)])
                        println("  shared   = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in _di_shared])
                      end
                      # SB_DENSEDENSE_DUMP=<path>: serialize (it, Hv) ITensor
                      # pairs to <path> for offline replay (test alternative
                      # orderings under real ITensors *). Capped via
                      # SB_DENSEDENSE_DUMP_MAX (default 100).
                      _ddd_path = get(ENV, "SB_DENSEDENSE_DUMP", "")
                      if !isempty(_ddd_path)
                          _ddd_max = parse(Int, get(ENV, "SB_DENSEDENSE_DUMP_MAX", "100"))
                          if _DENSEDENSE_DUMP_COUNT[] < _ddd_max
                              open(_ddd_path, "a") do io
                                  Serialization.serialize(io, (it = deepcopy(it), Hv = deepcopy(Hv)))
                              end
                              _DENSEDENSE_DUMP_COUNT[] += 1
                          end
                      end
                      _ppio = _permute_profile_io()
                      if _ppio !== nothing
                          _pp_inA = inds(it); _pp_inB = inds(Hv)
                          _pp_shared = ITensors.commoninds(it, Hv)
                          _pp_pA = [findfirst(==(s), _pp_inA) for s in _pp_shared]
                          _pp_pB = [findfirst(==(s), _pp_inB) for s in _pp_shared]
                          _pp_t0 = time_ns()
                          
                          @timeit PROJMPO_TIMER "matvec.denseH_denseV_$(idx)" begin
                              Hv = get(ENV, "SB_PLAN_B", "0") == "1" ? it * Hv : Hv * it
                          end
                          _pp_dt = (time_ns() - _pp_t0) / 1e9
                          println(_ppio, permute_profile_site(), "\tdenseH\t",
                                  length(_pp_inA), "\t", length(_pp_inB), "\t", ndims(Hv), "\t",
                                  length(_pp_shared), "\t",
                                  0.0, "\t", 0.0, "\t", 0.0, "\t", _pp_dt, "\t",
                                  join(_pp_pA, ";"), "\t", join(_pp_pB, ";"), "\t-\t",
                                  join(string.(size(it)), ","), "|", join(string.(size(Hv)), ","), "\t",
                                  join(string.(size(Hv)), ","))
                      else
                          if debug
                            println("Before multiplying dense H and dense V at position ", position, " and index ", idx)
                            println("inds(it) = ", inds(it))
                            println("inds(Hv) = ", inds(Hv))
                          end
                          @timeit PROJMPO_TIMER "matvec.denseH_denseV_$(idx)" begin
                              Hv = get(ENV, "SB_PLAN_B", "0") == "1" ? it * Hv : Hv * it
                          end
                          if debug
                                println("After multiplying dense H and dense V at position ", position, " and index ", idx)
                                println("inds(Hv) = ", inds(Hv))
                          end
                      end
                  end
              end
          end
          if do_trace && !(it isa OneITensor)
            _hv_storage_after = ITensors.has_external_storage(Hv) ?
                                string(typeof(ITensors.get_external_storage(Hv))) : "dense"
            _show_inds_after(t) = [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(t)]
            println("  Hv.inds (AFTER ) = ", _show_inds_after(Hv))
            println("  Hv storage after = $_hv_storage_after")
          end
      end
    end
    if debug
        println("------------------")
    end
    return Hv
end


# function classify_H(Hs, n)
#     if length(inds(Hs)) == 6
#         tag = ""
#         for i in inds(Hs)
#             if i.has_tags("Link")
#                 tag += "L,"
#             else
#                 tag += "s,"
#             end
#         end
#         return "H[$n]: sparse (lL,lR,s,s',dL,dR)"
#     elseif ndims(Hs) == 4
#         tag = ""
#         for i in inds(Hs)
#             if i.has_tags("Link")
#                 tag += "L,"
#             else
#                 tag += "s,"
#             end
#         end
#         return "H[$n]: dense ($tag)"
#     else
#         return "H[$n]: unknown layout"
#     end
# end


# function classify_H(Hs, n)

#     idxs = inds(Hs)

#     # collect label per index
#     tags = String[]

#     link_left = String[]
#     link_right = String[]
#     sites = String[]

#     for (k, ix) in enumerate(idxs)

#         ix_tags = string(ix)

#         if has_tag(ix, "Link")
#             # try to disambiguate left/right from tag string
#             if occursin("left", ix_tags) || occursin("l=", ix_tags)
#                 push!(tags, "L")
#                 push!(link_left, "k$k")
#             elseif occursin("right", ix_tags) || occursin("r=", ix_tags)
#                 push!(tags, "R")
#                 push!(link_right, "k$k")
#             else
#                 # unknown link → mark generic
#                 push!(tags, "L?")
#             end
#         else
#             push!(tags, "s")
#             push!(sites, "k$k")
#         end
#     end

#     tag_str = join(tags, ",")

#     if length(idxs) == 6
#         return "H[$n]: sparse (lL,lR,s,s',dL,dR) | $tag_str"
#     elseif length(idxs) == 4
#         return "H[$n]: dense (lL,lR,s,s') | $tag_str"
#     else
#         return "H[$n]: unknown layout ($tag_str)"
#     end
# end

# function ITensors.contract(P::AbstractProjMPO, v::ITensor)::ITensor
#     # Store (label, tensor)
#     itensor_map = Vector{Tuple{String,Union{ITensor,OneITensor}}}()

#     push!(itensor_map, ("LProj", lproj(P)))

#     # DMRG Hamiltonian terms
        
#     # for (n, s) in enumerate(site_range(P))
#     #     push!(itensor_map, (classify_H(P.H[s], n), P.H[s]))
#     # end
#     apppend!(itensor_map, [(classify_H(P.H[s], s), P.H[s]) for s in site_range(P)])

#     push!(itensor_map, ("RProj", rproj(P)))

#     # Reverse contraction order if needed
#     first_t = first(itensor_map)[2]

#     comp = if first_t isa OneITensor
#         1
#     elseif ITensors.has_external_storage(first_t)
#         prod(SparseBackends._dims(ITensors.get_external_storage(first_t)))
#     else
#         dim(first_t)
#     end

#     if comp == 1
#         reverse!(itensor_map)
#     end

#     Hv = v

#     @timeit PROJMPO_TIMER "ProjMPO.contract" begin
#         for (label, it) in itensor_map
#             if it isa OneITensor
#                 @timeit PROJMPO_TIMER "matvec.OneITensor" begin
#                     Hv *= it
#                 end

#             elseif ITensors.has_external_storage(it)
#                 @timeit PROJMPO_TIMER "matvec.sparse_$label" begin
#                     Hv *= it
#                 end

#             else
#                 @timeit PROJMPO_TIMER "matvec.dense_$label" begin
#                     Hv *= it
#                 end
#             end
#         end
#     end

#     return Hv
# end

"""
    product(P::ProjMPO,v::ITensor)::ITensor

    (P::ProjMPO)(v::ITensor)

Efficiently multiply the ProjMPO `P`
by an ITensor `v` in the sense that the
ProjMPO is a generalized square matrix
or linear operator and `v` is a generalized
vector in the space where it acts. The
returned ITensor will have the same indices
as `v`. The operator overload `P(v)` is
shorthand for `product(P,v)`.
"""
function product(P::AbstractProjMPO, v::ITensor)::ITensor
    Pv = contract(P, v)
    if order(Pv) != order(v)
        error(
            string(
                "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
                "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
                "(1) You are trying to multiply the ProjMPO with the $(nsite(P))-site wave-function at the wrong position.\n",
                "(2) `orthogonalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
                "P*v inds: $(inds(Pv)) \n\n",
                "v inds: $(inds(v))",
            ),
        )
    end
    Pv = noprime(Pv)
    # When v has BS storage, the contract output Pv may have a slightly
    # different BS axis order. KrylovKit's add!!(y, x, α) requires basis
    # vectors with IDENTICAL BS structure — recast Pv to v's structure to keep
    # the Lanczos basis type-stable.
    if ITensors.has_external_storage(v) && ITensors.has_external_storage(Pv)
        Tw = ITensors.get_external_storage(v)
        Cw = ITensors.get_external_storage(Pv)
        if Cw isa SparseBackends.WrappedBlockSparse && Tw isa SparseBackends.WrappedBlockSparse
            Pv = ITensors._itensor_from_external_storage(
                SparseBackends.recast_bs_to_template(Cw, Tw))
        end
    end
    return Pv
end

(P::AbstractProjMPO)(v::ITensor) = product(P, v)

"""
    eltype(P::ProjMPO)

Deduce the element type (such as Float64
or ComplexF64) of the tensors in the ProjMPO
`P`.
"""
function Base.eltype(P::AbstractProjMPO)::Type
    ElType = eltype(lproj(P))
    for j in site_range(P)
        ElType = promote_type(ElType, eltype(P.H[j]))
    end
    return promote_type(ElType, eltype(rproj(P)))
end

"""
    size(P::ProjMPO)

The size of a ProjMPO are its dimensions
`(d,d)` when viewed as a matrix or linear operator
acting on a space of dimension `d`.

For example, if a ProjMPO maps from a space with
indices `(a,s1,s2,b)` to the space `(a',s1',s2',b')`
then the size is `(d,d)` where
`d = dim(a)*dim(s1)*dim(s1)*dim(b)`
"""
function Base.size(P::AbstractProjMPO)::Tuple{Int, Int}
    d = 1
    for i in inds(lproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    for j in site_range(P)
        for i in inds(P.H[j])
            plev(i) > 0 && (d *= dim(i))
        end
    end
    for i in inds(rproj(P))
        plev(i) > 0 && (d *= dim(i))
    end
    return (d, d)
end

const _ENV_DBG_COUNT = Ref(0)

# Reorder an env tensor so that:
#   - axes tagged "FusedSparse" go LAST  (matches the matvec kernel's
#     canonical layout where shared_prefix is trailing).
#   - all other axes keep their relative order.
# This is a metadata + one-time data move during env construction; downstream
# matvec contractions then skip their per-call permute_B.
function _reorder_env_for_aliased(T::ITensor)
    get(ENV, "SB_FUSE_LINKS", "0") == "1" || return T
    all_inds = collect(inds(T))
    # Classify each axis of the env tensor:
    #   sparse  : tag contains "FusedSparse"  (sparse H-link)
    #   paired  : has a noprime sibling on T (ket↔bra link of psi)
    #               → split by plev  (0=ket, 1=bra)
    #   unpaired: dense Link without a sibling on T (dense H-link)
    is_fused(I) = contains(string(tags(I)), "FusedSparse")
    fused   = filter(is_fused, all_inds)
    nonfused = filter(I -> !is_fused(I), all_inds)
    function has_sibling(I)
        for J in nonfused
            J === I && continue
            if ITensors.noprime(I) == ITensors.noprime(J) &&
               ITensors.plev(I) != ITensors.plev(J)
                return true
            end
        end
        return false
    end
    paired_ket = filter(I -> has_sibling(I) && ITensors.plev(I) == 0, nonfused)
    paired_bra = filter(I -> has_sibling(I) && ITensors.plev(I) == 1, nonfused)
    unpaired   = filter(I -> !has_sibling(I), nonfused)
    # Order: [ket-link (shared with v),  dense-H-link,  sparse-H-link (FusedSparse),  bra-link (kept)]
    # so that `it * Hv` (Plan B direction) produces an output whose B side
    # starts with [dense-H, sparse-H, bra] = [red_dense, keepB-from-env] —
    # leaving only the v-side shared-prefix-site to be placed for the next
    # sparse-H × dense kernel.
    new_order = vcat(paired_ket, unpaired, fused, paired_bra)
    new_order == all_inds && return T
    return ITensors.permute(T, new_order...)
end

function _makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)::Union{ITensor, Nothing}
    # SB_ENV_DEBUG=1 turns on env structure prints for the first
    # SB_ENV_DEBUG_MAX (default 6) env-build steps.  Shows the index order
    # of L (and the substep results) so we can see what indices the
    # downstream matvec contraction will see.
    _env_dbg = get(ENV, "SB_ENV_DEBUG", "0") == "1"
    _env_dbg_max = parse(Int, get(ENV, "SB_ENV_DEBUG_MAX", "6"))
    # println("\t\tPositioning ProjMPO ", debug)
    # Save the last `L` that is made to help with caching
    # for DiskProjMPO
    ll = P.lpos
    if ll ≥ k
        # Special case when nothing has to be done.
        # Still need to change the position if lproj is
        # being moved backward.
        P.lpos = k
        return nothing
    end
    # Make sure ll is at least 0 for the generic logic below
    ll = max(ll, 0)
    L = lproj(P)
    while ll < k
        if debug
            L = L * P.H[ll + 1]
            println("========= ", ll + 1, " after multiplying H: ", ITensors.has_external_storage(P.H[ll + 1]))
            L = L * dag(prime(psi[ll + 1]))
            println("========= ", ll + 1, " after multiplying dag prime psi: ", ITensors.has_external_storage(L))
            L = L * psi[ll + 1]
            println("========= ", ll + 1, " after multiplying psi: ", ITensors.has_external_storage(L))
        else
            H_site = P.H[ll + 1]
            # if ITensors.has_external_storage(H_site)
            #     H_site = SparseBackends.to_dense_itensors(H_site)
            # end
            if _env_dbg && _ENV_DBG_COUNT[] < _env_dbg_max
                _ENV_DBG_COUNT[] += 1
                println("\n[SB_ENV_DEBUG #", _ENV_DBG_COUNT[], "] _makeL! step site=", ll+1, "  (target k=", k, ")")
                if L isa ITensor
                    println("  L (before) inds = ", inds(L),
                            "  external? ", ITensors.has_external_storage(L))
                else
                    println("  L (before) is OneITensor (scalar identity)")
                end
                println("  H_site inds       = ", inds(H_site),
                        "  external? ", ITensors.has_external_storage(H_site))
            end
            L = L * H_site
            if _env_dbg && _ENV_DBG_COUNT[] <= _env_dbg_max
                println("  L (after × H_site) inds = ", inds(L))
            end
            try
                L = L * dag(prime(psi[ll + 1]))
            catch e
                println("Error at position ll = ", ll)
                println("ll + 1 = ", ll + 1)
                println("\n--- Tensor L ---")
                println(L)
                println("inds(L) = ", inds(L))
                println("\n--- psi[ll + 1] ---")
                println(psi[ll + 1])
                println("inds(psi[ll + 1]) = ", inds(psi[ll + 1]))
                println("\n--- prime(psi[ll + 1]) ---")
                println(prime(psi[ll + 1]))
                println("inds(prime(psi[ll + 1])) = ", inds(prime(psi[ll + 1])))
                println("\n--- dag(prime(psi[ll + 1])) ---")
                println(dag(prime(psi[ll + 1])))
                println("inds(dag(prime(psi[ll + 1]))) = ", inds(dag(prime(psi[ll + 1]))))
                println("\n--- Error ---")
                rethrow(e)
            end
            # L = L * dag(prime(psi[ll + 1]))
            L = L * psi[ll + 1]
            # Reorder so "FusedSparse" axes go last → matvec kernel sees
            # canonical layout and skips its per-call permute_B.
            L = _reorder_env_for_aliased(L)
            if _env_dbg && _ENV_DBG_COUNT[] <= _env_dbg_max
                println("  L (after × dag(psi') × psi, post-reorder) FINAL inds = ", inds(L))
                println("    P.LR[", ll+1, "] stored.  external? ",
                        ITensors.has_external_storage(L))
            end
        end
        P.LR[ll + 1] = L
        ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return L
end

function makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)
    # println("\tPositioning ProjMPO ", debug)
    _makeL!(P, psi, k; debug=debug)
    return P
end

function _makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)::Union{ITensor, Nothing}
    # Save the last `R` that is made to help with caching
    # for DiskProjMPO
    rl = P.rpos
    if rl ≤ k
        # Special case when nothing has to be done.
        # Still need to change the position if rproj is
        # being moved backward.
        P.rpos = k
        return nothing
    end
    N = length(P.H)
    # Make sure rl is no bigger than `N + 1` for the generic logic below
    rl = min(rl, N + 1)
    R = rproj(P)
    while rl > k
        if debug
            println("========= ", rl, N+1)
            time = @elapsed begin
                R = R * P.H[rl - 1]
            end
            println("========= ", " after multiplying H: ", R)
            time = @elapsed begin
                R =  dag(prime(psi[rl - 1])) * R
            end
            println("========= ", " after multiplying psi: ", R)
            time = @elapsed begin
                R = psi[rl - 1] * R
            end
            println("========= ", " after multiplying dag psi: ", R)
        else
            _env_dbg2 = get(ENV, "SB_ENV_DEBUG", "0") == "1"
            _env_dbg2_max = parse(Int, get(ENV, "SB_ENV_DEBUG_MAX", "6"))
            H_site = P.H[rl - 1]
            # if ITensors.has_external_storage(H_site)
            #     H_site = SparseBackends.to_dense_itensors(H_site)
            # end
            if _env_dbg2 && _ENV_DBG_COUNT[] < _env_dbg2_max
                _ENV_DBG_COUNT[] += 1
                println("\n[SB_ENV_DEBUG #", _ENV_DBG_COUNT[], "] _makeR! step site=", rl-1, "  (target k=", k, ")")
                if R isa ITensor
                    println("  R (before) inds = ", inds(R),
                            "  external? ", ITensors.has_external_storage(R))
                else
                    println("  R (before) is OneITensor (scalar identity)")
                end
                println("  H_site inds       = ", inds(H_site),
                        "  external? ", ITensors.has_external_storage(H_site))
            end
            R = R * H_site
            if _env_dbg2 && _ENV_DBG_COUNT[] <= _env_dbg2_max
                println("  R (after × H_site) inds = ", inds(R))
            end
            R = dag(prime(psi[rl - 1])) * R
            R = psi[rl - 1] * R
            R = _reorder_env_for_aliased(R)
            if _env_dbg2 && _ENV_DBG_COUNT[] <= _env_dbg2_max
                println("  R (after × dag(psi') × psi, post-reorder) FINAL inds = ", inds(R))
            end
        end
        P.LR[rl - 1] = R
        # println(" check rl - 1 information ", rl - 1)
        # println("TENSOR IS ", R)
        rl -= 1
    end
    P.rpos = k
    return R
end

function makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false)
    _makeR!(P, psi, k; debug=debug)
    return P
end

"""
    position!(P::ProjMPO, psi::MPS, pos::Int)

Given an MPS `psi`, shift the projection of the
MPO represented by the ProjMPO `P` such that
the set of unprojected sites begins with site `pos`.
This operation efficiently reuses previous projections
of the MPO on sites that have already been projected.
The MPS `psi` must have compatible bond indices with
the previous projected MPO tensors for this
operation to succeed.
"""
function position!(P::AbstractProjMPO, psi::MPS, pos::Int; debug=false)
    # println("Positioning ProjMPO ", debug)
    makeL!(P, psi, pos - 1; debug=debug)
    makeR!(P, psi, pos + nsite(P); debug=debug)
    return P
end

"""
    noiseterm(P::ProjMPO,
              phi::ITensor,
              ortho::String)

Return a "noise term" or density matrix perturbation
ITensor as proposed in Phys. Rev. B 72, 180403 for aiding
convergence of DMRG calculations. The ITensor `phi`
is the contracted product of MPS tensors acted on by the
ProjMPO `P`, and `ortho` is a String which can take
the values `"left"` or `"right"` depending on the
sweeping direction of the DMRG calculation.
"""
function noiseterm(P::AbstractProjMPO, phi::ITensor, ortho::String)::ITensor
    if nsite(P) != 2
        error("noise term only defined for 2-site ProjMPO")
    end

    site_range_P = site_range(P)
    if ortho == "left"
        AL = P.H[first(site_range_P)]
        AL = lproj(P) * AL
        nt = AL * phi
    elseif ortho == "right"
        AR = P.H[last(site_range_P)]
        AR = AR * rproj(P)
        nt = phi * AR
    else
        error("In noiseterm, got ortho = $ortho, only supports `left` and `right`")
    end
    nt = nt * dag(noprime(nt))

    return nt
end

function checkflux(P::AbstractProjMPO)
    checkflux(P.H)
    for n in length(P.LR)
        if isassigned(P.LR, n)
            checkflux(P.LR[n])
        end
    end
    return nothing
end
