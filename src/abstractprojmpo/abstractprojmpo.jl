abstract type AbstractProjMPO end
import SparseBackends
import Serialization
using TimerOutputs: TimerOutput, @timeit, reset_timer!, print_timer

# Aliased-aware multiply: if at least one input has WrappedAliasedBlockSparse
# storage, route through wrapped_contract_aliased with preserve_bs_output=true
# so the result stays aliased (the default `*` path would densify). Otherwise
# fall back to the normal `*`. Mirrors `contract_preserve_bs` for BS psi.
# Used inside _makeL!/_makeR! and the matvec path.
@inline _is_aliased_itensor(T::ITensor) = ITensors.has_external_storage(T) &&
    T.tensor.data isa SparseBackends.WrappedAliasedBlockSparse

# ── φ-vs-Hv schema dump (gated SB_PHI_SCHEMA_DUMP=1) ───────────────────────────
# Prints the aliased schema (per-block key / alias_id / scalar, + which template
# ids are actually distinct) for a matvec input v and its output Hv, so we can SEE
# how applying H reshuffles keys and the template-sharing map. Latched to the
# first SB_PHI_SCHEMA_DUMP_MAX (default 1) product() calls.
const _PHI_SCHEMA_DUMP_COUNT = Ref(0)
function _dump_aliased_schema(label::String, T::ITensor; max_blocks::Int=40)
    if !_is_aliased_itensor(T)
        println("  [$label] NOT aliased: storage=",
                ITensors.has_external_storage(T) ? string(typeof(ITensors.get_external_storage(T))) : "dense",
                "  inds=", [ITensors.dim(I) for I in inds(T)])
        return
    end
    A = ITensors.get_external_storage(T).aliased
    nb = length(A.keys); nt = A.n_templates
    # hash each block's actual numeric template to confirm which are truly equal
    thash(tid) = begin
        off = (tid - 1) * A.blksize
        hash(@view A.templates[(off+1):(off+A.blksize)])
    end
    uniq_hashes = Set(thash(A.alias_ids[i]) for i in 1:nb)
    println("  [$label] dims=", collect(A.dims), " prefix=", length(A.keys) == 0 ? 0 : length(first(A.keys)),
            " blksize=", A.blksize, " nb=", nb, " nt=", nt,
            " dedup=", round(nb/max(nt,1), digits=2), "x  distinct_template_values=", length(uniq_hashes))
    println("       (block) key → alias_id [scalar]   (template value-hash)")
    for i in 1:min(nb, max_blocks)
        println("       #", lpad(i,3), "  ", A.keys[i], " → tid=", A.alias_ids[i],
                " [s=", round(A.scalars[i], digits=4), "]  vhash=", string(thash(A.alias_ids[i]), base=16)[1:6])
    end
    nb > max_blocks && println("       … (", nb - max_blocks, " more blocks)")
end

@inline function _mul_preserve_aliased(A::ITensor, B::ITensor; in_position::Bool=false)
    a_ali = _is_aliased_itensor(A)
    b_ali = _is_aliased_itensor(B)
    if !a_ali && !b_ali
        return A * B
    end
    Aw = a_ali ? ITensors.get_external_storage(A) :
                 SparseBackends.wrap_itensor(A; backend=:dense)
    Bw = b_ali ? ITensors.get_external_storage(B) :
                 SparseBackends.wrap_itensor(B; backend=:dense)
    # Compute a dense-inds hint so the output's classification stays
    # consistent with the aliased input: only the aliased input's dense-tail
    # Indices go into the output's dense tail; site/link indices from the
    # dense operand (e.g. H_site's site') get pulled into the sparse prefix.
    # This avoids cross-region mismatches between subsequent contracts.
    hint = if a_ali && b_ali
        Set{ITensors.Index}(SparseBackends.dense_inds(Aw)) ∪
        Set{ITensors.Index}(SparseBackends.dense_inds(Bw))
    elseif a_ali
        Set{ITensors.Index}(SparseBackends.dense_inds(Aw))
    else
        Set{ITensors.Index}(SparseBackends.dense_inds(Bw))
    end
    Cw = SparseBackends.wrapped_contract_aliased(Aw, Bw;
        preserve_bs_output=true, output_inds_hint=hint, in_position=in_position)
    C = Cw isa ITensor ? Cw : ITensors._itensor_from_external_storage(Cw)
    if SparseBackends.ALIASED_TRACE[]
        if ITensors.has_external_storage(C) &&
           C.tensor.data isa SparseBackends.WrappedAliasedBlockSparse
            ali = C.tensor.data.aliased
            nb = length(ali.keys); nt = ali.n_templates
            println("[SB_ALIASED_TRACE mul_preserve_aliased]  result=Aliased{N=$(ndims(ali)),N2=$(ndims(ali)-length(_p(ali)))} nb=$nb nt=$nt  compression=$(round(nb/max(nt,1),digits=2))x")
        else
            println("[SB_ALIASED_TRACE mul_preserve_aliased]  result=", ITensors.has_external_storage(C) ? typeof(C.tensor.data) : "dense")
        end
    end
    return C
end
_p(::SparseBackends.AliasedBlockSparse{T,N,N2,P,K}) where {T,N,N2,P,K} = ntuple(_->0, Val(P))

# Env-build multiply: when `keep` (BOTH ψ and the MPO H are aliased), route the
# Lenv/Renv contraction through `_mul_preserve_aliased` so the environment stays
# WrappedAliasedBlockSparse and carries an explicit prefix/dense classification.
# Otherwise (dense H, dense ψ, or BS) use the plain `*` — so the only-φ-aliased
# (aliased ψ × dense H) and only-H-aliased pathways are byte-identical. A scalar
# boundary `OneITensor` always uses `*`.
# MAC count of a pairwise contraction = ∏ over all DISTINCT indices of (it ∪ Hv)
# = output_size × contracted_size. Used by the FLOP counter for the dense matvec
# steps and the dense env-build contractions.
#
# CRITICAL: dedup on (id, plev), NOT id alone. In ITensors an index and its prime
# share the same `id` (priming only bumps `plev`), so keying on id would collapse
# a physical leg's `s` (plev 0, contracted) and `s'` (plev 1, output) into one and
# silently drop a factor = the site dim (×3 for S=1). That under-counts every
# H-step by the site dimension. Two indices are the same iff same (id, plev).
@inline function _union_dim_macs(A::ITensor, B::ITensor)
    seen = Set{Tuple{UInt64,Int}}(); m = 1
    @inbounds for I in inds(A)
        k = (ITensors.id(I), ITensors.plev(I))
        if !(k in seen); push!(seen, k); m *= ITensors.dim(I); end
    end
    @inbounds for I in inds(B)
        k = (ITensors.id(I), ITensors.plev(I))
        if !(k in seen); push!(seen, k); m *= ITensors.dim(I); end
    end
    return m
end

@inline function _env_mul(A, B, keep::Bool)
    # _env_mul is only ever called from _makeL!/_makeR! (position!'s
    # env-building code) — structurally always "position", so in_position=true
    # is hardcoded below rather than threaded (was SB_IN_POSITION env var).
    if keep && A isa ITensor && B isa ITensor
        return _mul_preserve_aliased(A, B; in_position=true)
    end
    # Honest FLOP accounting: count PURE dense×dense env-build contractions here.
    # If either operand is aliased, `A * B` routes through the aliased kernel,
    # which counts itself — gating on !aliased avoids double-counting.
    if SparseBackends._flop_count_enabled() && A isa ITensor && B isa ITensor &&
       !_is_aliased_itensor(A) && !_is_aliased_itensor(B)
        SparseBackends.add_dense_macs!(_union_dim_macs(A, B), true)
    end
    return A * B
end

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

# Permute profile logger, gated by the roofline switch (was SB_PERMUTE_PROFILE=<path>).
# Captures, for every matvec/position! kernel call, the input/output structure
# plus the time spent in permutes vs GEMM. Tagged with call-site (matvec | position).
const _IN_POSITION = Ref{Bool}(false)
const _BOND_POSITION = Ref{Int}(-1)   # set by dmrg.jl before each matvec
const _SWEEP_NUM = Ref{Int}(-1)
const _PERMUTE_PROFILE_IO = Ref{Union{Nothing,IO}}(nothing)
const _PERMUTE_PROFILE_INIT = Ref{Bool}(false)
const _RUN_LABEL = Ref{String}("?")   # set by dmrg.jl: "DENSE" or "SPARSE"
function _permute_profile_io()
    if !_PERMUTE_PROFILE_INIT[]
        _PERMUTE_PROFILE_INIT[] = true
        path = SparseBackends._roofline_on() ? SparseBackends._RF_PERMUTE_PATH[] : ""
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
function permute_profile_site(run_label::String="?")
    base = _IN_POSITION[] ? "position" : "matvec"
    bond = get(ENV, "SB_BOND", "-1")
    step = something(SparseBackends.CURRENT_STEP[], -1)
    return string(base, "|", run_label, "|", bond, "|", step)
end

# INDEX_DEBUG removed 2026-06 — was the master switch for DEBUG_FLAG, which gates
# a family of `if debug` prints in ITensors.contract below. Hardcoded off.
DEBUG_FLAG = false

# ── Env-footprint recorder (gated by the roofline switch) ────────────────────
# Records, per (CURRENT_RUN_LABEL, site), a snapshot of the env tensor stored in
# P.LR[site]: its leg structure (inds+dims), real bytes, storage kind, and the
# dense-equivalent bytes. Keyed by site so the LAST build (steady-state, largest
# bond dim) overwrites earlier sweeps. Call print_env_footprint() to dump a
# per-site ALI-vs-DEN comparison. Used to settle whether the both-aliased env is
# actually smaller than the dense-PHP × dense-ψ env (it carries bond legs from
# BOTH H and ψ, and may lose channel sparsity).
const _ENV_FP = Dict{Tuple{String,Int},NamedTuple}()

function _record_env_footprint(site::Int, T, label::String="?")
    (T isa ITensor) || return nothing
    is = collect(inds(T))
    dims = [ITensors.dim(I) for I in is]
    dense_equiv_bytes = (isempty(dims) ? 1 : prod(dims)) * 16   # ComplexF64 worst case
    bytes = Base.summarysize(T)
    kind = "dense"; nb = 0; nt = 0; ali_dims = Int[]; prefix = 0
    if ITensors.has_external_storage(T)
        s = ITensors.get_external_storage(T)
        if s isa SparseBackends.WrappedAliasedBlockSparse
            kind = "aliased"; ali = s.aliased
            nb = length(ali.keys); nt = ali.n_templates
            ali_dims = collect(ali.dims); prefix = SparseBackends._abs_head_len(s)
        elseif s isa SparseBackends.WrappedBlockSparse
            kind = "blocksparse"; bs = s.blocksparse; nb = length(bs.keys)
        end
    end
    _ENV_FP[(label, site)] = (; site, dims, bytes, dense_equiv_bytes, kind, nb, nt, prefix, ali_dims)
    return nothing
end

_kib(b) = round(b/1024, digits=2)
# Was gated on SB_ENV_FOOTPRINT env var; now (like show_roofline/show_cas_stats)
# just reports whatever _ENV_FP accumulated — empty unless a dmrg(...; roofline=true)
# call populated it via _record_env_footprint.
function print_env_footprint(; labelA="ALI", labelB="DEN")
    sites = sort(unique(k[2] for k in keys(_ENV_FP)))
    println("\n========== ENV FOOTPRINT (P.LR per site; latest/steady-state build) ==========")
    println("  legend: bytes = Base.summarysize of the env ITensor; dims = env leg dims ",
            "(ket-bond × H-bond × bra-bond, + any link strands); dense_equiv = ∏dims×16B")
    totA = 0; totB = 0
    for s in sites
        a = get(_ENV_FP, (labelA, s), nothing)
        b = get(_ENV_FP, (labelB, s), nothing)
        if a !== nothing
            totA += a.bytes
            extra = a.kind == "aliased" ? "  nb=$(a.nb) nt=$(a.nt) dedup=$(round(a.nb/max(a.nt,1),digits=2))x prefix=$(a.prefix) ali_dims=$(a.ali_dims)" :
                    a.kind == "blocksparse" ? "  nb=$(a.nb)" : ""
            println("  [$labelA] site $s  $(rpad(a.kind,11)) bytes=$(_kib(a.bytes))KiB  dims=$(a.dims)  dense_equiv=$(_kib(a.dense_equiv_bytes))KiB$extra")
        end
        if b !== nothing
            totB += b.bytes
            println("  [$labelB] site $s  $(rpad(b.kind,11)) bytes=$(_kib(b.bytes))KiB  dims=$(b.dims)  dense_equiv=$(_kib(b.dense_equiv_bytes))KiB")
        end
        if a !== nothing && b !== nothing
            println("        → ratio $labelA/$labelB bytes = $(round(a.bytes/max(b.bytes,1),digits=3))  (want < 1)")
        end
    end
    if totA > 0 && totB > 0
        println("  --- TOTAL env bytes: $labelA=$(_kib(totA))KiB  $labelB=$(_kib(totB))KiB  ratio=$(round(totA/max(totB,1),digits=3)) (want < 1) ---")
    else
        println("  --- TOTAL env bytes: $labelA=$(_kib(totA))KiB  $labelB=$(_kib(totB))KiB (one side missing — set SparseBackends.CURRENT_RUN_LABEL[] per run) ---")
    end
    return nothing
end

# The preferred-output ordering for the `it * Hv` result (so the NEXT
# contraction's `permB` is identity) is now derived kernel-side from the next
# operator itself — see `_canon_labels_for_next` in SparseBackends. The matvec
# loop passes the next chain operator as `next_op=` rather than recomputing the
# step's shared/output indices here.

# Dense-chain layout hint: order THIS step's output so the NEXT step's kernel
# `permA` (input-ψ reorg) is identity — the axes reduced at the next step (shared
# with it_next) go LAST, into the kernel's `red_dense` slot, so it needn't permute
# the input. Operates purely on index structure (the chain is deterministic), and
# unlike the kernel-side next_op ordering it handles a DENSE `it_next` (our
# denseH_wrapV matvec chain). Returns an ordered Vector{Index} or nothing.
function _dense_chain_hint(current_Hv, current_it, it_next)
    it_next isa OneITensor && return nothing
    shared = ITensors.commoninds(current_it, current_Hv)
    out_inds = ITensors.Index[]
    for I in inds(current_it); (I in shared) || push!(out_inds, I); end
    for I in inds(current_Hv); (I in shared) || push!(out_inds, I); end
    nxt = inds(it_next)
    next_red = ITensors.Index[I for I in out_inds if I in nxt]   # reduced at next step
    isempty(next_red) && return nothing
    keep = ITensors.Index[I for I in out_inds if !(I in nxt)]
    return vcat(keep, next_red)   # next-reduction axes LAST ⇒ next permA dense-tail identity
end

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

"""
Permute env tensor at P.LR[idx] so FusedSparse-tagged axes come LAST. When
the resulting env is then contracted with ψ via ITensor `*`, the FusedSparse
axes land at the end of the env's contribution to Hv — closer to the kernel's
preferred B layout [red_dense, keepB, ..., shared_prefix].

Mutates P.LR[idx] in place. Idempotent (skips if already canonical). Cost:
O(|env|) once per `position!` boundary; amortized across all matvecs at
that bond.
"""
# True iff any H-MPO site of this projector is aliased (dense-ψ × aliased-H run).
# Guarded: some projector types (e.g. excited-state wrappers) have no `.H`.
function _proj_h_aliased(P::AbstractProjMPO)
    hasproperty(P, :H) || return false
    H = P.H
    @inbounds for i in 1:length(H)
        t = H[i]
        !(t isa OneITensor) && ITensors.has_external_storage(t) && return true
    end
    return false
end

function _permute_env_to_canonical!(P::AbstractProjMPO, idx::Int)
    (idx <= 0 || idx > length(P.LR)) && return
    T = P.LR[idx]
    T isa OneITensor && return
    isnothing(T) && return
    cur_inds = collect(inds(T))
    is_fused(I) = contains(string(ITensors.tags(I)), "FusedSparse")
    fused     = [I for I in cur_inds if  is_fused(I)]
    non_fused = [I for I in cur_inds if !is_fused(I)]
    isempty(fused) && return
    # dense-ψ × aliased-H: the matvec swaps step 1 to `env * φ` and needs the env as
    # [ket, dense-H, fused, bra] (fused BEFORE bra) so T1 lands in "Layout C" and
    # step 2 reads B strided (K2). Distinguished from the aliased-ψ × dense-H path
    # (fused links come from ψ, H is dense) which keeps the original fused-LAST order.
    if _proj_h_aliased(P) && !_is_aliased_itensor(T)
        newT = _reorder_env_for_aliased(T, false, true)
        (newT !== T) && (P.LR[idx] = newT)
        return
    end
    target = vcat(non_fused, fused)
    cur_inds == target && return
    P.LR[idx] = permute(T, target...)
end

function ITensors.contract(P::AbstractProjMPO, v::ITensor; roofline::Bool=false, run_label::String="?")::ITensor
    global DEBUG_FLAG
    # Also set the Ref for contract_bs_dense.jl's shared writer, which is
    # reached via generic dispatch (no argument-carrying call chain from here).
    SparseBackends.CURRENT_RUN_LABEL[] = run_label
    # Pre-order each env (Lenv/Renv) so FusedSparse axes trail — once per bond
    # (idempotent: subsequent matvecs at the same bond find it canonical and skip),
    # so the per-matvec permB/conv on the env-contracting steps (1 & 4) is cheaper.
    # HARDENED 2026-07-05 (was SB_PREPERMUTE_ENVS): bit-identical E, conv −49% / permB
    # −28% on the dense-H aliased-ψ matvec; no-op for dense ψ (env has no FusedSparse
    # tags → early return in _permute_env_to_canonical!).
    _permute_env_to_canonical!(P, P.lpos)
    _permute_env_to_canonical!(P, P.rpos)
    itensor_map = Union{ITensor, OneITensor}[lproj(P)]

    sr = site_range(P)
    # push!(itensor_map, reduce(*, P.H[site_range(P)]))
    append!(itensor_map, P.H[sr])

    push!(itensor_map, rproj(P))
    SparseBackends.schema_dbg("HENV lproj (Lenv)", itensor_map[1])
    SparseBackends.schema_dbg("HENV rproj (Renv)", itensor_map[end])

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
    index = 0
    position = site_range(P).start
    debug = false
    if DEBUG_FLAG && position == 3
        debug = true
    end
    if false  # INDEX_DEBUG removed 2026-06 — flip to true here for debug output
        println("Contracting ProjMPO at position ", position, " ", debug, " ", DEBUG_FLAG)
    end

    # TRACE_BOND / TRACE_LABEL removed 2026-06: one-shot full-step dump of a single
    # product(P, v) call at a chosen bond. Debug-print only. do_trace hardcoded
    # false below.
    # TRACE (manually re-enabled): fire ONCE, on the first bulk bond (neither
    # env a OneITensor) whose chain contains an ALIASED operator — i.e. the H
    # sites are aliased while Lenv/Renv stay dense (a MIXED chain, not fully
    # aliased). This condition first becomes true in the aliased run's JIT
    # WARMUP pass (maxdim hardcoded to 10 there), so the dumped bond dims reflect
    # the warmup, not the requested --bd; the index STRUCTURE and permutes are
    # identical to the measured run. Skips the dense-PHP run (no aliased op) and
    # edge bonds. For --bd 10 the warmup maxdim == requested bd, so fully
    # representative; for larger bd only the dim magnitudes would differ.
    do_trace = (!_TRACE_BOND_FIRED[]) &&
               !(lproj(P) isa OneITensor) && !(rproj(P) isa OneITensor) &&
               any(t -> !(t isa OneITensor) && ITensors.has_external_storage(t) &&
                        (ITensors.get_external_storage(t) isa SparseBackends.WrappedAliasedBlockSparse),
                   itensor_map)
    if do_trace
      _TRACE_BOND_FIRED[] = true
      _show_inds(t) = [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in inds(t)]
      _v_storage = ITensors.has_external_storage(v) ?
                   string(typeof(ITensors.get_external_storage(v))) : "dense"
      println("\n========== [TRACE first aliased bulk bond] product(P, v) ==========")
      println("v storage = $_v_storage")
      println("v.inds = ", _show_inds(v))
      SparseBackends.check_image("v (input = M^{1/2}φ)", v)
    end

    # First real (non-OneITensor) operator index — used by the step-1 env-swap below.
    _first_real_idx = findfirst(x -> !(x isa OneITensor), itensor_map)
    # Step-1 operand swap (Plan B): the env is pre-ordered [ket, dense-H, fused,
    # bra] by _reorder_env_for_aliased for the `env * φ` direction. Running step 1
    # as `it * Hv` (env first) instead of `Hv * it` puts the env's dense-H leg
    # (red_dense) LEADING and keepB contiguous in T1 = "Layout C", so step 2's
    # aliased×dense kernel reads B strided (K2) with no permute_B. Only for an
    # aliased-H run (dense-H baselines stay byte-identical); at edges where the
    # geometry differs it just harmlessly falls back to permute_B. Label-based
    # contraction ⇒ energy is convergence-equivalent.
    _aliased_h_run = any(x -> !(x isa OneITensor) && ITensors.has_external_storage(x), itensor_map)
    # Bond-type for the static-perm table lookup (and the capture generator): the
    # only chain variation is the edge reversal — left edge (lproj scalar) vs right
    # edge (rproj scalar) vs bulk. Set in ENV for the wrapper to read.
    _bondtype = (lproj(P) isa OneITensor) ? :left :
                (rproj(P) isa OneITensor) ? :right : :bulk
    ENV["SB_BONDTYPE"] = string(_bondtype)
    @timeit PROJMPO_TIMER "ProjMPO.contract" begin
      for it in itensor_map
          idx += 1
          # Step context for permute-profile probes (read by SparseBackends kernel
          # too). DEBUGGING VAR — plain runtime Ref, not an ENV var.
          SparseBackends.CURRENT_STEP[] = idx
          # Canon-pool (dense-ψ × aliased-H): remember the operand THIS step consumes
          # so its dead dense buffer can be returned to the ali_dense_alloc pool after
          # the step. Only real steps; v and the produced Hv are guarded at recycle.
          _prev_dense = (_aliased_h_run && !(it isa OneITensor)) ? Hv : nothing
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
            # ACTUAL sparse-key vs dense-tail classification from the storage
            # (not inferred from inds order).
            _fmt(lst) = [(ITensors.dim(I), string(ITensors.tags(I)), ITensors.plev(I)) for I in lst]
            if ITensors.has_external_storage(Hv)
              _hvw = ITensors.get_external_storage(Hv)
              if _hvw isa SparseBackends.WrappedAliasedBlockSparse
                _den = SparseBackends.dense_inds(_hvw)
                _spk = [I for I in inds(Hv) if !(I in _den)]
                println("  Hv P=", SparseBackends._abs_head_len(_hvw),
                        "  SPARSE keys = ", _fmt(_spk),
                        "  DENSE tail = ", _fmt([I for I in inds(Hv) if I in _den]))
              end
            end
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
              # FLOP count (SB_FLOP_COUNT=1): for DENSE matvec steps (it not
              # aliased — env L/R steps, and every step of the dense-PHP run), the
              # GEMM MAC count is ∏ over all distinct indices of (it ∪ Hv). Aliased-it
              # steps are counted inside contract_aliased_dense_to_dense! instead, so
              # gating on !it_wrap avoids double-counting.
              if SparseBackends._flop_count_enabled() && !it_wrap
                  SparseBackends.add_dense_macs!(_union_dim_macs(it, Hv), false)
              end
              if it_wrap
                      # Was SB_SPARSEDENSE_DUMP(_MAX): serialize (it, Hv) ITensor pairs to a
                      # path for offline replay. Commented out, not deleted — uncomment to
                      # re-enable, giving `_sdd_path`/`_sdd_max` real values.
                      # _sdd_path = get(ENV, "SB_SPARSEDENSE_DUMP", "")
                      # if !isempty(_sdd_path)
                      #     _sdd_max = parse(Int, get(ENV, "SB_SPARSEDENSE_DUMP_MAX", "100"))
                      #     if _SPARSEDENSE_DUMP_COUNT[] < _sdd_max
                      #         open(_sdd_path, "a") do io
                      #             Serialization.serialize(io, (it = deepcopy(it), Hv = deepcopy(Hv)))
                      #         end
                      #         _SPARSEDENSE_DUMP_COUNT[] += 1
                      #     end
                      # end
                      if debug
                          println("Before multiplying sparse H and dense V at position ", position, " and index ", idx)
                          println("inds(it) = ", inds(it))
                          println("inds(Hv) = ", inds(Hv))
                      end
                      @timeit PROJMPO_TIMER "matvec.sparseH_denseV_$(idx)" begin
                        # Output order comes from the hardcoded dense-ψ table keyed by
                        # (bondtype, step); φ is pinned upstream so this is deterministic
                        # every sweep. Retires _canon_labels_for_next on the dense path
                        # (next_op no longer passed).
                        Hv = SparseBackends.contract_aliased_itensor(
                            it, Hv, :aliased, :dense;
                            preserve_bs_output = false,
                            output_perm = SparseBackends.static_output_perm_dense(_bondtype, idx),
                        )
                      end
                      if debug
                          println("After multiplying sparse H and dense V at position ", position, " and index ", idx)
                          println("inds(Hv) = ", inds(Hv))
                      end
              else
                      # DENSE_INDS_DEBUG=1: budgeted (idx, inds(it), inds(Hv), inds(Hv*it))
                      # dump so we can compare what natural-order ITensor produces
                      # for dense H·v contracts vs what the sparse hint kernel does.
                      if false && _DENSE_INDS_BUDGET[] > 0  # DENSE_INDS_DEBUG removed 2026-06 — flip to true here for debug output
                        _DENSE_INDS_BUDGET[] -= 1
                        _di_shared = ITensors.commoninds(it, Hv)
                        println("[DENSE_INDS] idx=$idx  shared=$(length(_di_shared))")
                        println("  inds(it) = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in inds(it)])
                        println("  inds(Hv) = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in inds(Hv)])
                        println("  shared   = ", [(ITensors.dim(I), ITensors.tags(I), ITensors.plev(I)) for I in _di_shared])
                      end
                      # Was SB_DENSEDENSE_DUMP(_MAX): serialize (it, Hv) ITensor pairs to a
                      # path for offline replay (test alternative orderings under real
                      # ITensors *). Commented out, not deleted — uncomment to re-enable,
                      # giving `_ddd_path`/`_ddd_max` real values.
                      # _ddd_path = get(ENV, "SB_DENSEDENSE_DUMP", "")
                      # if !isempty(_ddd_path)
                      #     _ddd_max = parse(Int, get(ENV, "SB_DENSEDENSE_DUMP_MAX", "100"))
                      #     if _DENSEDENSE_DUMP_COUNT[] < _ddd_max
                      #         open(_ddd_path, "a") do io
                      #             Serialization.serialize(io, (it = deepcopy(it), Hv = deepcopy(Hv)))
                      #         end
                      #         _DENSEDENSE_DUMP_COUNT[] += 1
                      #     end
                      # end
                      _ppio = _permute_profile_io()
                      if _ppio !== nothing
                          _pp_inA = inds(it); _pp_inB = inds(Hv)
                          _pp_shared = ITensors.commoninds(it, Hv)
                          _pp_pA = [findfirst(==(s), _pp_inA) for s in _pp_shared]
                          _pp_pB = [findfirst(==(s), _pp_inB) for s in _pp_shared]
                          _pp_t0 = time_ns()
                          
                          @timeit PROJMPO_TIMER "matvec.denseH_denseV_$(idx)" begin
                              Hv = (_aliased_h_run && idx == _first_real_idx) ?
                                   _reorder_env_for_aliased(it, false, true) * Hv : Hv * it
                          end
                          _pp_dt = (time_ns() - _pp_t0) / 1e9
                          println(_ppio, permute_profile_site(run_label), "\tdenseH\t",
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
                              # Step 1 = env * φ (swapped) so T1 = "Layout C"
                              # ([red | gapBefore | keepB(contig) | gapAfter]) and the
                              # step-2 kernel reads B strided (no permute_B). The env is
                              # canonicalized to [ket, dense-H, fused, bra] ONCE per bond
                              # by _permute_env_to_canonical! (see below).
                              Hv = (_aliased_h_run && idx == _first_real_idx) ? it * Hv : Hv * it
                          end
                          if debug
                                println("After multiplying dense H and dense V at position ", position, " and index ", idx)
                                println("inds(Hv) = ", inds(Hv))
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
            SparseBackends.check_image("step $idx out", Hv)
          end
          # Recycle the consumed operand's dead dense buffer into the ali_dense_alloc
          # pool. Guards: skip v (KrylovKit owns it) and skip the produced Hv (so the
          # last step's output — the returned matvec result — is never pooled).
          if _prev_dense !== nothing && _prev_dense !== Hv && _prev_dense !== v
              SparseBackends.recycle_dense_ctgt!(_prev_dense)
          end
      end
    end
    if debug
        println("------------------")
    end
    # (Removed the SB_ALIASED_SNAP matvec snap-to-schema experiment: recompressing
    # each H·v output onto v's key-set here drifted the energy ~1.7e-3, same as the
    # removed SB_SNAP_PHI. Lanczos uses the cross-schema merge; dedup is recovered
    # at factorize. _snap_to_schema stays as a primitive — used by snap_dense_to_aliased.)
    if SparseBackends.ALIASED_TRACE[] && _ALIASED_HV_TRACE_COUNT[] < 30
        _ALIASED_HV_TRACE_COUNT[] += 1
        v_st = ITensors.has_external_storage(v) ? typeof(v.tensor.data) : "dense"
        hv_st = ITensors.has_external_storage(Hv) ? typeof(Hv.tensor.data) : "dense"
        println("[SB_ALIASED_TRACE product matvec #$(_ALIASED_HV_TRACE_COUNT[])]  v storage=$v_st  Hv storage=$hv_st")
    end
    return Hv
end
const _ALIASED_HV_TRACE_COUNT = Ref(0)


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
function product(P::AbstractProjMPO, v::ITensor; debug::Bool=false, roofline::Bool=false, run_label::String="?")::ITensor
    Pv = contract(P, v; roofline=roofline, run_label=run_label)
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
    # Was SB_PHI_SCHEMA_DUMP(_MAX); pass debug=true at a call site to enable
    # (budget hardcoded to 1 call below, was SB_PHI_SCHEMA_DUMP_MAX).
    if debug && _PHI_SCHEMA_DUMP_COUNT[] < 1
        _PHI_SCHEMA_DUMP_COUNT[] += 1
        println("\n========== [SB_PHI_SCHEMA_DUMP #", _PHI_SCHEMA_DUMP_COUNT[],
                "] matvec input φ  vs  output Hv=Pφ ==========")
        _dump_aliased_schema("φ  IN ", v)
        _dump_aliased_schema("Hv OUT", Pv)
    end
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
function _reorder_env_for_aliased(T::ITensor, both_aliased::Bool=false, aliased_h::Bool=false)
    # A/B switch: SB_NO_ENV_REORDER=1 disables this standalone env permute so we
    # can measure whether it actually reduces the matvec's permute_B@step-2 (the
    # fire-counter says it does not) vs just adding a permute in position!.
    get(ENV, "SB_NO_ENV_REORDER", "0") == "1" && return T
    # Only canonicalize the env for an ALIASED-H run (the matvec will call the
    # aliased×dense kernel, whose per-call permute_B this reorder eliminates).
    # For dense-H / BS runs the env feeds a plain ITensors `*`, so reordering is
    # pointless and would perturb those (byte-identical) baselines. This is the
    # intrinsic replacement for the old SB_FUSE_LINKS gate — NOTE it must NOT be
    # gated on "has FusedSparse axes": at sizes with no multi-strand links to
    # fuse, the env carries plain `Link` tags (nfused=0) yet the [ket, dense-H,
    # bra] regrouping is still exactly what makes step-2's B canonical.
    aliased_h || return T
    # Both-aliased (aliased ψ × aliased H) ONLY: the env is aliased here (built by
    # _env_mul) and already carries the correct prefix/dense classification; this
    # reorder is only a perf canonicalization for the DENSE-env matvec kernel, and
    # ITensors.permute has no method for aliased external storage. So skip it for
    # the aliased env. Gated on `both_aliased` so the all-dense, aliased-ψ×dense-H,
    # and dense-ψ×aliased-H paths are byte-identical (they never set both_aliased).
    both_aliased && _is_aliased_itensor(T) && return T
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

function _makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false, roofline::Bool=false, run_label::String="?")::Union{ITensor, Nothing}
    # Was SB_ENV_DEBUG/_MAX — removed. Hardcoded off; the prints below (env
    # structure / index order for L and its substeps) are unreachable dead
    # code kept for reference. Flip _env_dbg to true to re-enable.
    _env_dbg = false
    _env_dbg_max = 6
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
            # Keep the env aliased when BOTH ψ and H are aliased (both-aliased
            # run), so its multiplicity axes stay dense in the matvec output.
            # Hardened 2026-06 — always on (was SB_ALIASED_AA_ENV, default-on knob).
            _keep_env = _is_aliased_itensor(H_site) && _is_aliased_itensor(psi[ll + 1])
            L = _env_mul(L, H_site, _keep_env)
            if _env_dbg && _ENV_DBG_COUNT[] <= _env_dbg_max
                println("  L (after × H_site) inds = ", inds(L))
            end
            try
                L = _env_mul(L, dag(prime(psi[ll + 1])), _keep_env)
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
            L = _env_mul(L, psi[ll + 1], _keep_env)
            # Reorder so "FusedSparse" axes go last → matvec kernel sees
            # canonical layout and skips its per-call permute_B. (Skipped for the
            # both-aliased env inside _reorder_env_for_aliased.)
            L = _reorder_env_for_aliased(L, _keep_env, _is_aliased_itensor(H_site))
            if _env_dbg && _ENV_DBG_COUNT[] <= _env_dbg_max
                println("  L (after × dag(psi') × psi, post-reorder) FINAL inds = ", inds(L))
                println("    P.LR[", ll+1, "] stored.  external? ",
                        ITensors.has_external_storage(L))
            end
        end
        P.LR[ll + 1] = L
        roofline && _record_env_footprint(ll + 1, L, run_label)
        ll += 1
    end
    # Needed when moving lproj backward.
    P.lpos = k
    return L
end

function makeL!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false, roofline::Bool=false, run_label::String="?")
    # println("\tPositioning ProjMPO ", debug)
    _makeL!(P, psi, k; debug=debug, roofline=roofline, run_label=run_label)
    return P
end

function _makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false, roofline::Bool=false, run_label::String="?")::Union{ITensor, Nothing}
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
            # Was SB_ENV_DEBUG/_MAX — removed. Hardcoded off; the prints below
            # are unreachable dead code kept for reference.
            _env_dbg2 = false
            _env_dbg2_max = 6
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
            # Hardened 2026-06 — always on (was SB_ALIASED_AA_ENV, default-on knob).
            _keep_env = _is_aliased_itensor(H_site) && _is_aliased_itensor(psi[rl - 1])
            R = _env_mul(R, H_site, _keep_env)
            if _env_dbg2 && _ENV_DBG_COUNT[] <= _env_dbg2_max
                println("  R (after × H_site) inds = ", inds(R))
            end
            R = _env_mul(dag(prime(psi[rl - 1])), R, _keep_env)
            R = _env_mul(psi[rl - 1], R, _keep_env)
            R = _reorder_env_for_aliased(R, _keep_env, _is_aliased_itensor(H_site))
            if _env_dbg2 && _ENV_DBG_COUNT[] <= _env_dbg2_max
                println("  R (after × dag(psi') × psi, post-reorder) FINAL inds = ", inds(R))
            end
        end
        P.LR[rl - 1] = R
        roofline && _record_env_footprint(rl - 1, R, run_label)
        # println(" check rl - 1 information ", rl - 1)
        # println("TENSOR IS ", R)
        rl -= 1
    end
    P.rpos = k
    return R
end

function makeR!(P::AbstractProjMPO, psi::MPS, k::Int; debug=false, roofline::Bool=false, run_label::String="?")
    _makeR!(P, psi, k; debug=debug, roofline=roofline, run_label=run_label)
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
function position!(P::AbstractProjMPO, psi::MPS, pos::Int; debug=false, roofline::Bool=false, run_label::String="?")
    # println("Positioning ProjMPO ", debug)
    makeL!(P, psi, pos - 1; debug=debug, roofline=roofline, run_label=run_label)
    makeR!(P, psi, pos + nsite(P); debug=debug, roofline=roofline, run_label=run_label)
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
