# dmrg_debug.jl — profiling / trace helpers factored out of the dmrg sweep loop.
# Every helper is a cheap no-op unless its debug switch is on (ENV SB_KEYTRACE,
# SparseBackends.ALIASED_TRACE), so the sweep loop can call them unconditionally.
# The Refs/timer they touch (_BOND_POSITION, _SWEEP_NUM, PROJMPO_TIMER,
# _EIGSOLVE_PHI_TRACE_COUNT) are defined alongside their readers in abstractprojmpo.jl.

# Record the (bond, sweep) the current matvec/position! runs at, for the
# permute-profile probes in abstractprojmpo.jl. (Was the SB_BOND env var + Refs.)
function _dmrg_set_bond_context!(b::Int, sw::Int)
  _BOND_POSITION[] = b
  _SWEEP_NUM[] = sw
  ENV["SB_BOND"] = string(b)
  return nothing
end

const _EIGSOLVE_PHI_TRACE_COUNT = Ref(0)   # budget for the eigsolve-φ storage-class trace

# Trace φ's storage class the first few times eigsolve returns (SB_ALIASED_TRACE only).
function _dmrg_trace_eigsolve_phi(phi, b::Int)
  (SparseBackends.ALIASED_TRACE[] && _EIGSOLVE_PHI_TRACE_COUNT[] < 5) || return nothing
  _EIGSOLVE_PHI_TRACE_COUNT[] += 1
  phi_st = ITensors.has_external_storage(phi) ? typeof(phi.tensor.data) : "dense"
  println("[SB_ALIASED_TRACE eigsolve returned #$(_EIGSOLVE_PHI_TRACE_COUNT[])]  phi storage=$phi_st  b=$b")
  return nothing
end

# The three sweep-loop probe points. Each bundles the schema/index-order dumps + traces
# for a stage; all are cheap no-ops unless SB_SCHEMA / SB_KEYTRACE / SB_ALIASED_TRACE is on.
_dmrg_probe_phi_in(phi, b::Int, sw::Int, ha::Int) = begin
  SparseBackends.schema_dbg("eigsolve-OPERAND phi b=$b", phi)   # sparse/dense index order + keys
  _dmrg_keytrace("1.phi_original", phi, b, sw, ha)
  nothing
end
_dmrg_probe_phi_out(phi, b::Int) = begin
  SparseBackends.schema_dbg("eigsolve-RESULT phi b=$b", phi)
  _dmrg_trace_eigsolve_phi(phi, b)
  nothing
end
_dmrg_probe_bond_out(psi, b::Int) = begin
  SparseBackends.schema_dbg("replacebond-OUT psi[$b]", psi[b])
  SparseBackends.schema_dbg("replacebond-OUT psi[$(b+1)]", psi[b+1])
  nothing
end

# KEYTRACE (ENV SB_KEYTRACE=1): dump prefix-key set + index order + dedup grouping for
# an aliased tensor at one traced bond (SB_KEYTRACE_BOND, sw==1, ha==1), following the
# M^{±1/2} key flow. No-op unless enabled and the bond/sweep/half match.
function _dmrg_keytrace(lbl, T, b::Int, sw::Int, ha::Int)
  _kt_bonds = Set(parse.(Int, split(get(ENV, "SB_KEYTRACE_BOND", "2"), ",")))
  (get(ENV, "SB_KEYTRACE", "0") == "1" && (b in _kt_bonds) && sw == 1 && ha == 1) ||
    return nothing
  if T isa ITensors.ITensor && ITensors.has_external_storage(T) &&
     ITensors.get_external_storage(T) isa SparseBackends.WrappedAliasedBlockSparse
    w = ITensors.get_external_storage(T); a = w.aliased
    P = SparseBackends._abs_head_len(w); Nn = length(w.inds)
    _ord = [(ITensors.dim(w.inds[i]), string(ITensors.tags(w.inds[i])), ITensors.plev(w.inds[i])) for i in 1:Nn]
    _dedup = round(length(a.keys) / max(a.n_templates, 1); digits=3)
    _keymap = [(a.keys[i], Int(a.alias_ids[i])) for i in eachindex(a.keys)]
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
