module RestrictedSparse

export restrictedsparse, from_dense, to_dense


"""
restrictedsparse{T,N}

- dims: full tensor dims
- diag_pairs: list of (a,b) among prefix axes 1:(N-2) requiring I[a]==I[b]
- reps / rep_of: union-find representation of diagonal constraints
- sel: selected column for each (prefix_state,row), 0 means no nonzero
- val: value for that selection

NOTE: Storage is prefix-major:
  slot s corresponds to (plin,row) with row varying fastest.
"""
struct restrictedsparse{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
  diag_pairs::Vector{Tuple{Int,Int}}
  reps::Vector{Int}      # representatives (subset of 1:(N-2))
  rep_of::Vector{Int}    # length N-2, maps prefix axis -> representative
  sel::Vector{Int}       # length = n_prefix_states * dims[N-1]
  val::Vector{T}         # same length
end

Base.eltype(::Type{restrictedsparse{T}}) where {T} = T
Base.eltype(x::restrictedsparse) = eltype(typeof(x))

# --------------------------
# Union-find over prefix axes 1:(N-2)
# --------------------------
function _build_rep_map(dims::NTuple{N,Int}, diag_pairs::Vector{Tuple{Int,Int}}) where {N}
  P = N - 2
  parent = collect(1:P)

  function find(x)
    while parent[x] != x
      parent[x] = parent[parent[x]]
      x = parent[x]
    end
    return x
  end

  function unite(a,b)
    ra, rb = find(a), find(b)
    ra == rb && return
    parent[rb] = ra
  end

  for (a,b) in diag_pairs
    @assert 1 <= a < b <= P
    @assert dims[a] == dims[b] "Diagonal constraint requires dims[$a]==dims[$b]"
    unite(a,b)
  end

  rep_of = [find(i) for i in 1:P]
  reps = sort!(unique(rep_of))
  return reps, rep_of
end

@inline _n_prefix_states(dims::NTuple{N,Int}, reps::Vector{Int}) where {N} =
  isempty(reps) ? 1 : prod(dims[r] for r in reps)

# decode plin -> rep_vals (column-major over reps)
@inline function _rep_decode!(rep_vals::Vector{Int}, plin::Int, rep_dims::Vector{Int})
  x = plin - 1
  @inbounds for m in 1:length(rep_dims)
    d = rep_dims[m]
    rep_vals[m] = (x % d) + 1
    x ÷= d
  end
  return rep_vals
end

# Precompute: rep_pos[r] = position m such that reps[m] == r (r in 1:P), else 0
function _rep_pos(reps::Vector{Int}, P::Int)
  rep_pos = zeros(Int, P)
  @inbounds for (m, r) in enumerate(reps)
    rep_pos[r] = m
  end
  return rep_pos
end

# build full prefix index vector from rep_vals without Dict
@inline function _prefix_from_rep!(prefix::Vector{Int},
                                  rep_of::Vector{Int},
                                  rep_pos::Vector{Int},
                                  rep_vals::Vector{Int})
  @inbounds for a in 1:length(prefix)
    r = rep_of[a]          # representative axis index in 1:P
    m = rep_pos[r]         # position in rep_vals
    prefix[a] = rep_vals[m]
  end
  return prefix
end

# --------------------------
# Constructor
# --------------------------
function restrictedsparse{T}(dims::NTuple{N,Int};
                             diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[]) where {T,N}
  @assert N >= 2 "Need at least 2 indices (row,col) in the last two axes"
  reps, rep_of = _build_rep_map(dims, diag_pairs)
  nps  = _n_prefix_states(dims, reps)
  rows = dims[N-1]
  sel = zeros(Int, nps * rows)
  val = zeros(T,  nps * rows)
  return restrictedsparse{T,N}(dims, diag_pairs, reps, rep_of, sel, val)
end

# --------------------------
# Dense conversion: storage -> Array
# --------------------------
function to_dense(rs::restrictedsparse{T,N}) where {T,N}
  dims = rs.dims
  out  = zeros(T, dims)

  P    = N - 2
  rows = dims[N-1]
  nps  = _n_prefix_states(dims, rs.reps)

  # Fast path: N == 2
  if P == 0
    for row in 1:rows
      s = row
      col = rs.sel[s]
      col == 0 && continue
      out[row, col] = rs.val[s]
    end
    return out
  end

  rep_dims = [dims[r] for r in rs.reps]
  rep_pos  = _rep_pos(rs.reps, P)

  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    base = (plin - 1) * rows

    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, rs.rep_of, rep_pos, rep_vals)
    else
      # no reps => single prefix-state; prefix content is irrelevant if P>0 can't happen unless dims weird,
      # but keep safe: set prefix to 1s
      fill!(prefix, 1)
    end

    for row in 1:rows
      s = base + row
      col = rs.sel[s]
      col == 0 && continue

      I = CartesianIndex(prefix..., row, col)
      out[I] = rs.val[s]
      # I = ntuple(i -> begin
      #   if i <= P
      #     prefix[i]
      #   elseif i == N-1
      #     row
      #   else
      #     col
      #   end
      # end, Val(N))
      # out[I...] = rs.val[s]
    end
  end

  return out
end

# --------------------------
# Dense conversion: Array -> storage (checks structure)
# --------------------------
function from_dense(A::AbstractArray{T,N};
                    diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
                    atol::Real=1e-12,
                    rtol::Real=0) where {T,N}

  dims = ntuple(i -> size(A, i), Val(N))
  rs   = restrictedsparse{T}(dims; diag_pairs=diag_pairs)

  P    = N - 2
  rows = dims[N-1]
  cols = dims[N]
  nps  = _n_prefix_states(dims, rs.reps)

  # Fast path: N == 2
  if P == 0
    @inbounds for row in 1:rows
      found_col = 0
      found_val = zero(T)
      for col in 1:cols
        v = A[row, col]
        is_nz = abs(v) > max(atol, rtol * abs(found_val))
        if is_nz
          if found_col != 0
            error("Structure violated: more than 1 nonzero in last axis for row=$row. Found at cols $found_col and $col.")
          end
          found_col = col
          found_val = v
        end
      end
      s = row
      rs.sel[s] = found_col
      rs.val[s] = found_col == 0 ? zero(T) : found_val
    end
    return rs
  end

  rep_dims = [dims[r] for r in rs.reps]
  rep_pos  = _rep_pos(rs.reps, P)

  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    base = (plin - 1) * rows

    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, rs.rep_of, rep_pos, rep_vals)
    else
      fill!(prefix, 1)
    end

    for row in 1:rows
      found_col = 0
      found_val = zero(T)

      for col in 1:cols
        I = ntuple(i -> begin
          if i <= P
            prefix[i]
          elseif i == N-1
            row
          else
            col
          end
        end, Val(N))
        v = A[I...]

        is_nz = abs(v) > max(atol, rtol * abs(found_val))
        if is_nz
          if found_col != 0
            error("Structure violated: more than 1 nonzero in last axis for (plin=$plin,row=$row). Found at cols $found_col and $col.")
          end
          found_col = col
          found_val = v
        end
      end

      s = base + row
      rs.sel[s] = found_col
      rs.val[s] = found_col == 0 ? zero(T) : found_val
    end
  end

  return rs
end

end # module


# # restrictedsparse/restricted_perm_last2_storage.jl
# #
# # Minimal structured storage:
# # - arbitrary prefix indices 1:(N-2), with optional diagonal constraints
# # - last two indices (N-1, N) are (row, col) and for each (prefix,row) there is <= 1 nonzero col
# #
# # This file is written to be included inside the NDTensors module,
# # but it also works standalone if you define `abstract type TensorStorage end`.

# module RestrictedSparse

# export restrictedsparse, from_dense, to_dense

# # If you include this inside NDTensors, comment out the next line
# abstract type TensorStorage end

# """
# restrictedsparse{T,N}

# - dims: full tensor dims
# - diag_pairs: list of (a,b) among prefix axes 1:(N-2) requiring I[a]==I[b]
# - reps / rep_of: union-find representation of diagonal constraints
# - sel: selected column for each (prefix_state,row), 0 means no nonzero
# - val: value for that selection
# """
# struct restrictedsparse{T,N} <: TensorStorage
#   dims::NTuple{N,Int}
#   diag_pairs::Vector{Tuple{Int,Int}}
#   reps::Vector{Int}      # representatives (subset of 1:(N-2))
#   rep_of::Vector{Int}    # length N-2, maps prefix axis -> representative
#   sel::Vector{Int}       # length = n_prefix_states * dims[N-1]
#   val::Vector{T}         # same length
# end

# Base.eltype(::Type{restrictedsparse{T}}) where {T} = T
# Base.eltype(x::restrictedsparse) = eltype(typeof(x))

# # --------------------------
# # Union-find over prefix axes 1:(N-2)
# # --------------------------
# function _build_rep_map(dims::NTuple{N,Int}, diag_pairs::Vector{Tuple{Int,Int}}) where {N}
#   P = N - 2
#   parent = collect(1:P)

#   function find(x)
#     while parent[x] != x
#       parent[x] = parent[parent[x]]
#       x = parent[x]
#     end
#     return x
#   end

#   function unite(a,b)
#     ra, rb = find(a), find(b)
#     ra == rb && return
#     parent[rb] = ra
#   end

#   for (a,b) in diag_pairs
#     @assert 1 <= a < b <= P
#     @assert dims[a] == dims[b] "Diagonal constraint requires dims[$a]==dims[$b]"
#     unite(a,b)
#   end

#   rep_of = [find(i) for i in 1:P]
#   reps = sort!(unique(rep_of))
#   return reps, rep_of
# end

# @inline _n_prefix_states(dims::NTuple{N,Int}, reps::Vector{Int}) where {N} =
#   isempty(reps) ? 1 : prod(dims[r] for r in reps)

# # column-major linearization for rep coordinates
# @inline function _rep_linear(rep_vals::AbstractVector{Int}, rep_dims::AbstractVector{Int})
#   lin = 1
#   stride = 1
#   @inbounds for m in 1:length(rep_vals)
#     v = rep_vals[m]; d = rep_dims[m]
#     @assert 1 <= v <= d
#     lin += (v - 1) * stride
#     stride *= d
#   end
#   return lin
# end

# # decode plin -> rep_vals (column-major)
# @inline function _rep_decode(plin::Int, rep_dims::AbstractVector{Int})
#   x = plin - 1
#   rep_vals = Vector{Int}(undef, length(rep_dims))
#   @inbounds for m in 1:length(rep_dims)
#     d = rep_dims[m]
#     rep_vals[m] = (x % d) + 1
#     x ÷= d
#   end
#   return rep_vals
# end

# # build full prefix index vector from rep_vals
# @inline function _prefix_from_rep!(prefix::Vector{Int},
#                                   reps::Vector{Int}, rep_of::Vector{Int},
#                                   rep_vals::Vector{Int})
#   # map rep -> value
#   rep_val = Dict{Int,Int}()
#   @inbounds for (m, r) in enumerate(reps)
#     rep_val[r] = rep_vals[m]
#   end
#   @inbounds for a in 1:length(prefix)
#     prefix[a] = rep_val[rep_of[a]]
#   end
#   return prefix
# end

# # --------------------------
# # Constructor
# # --------------------------
# function restrictedsparse{T}(dims::NTuple{N,Int};
#                                diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[]) where {T,N}
#   @assert N >= 2 "Need at least 2 indices (row,col) in the last two axes"
#   reps, rep_of = _build_rep_map(dims, diag_pairs)
#   nps = _n_prefix_states(dims, reps)
#   rows = dims[N-1]
#   sel = zeros(Int, nps * rows)
#   val = zeros(T, nps * rows)
#   return restrictedsparse{T,N}(dims, diag_pairs, reps, rep_of, sel, val)
# end

# # slot index for (plin,row): prefix varies fastest
# @inline function _slot(nps::Int, plin::Int, row::Int)
#   return plin + (row - 1) * nps
# end

# # --------------------------
# # Dense conversion: storage -> Array
# # --------------------------
# function to_dense(rs::restrictedsparse{T,N}) where {T,N}
#   dims = rs.dims
#   out = zeros(T, dims)

#   P = N - 2
#   nps = _n_prefix_states(dims, rs.reps)
#   rep_dims = [dims[r] for r in rs.reps]
#   rows = dims[N-1]

#   prefix = Vector{Int}(undef, max(P, 0))

#   for row in 1:rows
#     for plin in 1:nps
#       s = _slot(nps, plin, row)
#       col = rs.sel[s]
#       col == 0 && continue

#       rep_vals = isempty(rs.reps) ? Int[] : _rep_decode(plin, rep_dims)
#       if P > 0
#         _prefix_from_rep!(prefix, rs.reps, rs.rep_of, rep_vals)
#       end

#       # write out[prefix..., row, col] = val
#       if P == 0
#         out[row, col] = rs.val[s]
#       else
#         # build index tuple dynamically
#         I = ntuple(i -> begin
#           if i <= P
#             prefix[i]
#           elseif i == N-1
#             row
#           else
#             col
#           end
#         end, Val(N))
#         out[I...] = rs.val[s]
#       end
#     end
#   end

#   return out
# end

# # --------------------------
# # Dense conversion: Array -> storage (checks structure)
# # --------------------------
# function from_dense(A::AbstractArray{T,N};
#                     diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
#                     atol::Real=1e-12,
#                     rtol::Real=0) where {T,N}

#   dims = ntuple(i -> size(A, i), Val(N))
#   rs = restrictedsparse{T}(dims; diag_pairs=diag_pairs)

#   println("from_dense called with dims = ", dims, " ", N)
#   P = N - 2
#   nps = _n_prefix_states(dims, rs.reps)
#   rep_dims = [dims[r] for r in rs.reps]
#   rows = dims[N-1]
#   cols = dims[N]

#   prefix = Vector{Int}(undef, max(P, 0))

#   for row in 1:rows
#     for plin in 1:nps
#       rep_vals = isempty(rs.reps) ? Int[] : _rep_decode(plin, rep_dims)
#       if P > 0
#         _prefix_from_rep!(prefix, rs.reps, rs.rep_of, rep_vals)
#       end

#       println("===============  ", row, " ", plin, " ", prefix, " ")

#       found_col = 0
#       found_val = zero(T)

#       for col in 1:cols
#         v = if P == 0
#           A[row, col]
#         else
#           I = ntuple(i -> begin
#             if i <= P
#               prefix[i]
#             elseif i == N-1
#               row
#             else
#               col
#             end
#           end, Val(N))
#           A[I...]
#         end

#         is_nz = abs(v) > max(atol, rtol * abs(found_val))
#         print(is_nz, " ", found_col, ", ", abs(v), " ", atol, " ", max(atol, rtol * abs(found_val)), " | ")
#         if is_nz
#           if found_col != 0
#             error("Structure violated: more than 1 nonzero in last axis for (prefix_state=$plin,row=$row). " *
#                   "Found at cols $found_col and $col.")
#           end
#           found_col = col
#           found_val = v
#         end
#         println("")
#       end

#       s = _slot(nps, plin, row)
#       rs.sel[s] = found_col
#       rs.val[s] = found_col == 0 ? zero(T) : found_val
#     end
#   end

#   # Optional: validate that entries violating diag constraints are ~0.
#   # This is expensive; for small tests it's fine.
#   if !isempty(diag_pairs)
#     # brute-force check over all prefix indices: if violates, must be zero
#     # For correctness tests only.
#     if P > 0
#       prefix_dims = ntuple(i -> dims[i], Val(P))
#       for Iprefix in CartesianIndices(prefix_dims)
#         ok = true
#         for (a,b) in diag_pairs
#           if Iprefix[a] != Iprefix[b]
#             ok = false
#             break
#           end
#         end
#         if !ok
#           # any row/col should be ~0
#           for row in 1:rows, col in 1:cols
#             I = ntuple(i -> begin
#               if i <= P
#                 Iprefix[i]
#               elseif i == N-1
#                 row
#               else
#                 col
#               end
#             end, Val(N))
#             v = A[I...]
#             if abs(v) > atol
#               error("Entry violates diag constraints but is nonzero at index $I with value $v")
#             end
#           end
#         end
#       end
#     end
#   end

#   return rs
# end

# end # module


# # """
# # restrictedsparse storage:

# # - Last two axes (N-1, N) are (row, col).
# # - For each fixed (prefix_state, row) there is at most one nonzero column.
# # - Values are weighted.

# # Optional diagonal constraints among prefix indices 1:(N-2):
# #   diag_pairs = [(a,b), ...] meaning index[a] == index[b].

# # Storage:
# #   dims::NTuple{N,Int}
# #   diag_pairs::Vector{Tuple{Int,Int}}
# #   # For each (prefix_state, row) we store selected col and value
# #   sel::Vector{Int}        # 0 means empty, else 1..dims[N]
# #   val::Vector{T}

# # We treat the free degrees of freedom as a set of representatives
# # for the prefix indices after applying diag constraints.
# # """
# # struct restrictedsparse{T,N} <: TensorStorage
# #   dims::NTuple{N,Int}
# #   diag_pairs::Vector{Tuple{Int,Int}}
# #   reps::Vector{Int}         # representative indices (subset of 1:(N-2))
# #   rep_of::Vector{Int}       # length N-2, maps each prefix axis -> representative axis
# #   sel::Vector{Int}          # length = prefix_states * dims[N-1]
# #   val::Vector{T}
# # end

# # Base.eltype(::Type{restrictedsparse{T}}) where {T} = T
# # Base.eltype(x::restrictedsparse) = eltype(typeof(x))


# # function _build_rep_map(dims::NTuple{N,Int}, diag_pairs::Vector{Tuple{Int,Int}}) where {N}
# #   P = N - 2
# #   @assert P >= 0
# #   parent = collect(1:P)

# #   find(x) = begin
# #     while parent[x] != x
# #       parent[x] = parent[parent[x]]
# #       x = parent[x]
# #     end
# #     x
# #   end
# #   union(a,b) = begin
# #     ra, rb = find(a), find(b)
# #     ra == rb && return
# #     parent[rb] = ra
# #   end

# #   for (a,b) in diag_pairs
# #     @assert 1 <= a < b <= P
# #     @assert dims[a] == dims[b] "Diagonal constraint requires dims[$a]==dims[$b]"
# #     union(a,b)
# #   end

# #   rep_of = [find(i) for i in 1:P]
# #   reps = sort!(unique(rep_of))
# #   return reps, rep_of
# # end

# # function _n_prefix_states(dims::NTuple{N,Int}, reps::Vector{Int}) where {N}
# #   prod(dims[r] for r in reps)
# # end

# # # Compute linear index (1-based, column-major) of representative tuple
# # @inline function _rep_linear(rep_vals::NTuple{M,Int}, rep_dims::NTuple{M,Int}) where {M}
# #   lin = 1
# #   stride = 1
# #   @inbounds for m in 1:M
# #     v = rep_vals[m]; d = rep_dims[m]
# #     @assert 1 <= v <= d
# #     lin += (v - 1) * stride
# #     stride *= d
# #   end
# #   return lin
# # end

# # # Given full prefix indices I[1:P], check diag constraints and compute rep linear index.
# # function _prefix_linear(Iprefix::NTuple{P,Int}, dims_prefix::NTuple{P,Int},
# #                         reps::Vector{Int}, rep_of::Vector{Int}) where {P}
# #   # Validate diag constraints by checking all indices that share a rep match
# #   # We'll store rep values as we see them.
# #   rep_to_val = Dict{Int,Int}()
# #   @inbounds for a in 1:P
# #     v = Iprefix[a]
# #     @assert 1 <= v <= dims_prefix[a]
# #     r = rep_of[a]
# #     if haskey(rep_to_val, r)
# #       rep_to_val[r] == v || return 0  # indicates invalid (violates diag)
# #     else
# #       rep_to_val[r] = v
# #     end
# #   end

# #   M = length(reps)
# #   rep_vals = ntuple(m -> rep_to_val[reps[m]], Val(M))
# #   rep_dims = ntuple(m -> dims_prefix[reps[m]], Val(M))
# #   return _rep_linear(rep_vals, rep_dims)
# # end

# # # --------------------------
# # # Constructor
# # # --------------------------
# # function restrictedsparse{T}(dims::NTuple{N,Int};
# #                                diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[]) where {T,N}
# #   reps, rep_of = _build_rep_map(dims, diag_pairs)
# #   prefix_states = _n_prefix_states(dims, reps)
# #   rows = (N >= 1) ? dims[N-1] : 1  # only meaningful for N>=2; we’ll assert below
# #   @assert N >= 2 "Need at least 2 indices for last-2 (row,col) structure"
# #   rows = dims[N-1]
# #   len = prefix_states * rows
# #   sel = zeros(Int, len)
# #   val = zeros(T, len)
# #   return restrictedsparse{T,N}(dims, diag_pairs, reps, rep_of, sel, val)
# # end

# # # --------------------------
# # # Access helpers
# # # --------------------------
# # @inline function _slot(rs::restrictedsparse{T,N}, I::NTuple{N,Int}) where {T,N}
# #   dims = rs.dims
# #   P = N - 2
# #   # prefix tuple
# #   Iprefix = ntuple(i -> I[i], Val(P))
# #   dims_prefix = ntuple(i -> dims[i], Val(P))
# #   plin = _prefix_linear(Iprefix, dims_prefix, rs.reps, rs.rep_of)
# #   plin == 0 && return 0  # violates diag constraints

# #   r = I[N-1]
# #   @assert 1 <= r <= dims[N-1]
# #   # slot for (prefix_state, row) in column-major: prefix varies fastest
# #   return plin + (r - 1) * (_n_prefix_states(dims, rs.reps))
# # end

# # # --------------------------
# # # getindex / setindex!
# # # --------------------------
# # function Base.getindex(rs::restrictedsparse{T,N}, I::Vararg{Int,N}) where {T,N}
# #   dims = rs.dims
# #   It = ntuple(i -> I[i], Val(N))
# #   # Basic bounds check for last two
# #   @assert 1 <= It[N-1] <= dims[N-1]
# #   @assert 1 <= It[N]   <= dims[N]

# #   s = _slot(rs, It)
# #   s == 0 && return zero(T)

# #   c = rs.sel[s]
# #   return (c == It[N]) ? rs.val[s] : zero(T)
# # end

# # function Base.setindex!(rs::restrictedsparse{T,N}, x, I::Vararg{Int,N}) where {T,N}
# #   dims = rs.dims
# #   It = ntuple(i -> I[i], Val(N))

# #   s = _slot(rs, It)
# #   s == 0 && throw(ArgumentError("Indices violate diagonal constraints"))

# #   col = It[N]
# #   @assert 1 <= col <= dims[N]

# #   if iszero(x)
# #     if rs.sel[s] == col
# #       rs.sel[s] = 0
# #       rs.val[s] = zero(T)
# #     end
# #   else
# #     rs.sel[s] = col
# #     rs.val[s] = convert(T, x)
# #   end
# #   return rs
# # end

# # # --------------------------
# # # Dense materialization
# # # --------------------------
# # function array(rs::restrictedsparse{T,N}) where {T,N}
# #   dims = rs.dims
# #   out = zeros(T, dims)

# #   prefix_states = _n_prefix_states(dims, rs.reps)
# #   rows = dims[N-1]
# #   cols = dims[N]

# #   # Iterate slots; decode prefix-state back into rep indices is possible but slower.
# #   # For now we materialize by iterating all full indices would be too slow.
# #   # We'll instead decode rep coords, then expand to full prefix coords using rep_of,
# #   # and set the diagonal-related indices accordingly.
# #   #
# #   # NOTE: This is correctness-first. You can optimize later.
# #   rep_dims = map(r -> dims[r], rs.reps)
# #   M = length(rs.reps)

# #   for row in 1:rows
# #     for plin in 1:prefix_states
# #       s = plin + (row - 1) * prefix_states
# #       col = rs.sel[s]
# #       col == 0 && continue

# #       # decode plin -> rep coords (column-major)
# #       x = plin - 1
# #       rep_vals = Vector{Int}(undef, M)
# #       for m in 1:M
# #         dm = rep_dims[m]
# #         rep_vals[m] = (x % dm) + 1
# #         x ÷= dm
# #       end

# #       # build full prefix indices 1:(N-2)
# #       prefix_full = Vector{Int}(undef, N-2)
# #       # assign each representative value
# #       rep_val_map = Dict{Int,Int}()
# #       for (m, r) in enumerate(rs.reps)
# #         rep_val_map[r] = rep_vals[m]
# #       end
# #       for a in 1:(N-2)
# #         prefix_full[a] = rep_val_map[rs.rep_of[a]]
# #       end

# #       # assemble full index tuple and set
# #       fullI = ntuple(i -> begin
# #         if i <= N-2
# #           prefix_full[i]
# #         elseif i == N-1
# #           row
# #         else
# #           col
# #         end
# #       end, Val(N))

# #       out[fullI...] = rs.val[s]
# #     end
# #   end

# #   return out
# # end
