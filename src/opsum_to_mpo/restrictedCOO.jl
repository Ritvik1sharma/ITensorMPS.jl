# Minimal structured storage:
# - arbitrary prefix indices 1:(N-2), with optional diagonal constraints
# - last two indices (N-1, N) are COO (row, col)

module RestrictedCOO

using ITensors

export restrictedcoo, from_dense, to_dense


"""
restrictedcoo{T,N}

- dims: full tensor dims
- diag_pairs: list of (a,b) among prefix axes 1:(N-2) requiring I[a]==I[b]
- reps / rep_of: union-find representation of diagonal constraints
- sel: selected column for each (prefix_state,row), 0 means no nonzero
- val: value for that selection
"""
#  <: TensorStorage
struct restrictedcoo{T,N} <: AbstractArray{T,N}
  dims::NTuple{N,Int}
  diag_pairs::Vector{Tuple{Int,Int}}
  reps::Vector{Int}      # representatives (subset of 1:(N-2))
  rep_of::Vector{Int}    # length N-2, maps prefix axis -> representative
  coo_segment::Vector{Int}  # length = length of rep_of + 1
  sel::Vector{Tuple{Int, Int}}       # length = nnz of COO matrix
  val::Vector{T}         # same length as sel
end

Base.eltype(::Type{restrictedcoo{T}}) where {T} = T
Base.eltype(x::restrictedcoo) = eltype(typeof(x))

Base.size(A::restrictedcoo{T,N}) where {T,N} = A.dims
Base.axes(A::restrictedcoo{T,N}) where {T,N} = ntuple(d -> Base.OneTo(A.dims[d]), Val(N))
Base.IndexStyle(::Type{<:restrictedcoo}) = IndexCartesian()

@inline function _prefix_to_plin(dims::NTuple{N,Int},
                                 reps::Vector{Int},
                                 I::NTuple{P,Int}) where {N,P}
  # Column-major linearization over reps (matches your _rep_decode!)
  lin = 1
  stride = 1
  @inbounds for r in reps
    v = I[r]                 # reps are axis indices in 1:P
    d = dims[r]
    lin += (v - 1) * stride
    stride *= d
  end
  return lin
end

@inline function _check_diag(rep_of::Vector{Int}, I::NTuple{P,Int}) where {P}
  @inbounds for a in 1:P
    ra = rep_of[a]
    # representative is itself an axis index; enforce equality
    if I[a] != I[ra]
      return false
    end
  end
  return true
end

@inline function Base.getindex(A::RestrictedCOO{T,N}, I::Vararg{Int,N}) where {T,N}
  @boundscheck begin
    @assert length(I) == N
    for d in 1:N
      1 <= I[d] <= A.dims[d] || throw(BoundsError(A, I))
    end
  end

  P = N - 2
  row = I[N-1]
  col = I[N]

  # N == 2 case: only (row,col), single segment
  if P == 0
    lo = 1
    hi = length(A.sel)
    @inbounds for k in lo:hi
      if A.sel[k] == (row, col)
        return A.val[k]
      end
    end
    return zero(T)
  end

  prefix = ntuple(j -> I[j], Val(P))

  # Optional diag check (can be removed for speed)
  @boundscheck begin
    _check_diag(A.rep_of, prefix) || error("Index violates diagonal constraint: prefix=$prefix")
  end

  plin = _prefix_to_plin(A.dims, A.reps, prefix)

  # Segment offsets are stored 0-based in your constructor; convert to 1-based indices:
  start0 = A.coo_segment[plin]       # 0-based start
  stop0  = A.coo_segment[plin + 1]   # 0-based stop (end)
  lo = start0 + 1
  hi = stop0

  @inbounds for k in lo:hi
    if A.sel[k] == (row, col)
      return A.val[k]
    end
  end
  return zero(T)
end

function Base.setindex!(A::restrictedcoo{T,N}, v, I::Vararg{Int,N}) where {T,N}
  P = N - 2
  row = I[N-1]
  col = I[N]
  prefix = P == 0 ? () : ntuple(j -> I[j], Val(P))
  plin = P == 0 ? 1 : _prefix_to_plin(A.dims, A.reps, prefix)

  start0 = A.coo_segment[plin] + 1
  stop0  = A.coo_segment[plin + 1]
  # Find existing
  for k in start0:stop0
    if A.sel[k][1] == row && A.sel[k][2] == col
      A.val[k] = convert(T, v)
      return A
    elseif A.sel[k][1] > row || (A.sel[k][1] == row && A.sel[k][2] > col)
      insert!(A.sel, k, (row, col))
      insert!(A.val, k, convert(T, v))
      for j in (plin+1):length(A.coo_segment)
        A.coo_segment[j] += 1
      end
      return A
    end
  end
  insert!(A.sel, stop0+1, (row, col))
  insert!(A.val, stop0+1, convert(T, v))
  for j in (plin+1):length(A.coo_segment)
    A.coo_segment[j] += 1
  end
  return A
end

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
function restrictedcoo{T}(dims::NTuple{N,Int}, data::Vector{Vector{Tuple{Tuple{Int,Int}, T}}};
                             diag_pairs::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[]) where {T,N}
  @assert N >= 2 "Need at least 2 indices (row,col) in the last two axes"
  reps, rep_of = _build_rep_map(dims, diag_pairs)

  coords = Vector{Tuple{Int,Int}}()
  vals   = Vector{T}()
  segment = Vector{Int}()
  push!(segment, 0)
  for seg_data in data
    for data_entry in seg_data
        coords_entry, val_entry = data_entry
        push!(coords, coords_entry)
        push!(vals, val_entry)
    end
    push!(segment, length(coords))
  end
  return restrictedcoo{T,N}(dims, diag_pairs, reps, rep_of,
                            segment, coords, vals)
end


# --------------------------
# Dense conversion: storage -> Array
# --------------------------
function to_dense(rs::restrictedcoo{T,N}) where {T,N}
  dims = rs.dims
  out  = zeros(T, dims)

  P    = N - 2
  rows = dims[N-1]
  nps  = _n_prefix_states(dims, rs.reps)

  # Fast path: N == 2
  if P == 0
    for coords in rs.sel
      row, col = coords
      out[row, col] = rs.val[findfirst(==(coords), rs.sel)]
    end
    return out
  end

  rep_dims = [dims[r] for r in rs.reps]
  rep_pos  = _rep_pos(rs.reps, P)

  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, rs.rep_of, rep_pos, rep_vals)
    else
      # no reps => single prefix-state; prefix content is irrelevant if P>0 can't happen unless dims weird,
      # but keep safe: set prefix to 1s
      fill!(prefix, 1)
    end

    for (coords, val) in zip(rs.sel[rs.coo_segment[plin] .+ 1 : rs.coo_segment[plin+1]],
                             rs.val[rs.coo_segment[plin] .+ 1 : rs.coo_segment[plin+1]])
      row, col = coords
      I = CartesianIndex(prefix..., row, col)
      out[I] = val
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
  
  # Build diagonal constraint representation
  reps, rep_of = _build_rep_map(dims, diag_pairs)
  nps = _n_prefix_states(dims, reps)

  P    = N - 2
  rows = dims[N-1]
  cols = dims[N]

  data = Vector{Vector{Tuple{Tuple{Int,Int}, T}}}(undef, nps)
  # Fast path: N == 2
  if P == 0
    push!(data, Vector{Tuple{Tuple{Int,Int}, T}}())
    @inbounds for row in 1:rows
      found_val = zero(T)
      for col in 1:cols
        v = A[row, col]
        is_nz = abs(v) > max(atol, rtol * abs(found_val))        
        if is_nz
            push!(data[plin], ((row, col), v))
        end
      end
    end
    return restrictedcoo{T}(dims, data; diag_pairs=diag_pairs)
    # restrictedcoo{T,N}(dims, diag_pairs, reps, rep_of,
    #                           segment, coords, vals)
  end

  rep_dims = [dims[r] for r in reps]
  rep_pos  = _rep_pos(reps, P)
  prefix   = Vector{Int}(undef, P)
  rep_vals = Vector{Int}(undef, length(rep_dims))

  @inbounds for plin in 1:nps
    push!(data[plin], Vector{Tuple{Tuple{Int,Int}, T}}())
    if !isempty(rep_dims)
      _rep_decode!(rep_vals, plin, rep_dims)
      _prefix_from_rep!(prefix, rep_of, rep_pos, rep_vals)
    else
      fill!(prefix, 1)
    end

    for row in 1:rows
      for col in 1:cols
        I = CartesianIndex(prefix..., row, col)
        v = A[I...]
        is_nz = abs(v) > max(atol, rtol * abs(found_val))
        if is_nz
            push!(data[plin], ((row, col), v))
        end
      end
    end
  end
  return restrictedcoo{T}(dims, data; diag_pairs=diag_pairs)
end

function ITensors.ITensor(rs::restrictedcoo{T,N}, inds::Vararg{Index,N}) where {T,N}
  A = to_dense(rs)             # your N-d Array
  return ITensor(A, inds...)   # ITensors builds a dense ITensor from an Array
end

function restricted_from_itensor(T::ITensor;
                                 diag_pairs=Tuple{Int,Int}[],
                                 atol=1e-12,
                                 rtol=0.0)
  indsT = inds(T)  # returns the indices of T (some order)
  # Choose an order consistent with your convention:
  # (prefix..., row, col) i.e. last two are the permutation-matrix axes.
  # Suppose you already have them as i1..iN in the right order:
  A = Array(T, indsT...)  # convert ITensor -> Array in that explicit order
  return from_dense(A; diag_pairs=diag_pairs, atol=atol, rtol=rtol)
end

end # module