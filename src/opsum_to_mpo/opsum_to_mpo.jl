using NDTensors: using_auto_fermion
# include("restrictedsparse.jl")
# using .RestrictedSparse: restrictedsparse, from_dense, to_dense



# function build_MPO_with_support!(
#   H::MPO, tempMPO, sites, Vs::Vector{<:AbstractMatrix},
#   support::Union{Nothing,Vector{Int64}}
# )
#   N = length(H)
#   dual = support === nothing
#   supp = dual ? nothing : Set(support)

#   # Rank from SVD only
#   tdim_of = n -> size(Vs[n], 2)
#   # Bond dim = rank (min 1 so Index exists). No +1, no backbone.
#   bonddim(b) = if dual
#     left = clamp(b - 1, 1, N)
#     max(tdim_of(left), 1)
#   else
#     if b == 1 || b == N + 1
#       1
#     else
#       left, right = b - 1, b
#       ((left in supp) || (right in supp)) ? max(tdim_of(left), 1) : 1
#     end
#   end

#   llinks = [Index(bonddim(b), "Link,l=$(b-1)") for b in 1:(N + 1)]

#   # detect pure on-site Id at site n
#   _is_pure_on_site_Id = function(n, t)
#     ops = argument(t)
#     length(ops) == 1 && ITensors.name(only(ops)) == "Id" && only(site(only(ops))) == n
#   end

#   function infer_valtype(n)
#     println("Infering ValType for site $n ")
#     Tvs = promote_type(eltype(Vs[max(n-1,1)]), eltype(Vs[n]))
#     Tc  = Float64
#     for el in tempMPO[n]
#       Tc = promote_type(Tc, typeof(coefficient(el.val)))
#     end
#     return promote_type(Tvs, Tc)
#   end

#   function site_tensor(n, ll, rl)
#     println("current site: $n")

#     ValT = infer_valtype(n)
#     println("Inferred ValType: ", ValT)
  
#     # Outside support → strict 1×1 Id
#     if !dual && !(n in supp)
#       idM = zeros(ValT, dim(ll), dim(rl)); idM[1,1] = one(ValT)
#       return itensor(idM, ll, rl) * computeSiteProd(sites, Prod([Op("Id", n)]))
#     end

#     VL = (n > 1) ? Vs[n - 1] : zeros(ValT, 1, 1)
#     VR = Vs[n]

#     Tsum = ITensor()
#     for el in tempMPO[n]
#       println("PROCESSING TERM IN SITE $n: ", el)

#       t = el.val
#       # Skip pure on-site Id; if H contains Id, it creates its own corridor via keys/SVD.
#       if el.row == -1 && el.col == -1 && _is_pure_on_site_Id(n, t)
#         continue
#       end
#       abs(coefficient(t)) > eps() || continue

#       A_row, A_col = el.row, el.col
#       ct = convert(ValT, coefficient(t))
#       M = zeros(ValT, dim(ll), dim(rl))
#       if A_row == -1 && A_col == -1
#         continue
#       elseif A_row == -1
#         @inbounds for c in 1:size(VR, 2)
#           M[1, c] += ct * VR[A_col, c]
#         end
#       elseif A_col == -1
#         @inbounds for r in 1:size(VL, 2)
#           M[r, 1] += ct * conj(VL[A_row, r])
#         end
#       else
#         @inbounds for r in 1:size(VL, 2), c in 1:size(VR, 2)
#           M[r, c] += ct * conj(VL[A_row, r]) * VR[A_col, c]
#         end
#       end

#       println("MATRIX FOR SITE $n, TERM $t : ")
#       println(itensor(M, ll, rl))
#       println("------------")
#       println(computeSiteProd(sites, argument(t)))
#       println("=================================")
#       Tsum += itensor(M, ll, rl) * computeSiteProd(sites, argument(t))
#     end
#     println(Tsum)
#     error("STOPPING AFTER SITE $n")
#     return Tsum
#   end

#   for n in 1:N
#     H[n] = site_tensor(n, llinks[n], llinks[n + 1])
#   end

#   # simple left/right boundary
#   L = ITensor(llinks[1]); R = ITensor(llinks[N + 1])
#   L[1] = 1.0; R[1] = 1.0
#   H[1] *= L; H[N] *= R
#   return H
# end

# function from_itensor(T::ITensor;
#                       order_inds::Vector{<:Index}=collect(inds(T)),
#                       diag_pairs::Vector{<:Tuple{<:Integer,<:Integer}}=Tuple{Int,Int}[],
#                       atol::Real=1e-12, 
#                       rtol::Real=0)
  
#   println("from_itensor called and check the result here")

#   Tp = permute(T, order_inds...)          # ensure axis order is what you want
#   A = Array(Tp, inds(Tp))                 # uses Tp's current index order

#   # A  = Array(Tp)                          # dense materialization
#   A6 = from_dense(A; diag_pairs=diag_pairs, atol=atol, rtol=rtol)
#   B6 = to_dense(A6)
#   println("max|A6-B6| = ", maximum(abs.(A .- B6)))
#   @assert maximum(abs.(A .- B6)) == 0.0
#   error("matelem.jl test stopping after from_itensor")
#   # return from_dense(A; diag_pairs=diag_pairs, atol=atol, rtol=rtol), order_inds
# end


# function build_MPO_with_support!(
#   H::MPO, tempMPO, sites, Vs::Vector{<:AbstractMatrix},
#   support::Union{Nothing,Vector{Int64}}
# )
#   N = length(H)
#   dual = support === nothing
#   supp = dual ? nothing : Set(support)

#   # Rank from SVD only
#   tdim_of = n -> size(Vs[n], 2)

#   # Bond dim = SVD rank, but at least 1 so ITensors Index exists.
#   # In support-mode, bonds not touching support are forced to dim 1.
#   bonddim(b) = if dual
#     left = clamp(b - 1, 1, N)
#     max(tdim_of(left), 1)
#   else
#     if b == 1 || b == N + 1
#       1
#     else
#       left, right = b - 1, b
#       ((left in supp) || (right in supp)) ? max(tdim_of(left), 1) : 1
#     end
#   end

#   # llinks = [Index(bonddim(b), "Link,l=$(b-1)") for b in 1:(N + 1)]
#   llinks = [Index(bonddim(b+1), "Link,l=$b") for b in 0:N] 

#   function infer_valtype(n)
#     Tvs = promote_type(eltype(Vs[max(n-1, 1)]), eltype(Vs[n]))
#     Tc = Float64
#     for el in tempMPO[n]
#       Tc = promote_type(Tc, typeof(coefficient(el.val)))
#     end
#     return promote_type(Tvs, Tc)
#   end

#   function site_tensor(n, ll, rl)
#     ValT = infer_valtype(n)
#     if !dual && !(n in supp)
#       idM = zeros(ValT, dim(ll), dim(rl))
#       idM[1, 1] = one(ValT)
#       return itensor(idM, ll, rl) * computeSiteProd(sites, Prod([Op("Id", n)]))
#     end
#     VL = (n > 1) ? Vs[n - 1] : zeros(ValT, 1, 1)
#     VR = Vs[n] 
    
#     Tsum = ITensor()

#     for el in tempMPO[n]
#       t = el.val
#       ct = convert(ValT, coefficient(t))
#       # Skip exact zeros (use ValT epsilon)
#       abs(ct) > eps(real(one(ValT))) || continue
#       A_row, A_col = el.row, el.col
#       # Dense link-matrix for this single transition, then multiply by onsite operator
#       M = zeros(ValT, dim(ll), dim(rl))
#       if A_row == -1 && A_col == -1
#         M[1, 1] += ct
#       elseif A_row == -1
#         @inbounds for c in 1:size(VR, 2)
#           M[1, c] += ct * VR[A_col, c]
#         end
#       elseif A_col == -1
#         @inbounds for r in 1:size(VL, 2)
#           M[r, 1] += ct * conj(VL[A_row, r])
#         end
#       else
#         @inbounds for r in 1:size(VL, 2), c in 1:size(VR, 2)
#           M[r, c] += ct * conj(VL[A_row, r]) * VR[A_col, c]
#         end
#       end

#       # println("MATRIX FOR SITE $n, TERM $t : ")
#       # println(itensor(M, ll, rl))
#       # println("------------")
#       # println(computeSiteProd(sites, argument(t)))
#       # println("=================================")
#       Tsum += itensor(M, ll, rl) * computeSiteProd(sites, argument(t))
#       # Tsum += itensor(M, ll, rl) * computeSiteProd(sites, argument(t))
#     end
#     # println(Tsum)

#     typeof(Tsum) === ITensor || error("Expected ITensor, got $(typeof(Tsum))")

#     # from_itensor(Tsum)

#     # error("errr")
#     return Tsum
#   end

#   for n in 1:N
#     H[n] = site_tensor(n, llinks[n], llinks[n + 1])
#   end

#   # simple left/right boundary (scalar 1 on the boundary link indices)
#   L = ITensor(llinks[1]); R = ITensor(llinks[N + 1])
#   L[1] = 1.0; R[1] = 1.0
#   # L[1] = one(eltype(L)); R[1] = one(eltype(R))
#   H[1] *= L; H[N] *= R
#   return H
# end

function build_MPO_with_support!(
  H::MPO, tempMPO, sites, Vs::Vector{<:AbstractMatrix},
  support::Union{Nothing,Vector{Int64}}
)
  N = length(H)
  dual = support === nothing
  supp = dual ? nothing : Set(support)

  # Rank from SVD only
  tdim_of = n -> size(Vs[n], 2)

  # Bond dims for b = 1..N+1 (llinks[1] is left boundary, llinks[N+1] right boundary)
  # Fix: force boundaries to dim 1 in BOTH modes.
  bonddim(b) = if b == 1 || b == N + 1
    1
  elseif dual
    # bond b is between sites (b-1) and b, so take rank from Vs[b-1]
    max(tdim_of(b - 1), 1)
  else
    left, right = b - 1, b
    ((left in supp) || (right in supp)) ? max(tdim_of(left), 1) : 1
  end

  # Fix: make N+1 link indices; internal bond between sites b and b+1 has tag "Link,l=b".
  # llinks[b+1] corresponds to bond b for b=1..N-1
  llinks = Vector{ITensors.Index}(undef, N + 1)
  llinks[1] = Index(bonddim(1), "Link,l=0")           # left boundary
  for b in 1:(N - 1)
    llinks[b + 1] = Index(bonddim(b + 1), "Link,l=$b") # internal bond b
  end
  llinks[N + 1] = Index(bonddim(N + 1), "Link,l=$N")  # right boundary

  function infer_valtype(n)
    Tvs = promote_type(eltype(Vs[max(n - 1, 1)]), eltype(Vs[n]))
    Tc = Float64
    for el in tempMPO[n]
      Tc = promote_type(Tc, typeof(coefficient(el.val)))
    end
    return promote_type(Tvs, Tc)
  end

  function site_tensor(n, ll, rl)
    ValT = infer_valtype(n)

    if !dual && !(n in supp)
      idM = zeros(ValT, dim(ll), dim(rl))
      idM[1, 1] = one(ValT)
      return itensor(idM, ll, rl) * computeSiteProd(sites, Prod([Op("Id", n)]))
    end

    VL = (n > 1) ? Vs[n - 1] : zeros(ValT, 1, 1)
    VR = Vs[n]

    Tsum = ITensor()

    for el in tempMPO[n]
      t = el.val
      ct = convert(ValT, coefficient(t))
      abs(ct) > eps(real(one(ValT))) || continue

      A_row, A_col = el.row, el.col
      M = zeros(ValT, dim(ll), dim(rl))

      if A_row == -1 && A_col == -1
        M[1, 1] += ct
      elseif A_row == -1
        @inbounds for c in 1:size(VR, 2)
          M[1, c] += ct * VR[A_col, c]
        end
      elseif A_col == -1
        @inbounds for r in 1:size(VL, 2)
          M[r, 1] += ct * conj(VL[A_row, r])
        end
      else
        @inbounds for r in 1:size(VL, 2), c in 1:size(VR, 2)
          M[r, c] += ct * conj(VL[A_row, r]) * VR[A_col, c]
        end
      end
      # Fix: only add once (you had a duplicate line earlier)
      Tsum += itensor(M, ll, rl) * computeSiteProd(sites, argument(t))
    end
    typeof(Tsum) === ITensor || error("Expected ITensor, got $(typeof(Tsum))")
    return Tsum
  end

  @inbounds for n in 1:N
    H[n] = site_tensor(n, llinks[n], llinks[n + 1])
    # println("Hn for site $n : ", H[n])
  end

  # Scalar boundary tensors (now guaranteed boundary links are dim 1)
  L = ITensor(llinks[1]);  L[1] = 1.0
  R = ITensor(llinks[N + 1]); R[1] = 1.0
  H[1] *= L
  H[N] *= R
  # println("==================================")
  return H
end


function ctn_svdMPO(
  ValType::Type{<:Number}, os::OpSum{C}, sites,
  support::Union{Vector{Int64}, Nothing}=nothing;
  mindim=1, maxdim=typemax(Int), cutoff=1e-15
)::MPO where {C}
  # println("CTN compression based tensor ")
  N = length(sites)

  # Start with "no channels" everywhere; only SVD fills Vs[k]
  Vs = [zeros(ValType, 0, 0) for _ in 1:N]
  tempMPO = [MatElem{Scaled{C,Prod{Op}}}[] for _ in 1:N]
  rightmaps = [Dict{Vector{Op},Int}() for _ in 1:N]

  crosses_bond(t::Scaled{C,Prod{Op}}, n::Int) = (only(site(t[1])) <= n <= only(site(t[end])))

  _is_id(o::Op) = ITensors.name(o) == "Id"
  all_id(v::Vector{Op}) = !isempty(v) && all(_is_id, v)   # empty ≠ all-Id
  strip_id(v::Vector{Op}) = [o for o in v if !_is_id(o)]
  # Use a *bond-based* sentinel so both sides of a bond use the same key
  _ID_SENTINEL_BOND(b) = [Op("Id", b)]

  for n in 1:N
    leftbond_coefs = MatElem{ValType}[]
    leftmap = Dict{Vector{Op},Int}()

    for term in os
      crosses_bond(term, n) || continue
      left   = filter(t -> (only(site(t)) <  n), terms(term))
      onsite = filter(t -> (only(site(t)) == n), terms(term))
      right  = filter(t -> (only(site(t)) >  n), terms(term))

      # println("Processing term at site $n : ", term, " ", left, " | ", onsite, " | ", right)

      # Keys for bond (n-1)|n  and n|(n+1)
      left_key  = (n > 1 && all_id(left)) ? _ID_SENTINEL_BOND(n-1) : strip_id(left)
      mid_raw   = vcat(onsite, right)           # lives on bond (n-1)|n
      mid_key   = (n > 1 && all_id(mid_raw)) ? _ID_SENTINEL_BOND(n-1) : strip_id(mid_raw)
      right_key = (all_id(right)) ? _ID_SENTINEL_BOND(n) : strip_id(right)

      # println("  left_key: ", left_key, "  mid_key: ", mid_key, "  right_key: ", right_key, " mid raw: ", mid_raw)

      # Build SVD matrix rows/cols only if left side is nonempty (not first site)
      bond_row = -1
      bond_col = -1
      if !isempty(left_key)
        bond_row = posInLink!(leftmap, left_key)
        bond_col = (!isempty(mid_key) ? posInLink!(rightmaps[n-1], mid_key) : -1)
        push!(leftbond_coefs, MatElem(bond_row, bond_col, convert(ValType, coefficient(term))))
      end
      # Site wiring for the actual MPO tensor
      A_row = bond_col
      A_col = isempty(right_key) ? -1 : posInLink!(rightmaps[n], right_key)
      site_coef = (A_row == -1) ? coefficient(term) : one(C)

      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, Op("F", n))
        else
          push!(onsite, Op("Id", n))
        end
      end
      push!(tempMPO[n], MatElem(A_row, A_col, site_coef * Prod(onsite)))
    end
    # println(" leftbond_coefs at site $n : ", length(leftbond_coefs))
    # for me in leftbond_coefs
    #   println("   ", me)
    # end
    remove_dups!(tempMPO[n])
    if n > 1 && !isempty(leftbond_coefs)
      M = toMatrix(leftbond_coefs)   # rows: left_key, cols: mid_key
      # println(" SVD matrix at bond $(n-1)|$n : ", M)
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[n - 1] = (tdim == 0) ? zeros(ValType, 0, 0) : Matrix{ValType}(V[1:nc, 1:tdim])
    end    
  end

  H = MPO(sites)
  build_MPO_with_support!(H, tempMPO, sites, Vs, support)
  return H
end


# Original MPO construction implementation
# `ValType::Type{<:Number}` is used instead of `ValType::Type` for efficiency, possibly due to increased method specialization.
# See https://github.com/ITensor/ITensors.jl/pull/1183.
function svdMPO(
  ValType::Type{<:Number}, os::OpSum{C}, sites; mindim=1, maxdim=typemax(Int), cutoff=1e-15
)::MPO where {C}
  N = length(sites)

  # Specifying the element type with `Matrix{ValType}[...]` improves type inference and therefore efficiency.
  # See https://github.com/ITensor/ITensors.jl/pull/1183.
  Vs = Matrix{ValType}[Matrix{ValType}(undef, 1, 1) for n in 1:N]
  tempMPO = [MatElem{Scaled{C,Prod{Op}}}[] for n in 1:N]

  function crosses_bond(t::Scaled{C,Prod{Op}}, n::Int) where {C}
    return (only(site(t[1])) <= n <= only(site(t[end])))
  end

  rightmaps = [Dict{Vector{Op},Int}() for _ in 1:N]

  for n in 1:N
    leftbond_coefs = MatElem{ValType}[]

    leftmap = Dict{Vector{Op},Int}()
    for term in os
      crosses_bond(term, n) || continue

      left = filter(t -> (only(site(t)) < n), terms(term))
      onsite = filter(t -> (only(site(t)) == n), terms(term))
      right = filter(t -> (only(site(t)) > n), terms(term))

      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap, left)
        bond_col = posInLink!(rightmaps[n - 1], vcat(onsite, right))
        bond_coef = convert(ValType, coefficient(term))
        push!(leftbond_coefs, MatElem(bond_row, bond_col, bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(rightmaps[n], right)
      site_coef = one(C)
      if A_row == -1
        site_coef = coefficient(term)
      end
      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, Op("F", n))
        else
          push!(onsite, Op("Id", n))
        end
      end
      el = MatElem(A_row, A_col, site_coef * Prod(onsite))
      push!(tempMPO[n], el)
    end
    remove_dups!(tempMPO[n])
    if n > 1 && !isempty(leftbond_coefs)
      M = toMatrix(leftbond_coefs)
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[n - 1] = Matrix{ValType}(V[1:nc, 1:tdim])
    end
  end

  llinks = Vector{Index{Int}}(undef, N + 1)
  llinks[1] = Index(2, "Link,l=0")

  H = MPO(sites)

  for n in 1:N
    VL = Matrix{ValType}(undef, 1, 1)
    if n > 1
      VL = Vs[n - 1]
    end
    VR = Vs[n]
    tdim = isempty(rightmaps[n]) ? 0 : size(VR, 2)

    llinks[n + 1] = Index(2 + tdim, "Link,l=$n")

    ll = llinks[n]
    rl = llinks[n + 1]

    H[n] = ITensor()

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coefficient(t)) > eps()) || continue

      M = zeros(ValType, dim(ll), dim(rl))

      ct = convert(ValType, coefficient(t))
      if A_row == -1 && A_col == -1 #onsite term
        M[end, 1] += ct
      elseif A_row == -1 #term starting on site n
        for c in 1:size(VR, 2)
          z = ct * VR[A_col, c]
          M[end, 1 + c] += z
        end
      elseif A_col == -1 #term ending on site n
        for r in 1:size(VL, 2)
          z = ct * conj(VL[A_row, r])
          M[1 + r, 1] += z
        end
      else
        for r in 1:size(VL, 2), c in 1:size(VR, 2)
          z = ct * conj(VL[A_row, r]) * VR[A_col, c]
          M[1 + r, 1 + c] += z
        end
      end

      T = itensor(M, ll, rl)
      H[n] += T * computeSiteProd(sites, argument(t))
    end

    #
    # Special handling of starting and
    # ending identity operators:
    #
    idM = zeros(ValType, dim(ll), dim(rl))
    idM[1, 1] = 1.0
    idM[end, end] = 1.0
    T = itensor(idM, ll, rl)
    H[n] += T * computeSiteProd(sites, Prod([Op("Id", n)]))
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(llinks[N + 1])
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #svdMPO


function svdMPO(os::OpSum{C}, sites, support::Union{Vector{Int64}, Nothing}=nothing; kwargs...)::MPO where {C}
  # Function barrier to improve type stability
  ValType = determineValType(terms(os))
  if support === nothing
    return svdMPO(ValType, os, sites; kwargs...)
  else
    return ctn_svdMPO(ValType, os, sites, support; kwargs...)
  end
end