# rayleigh_ritz_sweep.jl — dmrg-side wiring for the Rayleigh-Ritz (:rr) run_mode.
#
# Split out of dmrg.jl (2026-07) to keep the RR path in one place. The RR
# numerical kernel is in SparseBackends/src/rayleigh_ritz.jl
# (rayleigh_ritz_local_eigsolve); this is the sweep-loop adapter that builds the
# H·v operator and recovers an aliased φ. Called from the `:rr` branch of the
# local eigensolve dispatch in dmrg.jl.
#
# Generalized Rayleigh–Ritz: project H_eff·φ = E·M·φ onto a small aliased Krylov
# subspace built from H·v (kept aliased via recast_to_phi), form k×k H_small /
# M_small by SCALAR inner products (M = RAW Lgram/Rgram — no square root, no rtol
# cut), solve the k×k generalized eig with a per-block null projection. The Ritz
# vector Σcᵢvᵢ is an aliased same-schema combo → stays aliased. Robust to a graded
# / ill-conditioned M (the case where the B_op rtol frame stalls). Uses the
# EXISTING (dense env-slice) grams: RR needs no aliased M — aliasing is preserved
# by the Ritz combo, M only enters as scalars ⟨vᵢ|M|vⱼ⟩.
#
# rr_dense_iter (diagnostic): run the subspace / H_small / M_small fully DENSE
# (densify φ + grams, no recast in Hop) and re-impose φ's aliased schema ONLY at
# recovery via snap_dense_to_aliased (the dense→aliased projection — recast_to_phi
# and _snap_to_schema both require an already-aliased source, which the dense Ritz
# vector isn't). Tests whether per-iteration aliasing is what caps the RR energy.
function rr_local_eigsolve(PH, phi, Lgram, Rgram, recast_to_phi;
                           rr_dense_iter::Bool,
                           which, tol, krylovdim, maxiter,
                           roofline::Bool, run_label::String,
                           b::Int, ha::Int, sw::Int, debug::Bool=false)
    if rr_dense_iter
        _dns(x) = ITensors.has_external_storage(x) ? SparseBackends.to_dense_itensors_unfused(x) : x
        _phid, _Lgd, _Rgd = _dns(phi), _dns(Lgram), _dns(Rgram)
        Hop_rr = v -> _dns(product(PH, v; roofline=roofline, run_label=run_label))
        vals, vecs = SparseBackends.rayleigh_ritz_local_eigsolve(
            Hop_rr, _phid, _Lgd, _Rgd;
            which=which, tol=tol, krylovdim=krylovdim, maxiter=maxiter,
            rtol=1e-8, b=b, ha=ha, sw=sw)
        vecs = [SparseBackends.snap_dense_to_aliased(vecs[1], phi)]
    else
        Hop_rr = v -> recast_to_phi(product(PH, v; roofline=roofline, run_label=run_label))
        vals, vecs = SparseBackends.rayleigh_ritz_local_eigsolve(
            Hop_rr, phi, Lgram, Rgram;
            which=which, tol=tol, krylovdim=krylovdim, maxiter=maxiter,
            rtol=1e-8, b=b, ha=ha, sw=sw)
    end
    if debug
        println("[BOND_EIG b=", b, " ha=", ha, " sw=", sw, "] RR smallest=", real(vals[1])); flush(stdout)
    end
    return vals, vecs
end
