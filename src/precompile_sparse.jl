#
# precompile_sparse.jl -- cache native code for the sparse-backend MPO/MPS paths.
#
# WHY THIS EXISTS
# Measured cold-process cost of a FOUR-SITE run (experiments/lgt_tests/sparse):
#
#     MPO(H)             112.68 s      attach_P (coo x dense -> aliased)   56.65 s
#     mul_coo(projs)     100.37 s      d4_mpo_layers                      168.47 s
#
# ~11 min of LLVM codegen to evolve 4 sites. Immediately afterwards, the SAME calls
# at Nm=4 and Nm=5 cost 0.02-0.83 s, so it is all first-call specialisation, paid
# once per PROCESS. Package precompilation did not cover it: the cached image holds
# the package BODY, not specialisations for the concrete types a caller instantiates.
# This workload instantiates them at precompile time so the cost is paid once ever.
# That matters most for HPC ensembles, where every job would otherwise pay it.
#
# WHY IT LIVES HERE, NOT IN SparseBackends
# `contract(A::MPO, psi::MPS, Abackend, Bbackend; Cbackend)` is defined in this
# package (src/mpo.jl). SparseBackends depends on ITensors only -- it cannot see
# MPO/MPS, and adding ITensorMPS to it would be a dependency cycle.
#
# SITES: stock "S=3/2" (d=4) interleaved with "S=1/2" (d=2)
# This is the same d=4/d=2 mixed chain the LGT tests use, but built from sitetypes
# ITensors already ships, so the workload does not depend on the experiment's
# Matter4 definitions. A 4-site chain is the smallest with both boundary (rank 2/3)
# and interior (rank 3/4) tensors, which is what drives specialisation -- an Index's
# dimension is a runtime field of `Index{Int64}`, not a type parameter.
#
# EVERY BLOCK IS try/catch-GUARDED ON PURPOSE
# @compile_workload does not swallow exceptions, and this file is in a submodule
# shared by other work: a throw here would fail precompilation of ITensorMPS and
# break the whole environment. Guarding degrades a broken block to "less code
# cached" instead. Code executed before a throw is still cached.
#
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    N = 4
    @compile_workload begin
        try
            # d=4 on odd sites, d=2 on even -- the LGT chain's layout.
            s = siteinds(n -> isodd(n) ? "S=3/2" : "S=1/2", N)

            # OpSum -> MPO: the 113 s item. Mixed 1-site and 3-site terms, matching
            # the LGT Hamiltonian's two commuting groups. Sz/S+/S- are defined for
            # every spin sitetype, so this works at both dimensions.
            os = OpSum()
            for j in 1:(N - 2)
                os += 1.0, "Sz", j, "Sz", j + 1, "Sz", j + 2
            end
            for j in 1:N
                os += 0.5, "S+", j
                os += 0.5, "S-", j
            end
            H = MPO(os, s)
            psi = random_mps(ComplexF64, s; linkdims = 2)

            # Dense reference paths (used by the dense drivers and by densify()).
            try
                contract(H, psi; alg = "naive")
            catch
            end

            # MPO x MPO through COO: builds P = prod_j Pi_j without densifying.
            try
                contract(H, prime(H, "Site"), :coo, :coo)
            catch
            end

            # THE expensive one: COO MPO x dense MPS -> AliasedBlockSparse, i.e.
            # psi = P * core. This is where a cold process spent >15 min in
            # jl_compile_codeinst_now for wrapped_contract_aliased.
            ali = contract(H, psi, :coo, :dense; Cbackend = :aliased)
            try
                orthogonalize!(ali, 1; maxdim = 8)
            catch
            end

            # Feeding an aliased MPS forward: one TEBD layer onto sparse storage.
            try
                ali2 = contract(H, ali, :dense, :aliased; Cbackend = :aliased)
                orthogonalize!(ali2, 1; maxdim = 8)
            catch
            end

            # Measurement helpers the drivers call every cycle.
            try
                maxlinkdim(ali)
                inner(psi, psi)
                norm(psi)
            catch
            end
        catch
        end
    end
end
