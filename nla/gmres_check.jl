using LinearAlgebra, KernelAbstractions

"""
ParallelGMRES{𝒮,𝒯,𝒱}
# Description

Launches n independent GMRES solves

# Members
- atol::ℱ (float) absolute tolerance
- rtol::ℱ (float) relative tolerance
- m::𝒮 (int) size of vector in each independent instance
- n::𝒮 (int) number of independent GMRES
- k_n::𝒮 (int) Krylov Dimension for each GMRES. It is also the number of GMRES iterations before nuking the subspace
- residual::𝒱 (vector) residual vector
- b::𝒱 (vector) permutation of the rhs
- x::𝒱 (vector) permutation of the initial guess
- sol::𝒱 (vector) solution vector
- rhs::𝒱 (vector) rhs vector
- cs::𝒱 (vector) Sequence of Gibbs Rotation matrices in compact form. This is implicitly the Qᵀ of the QR factorization of the upper hessenberg matrix H.
- H::𝒯 (array) Upper Hessenberg Matrix
- Q::𝒯 (array) Orthonormalized Krylov Subspace
- R::𝒯 (array) The R of the QR factorization of the UpperHessenberg matrix H

# Intended Use
Solving n linear systems iteratively

# Comments on Improvement
- Allocates all the memory at once: Could improve to something more dynamic
- Too much memory in H and R struct: Could use a sparse representation to cut memory use in half (or more)
- Needs to perform a transpose of original data structure into current data structure: Could perhaps do a transpose free version, but the code gets a bit clunkier and the memory would no longer be coalesced for the heavy operations
"""
struct ParallelGMRES{ℱ, 𝒮, 𝒯, 𝒱}
    atol::ℱ
    rtol::ℱ
    m::𝒮
    n::𝒮
    k_n::𝒮
    residual::𝒱
    b::𝒱
    x::𝒱
    sol::𝒱
    rhs::𝒱
    cs::𝒱
    H::𝒯  # A factor of two in memory can be saved here
    Q::𝒯
    R::𝒯 # A factor of two in memory can be saved here
end

"""
ParallelGMRES(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))) )

# Description
Constructor for the ParallelGMRES struct

# Arguments
- `Qrhs`: (array) Array structure that linear_operator! acts on

# Keyword Arguments
- `m`: (int) size of vector space for each independent linear solve. This is assumed to be the same for each and every linear solve. DEFAULT = length(Qrhs[:,1])
- `n`: (int) number of independent linear solves, DEFAULT = length(Qrhs[1,:])
- `atol`: (float) absolute tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `rtol`: (float) relative tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
"""
function ParallelGMRES(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))) )
    container = [
        atol,
        rtol,
        m,
        n,
        k_n = subspace_size,
        residual = zeros(eltype(Qrhs), (m, n)),
        b = zeros(eltype(Qrhs), (m, n)),
        x = zeros(eltype(Qrhs), (m, n)),
        sol = zeros(eltype(Qrhs), (k_n + 1, n)),
        rhs = zeros(eltype(Qrhs), (m, n)),
        cs = zeros(eltype(Qrhs), (2 * k_n, n)),
        Q = zeros(eltype(Qrhs), (m, k_n+1 , n)),
        H = zeros(eltype(Qrhs), (k_n+1, k, n)),
        R  = zeros(eltype(Qrhs), (k_n+1, k, n))
    ]
    ParallelGMRES(container...)
end
