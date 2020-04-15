using LinearAlgebra

"""
PrototypeRes{𝒮,𝒯,𝒱}

# Description
- A memory intensive struct for GMRES

# Members

- restart::𝒮 (int) number of GMRES iterations before wiping the subspace
- residual::𝒱 (vector) residual vector
- H::𝒯 (array) Upper Hessenberg Matrix
- Q::𝒯 (array) Orthonormalized Krylov Subspace
- KQ::𝒯 (array) The Q of the QR factorization of operator in Krylov Subspace
- KR::𝒯 (array) The R of the QR factorization of operator in Krylov Subspace

# Intended Use
- Solving linear systems iteratively
"""
struct PrototypeRes{𝒮,𝒯,𝒱}
    restart::𝒮
    residual::𝒱
    Q::𝒯
    H::𝒯 # A factor of two in memory can be saved here
    KQ::𝒯
    KR::𝒯 # A factor of two in memory can be saved here
end


"""
PrototypeRes(Q; restart = length(Q))

# Description
- Constructor for the PrototypeRes class

# Arguments
- `Q`: (array) ; Represents solution

# Keyword Arguments
- `k`: (int) ; Default = length(Q) ; How many krylove subspace iterations we keep

# Return
- An instance of the PrototypeRes class
"""
function PrototypeRes(Q; restart = length(Q))
    residual = similar(Q)
    k = restart
    Q = zeros(eltype(Q), (length(Q), k+1 ))
    H = zeros(eltype(Q), (k+1, k))
    KQ = zeros(eltype(Q), (k+1, k))
    KR  = zeros(eltype(Q), (k, k))
    container = [
        restart,
        residual,
        Q,
        H,
        KQ,
        KR
    ]
    return PrototypeRes(container...)
end



###
"""
arnoldi_update!(n, g, linear_operator!, b)

# Description
Perform an Arnoldi iteration

# Arguments
- `n`: current iteration number
- `g`: gmres struct that gets overwritten
- `linear_operator!`: (function) Action of linear operator on vector
- `b`: (vector). Initial guess

# linear_operator! Arguments
- `linear_operator!(x,y)`
# # Description
# - Performs Linear operation on vector and overwrites it
# # Arguments
# - `x`: (array) [OVERWRITTEN]
# - `y`: (array)
# # Return
# - Nothing

# Return
- nothing
"""
function arnoldi_update!(n, g, linear_operator!, b)
    if n==1
        g.Q[:,1] .= b / norm(b) # First Krylov vector
    end
    Aqⁿ = copy(b)
    linear_operator!(Aqⁿ, g.Q[:,n])
    for j in 1:n
        g.H[j, n] = dot(g.Q[:,j], Aqⁿ)
        Aqⁿ -= g.H[j, n] * g.Q[:,j]
    end
    if n+1 <= length(b)
        g.H[n+1, n] = norm(Aqⁿ)
        g.Q[:, n+1] .= Aqⁿ / g.H[n+1, n]
    end
    return nothing
end
