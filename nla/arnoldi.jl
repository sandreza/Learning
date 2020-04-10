using Revise, LinearAlgebra
"""
arnoldi(A, x⁰; iterations = m)

# Description
Perform an Arnoldi iteration

# Arguments
- `A`: (array). Matrix represented in the standard basis
- `b`: (vector). Initial guess

# Keyword Arguments

- columns: default = length(b). Number of columns in the Hessenberg matrix

# Return
- `H`: (array). upper hessenber matrix
- `Q`: (array). orthonormal basis of Krylov Subspace
"""
function arnoldi(A, b; columns = length(b))
    m = columns
    Q = zeros(eltype(A), (length(b), m+1))
    # H = [zeros(eltype(A), 1 + i) for i in 1:m-1]
    H = zeros(eltype(A), (m+1, m))
    Q[:,1] .= b / norm(b) # First Krylov vector
    Aqⁿ = copy(b)
    for n in 1:m
        Aqⁿ .= A * Q[:,n]
        for j in 1:n
            H[j, n] = Q[:,j]' * Aqⁿ
            Aqⁿ -= H[j, n] * Q[:,j]
        end
        if n+1 <= length(b)
            H[n+1, n] = norm(Aqⁿ)
            Q[:, n+1] .= Aqⁿ / H[n+1, n]
        end
    end
    return Q, H
end

###
n = 4
columns = 4
A = randn(n,n)
b = randn(n)

Qⁿ⁺¹, Hⁿ = arnoldi(A,b, columns = columns)

if columns == length(b)
    err = norm(A * Qⁿ⁺¹[:, 1:columns] - Qⁿ⁺¹[:, 1:columns] * Hⁿ) / norm(A)
    println("The relative error is " * string(err))
else
    err = norm(A * Qⁿ⁺¹[:, 1:columns] - Qⁿ⁺¹[:, 1:columns+1] * Hⁿ) / norm(A)
    println("The relative error is " * string(err))
end
