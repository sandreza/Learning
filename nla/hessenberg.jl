using Revise, LinearAlgebra
# Algorithm 26.1
# why have it go from 1:m in the second product?
# those entries are zero anyway
function upper_hessenberg(A)
    H = copy(A)
    m, n = size(H)
    v = [zeros(eltype(A), m-i) for i in 1:(n-2)]
    for i in 1:(m-2)
        x = H[i+1:m, i]
        v[i] .= x
        v[i][1] += sign(x[1]) * norm(x)
        v[i] /= norm(v[i])
        # make use of the fact zeros are introduced in the columns
        # note that we could also go from 1:n, but it wastes computation
        H[i+1:m, i:n] -= 2.0 * v[i] * (v[i]' * H[i+1:m, i:n])
        # zeros are not introduced into the rows, so we can't use the same trick
        H[1:m, i+1:n] -=  2.0 * (H[1:m, i+1:n] * v[i] ) * v[i]'
    end
    return H, v
end

function build_Q_H(v)
    H = zeros(eltype(v[1]), length(v)+2, length(v)+2) + I
    m, n = size(H)
    for i in 1:(m-2)
        H[i+1:m, i:n] -= 2.0 * v[i] * (v[i]' * H[i+1:m, i:n])
    end
    return H
end

n = 4
A = randn(n,n)
# A = A' * A
H, v = upper_hessenberg(A)
Q = build_Q_H(v)
norm(Q' * H * Q - A)
