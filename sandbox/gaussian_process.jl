using LinearAlgebra
using BenchmarkTools

"""
compute_kernel_matrix(k, x)

# Description
- Computes the kernel matrix for GPR

# Arguments
- k : (function) the kernel. Takes in two arguments and produce a real number
- x : (array of predictors). x[1] is a vector

# Return
- sK: a symmetric matrix with entries sK[i,j] = k(x[i], x[j])

"""
function compute_kernel_matrix(k, x)
    n = size(x)[1]
    K = zeros(n,n)
    for i in 1:n
        for j in i:n
            K[i,j] = k(x[i], x[j])
        end
    end
    sK = Symmetric(K)
    return sK
end
