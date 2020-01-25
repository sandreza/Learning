using LinearAlgebra
using BenchmarkTools

struct GPR{ℱ, 𝒮, 𝒰}
    kernel::ℱ
    kstar::𝒮
    kmat::𝒰
end



"""
compute_kernel_matrix(k, x)

# Description
- Computes the kernel matrix for GPR

# Arguments
- k : (function) the kernel. Takes in two arguments and produce a real number
- x : (array of predictors). x[1] is a vector

# Return
- sK: (symmetric matrix). A symmetric matrix with entries sK[i,j] = k(x[i], x[j]). This is only meaningful if k(x,y) = k(y,x) (it should)
"""
function compute_kernel_matrix(k, x; hyperparameters = [])
    n = size(x)[1]
    K = zeros(n,n)
    for i in 1:n
        for j in i:n
            if isempty(hyperparameters)
                K[i,j] = k(x[i], x[j])
            else
                K[i,j] = k(x[i], x[j], hyperparameters = hyperparameters)
            end
        end
    end
    sK = Symmetric(K)
    return sK
end


"""
gaussian_kernel(x,y; γ = 1.0, σ = 1.0)

# Description
- Outputs a Gaussian kernel with hyperparameter γ

# Arguments
- x: first coordinate
- y: second coordinate

# Keyword Arguments
-The first is γ, the second is σ where, k(x,y) = σ * exp(- γ * d(x,y))
- γ = 1.0: (scalar). hyperparameter in the Gaussian Kernel.
- σ = 1.0; (scalar). hyperparameter in the Gaussian Kernel.
"""
function gaussian_kernel(x,y; γ = 1.0, σ = 1.0)
    y = σ * exp(- γ * d(x,y))
    return y
end

"""
closure_gaussian_kernel(x,y; γ = 1.0, σ = 1.0)

# Description
- Outputs a function that computes a Gaussian kernel

# Arguments
- d: distance function. d(x,y)

# Keyword Arguments
-The first is γ, the second is σ where, k(x,y) = σ * exp(- γ * d(x,y))
- γ = 1.0: (scalar). hyperparameter in the Gaussian Kernel.
- σ = 1.0; (scalar). hyperparameter in the Gaussian Kernel.
"""
function closure_guassian_closure(d; hyperparameters = [1.0, 1.0])
    function gaussian_kernel(x,y)
        y = hyperparameters[1] * exp(- hyperparameters[2] * d(x,y))
        return y
    end
    return gaussian_kernel
end
