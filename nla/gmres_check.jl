include("gmres.jl")
using LinearAlgebra, Plots, Random

# for defining linear_operator
function closure_linear_operator_multi!(A, n)
    function linear_operator!(x, y)
        for i in 1:n
            mul!(view(x,:,i), view(A, :, :, i), view(y,:,i))
        end
    end
end

n  = 5  # size of vector space
ni = 3 # number of independent linear solves
Random.seed!(1235)
b = randn(n, ni) # rhs
x = randn(n, ni) # initial guess
A = randn((n,n, ni)) ./ sqrt(n) .* 1.0
gmres = ParallelGMRES(b)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += randn(n,ni) * 0.01 * maximum(abs.(x))
linear_operator! = closure_linear_operator_multi!(A, ni)


###
# unrolled
x_init = copy(x)
# TODO: make this line work with CLIMA
gmres.x .= x
# TODO: make linear_operator! work with CLIMA
linear_operator!(x, x_init)
r_vector = b - x
# Save first candidate Krylov vector
# TODO: make this work with CLIMA
gmres.b .= r_vector # perhaps throw in initialize?
# Save second candidate Krylov vector
# TODO: make linear_operator! work with CLIMA
linear_operator!(gmres.sol, r_vector)
