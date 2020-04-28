include("gmres2.jl")
using LinearAlgebra, Plots, Random, CuArrays

ArrayType = Array
# for defining linear_operator
function closure_linear_operator_multi!(A, n)
    function linear_operator!(x, y)
        for i in 1:n
            mul!(view(x,:,i), view(A, :, :, i), view(y,:,i))
        end
    end
end

n  = 100  # size of vector space
ni = 10 # number of independent linear solves
Random.seed!(1235)
b = ArrayType(randn(n, ni)) # rhs
x = ArrayType(randn(n, ni)) # initial guess
A = ArrayType(randn((n,n, ni)) ./ sqrt(n) .* 1.0)
for i in 1:n
    A[i,i,:] .+= 1.0
end
gmres = ParallelGMRES(b, ArrayType=ArrayType)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
linear_operator! = closure_linear_operator_multi!(A, ni)
solve!(x, b, linear_operator!, gmres)
###

linear_operator!(y, x)
println("The error is ")
display(norm(y - b) / norm(b))

# now solve again to make sure that the same structure can be
# reused
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
linear_operator! = closure_linear_operator_multi!(A, ni)
solve!(x, b, linear_operator!, gmres)

linear_operator!(y, x)
println("The error is ")
display(norm(y - b) / norm(b))

###
plot(log.(gmres.residual)/log(10), xlabel = "iteration", ylabel = "log10 residual", title = "gmres convergence", legend = false)
