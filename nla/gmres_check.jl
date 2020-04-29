include("gmres.jl")
using LinearAlgebra, Plots, Random, CuArrays

ArrayType = Array

@kernel function multiply_A_kernel!(x,A,y,n1,n2)
    I = @index(Global)
    for i in 1:n1
        tmp = zero(eltype(x))
        for j in 1:n2
            tmp += A[i, j, I] * y[j, I]
        end
        x[i, I] = tmp
    end
end

function multiply_by_A!(x, A, y, n1, n2; ndrange = size(x[1,:]), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(x,Array)
        kernel! = multiply_A_kernel!(CPU(), cpu_threads)
    else
        kernel! = multiply_A_kernel!(CUDA(), gpu_threads)
    end
    return kernel!(x, A, y, n1, n2, ndrange = ndrange)
end
# for defining linear_operator
function closure_linear_operator_multi!(A, n1, n2, n3)
    function linear_operator!(x, y)
        event = multiply_by_A!(x, A, y, n1, n2, ndrange = n3)
        wait(event)
        return nothing
    end
end

n  = 10  # size of vector space
ni = 10 # number of independent linear solves
Random.seed!(1235)
b = ArrayType(randn(n, ni)) # rhs
x = ArrayType(randn(n, ni)) # initial guess
A = ArrayType(randn((n,n, ni)) ./ sqrt(n) .* 1.0)
for i in 1:n
    A[i,i,:] .+= 0.0
end
gmres = ParallelGMRES(b, ArrayType=ArrayType)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
linear_operator! = closure_linear_operator_multi!(A, size(A)...)
solve!(x, b, linear_operator!, gmres)
###
linear_operator!(y, x)
println("The error is ")
display(norm(y - b) / norm(b))
println(maximum(abs.(y-b)) / maximum(abs.(b)))

# now solve again to make sure that the same structure can be
# reused
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += ArrayType(randn(n,ni) * 0.01 * maximum(abs.(x)))
y = copy(x)
solve!(x, b, linear_operator!, gmres)

linear_operator!(y, x)
println("The error is ")
display(norm(y - b) / norm(b))

i = minimum([size(x)[2], 30])
norm(A[:,:,i] * x[:,i] - b[:, i]) / norm(b[:,i])
###
plot(log.(gmres.residual)/log(10), xlabel = "iteration", ylabel = "log10 residual", title = "gmres convergence", legend = false)
