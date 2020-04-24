using KernelAbstractions, Random, LinearAlgebra
using BenchmarkTools

function c_backsolve!(vector, matrix, n, index_set)
    @inbounds for k in index_set
        @inbounds for i in n:-1:1
            vector[i, Tuple(k)...] /= matrix[i,i, Tuple(k)...]
            @inbounds for j in 1:i-1
                vector[j, Tuple(k)...] -= matrix[j,i, Tuple(k)...] * vector[i, Tuple(k)...]
            end
        end
    end
    return nothing
end

function t_backsolve!(vector, matrix, n, index_set)
    Threads.@threads for k in index_set
        @inbounds for i in n:-1:1
            vector[i, Tuple(k)...] /= matrix[i,i, Tuple(k)...]
            @inbounds for j in 1:i-1
                vector[j, Tuple(k)...] -= matrix[j,i, Tuple(k)...] * vector[i, Tuple(k)...]
            end
        end
    end
    return nothing
end


###
n = 2
dupl = 2
Random.seed!(1234)
x = randn((n,dupl))
b = copy(x)
A = randn((n,n,dupl))
c_backsolve!(x, A, n, 1:dupl)
cont = []
for i in 1:dupl
    println("error for " * string(i) * " is")
    push!(cont, norm(x[:,i] .- UpperTriangular(A[:,:,i]) \ b[:,i]) / norm(x[:,i]))
    println(cont[i])
end

###
#=
for i in CartesianIndices((1:2, 1:4))
      println("i is " * string(i))
end
for i in CartesianIndices(A)
    println(i)
end
size(A)
=#

###
@kernel function mul2(A)
  I = @index(Global, NTuple)
  println("space")
  println(I)
  println(Threads.threadid())
end

A = ones(2, 3)
kernel = mul2(CPU(), Threads.nthreads())
event = kernel(A, ndrange=(1,3,3))
wait(event)

@kernel function k_backsolve!(vector, @Const(matrix), n)
    k = @index(Global, NTuple)
    @inbounds for i in n:-1:1
        vector[i, k...] /= matrix[i,i, k...]
        @inbounds for j in 1:i-1
            vector[j, k...] -= matrix[j,i, k...] * vector[i, k...]
        end
    end
end

###
n = 100
dupl = (10^4,1)
Random.seed!(1234)
x = randn((n,dupl...))
b = copy(x)
A = randn((n,n,dupl...))

kernel2 = k_backsolve!(CPU(), Threads.nthreads())
event2 = kernel2(x, A, n, ndrange=(dupl...,))
wait(event2)


for i in CartesianIndices(dupl)
    println("error for " * string(i) * " is")
    println(norm(x[:,Tuple(i)...] .- UpperTriangular(A[:,:,Tuple(i)...]) \ b[:,Tuple(i)...]) / norm(x[:,Tuple(i)...]))
end


###
@benchmark wait(kernel2($x, $A, $n, ndrange=(dupl...,)))



###
@benchmark c_backsolve!(x, A, n, CartesianIndices(dupl))
@benchmark t_backsolve!(x, A, n, CartesianIndices(dupl))
