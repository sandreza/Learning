using KernelAbstractions

function c_backsolve!(vector, matrix, n, index_set)
    @inbounds for k in index_set
        @inbounds for i in n:-1:1
            vector[i, k...] /= matrix[i,i, k...]
            @inbounds for j in 1:i-1
                vector[j, k...] -= matrix[j,i, k...] * vector[i, k...]
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
for i in CartesianIndices((1:2, 1:4))
      println("i is " * string(i))
end
for i in CartesianIndices(A)
    println(i)
end
size(A)


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
