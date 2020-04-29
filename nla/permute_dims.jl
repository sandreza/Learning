A = reshape(Vector(1:8), (2,2,2))
println("The original")
display(A)
B = permutedims(A, [2, 1, 3])
println("The permuted")
display(B)

C = permutedims(A, [3, 2, 1])
println("The permuted 2")
display(C)

D = permutedims(A, [2, 3, 1])
println("The permuted direct")
display(D)

tmp = Vector(1:5)
display(tmp)
permute!(tmp, [3,5,1,2,4])
display(tmp)

[(1,2,3,4,5)...]

c = :( function test(x)
    x.a = 0.0
end )


@kernel function rearrange_kernel!(y, x, permutation)
    I = @index(Global, NTuple)
    permute_index = [I...]
    permute!(permute_index, [permutation...])
    y[permute_index...] = x[I...]
end

function rearrange!(y, x; permutation = Tuple(1:length(size(x))), ndrange = size(x), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(x, Array)
        kernel! = rearrange_kernel!(CPU(), cpu_threads)
    else
        kernel! = rearrange_kernel!(CUDA(), gpu_threads)
    end
    event = kernel!(y, x, permutation, ndrange = ndrange)
    return event
end

a1 = randn(3,3,3)
b1 = randn(3,3,3)

display(a1)
display(b1)

event = rearrange!(a1, b1)
wait(event)
a1 - b1


a = randn(3,4,5)
b = randn(5,4,3)
display(b)
event = rearrange!(a, b, permutation = (3, 2, 1))

wait(event)

###
###
@kernel function rearrange_kernel_2!(y, @Const(x), permutation)
    linear_I = @index(Global)
    tuple_I = @index(Global, NTuple)
    permute_I = [tuple_I...]
    permute!(permute_I, [permutation...])
    y[linear_I] = x[permute_I...]

end

function rearrange_2!(y, x; permutation = Tuple(1:length(size(x))), ndrange = size(x), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(x, Array)
        kernel! = rearrange_kernel_2!(CPU(), cpu_threads)
    else
        kernel! = rearrange_kernel_2!(CUDA(), gpu_threads)
    end
    permuted_ndrange = [ndrange...]
    permute!(permuted_ndrange, [permutation...])
    event = kernel!(y, x, permutation, ndrange = Tuple(permuted_ndrange)) #needs to be a tuple?
    return event
end


###

a1 = randn(3,3,3)
b1 = randn(3,3,3)

event = rearrange_2!(a1, b1)
wait(event)
println(norm(a1 - b1))


a = randn(3,4,5)
b = randn(5,4,3)
event = rearrange_2!(a, b, permutation = (3, 2, 1))
wait(event)
println(a[1,3,4] - b[4,3,1])
c = permutedims(b, (3,2,1))
norm(c-a)

c = zeros(length(b))
event = rearrange_2!(c, b, permutation = (3, 2, 1))
wait(event)
norm(c) - norm(b)
b[1,1,1]

###
x = a
y = b
permutation = (3,2,1)

permuted_ndrange = [size(x)...]
permute!(permuted_ndrange, [permutation...])



###
cc = 4
a = randn(3*8*cc,4*8*cc,5*8*cc)
b = randn(5*8*cc,4*8*cc,3*8*cc)
event = rearrange_2!(a, b, permutation = (3, 2, 1))
wait(event)
println(a[1,3,4] - b[4,3,1])
c = permutedims(b, (3,2,1))
norm(c-a)
#=
@benchmark wait(rearrange_2!(a, b, permutation = (3, 2, 1)))
@benchmark permutedims(b, (3,2,1))

@benchmark permutedims!(a, b, (3,2,1))

@benchmark c[:] .= b[:]
=#

function convert_structure!(x, y, reshape_tuple, permute_tuple)
    alias_y = reshape(y, reshape_tuple)
    permute_y = permutedims(alias_y, permute_tuple)
    x[:] .= permute_y[:]
    return nothing
end

convert_structure!(a, b, size(b), (3, 2, 1))
