using KernelAbstractions, Random, LinearAlgebra
using BenchmarkTools, Adapt
using CUDAapi

@kernel function basic!(a)
    I = @index(Global)
    a[I] = 0
end

@kernel function basic_struct!(a)
    I = @index(Global)
    a.a[I] = 0
end

@kernel function basic_struct_composition!(a)
    I = @index(Global)
    a.a.a[I] = 0
end

function test1(a; cpu_threads = Threads.nthreads(), gpu_threads = 256, ndrange = size(a))
    if isa(a, Array)
        kernel! = basic!(CPU(), cpu_threads)
    else
        kernel! = basic!(CUDA(), gpu_threads)
    end
    return kernel!(a, ndrange = ndrange)
end

function test2(a; cpu_threads = Threads.nthreads(), gpu_threads = 256, ndrange = size(a.a))
    if isa(a.a, Array)
        kernel! = basic_struct!(CPU(), cpu_threads)
    else
        kernel! = basic_struct!(CUDA(), gpu_threads)
    end
    return kernel!(a, ndrange = ndrange)
end

function test3(a; cpu_threads = Threads.nthreads(), gpu_threads = 256, ndrange = size(a.a.a))
    if isa(a.a.a, Array)
        kernel! = basic_struct_composition!(CPU(), cpu_threads)
    else
        kernel! = basic_struct_composition!(CUDA(), gpu_threads)
    end
    return kernel!(a, ndrange = ndrange)
end

struct MyStruct{T}
    a::T
end

struct MyStruct2{T}
    a::T
end

Adapt.adapt_structure(to, x::MyStruct) = MyStruct(adapt(to, x.a))
Adapt.adapt_structure(to, x::MyStruct2) = MyStruct2(adapt(to, x.a))

if CUDAapi.has_cuda_gpu()
    ArrayType = CuArray
else
    ArrayType = Array
end

# no layers
a = ArrayType(randn(3))
println(a)
event  = test1(a)
wait(event)
println(a)
# one layer
a = ArrayType(randn(3))
b = MyStruct(a)
println(b.a)
event  = test2(b)
wait(event)
println(b.a)
# two layers
a = ArrayType(randn(3))
b = MyStruct(a)
c = MyStruct2(b)
println(c.a.a)
event  = test3(c)
wait(event)
println(c.a.a)
