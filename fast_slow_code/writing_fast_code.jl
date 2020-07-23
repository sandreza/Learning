# going through the https://biojulia.net/post/hardware/ stuff

### Start
# If you don't already have these packages installed, outcomment these lines and run it:

using StaticArrays
using BenchmarkTools

"Print median elapsed time of benchmark"
function print_median(trial)
    println("Median time: ", BenchmarkTools.prettytime(median(trial).time))
end
###
# Create a file
a = randn(10^3);
open("test_file","a") do io
   println(io,"a=",a)
end
# Open a file
function test_file(path)
    open(path) do file
        # Go to 1000'th byte of file and read it
        seek(file, 1000)
        read(file, UInt8)
    end
end

@time test_file("test_file")

# Randomly access data N times
function random_access(data::Vector{UInt}, N::Integer)
    n = rand(UInt)
    mask = length(data) - 1
    @inbounds for i in 1:N
        n = (n >>> 7) ⊻ data[n & mask + 1]
    end
    return n
end
# >>> is the bit shift operator
data = rand(UInt, 2^24)
@time random_access(data, 1000000);
###
# linear access as opposed to random access
function linear_access(data::Vector{UInt}, N::Integer)
    n = rand(UInt)
    mask = length(data) - 1
    for i in 1:N
        n = (n >>> 7) ⊻ data[i & mask + 1]
    end
    return n
end

print_median(@benchmark random_access(data, 4096))
print_median(@benchmark linear_access(data, 4096))

###
function alignment_test(data::Vector{UInt}, offset::Integer)
    # Jump randomly around the memory.
    n = rand(UInt)
    mask = (length(data) - 9) ⊻ 7
    GC.@preserve data begin # protect the array from moving in memory
        ptr = pointer(data)
        iszero(UInt(ptr) & 63) || error("Array not aligned")
        ptr += (offset & 63)
        for i in 1:4096
            n = (n >>> 7) ⊻ unsafe_load(ptr, (n & mask + 1) % Int)
        end
    end
    return n
end
data = rand(UInt, 256 + 8);
print_median(@benchmark alignment_test(data, 0))
print_median(@benchmark alignment_test(data, 60))

###
data = rand(UInt, 1 << 24 + 8)
print_median(@benchmark alignment_test(data, 10))
print_median(@benchmark alignment_test(data, 60))
###
memory_address = reinterpret(UInt, pointer(data))
@assert iszero(memory_address % 64)
###
struct AlignmentTest
    a::UInt32 # 4 bytes +
    b::UInt16 # 2 bytes +
    c::UInt8  # 1 byte = 7 bytes?
end

function get_mem_layout(T)
    for fieldno in 1:fieldcount(T)
        println("Name: ", fieldname(T, fieldno), "\t",
                "Size: ", sizeof(fieldtype(T, fieldno)), " bytes\t",
                "Offset: ", fieldoffset(T, fieldno), " bytes.")
    end
end

println("Size of AlignmentTest: ", sizeof(AlignmentTest), " bytes.")
get_mem_layout(AlignmentTest)
###
# View assembly code generated from this function call
function foo(x)
    s = zero(eltype(x))
    @inbounds for i in eachindex(x)
        s = x[i ⊻ s]
    end
    return s
end

# Actually running the function will immediately crash Julia, so don't.
@code_native foo(data)
###
# bitshifts are faster than integer division
divide_slow(x) = div(x,8)
divide_fast(x) = x >>> 3;
@btime divide_slow(16);
@btime divide_fast(16);

###
# Garbage collector
thing = [1,2,3]
thing = nothing # goodbye!
GC.gc()
###
# allocating versus non-allocating

function increment(x::Vector{<: Integer})
    y = similar(x)
    @inbounds for i in eachindex(x)
        y[i] = x[i] + 1
    end
    return y
end

function increment!(x::Vector{<: Integer})
    @inbounds for i in eachindex(x)
        x[i] = x[i] + 1
    end
    return nothing
end
data = rand(UInt, 2^10)
@btime increment(data)
@btime increment!(data)
###
# Stack vs Heap
abstract type AllocatedInteger end
struct StackAllocated <: AllocatedInteger
    x::Int
end
mutable struct HeapAllocated <: AllocatedInteger
    x::Int
end

@code_native HeapAllocated(1)
@code_native StackAllocated(1)
###
# compare sums
Base.:+(x::Int, y::AllocatedInteger) = x + y.x
Base.:+(x::AllocatedInteger, y::AllocatedInteger) = x.x + y.x

data_stack = [StackAllocated(i) for i in rand(UInt16, 10^6)]
data_heap = [HeapAllocated(i.x) for i in data_stack]

@btime sum(data_stack)
@btime sum(data_heap)

println("First object of data_stack: ", data_stack[1])
println("First data in data_stack array: ", unsafe_load(pointer(data_stack)), '\n')

println("First object of heap_stack: ", data_heap[1])
first_data = unsafe_load(Ptr{UInt}(pointer(data_heap)))
println("First data in data_stack array: ", first_data, '\n')
println("Data at address ", repr(first_data), ": ", unsafe_load(Ptr{HeapAllocated}(first_data)))
###
# Register and SIMD
