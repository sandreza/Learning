# From
# https://github.com/FugroRoames/RoamesNotebooks/blob/master/A%20practical%20introduction%20to%20metaprogramming%20in%20Julia.ipynb

using  BenchmarkTools

# The surface-level expression for `1 + 1`.
# Each `Expr` contains one "head" and many "args" (arguments).
# For the case of `:call`, the first argument is the function being called
dump(:(1 + 1))


# This is looking *inside* the + function, for `+(::Int64, ::Int64)`.
# Internally, this calls another (built-in) function called `Base.add_int`
@code_lowered 1 + 1


# Inference has marked the return type of `+(::Int64, ::Int64)` as `Int64`.
# Base.add_int is a built-in function and cannot be optimized further.
@code_typed 1 + 1


# Here is the LLVM assembly for adding two `i64`s, using LLVM's built-in `add` function.
@code_llvm 1 + 1

# This is the native x64 assembly for adding two signed 64-bit integers, using the `leaq` instruction.
@code_native 1 + 1

### Exercise
# sin is complicated
@code_native sin(1.0)


### Create Runtime stuff

# Binary representation at the type level - no run-time data!

abstract type Bit; end

struct Zero <: Bit; end
struct One <: Bit; end

# OR and AND for two Bits

Base.:|(::Zero, ::Zero) = Zero()
Base.:|(::Zero, ::One)  = One()
Base.:|(::One,  ::Zero) = One()
Base.:|(::One,  ::One)  = One()

Base.:&(::Zero, ::Zero) = Zero()
Base.:&(::Zero, ::One)  = Zero()
Base.:&(::One,  ::Zero) = Zero()
Base.:&(::One,  ::One)  = One()

# For fun I'll add these
Base.:~(::One) = Zero()
Base.:~(::Zero) = One()

# 8 Bits make a Byte

struct Byte2
    bit1::Bit
    bit2::Bit
    bit3::Bit
    bit4::Bit
    bit5::Bit
    bit6::Bit
    bit7::Bit
    bit8::Bit
end

struct Byte{Bit1 <: Bit, Bit2 <: Bit, Bit3 <: Bit, Bit4 <: Bit, Bit5 <: Bit, Bit6 <: Bit, Bit7 <: Bit, Bit8 <: Bit}
    bit1::Bit1
    bit2::Bit2
    bit3::Bit3
    bit4::Bit4
    bit5::Bit5
    bit6::Bit6
    bit7::Bit7
    bit8::Bit8
end

function Base.:|(a::Byte2, b::Byte2)
    return Byte2(a.bit1 | b.bit1,
                 a.bit2 | b.bit2,
                 a.bit3 | b.bit3,
                 a.bit4 | b.bit4,
                 a.bit5 | b.bit5,
                 a.bit6 | b.bit6,
                 a.bit7 | b.bit7,
                 a.bit8 | b.bit8
                )
end

function Base.:&(a::Byte2, b::Byte2)
    return Byte2(a.bit1 & b.bit1,
                 a.bit2 & b.bit2,
                 a.bit3 & b.bit3,
                 a.bit4 & b.bit4,
                 a.bit5 & b.bit5,
                 a.bit6 & b.bit6,
                 a.bit7 & b.bit7,
                 a.bit8 & b.bit8
                )
end

function Base.:|(a::Byte, b::Byte)
    return Byte(a.bit1 | b.bit1,
                 a.bit2 | b.bit2,
                 a.bit3 | b.bit3,
                 a.bit4 | b.bit4,
                 a.bit5 | b.bit5,
                 a.bit6 | b.bit6,
                 a.bit7 | b.bit7,
                 a.bit8 | b.bit8
                )
end

function Base.:&(a::Byte, b::Byte)
    return Byte(a.bit1 & b.bit1,
                 a.bit2 & b.bit2,
                 a.bit3 & b.bit3,
                 a.bit4 & b.bit4,
                 a.bit5 & b.bit5,
                 a.bit6 & b.bit6,
                 a.bit7 & b.bit7,
                 a.bit8 & b.bit8
                )
end

function Base.:~(a::Byte)
    return Byte( ~a.bit1,
                 ~a.bit2,
                 ~a.bit3,
                 ~a.bit4,
                 ~a.bit5,
                 ~a.bit6,
                 ~a.bit7,
                 ~a.bit8
                )
end
###

# Testing the structure
byte1 = Byte(Zero(), Zero(), One(),  Zero(), Zero(), One(), One(),  One())

byte2 = Byte(Zero(), One(),  Zero(), Zero(), Zero(), One(), Zero(), One())

byte3 = byte1 | byte2

@code_typed byte1 | byte2

###
# Testing the structure 2, 10x slower
byte1 = Byte2(Zero(), Zero(), One(),  Zero(), Zero(), One(), One(),  One())

byte2 = Byte2(Zero(), One(),  Zero(), Zero(), Zero(), One(), Zero(), One())

byte3 = byte1 | byte2

@code_typed byte1 | byte2

###
unsafe_load(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(Byte{Zero,One,One,Zero,Zero,One,One,One}) + 0x28))
###
byte1 = Byte(Zero(), Zero(), One(),  Zero(), Zero(), One(), One(),  One())

@code_typed ~byte1

### Promotion system

@code_lowered +(1, 3.14159)

###
Base.promote_rule(::Type{Zero}, ::Type{One}) = Int
Base.convert(::Type{Int}, ::Zero) = false
Base.convert(::Type{Int}, ::One) =  true
Base.:+(b1::Bit, b2::Bit) = +(promote(b1, b2)...)

Zero() + One()

One() + Zero()

One() + One()

### Simplified Promotion
function promote2(a,b)
    T = promote_type2(typeof(a), typeof(b))
    return (convert(T, a), convert(T, b))
end

promote_type2(::Type{T}, ::Type{T}) where {T} = T

function promote_type2(::Type{T1}, ::Type{T2}) where {T1, T2}
    T3 = promote_rule(T1, T2)
    if T3===Union{}
        return promote_rule(T1, T2)
    else
        return T3
    end
end
