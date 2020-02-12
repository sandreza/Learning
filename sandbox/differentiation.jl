using ForwardDiff, StaticArrays, BenchmarkTools


test_f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x);

x = rand(5)

test_g = x -> ForwardDiff.gradient(test_f, x);

#=
@show test_g(x)
@btime test_g(x)
@btime test_f(x)
=#


test_f1(x) = sum(sin, x)
# gradient assumes f maps to real numbers
# jacobian assumes output of f is an array
test_g1 = x -> ForwardDiff.gradient(test_f1, x);


x = 3.0
X = [x 1.0; 0 x]
sX = @SArray [x 1.0; 0 x]
test_f1(x)
test_g1([x])


@btime test_f1(x)
@btime test_f1(X)
@btime test_f1(sX)
tmp = [x]
@btime test_g1(tmp)


###
import Base.+, Base.*, Base./, Base.convert, Base.promote_rule

struct DualNumber{T} <: Number
    x::T
    dx::T
end

function *(x::DualNumber{T}, y::DualNumber{T}) where T
    return DualNumber{T}(x.x * y.x, x.dx * y.x + x.x * y.dx)
end

function +(x::DualNumber{T}, y::DualNumber{T}) where T
    return DualNumber{T}(x.x + y.x, x.dx + y.dx)
end

function /(x::DualNumber{T}, y::DualNumber{T}) where T
    return DualNumber{T}(x.x / y.x, (x.dx * y.x - x.x * y.dx)/(y.x + y.x))
end

a = DualNumber(3.0, 1.0)
b = DualNumber(1.0, 0.0)
convert(::Type{DualNumber{Float64}}, x::Float64) = DualNumber{Float64}(x, zero(x))
promote_rule(::Type{DualNumber{Float64}}, ::Type{Float64}) = DualNumber{Float64}
DualNumber(x) = DualNumber(x, zero(typeof(x)))
println(b / a)
