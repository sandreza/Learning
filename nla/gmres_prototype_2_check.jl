include("gmres_prototype_v2.jl")
using LinearAlgebra, Plots, Random
n = 100 # size of vector space
Random.seed!(1235)
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n)) ./ sqrt(n) .* 1.0 + 0.8I
gmres2 = ProtoRes(b)
x = A\b
x += randn(n) * 0.01 * maximum(abs.(x))
linear_operator! = closure_linear_operator!(A)
r2 = solve!(x, b, linear_operator!, gmres2; iterations = length(b), residual = true)

###
# Now just plot the convergence rate
scatter(log.(r2)/log(10), xlabel = "iteration", ylabel = "log10 residual", title = "gmres convergence", legend = false)
