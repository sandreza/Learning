include("gmres_prototype.jl")
using LinearAlgebra, Plots, Random
n = 100 # size of vector space
Random.seed!(1235)
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n)) ./ sqrt(n) .* 1.0 + I
gmres = PrototypeRes(b)
x = rand(n) * 0.0
linear_operator! = closure_linear_operator!(A)
r = solve!(x, b, linear_operator!, gmres; iterations = length(b), residual = true)

###
# Now just plot the convergence rate
scatter(log.(r)/log(10), xlabel = "iteration", ylabel = "log10 residual", title = "gmres convergence", legend = false)
