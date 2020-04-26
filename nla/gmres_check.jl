include("gmres.jl")
using LinearAlgebra, Plots, Random
n = 5 # size of vector space
ni = 3 # number of independent linear solves
Random.seed!(1235)
b = randn(n, ni) # rhs
x = randn(n, ni) # initial guess
A = randn((n,n, ni)) ./ sqrt(n) .* 1.0
gmres = ParallelGMRES(b)
