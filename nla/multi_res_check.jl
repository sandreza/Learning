# include("multi_res_prototype.jl")
using LinearAlgebra, Plots, Random
n = 3 # size of vector space
ni = 2 # number of independent linear solves
Random.seed!(1235)
b = randn(n, ni) # rhs
x = randn(n, ni) # initial guess
A = randn((n,n, ni)) ./ sqrt(n) .* 1.0
multires = MultiRes(b, n, ni)
for i in 1:ni
    x[:,i] = A[:, :, i] \ b[:, i]
end
sol = copy(x)
x += randn(n,ni) * 0.01 * maximum(abs.(x))
linear_operator! = closure_linear_operator_multi!(A, ni)
r2 = solve!(x, b, linear_operator!, multires; iterations = length(b), residual = true)
###
tmpsol = multires.sol
tmprhs = multires.Q[:, n, :]
linear_operator!(tmpsol, tmprhs)

event = arnoldi_update!(1, multires, tmpsol, ndrange = length(multires.sol[:,1]))
wait(event)

###
