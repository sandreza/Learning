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
Random.seed!(1235)
b = randn(n, ni) # rhs
x = randn(n, ni) # initial guess
A = randn((n,n, ni)) ./ sqrt(n) .* 1.0
gmres = multires
x += randn(n,ni) * 0.01 * maximum(abs.(x))
x_init = copy(x)
linear_operator!(x, x_init)
r_vector = b - x
ind_events = length(r_vector[1,:])
# First Initialize
event = initialize_arnoldi!(gmres, r_vector, ndrange = ind_events)
wait(event)
linear_operator!(gmres.sol, view(gmres.Q, :, 1, :))
event = arnoldi_update!(1, gmres, r_vector, ndrange = ind_events)
wait(event)
event = initialize_QR!(gmres, ndrange = ind_events)
wait(event)
event = update_QR!(gmres, 1, ndrange = ind_events)
wait(event)
###


###
