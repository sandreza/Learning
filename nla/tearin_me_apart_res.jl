include("gmres_prototype.jl")
using LinearAlgebra, Plots, Random, BenchmarkTools
n = 4 # size of vector space
Random.seed!(1235)
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n)) ./ sqrt(n) .* 1.0 + 0.8I
x = A \ b
x += randn(n) * 0.01 * maximum(abs.(x))
linear_operator! = closure_linear_operator!(A)

###
# Breakin apart the different stages in solve!
stage_stop = length(b)
gmres = PrototypeRes(b)
# boiler
x_init = copy(x)
linear_operator!(x, x_init)
r_vector = b - x
record_sc = zeros(eltype(b), 2 * stage_stop)
# now for the core
arnoldi_update!(1, gmres, linear_operator!, r_vector)
tmpKR, tmpKQ = gibbs_rotation(gmres.H[1:2,1])
record_sc[1:2] = tmpKQ[1:2,1]
gmres.KR[1:1,1] .= tmpKR
gmres.KQ[1:2,1:2]  = tmpKQ
solve_optimization!(1, gmres, r_vector, x)

for i in 2:stage_stop-1
    arnoldi_update!(i, gmres, linear_operator!, r_vector)
    # Apply previous Q to new column
    n = i
    tmp = gmres.KQ[1:n, 1:n] * gmres.H[1:n, n]
    # Construct vector that needs to be rotated
    v = [tmp[n]; gmres.H[n+1,n]]
    # Now get new rotation for update
    norm_v, Ω = gibbs_rotation(v)
    record_sc[1 + 2*(i-1):2*i] = Ω[1:2,1]
    # Create new Q
    gmres.KQ[n+1,n+1] = 1.0
    gmres.KQ[n:n+1,:]  = Ω * gmres.KQ[n:n+1,:]
    # Create new R
    gmres.KR[1:n-1,n] = tmp[1:n-1]
    gmres.KR[n,n] = norm_v
    # The line above should be checked so as to be more efficient
    solve_optimization!(i, gmres, r_vector, x)
end

i = stage_stop
n = stage_stop
arnoldi_update!(i, gmres, linear_operator!, r_vector)
# Apply previous Q to new column
tmp = gmres.KQ[1:n, 1:n] * gmres.H[1:n, n]
# Construct vector that needs to be rotated
v = [tmp[n]; gmres.H[n+1,n]]
# Now get new rotation for update
norm_v, Ω = gibbs_rotation(v)
record_sc[1 + 2*(i-1):2*i] = Ω[1:2,1]
# Create new Q
gmres.KQ[n+1,n+1] = 1.0
gmres.KQ[n:n+1,:]  = Ω * gmres.KQ[n:n+1,:]
# Create new R
gmres.KR[1:n-1,n] = tmp[1:n-1]
gmres.KR[n,n] = norm_v
# The line above should be checked so as to be more efficient
solve_optimization!(i, gmres, r_vector, x)

x .= x_init + x

println(norm(x - A\b) / norm(x))
###
 gmres.KR[1,1] .- gmres.KQ[1:2,1:2] * gmres.H[1:2,1]



###
function apply_rotation!(vector, record_sc, n)
    for i in 1:n
        tmp1 = record_sc[1 + 2*(i-1)] * vector[i] - record_sc[2*i] * vector[i+1]
        tmp2 = record_sc[2*i] * vector[i] + record_sc[1 + 2*(i-1)] * vector[i+1]
        vector[i] = tmp1
        vector[i+1] = tmp2
    end
end
###
test_vec = zeros(stage_stop + 1)
test_vec[4] = 1.0
test_vec2 = copy(test_vec)

apply_rotation!(test_vec, record_sc, stage_stop)
