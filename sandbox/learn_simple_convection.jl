include("../sandbox/oceananigans_converter.jl")

filename = "./data/high_res_general_strat_16_profiles.jld2"
data = OceananigansData(filename)

t = data.t
m,n = size(data.T)
gp = 16 #gridpoints
zavg = avg(data.z, gp)
x = [avg(data.T[:,j], gp) for j in 1:(n-1)]
y = [avg(data.T[:,j], gp) for j in 2:n]



###
n = length(t)
end_n = floor(Int, n/4)
end_n2 = floor(Int, n/2)
subsample = 1:10:end_n

x_data = x[subsample]
y_data = y[subsample]
const γ1 = 0.01
const σ1 = 1.0
k(x,y) = σ1 * exp(- γ1 * norm(x-y)^2 )
d(x,y) = norm(x-y)^2
cc = closure_guassian_closure(d, hyperparameters = [γ1,σ1])
𝒢 = construct_gpr(x_data, y_data, k)

index_check = 1
y_prediction = prediction([x_data[index_check]], 𝒢)
norm(y_prediction - y_data[index_check])

indices = end_n+1:1:end_n2
# indices = subsample
error = collect(indices)*1.0
for j in eachindex(indices)
    test_index = indices[j]
    y_prediction = prediction([x[test_index]], 𝒢)
    δ = norm(y_prediction - y[test_index])
    # println(δ)
    error[j] = δ
end
histogram(error)
println("The mean error is " * string(sum(error)/length(error)))
println("The maximum error is " * string(maximum(error)))


###
test_index = 100
gpr_y = prediction([x[test_index]], 𝒢)
norm(gpr_y - y[test_index])
scatter(gpr_y,zavg)
