using Plots
using JLD2
include("../sandbox/gaussian_process.jl")
data = jldopen("./data/Lorenz63.jld2")
x = data["x"]
y = data["y"]
t = data["t"]
close(data)

n = length(t)
end_n = floor(Int, n/4)
end_n2 = floor(Int, n/2)
subsample = 1:10:end_n

x_data = x[subsample]
y_data = y[subsample]
const γ1 = 0.0001
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

x_lorenz = [x[i][1] for i in eachindex(x)]

plot(x_lorenz)

indices = end_n+1:1:end_n2
gpr_prediction = similar(y[indices])
starting = x[end_n+1]
gpr_prediction[1] = starting
n = length(y[indices])
for i in 1:(n-1)
    gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
end


x_lorenz_truth = [y[i][1] for i in indices]
x_lorenz_prediction = [gpr_prediction[i][1] for i in 1:n]
tvals = t[indices]
plot(tvals, x_lorenz_truth, label = "truth", xlabel = "time", ylabel = "x", title = "Lorenz equation prediction")
plot!(tvals, x_lorenz_prediction, label = "prediction")
