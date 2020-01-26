include("../sandbox/oceananigans_converter.jl")
const save_figure = true
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
total_set = 1:(n-1)
training_set = 1:4:(n-1) # 25% of the data, but the entire interval
verification_set = setdiff(total_set, training_set)

x_data = x[training_set]
y_data = y[training_set]
# these are the hyperparameter nobs
const γ1 = 0.0001
const σ1 = 1.0
k(x,y) = σ1 * exp(- γ1 * norm(x-y)^2 )
d(x,y) = norm(x-y)^2
cc = closure_guassian_closure(d, hyperparameters = [γ1,σ1])
𝒢 = construct_gpr(x_data, y_data, k)

index_check = 1
y_prediction = prediction([x_data[index_check]], 𝒢)
norm(y_prediction - y_data[index_check])


error = collect(verification_set)*1.0
# greedy check
for j in eachindex(verification_set)
    test_index = verification_set[j]
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
###
# the true check
# time evolution given the same initial condition
n = length(data.t)
set = 1:(n-400)
gpr_prediction = similar(y[total_set])
starting = x[1]
gpr_prediction[1] = starting
n = length(y[set])
for i in set
    gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
end
animation_set = 1:50:(n-400)
anim = @animate for i in animation_set
    exact = data.T[:,i+1]
    day_string = string(floor(Int, data.t[i]/86400))
    p1 = scatter(gpr_prediction[i+1], zavg, label = "GP")
    plot!(exact,data.z, legend = :topleft, label = "LES", xlabel = "temperature", ylabel = "depth", title = "day " * day_string)
    display(p1)
end
if save_figure == true
    gif(anim, pwd() * "/figures/gp_emulator.gif", fps = 60)
end
