using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using JLD2, LinearAlgebra, Plots

include("../sandbox/oceananigans_converter.jl")
include("../sandbox/gaussian_process.jl")
const save_figure = true
filename = "./data/high_res_general_strat_16_profiles.jld2"
les = OceananigansData(filename)

t = les.t
m,n = size(les.T)
gp = 16 #gridpoints
z = avg(les.z, gp)
x = [avg(les.T[:,j], gp) for j in 1:(n-1)]
y = [avg(les.T[:,j], gp) for j in 2:n]

# now define training data
n = length(t)
total_set = 1:(n-1)
training_set = 1:4:(n-1) # 25% of the data, but the entire interval
verification_set = setdiff(total_set, training_set)
tp = length(training_set) #training points
X = randn((gp, tp))
Y = similar(X)

for i in 1:tp
    for j in 1:gp
        X[j,i] = x[training_set[i]][j]
        Y[j,i] = y[training_set[i]][j]-x[training_set[i]][j]
    end
end

# define NN structure
hls = 2^6 # hidden layer size
m = Chain(
  Dense(gp, hls, relu),
  Dense(hls, gp)
  )

# Flux.params(m).order[1] is the first matrix in the NN
# Flux.params(m).order[2] is the first bias in the NN
# Flux.params(m).order[3] is the second matrix in the NN
# Flux.params(m).order[4] is the second bias in the NN

const ampl = norm(X[:,1])^2
loss(x, y) = norm(m(x) - y)^2 / ampl


# train
###
dataset = repeated((X, Y), 10000)
evalcb = () -> @show(loss(X, Y))
opt = ADAM(10^(-4), (0.9, 0.8))
# opt = Descent(10^(-6))
Flux.train!(loss, Flux.params(m), dataset, opt, cb = throttle(evalcb, 10))

index = 10 #288 max
day_string = string(floor(Int,  t[training_set[index]]/86400))
loss(X[:,index], Y[:,index])
plot(Y[:,index] .+ X[:,index], z, label = "Truth", legend = :topleft,  xlabel = "Temperature", ylabel = "depth", title = "day " * day_string, xlims = (19,20), gridalpha = 0.25, framestyle = :box )
plot!(m(X[:,index]) .+ X[:,index] , z, label = "NN", xlabel = "Temperature", ylabel = "Depth")
