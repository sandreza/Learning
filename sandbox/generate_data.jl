using DifferentialEquations
using Plots
using JLD2
include("../sandbox/lorenz.jl")



u0 = [3.715076285062597, 5.131832744898209, 17.856701164345996] # a point on the attractor
tspan = (0.0,100.0)
p = [10.0, 28.0, 8/3]
prob = ODEProblem(lorenz63!,u0,tspan,p)
dt = 0.01
sol  = solve(prob, Tsit5(), dt=dt, adaptive = false); # AB3(), RK4(), Tsit5(), Heun()
# sol = solve(prob)

scatter(sol[1:100:end],vars=(1))

x = [sol.u[i][1] for i in eachindex(sol.u) ]
y = [sol.u[i][2] for i in eachindex(sol.u) ]
z = [sol.u[i][3] for i in eachindex(sol.u) ]
histogram(z)


x = sol.u[1:end-1]
y = sol.u[2:end]
t = sol.t
@save "./data/Lorenz63.jld2" x y p t


###
using DifferentialEquations
using Plots
using JLD2
include("../sandbox/lorenz.jl")


K = 10
u0 = [7.067136014191715, -6.363227559831556, 4.571307538330193, 6.226056927031731, 6.306426977360508, 0.3478399780159818, -1.2886231806357844, 3.3441916887501772, 5.470673623141801, 7.5514297305615115] # a point on the attractor
tspan = (0.0,100.0)
F = 10.0
p = (K, F)
prob = ODEProblem(lorenz96!, u0, tspan, p)
dt = 0.0001
sol  = solve(prob, Tsit5(), dt=dt, adaptive = false); # AB3(), RK4(), Tsit5(), Heun()
# sol = solve(prob)

plot(sol[1:100:end],vars=(1))

x = [sol.u[i][1] for i in eachindex(sol.u) ]
y = [sol.u[i][2] for i in eachindex(sol.u) ]
z = [sol.u[i][3] for i in eachindex(sol.u) ]
histogram(z)


x = sol.u[1:end-1]
y = sol.u[2:end]
t = sol.t
@save "./data/Lorenz96.jld2" x y p t
