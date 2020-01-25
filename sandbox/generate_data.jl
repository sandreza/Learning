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
@save "./data/Lorenz63.jld2" x y t
