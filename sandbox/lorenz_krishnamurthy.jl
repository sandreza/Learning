
function lorenz_krishnamurthy!(ṡ, s, params, t)
   # for convenience
   # slow variables
   u = s[1]
   v = s[2]
   w = s[3]
   # fast variables
   x = s[4]
   y = s[5]
   # parameters
   ε = params[1] # U / (f L √(1+b²)) , frequency separation parameter
   b = params[2] # √(gH) / (fL)

   # the Lorenz equations
   u̇ = - v * w + ε * b * v * y
   v̇ =   u * w - ε * b * u * y
   ẇ = - u * v
   ẋ = - 1/ε * y
   ẏ =   1/ε * x + b/ε * u * v

   # passing the arguments back in
   ṡ[1] = u̇
   ṡ[2] = v̇
   ṡ[3] = ẇ
   ṡ[4] = ẋ
   ṡ[5] = ẏ
   return nothing
end


###

using DifferentialEquations, Plots, JLD2, BenchmarkTools

U = 0.1 # [m/s] charecteristic velocity
f = 10^(-4) # [1/s] coriolis parameter
L = 10^4 #10^8 # [m] characteristic length, diameter of earth
H = 10^3 # [m] depth of ocean
g = 10   # [m/s²] gravitatonal acceleration
b = √(g*H) / (f*L) #dimensionless
ε = U / (f*L*√(1+b^2))


s0 = [1.0, 0, 1.0, 0.0, 0.1]
tspan = (0.0, 100.0)
p = [ε, b]
prob = ODEProblem(lorenz_krishnamurthy!, s0, tspan,p)
dt = p[1] * 0.1
sol  = solve(prob, Tsit5(), dt=dt, adaptive = false);

# Define plots, p1, p2, p3, p4, p5
for v in [:1, :2, :3, :4, :5]
   plabel = Meta.parse("p$v")
   @eval $plabel = plot(sol[1:10:end], vars = ($v))
end

plot(p1,p5)
