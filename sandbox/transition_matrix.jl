# # Set up the transition matrix for the markov chain
using Plots, Statistics
# State space length
state_n = 17

# Setup the transition matrix
FT = typeof(1.0)
T = zeros(FT, (state_n, state_n))

p = 0.6 # probability of "winning"

# Define the matrix
T[1,1] = 1-p # losing at the bottom
T[2,1] = p   # winning

# Loop over other states
for i in 2:(state_n-1)
    T[i-1,i] = 1-p # losing
    T[i+1,i] = p   # winning
end

T[state_n, state_n] = 1 # once at top stay at top


# Define starting vector
s = zeros(FT, state_n)
s[1] = 1 # at the bottom

container_x = []
container_y = []
# number of games
anim = @animate for g in 1:5:300
    end_s = T^g * s
    # probability of reaching top
    println(end_s[end]);
    p1 = scatter(end_s, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box, legend = false, title = "Rank after " * string(g) * " games ", xlabel = "rank ", ylabel = "probability" , ylims = (-0.01, 1.0));
    push!(container_x, g)
    push!(container_y, end_s[end])
    p2 = plot(container_x, container_y, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box, legend = false, title = "winning probability = " * string(p), ylabel = " Highest rank probability ", xlabel = "games" , ylims = (-0.01, 1.0))
    display(plot(p1,p2));
end

gif(anim, pwd() * "/win_" * string(p) * ".gif", fps = 15)
