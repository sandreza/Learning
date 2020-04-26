using BenchmarkTools, Random, Plots
include("gmres_prototype.jl")

function closure_linear_operator!(A)
    function linear_operator!(x,y)
        mul!(x,A,y)
        return nothing
    end
    return linear_operator!
end
###
n = 10 # size of vector space
Random.seed!(1235)
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n)) ./ sqrt(n) + I
gmres = PrototypeRes(b)
linear_operator! = closure_linear_operator!(A)

println("Currently the operators are as follows")
println("Q")
display(gmres.Q)
println("H")
display(gmres.H)
linear_operator!(x, b)# residual
r = b .- x
printstuff = false
printstuff_2 = false
keep_residual = true
residual = []
for i in 1:n
    iteration = i
    # Step 1: Get the Arnoldi Update
    arnoldi_update!(iteration, gmres, linear_operator!, b)
    # Print stuff for debugging
    if printstuff
        println("After " * string(iteration) * " iteration(s) it is ")
        println("Q")
        display(gmres.Q)
        println("H")
        display(gmres.H)
    end
    # Step 2: Solve the minimization problem
    sol = solve_optimization(iteration, gmres, b)
    if printstuff_2
        println("After " * string(iteration) * " iteration(s) the norm of the residual is ")
        println("residual")
        println(r)
        println("H")
        display(gmres.H[i+1,i])
        println("The solution guestimate is")
        println(sol)
    end
    if keep_residual
        r = norm(A * sol - b)
        push!(residual, r)
    end
end
###
n = 200 # size of vector space
Random.seed!(1235)
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n)) ./ sqrt(n) + I
gmres = PrototypeRes(b)
linear_operator! = closure_linear_operator!(A)

r = solve!(x, b, linear_operator!, gmres; iterations = length(b), residual = true)
###
scatter(log.(r)/log(10), xlabel = "iteration", ylabel = "log10 residual", title = "gmres convergence", legend = false)
###
# At the end of iteration n we must solve the minimization problem
# ```math
#   min_y \| Q^{n+1} H^n y - b \|
# ```
# where $b \in \mathbb{R}^m, y \in \mathbb{R}^n$, $H^n \in \mathbb{R}^{n+1 \times n}, $ and $Q^{n+1} \in \mathbb{R}^{m \times n+1}$.
# Here we assume that $m >> n$ So that the minimization problem is nice
# Furthermore we know (based off of the Krylov subspace iteration)
# that Q' * b = \|b \| e_1 where $e_1$ is the unit vector with zeros in all the components except for the first one
# this is implicitly a vector in $n+1$
# The minimization problem is overconstrained since $y$ lies in a lower
# dimensional vector space (n instead of n+1). Thus we need to solve it in the least squares sense.
# One way to solve the least squares problem is to perform a
# QR factorization of the H matrix, multiply through by Q
# and then backsolve the resulting R factor
# The reason that this is potentially nice is that the QR
# factorization of H^n can be obtained via a minor update of the QR factorization of H^{n-1}

# did I need to do it this way? No
# is it cool? oh yeah ... (to me it is ...)
###
for counter in 1:3
    i = counter
    Q_ = Meta.parse("Q"*string(i))
    R_ = Meta.parse("R"*string(i))
    H_ = Meta.parse("H"*string(i))
    i_ = Meta.parse("$counter")
    @eval $H_ = gmres.H[1:$i_+1,1:$i_]
    @eval $Q_, $R_ = qr($H_)
end

###





###
n = 1
tmpKR, tmpKQ = gibbs_rotation(gmres.H[1:n+1,n])
gmres.KR[1:n,n] .= tmpKR
gmres.KQ[1:n+1,1:n+1]  = tmpKQ
tmpv = [norm(b); 0]
tmpv = tmpKQ * tmpv
backsolve!(tmpv, gmres.KR[1:n,1:n], n)
sol = gmres.Q[:,1:n] * tmpv[1:n]
###
n = 3
# Apply previous Q
tmp = gmres.KQ[1:n,1:n] * gmres.H[1:n, n]
v = [tmp[n]; gmres.H[n+1,n]]
# Now get new rotation for update
norm_v, Ω = gibbs_rotation(v)
# Create new Q
gmres.KQ[n+1,n+1] = 1.0
gmres.KQ[n:n+1,:]  = Ω * gmres.KQ[n:n+1,:]
# Create new R, (only add last column)
gmres.KR[1:n,n] = tmp
gmres.KR[n+1,n] = gmres.H[n+1,n]
gmres.KR[n:n+1, :] = Ω * gmres.KR[n:n+1, :] #CHECK THIS, perhaps only need last two ros
# Now that we have the QR decomposition, we solve
tmpv = gmres.KQ[1:n+1,1] * norm(b)
backsolve!(tmpv, gmres.KR[1:n,1:n], n)
sol = gmres.Q[:,1:n] * tmpv[1:n]

###
# Recursive backsolve check
n = 500
vec = randn(n)
mat = UpperTriangular(randn(n,n))
sol = copy(vec)
sol = mat \ vec
backsolve!(vec, mat, n)
norm(sol - vec) / norm(vec)

##
