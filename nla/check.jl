using BenchmarkTools
include("gmres_prototype.jl")

function closure_linear_operator!(A)
    function linear_operator!(x,y)
        mul!(x,A,y)
        return nothing
    end
    return linear_operator!
end
###
n = 3 # size of vector space
b = randn(n) # rhs
x = randn(n) # initial guess
A = randn((n,n))
gmres = PrototypeRes(b)
linear_operator! = closure_linear_operator!(A)

println("Currently the operators are as follows")
println("Q")
display(gmres.Q)
println("H")
display(gmres.H)
linear_operator!(x, b)# residual
r = b .- x
printstuff = true
for i in 1:3
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
    # minimize(iteration, gmres, b)
end


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
tmp = Q1' * H2[1:2, 2]
v = [tmp[2]; H2[3,2]]
norm_v, Ω = gibbs_rotation(v)
newQ = zeros(size(Q1) .+ 1)
newQ[1:2,1:2] .= Q1'
newQ[3,3] = 1
newQ[2:3,:]  = Ω * newQ[2:3,:]
tmpQ[1:1,1:1] += I
tmpQ[2:3,2:3] .= Ω
# newQ[2:3,2:3]  = new * Ω'
# Apply rotation to last column
# Apply rotation last two entries
# Back solve


function solve_optimization(iteration, gmres, b)
    if iteration==1
        n = 1
        tmpKR, tmpKQ = gibbs_rotation(gmres.H[1:n+1,n])
        gmres.KR[1:n,n] .= tmpKR
        gmres.KQ[1:n+1,1:n+1]  = tmpKQ
        tmpv = [norm(b); 0]
        tmpv = gmres.KQ[1:n+1,1:n+1] * tmpv
        backsolve!(tmpv, gmres.KR[1:n,1:n], n)
        sol = gmres.Q[:,1:n] * tmpv[1:n]
    else
        n = iteration
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
    end
    return sol
end


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
n = 50
vec = randn(n)
mat = UpperTriangular(randn(n,n))
sol = copy(vec)
sol = mat \ vec
backsolve!(vec, mat, n)
norm(sol - vec) / norm(vec)
