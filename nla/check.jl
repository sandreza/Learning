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
for i in 1:1
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
# dimensional vector space. Thus we need to solve it in the least squares sense.
# One way to solve the least squares problem is to perform a
# QR factorization of the H matrix, multiply through by Q
# and then backsolve the resulting R factor
# The reason that this is potentially nice is that the QR
# factorization of H^n can be obtained via a minor update of the QR factorization of H^{n-1}
