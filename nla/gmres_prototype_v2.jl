using LinearAlgebra

"""
PrototypeRes{𝒮,𝒯,𝒱}

# Description
- A (less?) memory intensive struct for GMRES

# Members

- restart::𝒮 (int) number of GMRES iterations before nuking the subspace
- residual::𝒱 (vector) residual vector
- sol::𝒱 (vector) solution vector
- rhs::𝒱 (vector) rhs vector
- cs::𝒱 (vector) Sequence of Gibbs Rotation matrices in compact form
- H::𝒯 (array) Upper Hessenberg Matrix
- Q::𝒯 (array) Orthonormalized Krylov Subspace
- R::𝒯 (array) The R of the QR factorization of the UpperHessenberg matrix H

# Intended Use
- Solving linear systems iteratively
"""
struct ProtoRes{𝒮,𝒯,𝒱}
    restart::𝒮
    residual::𝒱
    sol::𝒱
    rhs::𝒱
    cs::𝒱
    H::𝒯  # A factor of two in memory can be saved here
    Q::𝒯
    R::𝒯 # A factor of two in memory can be saved here
end


"""
ProtoRes(Q; restart = length(Q))

# Description
- Constructor for the PrototypeRes class

# Arguments
- `Q`: (array) ; Represents solution

# Keyword Arguments
- `k`: (int) ; Default = length(Q) ; How many krylove subspace iterations we keep

# Return
- An instance of the ProtoRes class
"""
function ProtoRes(Q; restart = length(Q))
    residual = similar(Q)
    k = restart
    sol = similar(Q)
    rhs = zeros(eltype(Q), k+1)
    cs = zeros(eltype(Q), 2 * k)
    kQ = zeros(eltype(Q), (length(Q), k+1 ))
    H = zeros(eltype(Q), (k+1, k))
    R  = zeros(eltype(Q), (k+1, k))
    container = [
        restart,
        residual,
        sol,
        rhs,
        cs,
        H,
        kQ,
        R
    ]
    return ProtoRes(container...)
end



###
"""
arnoldi_update!(n, g, linear_operator!, b)

# Description
Perform an Arnoldi iteration

# Arguments
- `n`: current iteration number
- `g`: gmres struct that gets overwritten
- `linear_operator!`: (function) Action of linear operator on vector
- `b`: (vector). Initial guess

# Return
- nothing

# linear_operator! Arguments
- `linear_operator!(x,y)`
# Description
- Performs Linear operation on vector and overwrites it
# Arguments
- `x`: (array) [OVERWRITTEN]
- `y`: (array)
# Return
- Nothing
# Comment
There seems to be type instability here associated with a loop
"""
function arnoldi_update!(n::Int, g::ProtoRes, linear_operator!, b)
    if n==1
        # set everything to zero to be sure
        g.rhs .= 0.0
        g.Q .= 0.0
        g.R .= 0.0
        g.cs .= 0.0
        g.sol .= 0.0
        g.H .= 0.0
        # now start computations
        g.rhs[1] = norm(b) # for later
        g.Q[:,1] .= b / g.rhs[1] # First Krylov vector
    end
    linear_operator!(g.sol, g.Q[:,n])
    @inbounds for j in 1:n
        g.H[j, n] = 0
        @inbounds for i in eachindex(g.sol)
            g.H[j, n] += g.Q[i,j] * g.sol[i]
        end
        @inbounds for i in eachindex(g.sol)
            g.sol[i] -= g.H[j, n] * g.Q[i,j]
        end
    end
    if n+1 <= length(b)
        g.H[n+1, n] = norm(g.sol)
        g.Q[:, n+1] .= g.sol / g.H[n+1, n]
    end

    return nothing
end


"""
backsolve!(vector, matrix, n)

# Description
Recursively performs a backsolve

# Arguments
- `vector` (array) [OVERWRITTEN] the b in Ax = b gets overwitten with the solution x
- `matrix` (array) uppertriangular matrix for performing the backsolve

# Return
- Nothing

# Comment
- using vector[1:(n-1)] .-= matrix[1:(n-1),n] * vector[n]
instead of a for loop lead to a code that was 3 times slower
and had much more memory allocation
"""
function backsolve!(vector, matrix, n)
    vector[n] /= matrix[n,n]
    if n>1
        @inbounds for j in 1:n-1
            vector[j] -= matrix[j,n] * vector[n]
        end
        backsolve!(vector, matrix, n-1)
    end
    return nothing
end

"""
backsolve_2!(vector, matrix, n)

# Description
Performs a backsolve the usual way

# Arguments
- `vector` (array) [OVERWRITTEN] the b in Ax = b gets overwitten with the solution x
- `matrix` (array) uppertriangular matrix for performing the backsolve

# Return
- Nothing

# Comment
- this is slower than the recursive version
"""
function backsolve_2!(vector, matrix, n)
    @inbounds for i in n:-1:1
        vector[i] /= matrix[i,i]
        @inbounds for j in 1:i-1
            vector[j] -= matrix[j,i] * vector[i]
        end
    end
    return nothing
end

"""
apply_rotation!(vector, cs, n)

# Description
Apply sequences of givens rotation with compact representation given by cs

# Arguments
- `vector`: (vector) [OVERWITTEN]
- `cs`: (vector)
- `n`: (int)

# Return
Nothing
"""
function apply_rotation!(vector, cs, n)
    @inbounds for i in 1:n
        tmp1 = cs[1 + 2*(i-1)] * vector[i] - cs[2*i] * vector[i+1]
        tmp2 = cs[2*i] * vector[i] + cs[1 + 2*(i-1)] * vector[i+1]
        vector[i] = tmp1
        vector[i+1] = tmp2
    end
    return nothing
end


"""
update_QR!(gmres, n)

# Description
Given a QR decomposition of the first n-1 columns of an upper hessenberg matrix, this computes the QR decomposition associated with the first n columns

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the struct has factors that are updated
- `n`: (integer) column that needs to be updated

# Return
- nothing

# Comment
What is actually produced by the algorithm isn't the Q in the QR decomposition but rather Q^*. This is convenient since this is what is actually needed to solve the linear system

"""
function update_QR!(gmres::ProtoRes, n)
    if n==1
        gmres.cs[1] = gmres.H[1,1]
        gmres.cs[2] = gmres.H[2,1]
        gmres.R[1,1] = sqrt(gmres.cs[1]^2 + gmres.cs[2]^2)
        gmres.cs[1] /= gmres.R[1,1]
        gmres.cs[2] /= -gmres.R[1,1]
    else
        # Apply previous Q to new column
        gmres.R[1:n,n] .= gmres.H[1:n, n]
        apply_rotation!(view(gmres.R, 1:n, n), gmres.cs, n-1)
        # Now update
        gmres.cs[1+2*(n-1)] = gmres.R[n,n]
        gmres.cs[2*n] = gmres.H[n+1,n]
        gmres.R[n,n] = sqrt(gmres.cs[1+2*(n-1)]^2 + gmres.cs[2*n]^2)
        gmres.cs[1+2*(n-1)] /= gmres.R[n,n]
        gmres.cs[2*n] /= -gmres.R[n,n]
    end
    return nothing
end


"""
solve_optimization!(iteration, gmres)

# Description
Solves the optimization problem in GMRES

# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
- `b`: (array), rhs of lienar system
"""
function solve_optimization!(n, gmres)
    # just need to update rhs from previous iteration
    tmp1 = gmres.cs[1 + 2*(n-1)] * gmres.rhs[n] - gmres.cs[2*n] * gmres.rhs[n+1]
    gmres.rhs[n+1] = gmres.cs[2*n] * gmres.rhs[n] + gmres.cs[1 + 2*(n-1)] * gmres.rhs[n+1]
    gmres.rhs[n] = tmp1

    # note that gmres.rhs[iteration+1] is the residual
    gmres.sol[1:n] .= gmres.rhs[1:n]
    backsolve!(gmres.sol, gmres.R, n)
    return nothing
end


"""
solve!(x, b, linear_operator!, gmres; iterations = length(b), residual = false)

# Description
Solves a linear system using gmres

# arguments
- `x`: (array) [OVERWRITTEN] initial guess
- `b` (array) rhs
- `linear_operator!`: (function) represents action of linear oeprator. assumed arguments:: (x,y) where x gets overwritten
- `gmres`: (struct) the gmres struct that keeps track of krylov subspace information

# Keyword arguments
- `iterations`: (int) how many iterations to perform. DEFAULT = length(b)
- `residual`: (bool) whether or not to keep track of residual norm throughout the iterations. DEFAULT = false

# Return
- Nothing if keyword argument residual = false, otherwise returns an array of numbers corresponding to the residual at each iteration
"""
function solve!(x, b, linear_operator!, gmres::ProtoRes; iterations = length(b), residual = false)
    x_init = copy(x)
    linear_operator!(x, x_init)
    r_vector = b - x
    if residual
        r = zeros(eltype(x), iterations+1)
        r[1] = norm(r_vector)
    end
    @inbounds for i in 1:iterations
        arnoldi_update!(i, gmres, linear_operator!, r_vector)
        update_QR!(gmres, i)
        solve_optimization!(i, gmres)
        if residual
            r[i+1] = abs(gmres.rhs[i+1])
        end
    end
    tmp = gmres.Q[:, 1:iterations] *  gmres.sol[1:iterations]
    x .= x_init + tmp
    if residual
        return r
    else
        return nothing
    end
end

function closure_linear_operator!(A)
    function linear_operator!(x,y)
        mul!(x,A,y)
        return nothing
    end
    return linear_operator!
end
