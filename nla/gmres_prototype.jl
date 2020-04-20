using LinearAlgebra

"""
PrototypeRes{𝒮,𝒯,𝒱}

# Description
- A memory intensive struct for GMRES

# Members

- restart::𝒮 (int) number of GMRES iterations before wiping the subspace
- residual::𝒱 (vector) residual vector
- H::𝒯 (array) Upper Hessenberg Matrix
- Q::𝒯 (array) Orthonormalized Krylov Subspace
- KQ::𝒯 (array) The Q of the QR factorization of operator in Krylov Subspace
- KR::𝒯 (array) The R of the QR factorization of operator in Krylov Subspace

# Intended Use
- Solving linear systems iteratively
"""
struct PrototypeRes{𝒮,𝒯,𝒱}
    restart::𝒮
    residual::𝒱
    Q::𝒯
    H::𝒯  # A factor of two in memory can be saved here
    KQ::𝒯
    KR::𝒯 # A factor of two in memory can be saved here
end


"""
PrototypeRes(Q; restart = length(Q))

# Description
- Constructor for the PrototypeRes class

# Arguments
- `Q`: (array) ; Represents solution

# Keyword Arguments
- `k`: (int) ; Default = length(Q) ; How many krylove subspace iterations we keep

# Return
- An instance of the PrototypeRes class
"""
function PrototypeRes(Q; restart = length(Q))
    residual = similar(Q)
    k = restart
    Q = zeros(eltype(Q), (length(Q), k+1 ))
    H = zeros(eltype(Q), (k+1, k))
    KQ = zeros(eltype(Q), (k+1, k+1))
    KR  = zeros(eltype(Q), (k+1, k))
    container = [
        restart,
        residual,
        Q,
        H,
        KQ,
        KR
    ]
    return PrototypeRes(container...)
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
function arnoldi_update!(n, g, linear_operator!, b)
    if n==1
        g.Q[:,1] .= b / norm(b) # First Krylov vector
    end
    Aqⁿ = copy(b)
    linear_operator!(Aqⁿ, g.Q[:,n])
    for j in 1:n
        g.H[j, n] = dot(g.Q[:,j], Aqⁿ)
        Aqⁿ -= g.H[j, n] * g.Q[:,j]
    end

    if n+1 <= length(b)
        g.H[n+1, n] = norm(Aqⁿ)
        g.Q[:, n+1] .= Aqⁿ / g.H[n+1, n]
    end
    return nothing
end


"""
gibbs_rotation(v)

# Description
Takes a vector v and finds a rotation matrix that produces the vector [norm_v; 0]

# Argument
- `v`: (vector) a vector with two components

# Return
- `norm_v`: magnitude of the vector v
- `Ω`: rotation matrix
"""
function gibbs_rotation(v)
    norm_v = sqrt(v[1]^2 + v[2]^2)
    c = v[1] / norm_v
    s = - v[2] / norm_v
    return norm_v, [c -s; s c]
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
update_QR!(gmres, n)

# Description
- Given a QR decomposition of the first n-1 columns of a matrix, this computes the QR decomposition associated with the first n columns

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the struct has factors that are updated
- `n`: (integer) column that needs to be updated

# Return
- nothing

# Comment
What is actually produced by the algorithm isn't the Q in the QR decomposition but rather Q^*. This is convenient since this is what is actually need to solve the linear system

"""
function update_QR!(gmres, n)
    if n==1
        tmpKR, tmpKQ = gibbs_rotation(gmres.H[1:2,1])
        gmres.KR[1:1,1] .= tmpKR
        gmres.KQ[1:2,1:2]  = tmpKQ
    else
        # Apply previous Q to new column
        tmp = gmres.KQ[1:n, 1:n] * gmres.H[1:n, n]
        # Construct vector that needs to be rotated
        v = [tmp[n]; gmres.H[n+1,n]]
        # Now get new rotation for update
        norm_v, Ω = gibbs_rotation(v)
        # Create new Q
        gmres.KQ[n+1,n+1] = 1.0
        gmres.KQ[n:n+1,:]  = Ω * gmres.KQ[n:n+1,:]
        # Create new R, (only add last column)
        gmres.KR[1:n,n] = tmp
        gmres.KR[n+1,n] = gmres.H[n+1,n]
        gmres.KR[n:n+1, :] = Ω * gmres.KR[n:n+1, :]
        # The line above should be checked so as to be more efficient
    end
    return nothing
end

"""
solve_optimization(iteration, gmres, b)

# Description
Solves the optimization problem in GMRES

# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct)
- `b`: (array), rhs of lienar system
"""
function solve_optimization(iteration, gmres, b)
    rhs = gmres.KQ[1:iteration+1,1] * norm(b)
    backsolve!(rhs, gmres.KR[1:iteration,1:iteration], iteration)
    sol = gmres.Q[:, 1:iteration] * rhs[1:iteration]
    return sol
end

"""
solve_optimization!(iteration, gmres, b, x)

# Description
Solves the optimization problem in GMRES

# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
- `b`: (array), rhs of lienar system
"""
function solve_optimization!(iteration, gmres, b, x)
    rhs = gmres.KQ[1:iteration+1,1] * norm(b)
    backsolve!(rhs, gmres.KR[1:iteration,1:iteration], iteration)
    sol = gmres.Q[:, 1:iteration] * rhs[1:iteration]
    x .= sol
    # mul!(x, gmres.Q[:, 1:iteration], rhs[1:iteration])
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
function solve!(x, b, linear_operator!, gmres; iterations = length(b), residual = false)
    x_init = copy(x)
    linear_operator!(x, x_init)
    r_vector = b - x
    if residual
        r = zeros(eltype(x), iterations+1)
        r[1] = norm(r_vector)
    end
    for i in 1:iterations
        iteration = i
        # Step 1: Get the Arnoldi Update
        arnoldi_update!(iteration, gmres, linear_operator!, r_vector)
        # Step 2: Update the QR decomposition
        update_QR!(gmres, iteration)
        # Step 3: Solve the minimization problem
        solve_optimization!(iteration, gmres, r_vector, x)
        # Record Resiual
        if residual
            r[i+1] = norm(A * x - r_vector)
        end
    end
    x .= x_init + x
    if residual
        return r
    else
        return nothing
    end
end
