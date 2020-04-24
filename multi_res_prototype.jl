using LinearAlgebra, KernelAbstractions

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
struct MultiRes{𝒮,𝒯,𝒱}
    restart::𝒮
    residual::𝒱
    b_permute::𝒱
    Q_permute::𝒱
    sol::𝒱
    rhs::𝒱
    cs::𝒱
    H::𝒯  # A factor of two in memory can be saved here
    Q::𝒯
    R::𝒯 # A factor of two in memory can be saved here
end


"""
MultiRes(Q; restart = length(Q))

# Description
- Constructor for the MultiRes class

# Arguments
- `Q`: (array) ; Represents solution

# Keyword Arguments
- `k`: (int) ; Default = length(Q) ; How many krylove subspace iterations we keep

# Return
- An instance of the ProtoRes class
"""
function MultiRes(Q, m, threads; restart = m, permute_f = [])
    residual = zeros(eltype(Q), (m, threads))
    k = restart
    sol = zeros(eltype(Q), (m, threads))
    rhs = zeros(eltype(Q), k+1, threads)
    cs = zeros(eltype(Q), 2 * k, threads)
    kQ = zeros(eltype(Q), (m, k+1 , threads))
    H = zeros(eltype(Q), (k+1, k, threads))
    R  = zeros(eltype(Q), (k+1, k, threads))
    Q_permute = restructure_vec(Q, permute_f)
    b_permute = copy(Q_permute)
    # create permute_b (For permute back) here
    # the other is permute forward
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
    return MultiRes(container...)
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
@kernel function arnoldi_update!(n::Int, g::ProtoRes, linear_operator!, @Const(b))
    I = @index(Global)
    if n==1
        # set everything to zero to be sure
        g.rhs[:, I] .= 0.0
        g.Q[:,:, I] .= 0.0
        g.R[:,:, I] .= 0.0
        g.cs[:,  I] .= 0.0
        g.sol[:, I] .= 0.0
        g.H[:,:, I] .= 0.0
        # now start computations
        restructure_vec!(b_permute, b, permute_f)
        g.rhs[1,I] = norm(b[:,I]) # for later
        g.Q[:,1,I] .= b[:,I] / g.rhs[1, I] # First Krylov vector
    end

    linear_operator!(g.sol, g.Q[:,n])

    @inbounds for j in 1:n
        g.H[j, n, I] = 0
        @inbounds for i in eachindex(g.sol)
            g.H[j, n, I] += g.Q[i,j, I] * g.sol[i, I]
        end
        @inbounds for i in eachindex(g.sol)
            g.sol[i, I] -= g.H[j, n, I] * g.Q[i,j, I]
        end
    end
    if n+1 <= length(b)
        g.H[n+1, n] = norm(g.sol)
        g.Q[:, n+1] .= g.sol / g.H[n+1, n]
    end
end


"""
backsolve!(vector, matrix, n)

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
@kernel function backsolve!(vector, matrix, n)
    I = @index(Global)
    @inbounds for i in n:-1:1
        vector[i, I] /= matrix[i,i, I]
        @inbounds for j in 1:i-1
            vector[j, I] -= matrix[j,i, I] * vector[i, I]
        end
    end
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
function apply_rotation!(vector, cs, n, I)
    @inbounds for i in 1:n
        tmp1 = cs[1 + 2*(i-1), I] * vector[i, I] - cs[2*i, I] * vector[i+1, I]
        vector[i+1, I] = cs[2*i, I] * vector[i, I] + cs[1 + 2*(i-1), I] * vector[i+1, I]
        vector[i, I] = tmp1
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
@kernel function update_QR!(gmres::ProtoRes, n)
    I = @index(Global)
    if n==1
        gmres.cs[1, I] = gmres.H[1,1, I]
        gmres.cs[2, I] = gmres.H[2,1, I]
        gmres.R[1,1, I] = sqrt(gmres.cs[1, I]^2 + gmres.cs[2, I]^2)
        gmres.cs[1, I] /= gmres.R[1,1, I]
        gmres.cs[2, I] /= -gmres.R[1,1, I]
    else
        # Apply previous Q to new column
        @inbounds for i in 1:n
            gmres.R[i, n, I] = gmres.H[i, n, I]
        end
        apply_rotation!(view(gmres.R, 1:n, n, I), view(gmres.cs,1:2*n, I), n-1, I)
        # Now update
        gmres.cs[1+2*(n-1), I] = gmres.R[n,n, I]
        gmres.cs[2*n, I] = gmres.H[n+1,n, I]
        gmres.R[n,n, I] = sqrt(gmres.cs[1+2*(n-1), I]^2 + gmres.cs[2*n, I]^2)
        gmres.cs[1+2*(n-1), I] /= gmres.R[n,n, I]
        gmres.cs[2*n, I] /= -gmres.R[n,n, I]
    end
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
@kernel function solve_optimization!(n, gmres)
    I = @index(Global)
    # just need to update rhs from previous iteration
    tmp1 = gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n, I] - gmres.cs[2*n, I] * gmres.rhs[n+1, I]
    gmres.rhs[n+1, I] = gmres.cs[2*n, I] * gmres.rhs[n, I] + gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n+1, I]
    gmres.rhs[n, I] = tmp1

    # note that gmres.rhs[iteration+1] is the residual
    for i in 1:n
        gmres.sol[i, I] = gmres.rhs[i, I]
    end
    # do the backsolve
    @inbounds for i in n:-1:1
        gmres.sol[i, I] /= gmres.R[i,i, I]
        @inbounds for j in 1:i-1
            gmres.sol[j, I] -= gmres.R[j,i, I] * gmres.sol[i, I]
        end
    end
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
