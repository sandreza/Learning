using LinearAlgebra, KernelAbstractions

"""
ParallelGMRES{𝒮,𝒯,𝒱}
# Description

Launches n independent GMRES solves

# Members
- atol::ℱ (float) absolute tolerance
- rtol::ℱ (float) relative tolerance
- m::𝒮 (int) size of vector in each independent instance
- n::𝒮 (int) number of independent GMRES
- k_n::𝒮 (int) Krylov Dimension for each GMRES. It is also the number of GMRES iterations before nuking the subspace
- residual::𝒱 (vector) residual values for each independent linear solve
- b::𝒱 (vector) permutation of the rhs
- x::𝒱 (vector) permutation of the initial guess
- sol::𝒱 (vector) solution vector, it is used twice. First to represent Aqⁿ (the latest Krylov vector without being normalized), the second to represent the solution to the linear system
- rhs::𝒱 (vector) rhs vector
- cs::𝒱 (vector) Sequence of Gibbs Rotation matrices in compact form. This is implicitly the Qᵀ of the QR factorization of the upper hessenberg matrix H.
- H::𝒯 (array) Upper Hessenberg Matrix
- Q::𝒯 (array) Orthonormalized Krylov Subspace
- R::𝒯 (array) The R of the QR factorization of the UpperHessenberg matrix H

# Intended Use
Solving n linear systems iteratively

# Comments on Improvement
- Allocates all the memory at once: Could improve to something more dynamic
- Too much memory in H and R struct: Could use a sparse representation to cut memory use in half (or more)
- Needs to perform a transpose of original data structure into current data structure: Could perhaps do a transpose free version, but the code gets a bit clunkier and the memory would no longer be coalesced for the heavy operations
"""
struct ParallelGMRES{ℱ, 𝒮, 𝒯, 𝒱}
    atol::ℱ
    rtol::ℱ
    m::𝒮
    n::𝒮
    k_n::𝒮
    residual::𝒱
    b::𝒱
    x::𝒱
    sol::𝒱
    rhs::𝒱
    cs::𝒱
    Q::𝒯
    H::𝒯  # A factor of two in memory can be saved here
    R::𝒯 # A factor of two in memory can be saved here
end

"""
ParallelGMRES(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))) )

# Description
Constructor for the ParallelGMRES struct

# Arguments
- `Qrhs`: (array) Array structure that linear_operator! acts on

# Keyword Arguments
- `m`: (int) size of vector space for each independent linear solve. This is assumed to be the same for each and every linear solve. DEFAULT = length(Qrhs[:,1])
- `n`: (int) number of independent linear solves, DEFAULT = length(Qrhs[1,:])
- `atol`: (float) absolute tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))
- `rtol`: (float) relative tolerance. DEFAULT = sqrt(eps(eltype(Qrhs)))

# Return
ParalellGMRES struct
"""
function ParallelGMRES(Qrhs; m = length(Qrhs[:,1]), n = length(Qrhs[1,:]), subspace_size = m, atol = sqrt(eps(eltype(Qrhs))), rtol = sqrt(eps(eltype(Qrhs))), ArrayType = Array)
    k_n = subspace_size
    residual = ArrayType(zeros(eltype(Qrhs), (k_n, n)))
    b = ArrayType(zeros(eltype(Qrhs), (m, n)))
    x = ArrayType(zeros(eltype(Qrhs), (m, n)))
    sol = ArrayType(zeros(eltype(Qrhs), (m, n)))
    rhs = ArrayType(zeros(eltype(Qrhs), (k_n + 1, n)))
    cs = ArrayType(zeros(eltype(Qrhs), (2 * k_n, n)))
    Q = ArrayType(zeros(eltype(Qrhs), (m, k_n+1 , n)))
    H = ArrayType(zeros(eltype(Qrhs), (k_n+1, k_n, n)))
    R  = ArrayType(zeros(eltype(Qrhs), (k_n+1, k_n, n)))
    container = [
        atol,
        rtol,
        m,
        n,
        k_n,
        residual,
        b,
        x,
        sol,
        rhs,
        cs,
        Q,
        H,
        R
    ]
    ParallelGMRES(container...)
end

# First define all the initialization funtions
"""
initialize_gmres_kernel!(gmres)

# Description
Initializes the gmres struct by calling
- initialize_arnoldi
- initialize_QR!
- update_arnoldi!
- update_QR
It is assumed that the first two krylov vectors are already constructed

# Arguments
- `gmres`: (struct) gmres struct

# Return
(implicitly) kernel abstractions function closure
"""
@kernel function initialize_gmres_kernel!(gmres::ParallelGMRES)
    I = @index(Global)
    # Now we initialize
    initialize_arnoldi!(gmres, I)
    update_arnoldi!(1, gmres, I)
    initialize_QR!(gmres, I)
    update_QR!(1, gmres, I)
    solve_optimization!(1, gmres, I)
end

"""
initialize_gmres!(gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Uses the initialize_gmres_kernel! for initalizing

# Arguments
- `gmres`: (struct) [OVERWRITTEN]

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function initialize_gmres!(gmres::ParallelGMRES; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = initialize_gmres_kernel!(CPU(), cpu_threads)
    else
        kernel! = initialize_gmres_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(gmres, ndrange = ndrange)
    return event
end


"""
initialize_arnoldi!(g, I)

# Description
- First step of Arnoldi Iteration is to define first Krylov vector. Additionally sets things equal to zero

# Arguments
- `g`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_arnoldi!(gmres::ParallelGMRES, I)
    # set (almost) everything to zero to be sure
    gmres.rhs[:, I] .= 0.0
    gmres.Q[:,:, I] .= 0.0
    gmres.R[:,:, I] .= 0.0
    gmres.cs[:,  I] .= 0.0
    gmres.H[:,:, I] .= 0.0
    # gmres.x was initialized as the initial x
    # gmres.sol was initialized right before this function call
    # gmres.b was initialized right before this function call
    gmres.sol[:, I] ./= norm(gmres.b[:, I])
    # now start computations
    gmres.rhs[1, I]   = norm(gmres.b[:, I]) # for later
    gmres.Q[:, 1, I] .= gmres.b[:, I] / gmres.rhs[1, I] # First Krylov vector
    return nothing
end

"""
initialize_QR!(gmres::ParallelGMRES, I)

# Description
initializes the QR decomposition of the UpperHesenberg Matrix

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the gmres struct
- `I`: (int) thread index

# Return
nothing
"""
@inline function initialize_QR!(gmres::ParallelGMRES, I)
    gmres.cs[1, I] = gmres.H[1,1, I]
    gmres.cs[2, I] = gmres.H[2,1, I]
    gmres.R[1, 1, I] = sqrt(gmres.cs[1, I]^2 + gmres.cs[2, I]^2)
    gmres.cs[1, I] /= gmres.R[1,1, I]
    gmres.cs[2, I] /= -gmres.R[1,1, I]
    return nothing
end

# The meat of gmres with updates that leverage information from the previous iteration
"""
update_arnoldi!(n, g, linear_operator!, b)
# Description
Perform an Arnoldi iteration update

# Arguments
- `n`: current iteration number
- `g`: gmres struct that gets overwritten
- `I`: (int) thread index
# Return
- nothing
# linear_operator! Arguments
- `linear_operator!(x,y)`
# Description
- Performs Linear operation on vector and overwrites it
# Arguments
- `y`: (array)
# Return
nothing

"""
@inline function update_arnoldi!(n, g::ParallelGMRES, I)
    @inbounds for j in 1:n
        g.H[j, n, I] = 0
        # dot products
        @inbounds for i in 1:g.m
            g.H[j, n, I] += g.Q[i,j, I] * g.sol[i, I]
        end
        # orthogonalize latest Krylov Vector
        @inbounds for i in 1:g.m
            g.sol[i, I] -= g.H[j, n, I] * g.Q[i,j, I]
        end
    end
    # just to prevent some indexing errors
    if n+1 <= g.m
        g.H[n+1, n, I] = norm(g.sol[:,I])
        g.Q[:, n+1, I] .= g.sol[:,I] ./ g.H[n+1, n, I]
    end
    return nothing
end

"""
update_QR!(gmres, n, I)

# Description
Given a QR decomposition of the first n-1 columns of an upper hessenberg matrix, this computes the QR decomposition associated with the first n columns
# Arguments
- `gmres`: (struct) [OVERWRITTEN] the struct has factors that are updated
- `n`: (integer) column that needs to be updated
- `I`: (int) thread index
# Return
- nothing

# Comment
What is actually produced by the algorithm isn't the Q in the QR decomposition but rather Q^*. This is convenient since this is what is actually needed to solve the linear system
"""
@inline function update_QR!(n, gmres::ParallelGMRES, I)
    # Apply previous Q to new column
    @inbounds for i in 1:n
        gmres.R[i, n, I] = gmres.H[i, n, I]
    end
    apply_rotation!(view(gmres.R, 1:n, n, I), gmres.cs, n-1, I)
    # Now update
    gmres.cs[1+2*(n-1), I] = gmres.R[n, n, I]
    gmres.cs[2*n, I] = gmres.H[n+1,n, I]
    gmres.R[n, n, I] = sqrt(gmres.cs[1+2*(n-1), I]^2 + gmres.cs[2*n, I]^2)
    gmres.cs[1+2*(n-1), I] /= gmres.R[n, n, I]
    gmres.cs[2*n, I] /= -gmres.R[n, n, I]
    return nothing
end

"""
solve_optimization!(iteration, gmres, I)

# Description
Solves the optimization problem in GMRES
# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
- `I`: (int) thread index
# Return
nothing
"""
@inline function solve_optimization!(n, gmres::ParallelGMRES, I)
    # just need to update rhs from previous iteration
    tmp1 = gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n, I] - gmres.cs[2*n, I] * gmres.rhs[n+1, I]
    gmres.rhs[n+1, I] = gmres.cs[2*n, I] * gmres.rhs[n, I] + gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n+1, I]
    gmres.rhs[n, I] = tmp1

    # gmres.rhs[iteration+1] is the residual
    gmres.residual[n, I] = abs.(gmres.rhs[n+1, I])

    # copy for performing the backsolve and saving gmres.rhs
    @inbounds for i in 1:n
        gmres.sol[i, I] = gmres.rhs[i, I]
    end

    # do the backsolve
    @inbounds for i in n:-1:1
        gmres.sol[i, I] /= gmres.R[i,i, I]
        @inbounds for j in 1:i-1
            gmres.sol[j, I] -= gmres.R[j,i, I] * gmres.sol[i, I]
        end
    end

    return nothing
end

"""
gmres_update_kernel!(i, gmres, I)

# Description
kernel that calls
- update_arnoldi!
- update_QR!
- solve_optimization!
Which is the heart of the gmres algorithm

# Arguments
- `i`: (int) interation index
- `gmres`: (struct) gmres struct
- `I`: (int) thread index

# Return
kernel object from KernelAbstractions
"""
@kernel function gmres_update_kernel!(i, gmres::ParallelGMRES)
    I = @index(Global)
    update_arnoldi!(i, gmres, I)
    update_QR!(i, gmres, I)
    solve_optimization!(i, gmres, I)
end


"""
gmres_update!(i, gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Calls the gmres_update_kernel!

# Arguments
- `i`: (int) iteration number
- `gmres`: (struct) gmres struct

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function gmres_update!(i, gmres; ndrange = gmres.n, cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = gmres_update_kernel!(CPU(), cpu_threads)
    else
        kernel! = gmres_update_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(i, gmres, ndrange = ndrange)
    return event
end

# Several utility functions
"""
apply_rotation!(vector, cs, n, I)
# Description
Apply sequences of givens rotation with compact representation given by cs
# Arguments
- `vector`: (vector) [OVERWITTEN]
- `cs`: (vector)
- `n`: (int)
- `I`: (int) thread index
# Return
Nothing
"""
@inline function apply_rotation!(vector, cs, n, I)
    @inbounds for i in 1:n
        tmp1 = cs[1 + 2*(i-1), I] * vector[i, I] - cs[2*i, I] * vector[i+1, I]
        vector[i+1, I] = cs[2*i, I] * vector[i, I] + cs[1 + 2*(i-1), I] * vector[i+1, I]
        vector[i, I] = tmp1
    end
    return nothing
end

"""
construct_solution_kernel!(i, gmres)

# Description
given step i of the gmres iteration, constructs the "best" solution of the linear system for the given Krylov subspace

# Arguments
- `i`: (int) gmres iteration
- `gmres`: (struct) gmres struct

# Return
kernel object from KernelAbstractions
"""
@kernel function construct_solution_kernel!(i, gmres)
    M, I = @index(Global, NTuple)
    tmp = zero(eltype(gmres.b))
    @inbounds for j in 1:i
        tmp += gmres.Q[M, j, I] *  gmres.sol[j, I]
    end
    gmres.x[M , I] += tmp # since previously gmres.x held the initial value
end


"""
construct_solution!(i, gmres; ndrange = size(gmres.x), cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
Calls construct_solution_kernel! for constructing the solution

# Arguments
- `i`: (int) iteration number
- `gmres`: (struct) gmres struct

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function construct_solution!(i, gmres; ndrange = size(gmres.x), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.b, Array)
        kernel! = construct_solution_kernel!(CPU(), cpu_threads)
    else
        kernel! = construct_solution_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(i, gmres, ndrange = ndrange)
    return event
end

"""
solve!(x, b, linear_operator!, gmres; iterations = gmres.k_n)
# Description
Solves a linear system using gmres
# arguments
- `x`: (array) [OVERWRITTEN] initial guess
- `b` (array) rhs
- `linear_operator!`: (function) represents action of linear oeprator. assumed arguments:: (x,y) where x gets overwritten
- `gmres`: (struct) the gmres struct that keeps track of krylov subspace information
# Keyword arguments
- `iterations`: (int) how many iterations to perform. DEFAULT = gmres.k_n
# Return
Nothing if keyword argument residual = false, otherwise returns an array of numbers corresponding to the residual at each iteration
"""
function solve!(x, b, linear_operator!, gmres::ParallelGMRES; iterations = gmres.k_n)
    x_init = copy(x)
    # TODO: make this line work with CLIMA
    gmres.x .= x # MODIFY THIS LINE
    # TODO: make linear_operator! work with CLIMA
    linear_operator!(x, x_init)
    r_vector = b - x
    # Save first candidate Krylov vector
    # TODO: make this work with CLIMA
    gmres.b .= r_vector # MODIFY THIS LINE
    # Save second candidate Krylov vector
    # TODO: make linear_operator! work with CLIMA
    linear_operator!(gmres.sol, r_vector) #MODIFY THIS LINE
    # Initialize
    event = initialize_gmres!(gmres)
    wait(event)
    # Now we can actually start on the iterations
    @inbounds for i in 2:iterations
        # TODO: make linear_operator! work with CLIMA
        linear_operator!(gmres.sol, view(gmres.Q, :, i, :))
        event = gmres_update!(i, gmres)
        wait(event)
    end
    event = construct_solution!(iterations, gmres)
    wait(event)
    # TODO: make the following line work with CLIMA
    # note that if we don't wait for the event to finish
    # the following line may not work as expected
    x .= gmres.x
    return nothing
end
