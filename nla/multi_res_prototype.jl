using LinearAlgebra, KernelAbstractions

"""
MultiRes{𝒮,𝒯,𝒱}
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
function MultiRes(b, m, threads; restart = m, permute_f = [])
    residual = zeros(eltype(b), (m, threads))
    k = restart
    sol = zeros(eltype(b), (m, threads))
    rhs = zeros(eltype(b), (k+1, threads))
    # Q_permute = restructure_vec(Q, permute_f)
    # b_permute = copy(Q_permute)
    Q_permute = zeros(eltype(b), (m, threads))
    b_permute = zeros(eltype(b), (m, threads))
    cs = zeros(eltype(b), 2 * k, threads)
    Q = zeros(eltype(b), (m, k+1 , threads))
    H = zeros(eltype(b), (k+1, k, threads))
    R  = zeros(eltype(b), (k+1, k, threads))
    # The k+1 are not bugs, it is just more convenient to make them a slight bit larger
    # create permute_b (For permute back) here
    # the other is permute forward
    container = [
        restart,
        residual,
        Q_permute,
        b_permute,
        sol,
        rhs,
        cs,
        H,
        Q,
        R
    ]
    return MultiRes(container...)
end

###
"""
initialize_arnoldi_kernel!(g, b)

# Description
- First step of Arnoldi Iteration is to define first Krylov vector. Additionally sets things equal to zero

# Arguments
- `g`: (struct) [OVERWRITTEN] the gmres struct
- `b`: (array) The right hand side

# Return
nothing
"""
@kernel function initialize_arnoldi_kernel!(g, b)
    I = @index(Global)
    # set everything to zero to be sure
    g.rhs[:, I] .= 0.0
    g.Q[:,:, I] .= 0.0
    g.R[:,:, I] .= 0.0
    g.cs[:,  I] .= 0.0
    g.sol[:, I] .= 0.0
    g.H[:,:, I] .= 0.0
    g.Q_permute[:, I] .= 0.0
    g.b_permute[:, I] .= 0.0
    # now start computations
    # restructure_vec!(b_permute, b, permute_f)
    g.rhs[1, I]   = norm(b[:, I]) # for later
    g.Q[:, 1, I] .= b[:,I] / g.rhs[1, I] # First Krylov vector
end

"""
initialize_arnoldi!(g, b)

# Description
- First step of Arnoldi Iteration is to define first Krylov vector. Additionally sets things equal to zero. Calls the initial_arnoldi_kernel!

# Arguments
- `g`: (struct) [OVERWRITTEN] the gmres struct
- `b`: (array) The right hand side

# Keyword Arguments
- `cpu_threads`: (int) number of cpu threads to use default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads to use. default = 256
- `ndrange`: (tuple or int) number of independent tasks. default = (1,)

# Return
event: a KernelAbstractions structure
"""
function initialize_arnoldi!(g,b; ndrange = (1,), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(b,Array)
        kernel! = initialize_arnoldi_kernel!(CPU(), cpu_threads)
    else
        kernel! = initialize_arnoldi_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(g, b, ndrange = ndrange)
    return event
end

###


###
"""
arnoldi_update_kernel!(n, g, linear_operator!, b)
# Description
Perform an Arnoldi iteration
# Arguments
- `n`: current iteration number
- `g`: gmres struct that gets overwritten
- `linear_operator!`: (function) Action of linear operator on vector
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
@kernel function arnoldi_update_kernel!(n::Int, g::MultiRes)
    I = @index(Global)

    @inbounds for j in 1:n
        g.H[j, n, I] = 0
        # dot products
        @inbounds for i in eachindex(g.sol[:,I])
            g.H[j, n, I] += g.Q[i,j, I] * g.sol[i, I]
        end
        # orthogonalize latest Krylov Vector
        @inbounds for i in eachindex(g.sol[:,I])
            g.sol[i, I] -= g.H[j, n, I] * g.Q[i,j, I]
        end
    end

    if n+1 <= length(g.sol[:,I])
        g.H[n+1, n, I] = norm(g.sol[:,I])
        g.Q[:, n+1, I] .= g.sol[:,I] ./ g.H[n+1, n, I]
    end
end

"""
arnoldi_update!
# Description
wrapper function around arnoldi_update_kernel! for specific architectures
# Arguments
- `n`: (int) current iteration number
- `g`: (struct) gmres struct that gets overwritten

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
- event. A KernelAbstractions object
"""
function arnoldi_update!(n::Int, g::MultiRes; ndrange = (1), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(b, Array)
        kernel! = arnoldi_update_kernel!(CPU(), cpu_threads)
    else
        kernel! = arnoldi_update_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(n, g, ndrange = ndrange)
    return event
end

###
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

###
"""
initialize_QR_kernel!(gmres::MultiRes)

# Description
initializes the QR decomposition of the UpperHesenberg Matrix

# Arguments
- `gmres`: (struct) [OVERWRITTEN] the gmres struct

# Return
nothing
"""
@kernel function initialize_QR_kernel!(gmres::MultiRes)
    I = @index(Global)
    gmres.cs[1, I] = gmres.H[1,1, I]
    gmres.cs[2, I] = gmres.H[2,1, I]
    gmres.R[1, 1, I] = sqrt(gmres.cs[1, I]^2 + gmres.cs[2, I]^2)
    gmres.cs[1, I] /= gmres.R[1,1, I]
    gmres.cs[2, I] /= -gmres.R[1,1, I]
end

"""
initialize_QR!(gmres::MultiRes; ndrange = (1,), cpu_threads = Threads.nthreads(), gpu_threads = 256)

# Description
utilizes initialize_QR_kernel! to initialize the QR decomposition for the upper hessenberg matrix

# Arguments
- `gmres`: (struct) [OVERWRITTEN]

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256
# Return
- event. A KernelAbstractions object
"""
function initialize_QR!(gmres::MultiRes; ndrange = (1,), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.cs, Array)
        kernel! = initialize_QR_kernel!(CPU(), cpu_threads)
    else
        kernel! = initialize_QR_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(gmres, ndrange = ndrange)
    return event
end

###
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
@kernel function update_QR_kernel!(gmres::MultiRes, n)
    I = @index(Global)
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

end


"""
update_QR!

# Description
wrapper function update_QR_kernel!

# Arguments
- `gmres`: (struct) [OVERWRITTEN]
- `n`: (int) iteration number

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256
"""
function update_QR!(gmres::MultiRes, n; ndrange = (1,), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.sol, Array)
        kernel! = update_QR_kernel!(CPU(), cpu_threads)
    else
        kernel! = update_QR_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(gmres::MultiRes, n, ndrange = ndrange)
    return event
end


"""
solve_optimization_kernel!(iteration, gmres)
# Description
Creates the kernel for solving the optimization problem in GMRES
# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]
# Return
nothing
"""
@kernel function solve_optimization_kernel!(n, gmres::MultiRes)
    I = @index(Global)
    # just need to update rhs from previous iteration
    tmp1 = gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n, I] - gmres.cs[2*n, I] * gmres.rhs[n+1, I]
    gmres.rhs[n+1, I] = gmres.cs[2*n, I] * gmres.rhs[n, I] + gmres.cs[1 + 2*(n-1), I] * gmres.rhs[n+1, I]
    gmres.rhs[n, I] = tmp1

    # gmres.rhs[iteration+1] is the residual
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
end
###

"""
solve_optimization!
# Description
Uses the solve_optimization_kernel! for solving the optimization problem in GMRES

# Arguments
- `iteration`: (int) current iteration number
- `gmres`: (struct) [OVERWRITTEN]

# Keyword Arguments
- `ndrange`: (int) or (tuple) thread structure to iterate over
- `cpu_threads`: (int) number of cpu threads. default = Threads.nthreads()
- `gpu_threads`: (int) number of gpu threads. default = 256

# Return
event. A KernelAbstractions object
"""
function solve_optimization!(n, gmres::MultiRes; ndrange = (1,), cpu_threads = Threads.nthreads(), gpu_threads = 256)
    if isa(gmres.sol,Array)
        kernel! = solve_optimization_kernel!(CPU(), cpu_threads)
    else
        kernel! = solve_optimization_kernel!(GPU(), gpu_threads)
    end
    event = kernel!(n, gmres, ndrange = ndrange)
    return event
end

###
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
function solve!(x, b, linear_operator!, gmres::MultiRes; iterations = length(b), residual = false)
    x_init = copy(x)
    # TODO: make linear_operator! work with CLIMA
    linear_operator!(x, x_init)
    r_vector = b - x
    # independent events
    ind_events = length(r_vector[1,:])
    # First Initialize
    # TODO: fuse initialization
    event = initialize_arnoldi!(gmres, r_vector, ndrange = ind_events)
    wait(event)
    linear_operator!(gmres.sol, view(gmres.Q, :, 1, :))
    event = arnoldi_update!(1, gmres, ndrange = ind_events)
    wait(event)
    event = initialize_QR!(gmres, ndrange = ind_events)
    wait(event)
    event = update_QR!(gmres, 1)
    wait(event)
    event = solve_optimization!(1, gmres, ndrange = ind_events)
    wait(event)
    if residual
        residual_container = []
        push!(residual_container, abs.(gmres.rhs[2,:]))
    end
    # Now we can actually start on the iterations
    @inbounds for i in 2:iterations
        # TODO: make linear_operator! work with CLIMA
        linear_operator!(gmres.sol, view(gmres.Q, :, i, :))
        # TODO: fuse these
        event = arnoldi_update!(i, gmres, ndrange = ind_events)
        wait(event)
        event = update_QR!(gmres, i, ndrange = ind_events)
        wait(event)
        event = solve_optimization!(i, gmres, ndrange = ind_events)
        wait(event)
        if residual
            push!(residual_container, abs.(gmres.rhs[i+1,:]))
        end
    end
    # TODO: make this into a Kernel
    for i in 1:ind_events
        tmp = gmres.Q[:, 1:iterations, i] *  gmres.sol[1:iterations, i]
        x[:,i] .= x_init[:, i] + tmp
    end
    if residual
        return residual_container
    end
    return nothing
end

###
function closure_linear_operator_multi!(A, n)
    function linear_operator!(x, y)
        for i in 1:n
            mul!(view(x,:,i), view(A, :, :, i), view(y,:,i))
        end
    end
end
