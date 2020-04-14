# The goal is to write an inefficient version of GMRES
# We can then compare the performance to an efficient version

using LinearAlgebra, BenchmarkTools, Polynomials


function closure_linear_operator!(A)
    function linear_operator!(x,y)
        mul!(x,A,y)
    end
    return linear_operator!
end

###
# # Arnoldi Iteration

"""
arnoldi!(linear_operator!, b; iterations = m)

# Description
Perform an Arnoldi iteration

# Arguments
- `A`: (array). Matrix represented in the standard basis
- `b`: (vector). Initial guess

# Keyword Arguments

- columns: default = length(b). Number of columns in the Hessenberg matrix

# Return
- `H`: (array). upper hessenber matrix
- `Q`: (array). orthonormal basis of Krylov Subspace
"""
function arnoldi!(linear_operator!, b; columns = length(b))
    m = columns
    Q = zeros(eltype(b), (length(b), m+1))
    # H = [zeros(eltype(A), 1 + i) for i in 1:m-1]
    H = zeros(eltype(b), (m+1, m))
    Q[:,1] .= b / norm(b) # First Krylov vector
    Aqⁿ = copy(b)
    for n in 1:m
        linear_operator!(Aqⁿ, Q[:,n])
        for j in 1:n
            H[j, n] = Q[:,j]' * Aqⁿ
            Aqⁿ -= H[j, n] * Q[:,j]
        end
        if n+1 <= length(b)
            H[n+1, n] = norm(Aqⁿ)
            Q[:, n+1] .= Aqⁿ / H[n+1, n]
        end
    end
    return Q, H
end

###
# First define the linear operator
n = 20
A = randn(n,n)
A = A' * A + 10I


b = randn(n)
x = randn(n)
linear_operator! = closure_linear_operator!(A)
linear_operator!(x,b)
norm(A*b - x)

residual = []
coeffs = []
estimate = []
for i in 1:length(b)
    Q, H = arnoldi!(linear_operator!, b, columns = i)
    # Note that Q here is Qⁿ⁺¹
    # norm(A * Q[:,1:end-1] - Q * H) # should be zero
    QH, RH = qr(H)
    rotated_b = Q' * b
    kc = H \ (rotated_b) # Krylov coefficients
    cg = Q[:, 1:end-1] * kc # current guess
    push!(coeffs, kc)
    push!(estimate, eigvals(H[1:i,1:i]))
    push!(residual, norm(A * cg - b) )
end

scatter(log.(residual) ./ log(10), legend = false)

eigA = eigvals(A)

for i in 1:length(b)
    xmax = maximum(real.(eigvals(A))) * 1.1 + 0.1
    xmin = minimum(real.(eigvals(A))) * 1.1 - 0.1
    ymax = maximum(imag.(eigvals(A))) * 1.1 + 0.1
    ymin = minimum(imag.(eigvals(A))) * 1.1 - 0.1
    p1 = scatter(real.(eigA), imag.(eigA), label = "Eigvals A", legend = :topright, xlabel = "Real", ylabel = "Imaginary", title = "Eigenvalues of operator", xlims = (xmin, xmax), ylims = (ymin, ymax), gridalpha = 0.25, framestyle = :box )
    p2 = scatter!(real.(estimate[i]), imag.(estimate[i]), marker = :x, color = :black, label = "approximation " * string(i))
    display(p2)
    sleep(0.0)
end

###
i = length(b)-0
Q, H = arnoldi!(linear_operator!, b, columns = i)
# Note that Q here is Qⁿ⁺¹
# norm(A * Q[:,1:end-1] - Q * H) # should be zero
QH, RH = qr(H)
rotated_b = Q' * b
kc = H \ (rotated_b) # Krylov coefficients
cg = Q[:, 1:end-1] * kc # current guess

norm(A * Q[:,1:end-1] - Q * H)


norm(Q[:,1:end-1]' * Q[:,1:end-1] - I)

heatmap(log.(abs.(Q[:,1:end-1]' * Q[:,1:end-1] - I)))
