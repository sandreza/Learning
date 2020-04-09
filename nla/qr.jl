# # Gram-Schmidt
# Given a sequence of vectors, generate another sequence that spans the same space, but is orthogonal

# First load some packages

using LinearAlgebra, BenchmarkTools
timings = false
# We proceed in order. We start with vectors[1] as our generator. We normalize it and make the rest orthogonal to it
function gram_schmidt(vectors)
    # initialize new basis
    q = [vectors[i] for i in eachindex(vectors)]
    # initialize r
    m = length(vectors[1])
    n = length(vectors)
    r = zeros(eltype(vectors[1]), (m,n))
    for i in eachindex(vectors)
        q[i] = vectors[i]
        for j in 1:i-1
            r[j,i] = dot(q[j], vectors[i])
            q[i]  -= r[j,i] * q[j]
        end
        r[i,i] = norm(q[i])
        q[i] /= norm(q[i])
    end
    return q, r
end

# We proceed in order. We start with vectors[1] as our generator. We normalize it and make the rest orthogonal to it.
function modified_gram_schmidt(vectors)
    # initialize new basis
    q = [vectors[i] for i in eachindex(vectors)]
    # initialize r
    m = length(vectors[1])
    n = length(vectors)
    r = zeros(eltype(vectors[1]), (m,n))
    for i in eachindex(vectors)
        r[i,i] = norm(q[i])
        q[i] = q[i] / r[i,i]
        for j in i+1:length(vectors)
            r[i,j] = dot(q[i], vectors[j])
            q[j] -= r[i,j] * q[i]
        end
    end
    return q, r
end

# The differences between modified gram schmidt and regular
# only show up once one has 3 or more vectors
# The biggest difference for the 3rd vector is that
#```math
# w_3 = (I - q_2 q_2')*(I - q_1 q_1')*v_3
# q_3 = w_3 / norm(w_3)
#```
# as opposed to
#```math
# w_3 = (I - q_1 q_1' - q_2 q_2')*v_3
# q_3 = w_3 / norm(w_3)
#```
# in the usual procedure

# Now we do it the householder way
function householder(A)
    R = copy(A)
    m, n = size(R)
    v = [zeros(eltype(R), m-i+1) for i in 1:m]
    for j in 1:n
        x = R[j:m, j]
        v[j] .= x
        v[j][1] += norm(x) * sign(x[1])
        v[j] /= norm(v[j])
        R[j:m, j:n] -= 2.0 * v[j] * (v[j]' * R[j:m, j:n])
    end
    return R, v
end

# action of Q on vector x
function linear_operator_Q!(x, v)
    for i in length(v):-1:1
        x[i:length(v)] -= 2 * v[i] * dot(v[i], x[i:length(v)])
    end
end

function build_Q(v)
    x = zeros(length(v))
    Q = zeros(length(x), length(x))
    for i in eachindex(v)
        x .= 0.0
        x[i] = 1.0
        linear_operator_Q!(x, v)
        Q[:, i] .= x
    end
    return Q
end

# # Test
#
# Here we will assume that the data is of the form
#```julia
# data[i] = vector
#
#```

n = 300
vectors = [randn(n) for i in 1:n]

Q, R = gram_schmidt(vectors)
Q2, R2 = modified_gram_schmidt(vectors)

function make_matrix(Q)
    m = length(Q[1])
    n = length(Q)
    matQ = zeros(eltype(Q[1]), (m,n))
    for j in 1:n
        matQ[:,j] .= Q[j]
    end
    return matQ
end


mQ = make_matrix(Q)
mQ2 = make_matrix(Q2)
mV = make_matrix(vectors)

hR, hV = householder(mV)
hQ = build_Q(hV)

# The following is should be machine precision
norm(mQ * R - mV) / norm(mV)
norm(mQ2 * R2 - mV) / norm(mV)
norm(hQ * hR - mV) / norm(mV)


###
# timings
if timings
println("GS")
@btime Q, R = gram_schmidt(vectors);
println("mGS")
@btime Q2, R2 = modified_gram_schmidt(vectors);
println("householder")
@btime hR, hV = householder(mV);
println("qr")
@btime qrV = qr(mV);
end
