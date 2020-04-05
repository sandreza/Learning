function gmres(linear_operator!, x, b, max_iterations, threshold; n = length(x))
    m = max_iterations;
    Ax = similar(x)
    linear_operator!(Ax, x)
    r = b - Ax
    b_norm = norm(b)
    error = norm(r) / b_norm
    e = []
    push!(e, error)
    Q = zeros(n,m)
    sn = zeros(eltype(x), m)
    cs = copy(sn)
    el = copy(cs)
    el[1] = 1.0;
    r_norm = norm(r)
    Q[:,1] = r ./ r_norm
    β = r_norm .* el;

    for k in 1:m
        H[1:k+1, k] Q[:, k+1] = arnoldi(linear_operator!, Q, k)
        [H[1:k+1, k] cs[k] sn[k]] = givens_rotation(H[1:k+1, k], cs, sn, k)
        β[k+1] = -sn[k] * β[k]
        β[k] = cs[k] * β[k]
        error = abs(β[k+1]/b_norm)
        push!(e, error)
        if (error <= threshold)
            break;
        end
    end
    y = H[1:k, 1:k] \ β[1:k]
    x = x + Q[:, 1:k] * y
end

function arnoldi(linear_operator!, Q, k)
    q # define me as appropriate length
    linear_operator!(q, Q[:,k])
    for i in 1:k
        h[i] = q' * Q[:,i]
        q = q - h[i] * Q[:,i]
    end
    h[k+1] = norm(q)
    q = q / h[k+1]
end

function givens_rotation(h, cs, sn, k)
    for i in 1:k-1
        tmp = cs[i] * h[i] + sn[i] * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i] = tmp
    end
    [cs_k sn_k] = rotation(h[k], h[k+1])
    h[k] = cs_k * h[k] + sn_k * h[k+1]
    h[k+1] = 0.0
end

function rotation(v1,v2)
    if v1 == 0
        cs = 0;
        sn = 1;
    else
        t = sqrt(v1^2 + v2^2)
        cs = abs(v1) / t
        sn = cs * v2 / v1
    end
    return cs, sn
end
