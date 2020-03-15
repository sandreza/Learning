
function f(x; α = 1.1)
        return α * x
end

function g(x; N = 10)
        for i in 1:N
                x = f(x)
        end
        return x
end

function loss(x; N = 10)
        return (g(x; N = N)-x)^2
end

loss(1.0)

∇loss = x -> ForwardDiff.gradient(loss, x);
###
# ∇loss(1.0)
a = [DualNumber(1.0, 1.0)]
loss(a[1])
for i in 1:100
        a[1] = a[1]- 0.1 * loss(a[1]).dx
end
# tada minimized!
