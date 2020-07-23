using SymbolicUtils
# example from website
@syms x::Real y::Real z::Complex f(::Number)::Real

2*x^2 - y + x^2

f(sin(x)^2 + cos(x)^2) + z

r = @rule sinh(im * ~x) => sin(~x)

r(sinh(im * y))

simplify(cos(y)^2 + sinh(im * y)^2, RuleSet([r]))

###
using SymbolicUtils: symtype
@syms w z α::Real β::Real

symtype(w), symtype(z), symtype(α), symtype(β)
w isa Number
α isa Real

expr1 = α * sin(w)^2 + β * cos(z)^2
expr2 = α * cos(z)^2 + β * sin(w)^2

expr1 + expr2
showraw(expr1 + expr2)

###
# function like symbols
@syms f(x) g(x::Real, y::Real)::Real

f(z) + g(1,α) + sin(w)
sin(w) + f(z) + g(1, α)

g(2//5, g(1,β))
###
# Rule based rewrite
r1 = @rule ~x + ~x => 2 * (~x)

showraw(r1(sin(1+z) + sin(1+z)))

r1(sin(1+z) + sin(1+w)) === nothing
###
# segment variables
@rule(+(~~xs) => ~~xs)(x+y+z)

r2 = @rule ~x * +(~~ys) => sum(map(y -> ~x * y, ~~ys));

showraw(r2(2 * (w + w + α + β)))

###
# Ruleset
# won't work
showraw(r1(r2(2*(w+w+α+β))))
# won't work
rset = RuleSet([r1, r2])
showraw(rset(2*(w+w+α+β)))
# will work
showraw(rset(rset(2*(w+w+α+β))))
# convenience function
using SymbolicUtils: fixpoint
fixpoint(rset, 2 * (w + w + α + β), 0)
