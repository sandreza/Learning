# define some differential equations
function lorenz63!(ṡ, s, params, t)
   # for convenience
   x = s[1]
   y = s[2]
   z = s[3]
   σ = params[1]
   ρ = params[2]
   β = params[3]

   # the Lorenz equations from 1963
   ẋ = σ * (y-x)
   ẏ = - y - x * z + ρ * x
   ż = - β * z + x*y

   # passing the arguments back in
   ṡ[1] = ẋ
   ṡ[2] = ẏ
   ṡ[3] = ż

   return nothing
end
