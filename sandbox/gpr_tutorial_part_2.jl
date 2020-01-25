using Plots
using LinearAlgebra
#The first step will be to samples of a multidimensional Gaussian Distribution on a single axis; no correlation
T = 9 #the time interval
n = 10 #number of generator gpr samples on axis
nn = 500 #number of grid points that we will interplote to
m = 3  #number of instances of GPR
gprvec = randn((n,m))
function uniform_grid(n,a,b)
    y = a .+ collect(0:(n-1))/(n-1) .* (b-a)
    return y
end
tval = uniform_grid(n, 0, T)
tvalf = uniform_grid(nn, 0, T)

#we can assume this correlation matrix (or we can change this out for whatever)
γ = 1.0 #a 'hyperparameter', the value 1.0 works well for this example. Try gamma = 10.0 and 0.1 to see more
σ = 500.0 #a 'hyperparameter', the value 500 works well for this example.
δ = 0.0
function k(x,y; σ=500.0, γ = 1.0, δ = 0.00)
    z = σ*exp(-(x-y)^2 / 2^2 / γ^2)
    #z = 0.5*np.exp(-np.abs(x-y) / 2 / gamma)
    #z = 0.1*np.abs((x-y))
    #z = -0.1*np.abs((x-y))**(0.5)
    if x==y
        z+= δ #take into account noise in the data, 0.05 is a good value
    else
        z += 0.0
    end
    return z
end

#the function that we will be interploting
function g(x)
    return sin(x+pi/1.4)
end

yval = g.(tval)
yvalf = g.(tvalf)

scatter(tval, yval, color = :red, title = "given data", xlabel = "time", ylabel = "position" )

#solve for the coefficients
kmat = ones((n,n))
for i in 1:n
    for j in 1:n
        kmat[i,j] = k(tval[i],tval[j])
    end
end #kernel matrix

scoeff = kmat \ yval #solve for the coefficients
#the new mean functions
function gpr_mean(tt , tval , scoeff)
    y = 0.0
    n = length(tval)
    for i in 1:n
        y += scoeff[i] * k(tt,tval[i])
    end
    return y
end


interpol = zeros(nn) #now construct interpolant on fine grid

for i in 1:nn
    interpol[i] = gpr_mean(tvalf[i],tval,scoeff)
end

#plot mean
scatter(tval, yval, color = :red, label = "given data", xlabel = "time", ylabel = "position", gridalpha = 0.25, framestyle = :box, legend = false )
plot!(tvalf, interpol, color = :blue, line = :dashed, label = "interpolant")


#now we construct the covariance matrix and obtain uncertainty estimates for the points in between.
tmpv = zeros((n,nn))
tmpv2 = zeros((n,nn))
covar = zeros((nn,nn))
var = zeros(nn)
for i in 1:nn
    for j in 1:n
        tmpv[j,i] = k(tvalf[i],tval[j]) #evaluate the kernel at this particular location
    end
    tmpv2[:,i] = kmat\tmpv[:,i]
    var[i] = k(tvalf[i],tvalf[i]) - dot(tmpv[:,i],tmpv2[:,i])
end

scatter(tval, yval, color = :red, title = "Interpolant and Data", xlabel = "time", ylabel = "position" , label = "data")
plot!(tvalf, interpol, color = :orange, line = :dashed, label = "uncertainty", ribbon = var, width = 0.1)
plot!(tvalf, interpol, color = :blue, line = :dashed, label = "interpolant", width = 1)


𝒢 = construct_gpr(tval, yval, k)
tmp = randn(length(tvalf))
cov = similar(tmp)
for i in 1:length(tmp)
    tmp[i] = gpr_mean(tvalf[i], 𝒢)
    cov[i] = gpr_covariance(tvalf[i], 𝒢)
end
plot(tvalf,tmp, ribbons = cov)
