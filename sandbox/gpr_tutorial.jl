using Plots
using LinearAlgebra
#The goal of this julia script is to introduce one to GPR

#The first step will be to samples of a multidimensional Gaussian Distribution on a single axis; no correlation
n = 100 #number of generator gpr samples on axis
m = 4  #number of instances of GPR
gprvec = randn((n,m))
tval = collect(1:n)
p1 = plot(tval,gprvec, title = "Uncorrelated GP", xlabel = "time", ylabel = "position", grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box, legend = false, layout = 4)


#The first step will be to samples of a multidimensional Gaussian Distribution on a single axis; with correlation matrix
#first define formula for correlation matrix
function k(x,y)
    z = exp(-(x-y)^2 / 2^2)
    return z
end

n = 100 #number of generator gpr samples on axis
m = 4  #number of instances of GPR
gprvec = randn((n,m))
tval = collect(1:n)
kmat = ones((n,n))
for i in 1:n
    for j in 1:n
        kmat[i,j] = k(tval[i],tval[j])
    end
end #kernel matrix

heatmap(kmat, title = "Correlation Matrix")

L = cholesky(kmat) #need to take ``square root" of the matrix for the change of variables formula
gprvecs = L.L*gprvec    #smooth gpr vec

p2 = plot(tval, gprvecs, title = "Correlated GP", xlabel = "time", ylabel = "position", grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box, legend = false, layout = 4)

plot(p1,p2)

##
