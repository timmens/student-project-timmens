plot_AW <- function(lambda){

param <- as.list(match.call())  
lambda. <- param$lambda

tau <- function(x1,x2, lambda = lambda.){
  #(1+(1+exp(-20*(x1-1/3)))^(-1))*(1+(1+exp(-20*(x2-1/3)))^(-1))
  (1+(1+exp(-lambda*(x1-1/3)))^(-1))*(1+(1+exp(-lambda*(x2-1/3)))^(-1))
}

x1 <- seq(-1,1, length= 50)
x2 <- x1
z <- outer(x1, x2, tau)
?outer
z[is.na(z)] <- 1

#outer(c(1,2),c(3,4))
wireframe(z, drape=T, col.regions=rainbow(100),
          xlab="x1", ylab="x2", zlab="tau")
}