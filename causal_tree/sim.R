### Prepare for function load 

packages <- c("tidyverse", "FNN", "np")
for (i in seq_along(packages)) {
  if (!(packages[i] %in% rownames(installed.packages()))) {
    invisible(install.packages(packages[i]))
  }  
}
suppressMessages(library("tidyverse"))
suppressMessages(library("FNN"))
suppressMessages(library("np"))

### Actual function 

sim <- function(n, het_linear = FALSE, random_assignment = TRUE, k=5, kC=5,
                non_linearY = FALSE, non_linearD = FALSE, diff_error = TRUE,
                np_ps = FALSE,
                gamma0=1, gamma1=3, gamma2=0, gamma3=1,
                phi0=5, phi1=3, phi1p=5, phi2=0, phi3=5,
                sdeps0 = 1, sdeps1 = 1){
  x <- runif(n)
  eps0 <- rnorm(n)
  ifelse(diff_error, eps1 <- rnorm(n), eps1 <- eps0)
  #
  if(random_assignment){
    D <- rbinom(n,size=1,prob = 0.5)
    trueps <- rep(0.5,n)
  }
  else{
    D <- rep(0,n)
    if(non_linearD){
      prt <- function(y){
        (y-0.05)^2 -0.25*max(0.2, (abs(y) -0.05))^2 + 0.7*max(0.2, (abs(y)-0.6)^2)  
      }
      trueps <- sapply(x, prt)
      #trueps <- rep(0,n)
      #for(j in seq_along(x)){trueps[j] <- prt(x[j])}
    }
    else{
      prt <- function(y){y}
      trueps <- x
    }
    for(j in seq_along(D)){ D[j] <- rbinom(1, size=1, prob=prt(x[j])) }
  }
  if(non_linearY){
    nl <- function(z,const,t=1){const+3*z+20*sin(pi*z*t)}
    y1 <- nl(z=x,const=phi0,t=phi3) + eps1
    y0 <- nl(z=x,const=gamma0,t=gamma3) + eps0
    tc <- function(z){phi0 - gamma0 + 20*(sin(pi*z*phi3) - sin(pi*z))}
    truecate <- tc(x)
  }
  else{
    if(het_linear){
      y1 <- phi0 + phi1p*x + eps1
      truecate <- phi0-gamma0 + (phi1p-gamma1)*x
    }
    else{
      y1 <- phi0 + phi1*x + eps1
      truecate <- phi0 - gamma0
      }
    y0 <- gamma0 + gamma1*x + eps0
  }
  res <- tibble::tibble(Y0=y0,Y1=y1,X=x,D=D,Y_obs=rep(0,n),IntXD=x*D, CATE=truecate)
  res$Y_obs[res$D==1] <- res$Y1[res$D==1]
  res$Y_obs[res$D==0] <- res$Y0[res$D==0]
  #propensity score estimation
  res <- tibble::add_column(res, trueps = trueps)
  res <- tibble::add_column(res, pslog=glm(res$D~res$X, family = "binomial"(link="logit"))$fitted.values)
  res <- tibble::add_column(res, psprob=glm(res$D~res$X, family = "binomial"(link="probit"))$fitted.values)
  if(np_ps){
    res <- tibble::add_column(res, psnp = fitted(np::npreg(res$D~res$X, regtype = "ll", bwmethod = "cv.aic")))
  }
  else{}
  #
  #get nearest neighbors to do k-NN matching
  res <- tibble::add_column(res, Yi0hat=rep(0,n))
  res <- tibble::add_column(res, Yi1hat=rep(0,n))
  res$Yi0hat[res$D==0] <- res$Y_obs[res$D==0]
  res$Yi1hat[res$D==1] <- res$Y_obs[res$D==1]
  #
  Tg <- res$X[res$D==1]
  Cg <- res$X[res$D==0]
  
  Tgy <- res$Y_obs[res$D==1]
  Cgy <- res$Y_obs[res$D==0]
  
  nnTg <- FNN::get.knnx(data = Tg, query = Cg, k)
  
  nnCg <- FNN::get.knnx(data = Cg, query = Tg, k)
  
  esty0 <- rep(0,sum(res$D==1))
  for(j in 1:sum(res$D==1)){
    esty0[j] <- mean(Cgy[nnCg$nn.index[j,]])
  }
  
  esty1 <- rep(0,sum(res$D==0))
  for(j in 1:sum(res$D==0)){
    esty1[j] <- mean(Tgy[nnTg$nn.index[j,]])
  }
  
  res$Yi0hat[res$D==1] <- esty0
  res$Yi1hat[res$D==0] <- esty1
  
  res <- tibble::add_column(res, tauihat = res$Yi1hat-res$Yi0hat)
  res <- res %>% dplyr::select(-one_of("Yi1hat", "Yi0hat"))
  
  #
  #get k-nn for CATE eq 26 in JASA#
  nnTg2 <- FNN::get.knnx(data = Tg, query = res$X, kC)
  treat <- apply(matrix(data = Tgy[nnTg2$nn.index], dim(nnTg2$nn.index)), 1, mean)
  
  nnCg2 <- FNN::get.knnx(data = Cg, query = res$X, kC)
  cont <- apply(matrix(data = Cgy[nnCg2$nn.index], dim(nnCg2$nn.index)), 1, mean)
  #
  res <- tibble::add_column(res, tauhatknn = treat-cont)
  return(res)
}