simAW <- function(n, kC, d, sd = sqrt(1/2), smoothness=20){
  add <- FALSE
  D <- rbinom(n, size = 1, prob = 0.5)
  x <- matrix(runif(n*d), n, d)
  Int <- matrix(rep(0,n*d), n, d)
  for(j in seq_along(D)){Int[j,] <- D[j]*(x[j,])}
  eps0 <- rnorm(n, sd = sd)
  eps1 <- rnorm(n, sd = sd)
  #generate cate function.
  tau <- function(x1,x2){
    (1+(1+exp(-smoothness*(x1-1/3)))^(-1))*(1+(1+exp(-smoothness*(x2-1/3)))^(-1))
  }
  if(add){
    betas0 <- matrix(runif(n = d, min = 1, max = 30), nrow = d, ncol = 1)
    betas1 <- matrix(runif(n = d, min = 1, max = 30), nrow = d, ncol = 1)
    y0 <- x %*% betas0 + eps0
    y1 <- x %*% betas1 + eps1
  }
  else{
    y0 <- -0.5*(tau(x[,1], x[,2])) + eps0
    y1 <- 0.5*(tau(x[,1], x[,2])) + eps1  
  }
  Xs <- rep(0, d); nam <- rep(0,d)
  for(j in 1:d){
    Xs[j] <- paste("res$X",j, sep = "")
    nam[j] <- paste("I", j, sep="")
  }
  res <- data.frame(x)
  res <- as_tibble(res)
  Int_XD <- data.frame(Int)
  colnames(Int_XD) <- nam
  Int_XD <- as_tibble(Int_XD)
  res <- bind_cols(res, Int_XD)
  res <- add_column(res, D=D, Y0 = y0, Y1 = y1, Y_obs = rep(0,n))
  res$Y_obs[res$D==1] <- res$Y1[res$D==1]
  res$Y_obs[res$D==0] <- res$Y0[res$D==0]
  #logit + probit
  # fmla <- as.formula(paste("res$D ~ ", paste(Xs, collapse = "+")))
  # res <- add_column(res, pslog = glm(fmla, family = "binomial"(link="logit"))$fitted.values)
  # res <- add_column(res,psprob=glm(fmla, family = "binomial"(link="probit"))$fitted.values)
  #
  #res <- add_column(res, Yi0hat=rep(0,n))
  #res <- add_column(res, Yi1hat=rep(0,n))
  #res$Yi0hat[res$D==0] <- res$Y_obs[res$D==0]
  #res$Yi1hat[res$D==1] <- res$Y_obs[res$D==1]
  #
  allX <- grep("^[X]", names(res), value=TRUE)
  
  #k-NN for ATE
   Tg <- res[res$D==1, allX]
   Cg <- res[res$D==0, allX]
  # 
   Tgy <- res$Y_obs[res$D==1]
   Cgy <- res$Y_obs[res$D==0]
  # 
  # nnTg <- get.knnx(data = Tg, query = Cg, k)
  # 
  # nnCg <- get.knnx(data = Cg, query = Tg, k)
  # 
  # esty0 <- rep(0,sum(res$D==1))
  # for(j in 1:sum(res$D==1)){
  #   esty0[j] <- mean(Cgy[nnCg$nn.index[j,]])
  # }
  # 
  # esty1 <- rep(0,sum(res$D==0))
  # for(j in 1:sum(res$D==0)){
  #   esty1[j] <- mean(Tgy[nnTg$nn.index[j,]])
  # }
  # 
  # res$Yi0hat[res$D==1] <- esty0
  # res$Yi1hat[res$D==0] <- esty1
  
  #res <- add_column(res, tauihat = res$Yi1hat-res$Yi0hat)
  #res <- res %>%
  #  dplyr::select(-one_of("Yi1hat", "Yi0hat"))
  #
  #get k-nn for CATE eq 26 in JASA#
  nnTg2 <- get.knnx(data = Tg, query = res[, allX], kC)
  treat <- apply(matrix(data = Tgy[nnTg2$nn.index], dim(nnTg2$nn.index)), 1, mean)
  
  nnCg2 <- get.knnx(data = Cg, query = res[, allX], kC)
  cont <- apply(matrix(data = Cgy[nnCg2$nn.index], dim(nnCg2$nn.index)), 1, mean)
  #get variance as in JASA
  vhat <- (var(res$Y_obs[nnTg2$nn.index]) + var(res$Y_obs[nnCg2$nn.index]))/(kC*(kC-1))
  #
  res <- add_column(res, tauhatknn=treat-cont)
  res <- add_column(res, vartauhatknn = vhat)
  
  #do ols
  allX <- grep("^[X]", names(res), value=TRUE)
  allDX <- grep("^[I]", names(res), value=TRUE, ignore.case = FALSE)
  all <- c(allX,allDX)
  fmla <- as.formula(paste("Y_obs ~ D + ", paste(all, collapse= "+")))
  reslm <- lm(fmla, data = res)
  coef <- matrix(reslm$coefficients[c(allDX, "D")], nrow = d+1, ncol = 1)
  const <- matrix(rep(1, n), nrow = n, ncol=1)
  X <- as.matrix(cbind(res[, c(allX)], const))
  C_ols <- as.vector(t(coef)%*%t(X))
  res <- add_column(res, Cate_ols=C_ols)
  #compute true cate
  if(add){res <- add_column(res, CATE = x%*%(betas1-betas0))}
  else{res <- add_column(res, CATE = tau(res$X1, res$X2))}
  #
  return(res)
}