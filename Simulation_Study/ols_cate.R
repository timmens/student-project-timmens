ols_cate <- function(data, vcovmy = vcovHC, C_ols_se5 = C_ols_se5.){
  
  MSE <- function(est, true){
    y <- mean((est-true)^2)
  }
  
  data2 <- data %>% dplyr::select(Y_obs, D, X, IntXD, CATE)
  reslm <- lm(Y_obs ~ D + X + IntXD, data = data2)
  cate_ols <- reslm$coefficients[2] + reslm$coefficients[4]*(data2$X - mean(data2$X))
  
  if(C_ols_se5){
    varb <- vcovmy(reslm, type = "const")
    ols_se <- rep(0, nrow(data))
    for(j in seq_along(ols_se)){
      zw <- matrix(c(1, data2$D[j], data2$X[j], data2$IntXD[j]), nrow = 4, ncol = 1)
      ols_se[j] <- sqrt(t(zw)%*%varb%*%zw)
    }
    avg_se <- mean(ols_se)
  }
  else{avg_se <- NA}
  
  return(list(MSE=MSE(cate_ols, data2$CATE), se=avg_se))
}
