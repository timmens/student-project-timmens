ols_ate <- function(data, method_ols, ols_se5 = ols_se5., myvcov = sandwich::vcovHC){
  data2 <- data %>% dplyr::select(Y_obs, D, X, IntXD)
  
  if(method_ols=="D_only"){reslm <- lm(Y_obs ~ D, data = data2)}
  else if(method_ols=="D_X"){reslm <- lm(Y_obs ~ D + X, data = data2)}
  else{reslm <- lm(Y_obs ~ D + X + IntXD, data = data2)}
  
  tauhat <- reslm$coefficients[2]
  if(ols_se5){
    ols_se <- sqrt(diag(myvcov(reslm, type="const"))[2])
  }
  else{ols_se <- NA}
  return(list(ATE=tauhat, se=ols_se))
}
