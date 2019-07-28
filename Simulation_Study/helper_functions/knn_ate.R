knn_ate <- function(data, knn_se5 = knn_se5., mymatch = Matching::Match, k){
  tauhat <- mean(data$tauihat)
  
  if(knn_se5){
    rr  <- mymatch(Y=data$Y_obs, Tr=data$D, X=data$X, estimand = "ATE", M=k, replace = TRUE)
    tauhatM <- rr$est
    se <- rr$se
  }
  else{tauhatM <- NA ; se <- NA}
  return(list(ATE=tauhat, ATEM=tauhatM, se=se))
}