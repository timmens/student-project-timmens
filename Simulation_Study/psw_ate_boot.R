psw_ate_boot <- function(data, method, B, boot5 = boot5.){

  estim_hir <- function(dataset, index, methodi = method){
    
    if(methodi=="probit"){ehat <- dataset[index,"psprob"]}
    else if(methodi=="logit"){ehat <- dataset[index,"pslog"]}
    else if(methodi=="np"){ehat <- dataset[index, "psnp"]}
    else {ehat <- dataset[index, "trueps"]}
    ehat <- ehat[index]
    Tr <- dataset$D[index]
    Y <- dataset$Y_obs[index]
    #ehat <- dataset$pslog
    return( (sum( (Tr*Y) / ehat) / sum( Tr / ehat )) -
              (sum( ((1-Tr)*Y) / (1-ehat) ) / sum( (1-Tr) / (1-ehat) )) )
  }
  
  
  tauhat2 <- estim_hir(dataset = data, index = 1:nrow(data), method)
  #bootstrap for se#
  if(boot5){test <- sd(boot(data, R = B, estim_hir)$t)}
  else{test <- NA}
  return(list(ATE1=tauhat2, ATE1se=test))
}