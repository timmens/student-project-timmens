ols_cate_JASA <- function(data){
  
  MSE <- function(est, true){
    y <- mean((est-true)^2)
  }
  
  return(list(MSE=MSE(data$Cate_ols, data$CATE)))
}
