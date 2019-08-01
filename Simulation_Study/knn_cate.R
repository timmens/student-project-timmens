knn_cate <- function(data, C_knn_se5 = C_knn_se5.){
  MSE <- function(est, true){
    y <- mean((est-true)^2)
  }
  
  if(C_knn_se5){
    avg_se <- mean(sqrt(data$vartauhatknn))
  }
  else{avg_se <- NA}
  
  return(list(MSE=MSE(data$tauhatknn, data$CATE), se=avg_se))
}