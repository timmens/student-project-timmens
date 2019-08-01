rf_ate <- function(data, num_trees){
  #data2 <- data %>% dplyr::select(X, Y_obs, D)
  
  forest <- grf::causal_forest(X=as.matrix(data$X),
                               Y=as.matrix(data$Y_obs),
                               W=as.matrix(data$D),
                               num.trees = num_trees,
                               honesty = TRUE,
                               min.node.size = 1,
                               seed=123)
  
  ate_cf_robust <- grf::average_treatment_effect(forest)
  
  tauhat <- ate_cf_robust[1]
  forest_se <- ate_cf_robust[2]
  return(list(ATE=tauhat, se=forest_se))
}