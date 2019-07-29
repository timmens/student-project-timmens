create_output_table_JASA <- function(n, N, d, num_trees, kC,
                                   para = FALSE, C_knn_se5 = FALSE ){

#source("simAW.R")
#source("MSE.R")
#source("knn_cate.R")
#source("rf_cate_JASA.R")
#source("ols_cate_JASA.R")
  
  
  param <- as.list(match.call())
  N <- as.numeric(param$N)
  n <- as.numeric(param$n)
  d <- as.numeric(param$d)
  num_trees <- as.numeric(param$num_trees)
  kC <- as.numeric(param$kC)
  para <- param$para
  if(para == TRUE){
    nodes <- detectCores()
    cl <- makeCluster(nodes)
    registerDoParallel(cl)}
  else{}
  
  #KNN
  C_knn_se5. <- param$C_knn_se5
  
  
  system.time(test <- replicate(n = N,
                                simAW(n, kC, d, add = FALSE),
                                simplify = FALSE))
  for(j in 1:N){test[[j]] <- add_column(test[[j]], id = rep(j, n))}
  test <- bind_rows(test)
  
  
  system.time( est_C_knn <- dlply(test, .(id), knn_cate, C_knn_se5 = C_knn_se5.,
                                  .progress = "text",
                                  .parallel = para,
                                  .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
  
  C_knn <- as.numeric(map(est_C_knn, "MSE"))
  C_knn_se <- as.numeric(map(est_C_knn, "se"))
  
  #tbp
  C_mean_mse_knn <- mean(C_knn)
  C_mean_se_knn <- mean(C_knn_se)
  
  
  
  system.time( est_C_ols <- dlply(test, .(id), ols_cate_JASA,
                                  .progress = "text",
                                  .parallel = para,
                                  .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
  C_ols <- as.numeric(map(est_C_ols, "MSE"))
  
  #tbp
  C_mean_mse_ols <- mean(C_ols)
  
  
  system.time( est_C_rf <- dlply(test, .(id), rf_cate_JASA, num_trees, MSE_fun = MSE,
                                 .progress = "text",
                                 .parallel = para,
                                 .paropts = list(.packages = c("httr", "jsonlite", "dplyr"),
                                                 .export=c('data','rf_cate','MSE', 'predict'))))
  
  #stopCluster(cl)
  
  C_rf <- as.numeric(map(est_C_rf, "MSE"))
  C_rf_se <- as.numeric(map(est_C_rf, "se"))
  
  #tbp
  C_mean_mse_rf <- mean(C_rf)
  C_mean_se_rf <- mean(C_rf_se)
  
  
  
  #output
  res_ols_cate <- c(C_mean_mse_ols)
  res_knn_cate <- c(C_mean_mse_knn)
  res_rf_cate <- c(C_mean_mse_rf)
  
  rownames_cate <- c("avg_mse")
  res_df_cate <- data.frame(Description = rownames_cate, OLS = res_ols_cate,
                            k_NN = res_knn_cate, RF = res_rf_cate)
  disp2 <- format(res_df_cate, justify = "right", digit = 3, trim = FALSE, width = 10)
  disp2
  
  
  
  
}