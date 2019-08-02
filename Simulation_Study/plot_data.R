plot_data <- function(n, num_trees, kC,
                      het_linear, random_assignment,
                      non_linearY, non_linearD){
  

    
    #rm(list=ls())
    #setwd("C:\\Users\\Flori\\Documents\\Project_Microeconometrics\\student-project-timmens\\Simulation_Study\\helper_functions")
    #source("sim.R")
    #source("naive_ate.R")
    #source("psw_ate_boot.R")
    #source("ols_ate.R")
    #source("knn_ate.R")
    #source("rf_ate.R")
    #source("MSE.R")
    #source("ols_cate.R")
    #source("knn_cate.R")
    #source("rf_cate.R")
    
    
    param <- as.list(match.call())
    n. <- as.numeric(param$n)
    num_trees. <- as.numeric(param$num_trees)
    kC. <- as.numeric(param$kC)
    random_assignment. <- param$random_assignment
    non_linearY. <- param$non_linearY
    non_linearD. <- param$non_linearD
    het_linear. <- param$het_linear
    
      
  
  df <- sim(n = n., het_linear = het_linear., random_assignment= random_assignment.,
            non_linearD = non_linearD., np_ps = FALSE, k = 1, kC = kC.,
            non_linearY = non_linearY., diff_error = TRUE)
  df_part <- modelr::resample_partition(df, c(train = 0.5, test = 0.5))
  df_train <- as.data.frame(df_part$train)
  df_test <- as.data.frame(df_part$test)
  
  set.seed(1001)
  cf <- grf::causal_forest(X = as.matrix(df_train$X),
                           Y = as.matrix(df_train$Y_obs),
                           W = as.matrix(df_train$D),
                           num.trees = num_trees., # Make this larger for better acc.
                           num.threads = 1,
                           honesty = TRUE)
  names(cf)
  
  # Predict CATE and its std error for each individual on the dataset
  cf_res <- predict(cf, as.matrix(df_test$X), estimate.variance = TRUE)
  tauhatx_cf <- cf_res$predictions %>% as.numeric()
  
  reslm <- lm(Y_obs ~ D + X + IntXD, data = df)
  fitval <- reslm$fitted.values
  df[, "fitval"] <- reslm$coefficients[2] + df$X * reslm$coefficients[4]
  
  idtest <- df_part$test$idx
  df <- df[idtest, ]
  #olsx <- sort(df$X)
  #olsy <- df$fitval[order(df$X)]
  ggplot(df, aes(x=X), legend = TRUE) +
    geom_point(aes(y=Y_obs, col=factor(D)), size=1) +
    geom_line(aes(y=tauhatknn, col="k-NN"), size = 1) +
    geom_smooth(aes(y=(Y1-Y0), col="OLS"), method="lm", se = FALSE) + 
    #geom_point(aes(y=fitval, col="OLS"), size = 1) + 
    geom_line(aes(y=tauhatx_cf, col="RF"), size=1) + 
    geom_line(aes(y=CATE), linetype = "dotted", col="black", size = 1.25) + 
    scale_x_continuous(breaks = round(seq(0, 1, by = 0.1),1)) + 
    if(non_linearY.){
    scale_y_continuous(breaks = round(seq(min(df$CATE), max(df$Y_obs), by = 5),0))
    } 
   else{scale_y_continuous(breaks = round(seq(min(df$Y_obs), max(df$Y_obs), by = 1),0))}

  
}