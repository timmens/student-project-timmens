create_output_table <- function(n, N, B, num_trees, k, kC,
                                het_linear, random_assignment,
                                non_linearY, non_linearD,
                                method_ols,
                                knn_se5, ols_se5, C_ols_se5, C_knn_se5, boot5,
                                para){

#rm(list=ls())
#setwd("C:\\Users\\Flori\\Documents\\Project_Microeconometrics\\functions_sim")
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
N <- as.numeric(param$N)
n <- as.numeric(param$n)
B <- as.numeric(param$B)
num_trees. <- as.numeric(param$num_trees)
k. <- as.numeric(param$k)
kC. <- as.numeric(param$kC)
random_assignment. <- param$random_assignment
non_linearY. <- param$non_linearY
non_linearD. <- param$non_linearD
het_linear. <- param$het_linear
para <- param$para

knn_se5. <- param$knn_se5
#PSW
method. <- "logit"
boot5. <- param$boot5
#OLS
method_ols. <- param$method_ols
ols_se5. <- param$ols_se5


#CATE
#OLS
C_ols_se5. <- param$C_ols_se5
#KNN
C_knn_se5. <- param$C_knn_se5
  
test <- replicate(N,
                    sim(n, het_linear = het_linear., random_assignment = random_assignment.,
                        non_linearD = non_linearD., np_ps = FALSE, k = k., kC = kC.,
                        non_linearY = non_linearY., diff_error = TRUE),
                    simplify = FALSE)
for(j in 1:N){test[[j]] <- add_column(test[[j]], id = rep(j, n))}
test <- bind_rows(test)


#NAIVE
system.time(est_naive <- dlply(test, .(id), naive_ate,
                               .progress = "text",
                               .parallel = para,
                               .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
naive <- as.numeric(map(est_naive, "ATE"))
naive_se <- as.numeric(map(est_naive, "se"))

#tbp#
mean_naive <- mean(naive)
sd_naive <- sd(naive)
mean_se_naive <- mean(naive_se)
#
#plot(density(naive_se))

#PSW
system.time(est_psw_log_2 <- dlply(test, .(id), psw_ate_boot, method = method., B, boot5 = boot5.,
                                   .progress = "text",
                                   .parallel = para,
                                   .paropts = list(.packages = c("httr", "jsonlite", "dplyr"),
                                                   .export = c("boot"))))
psw_log_2 <- as.numeric(map(est_psw_log_2, "ATE1"))
psw_log_se_2 <- as.numeric(map(est_psw_log_2, "ATE1se"))

#tbp#
mean_psw_log_2 <- mean(psw_log_2)
sd_psw_log_2 <- sd(psw_log_2)
mean_se_psw_log_2 <- mean(psw_log_se_2)
#plot(density(psw_log))
#plot(density(psw_log_se))

#OLS
system.time( est_ols <- dlply(test, .(id), ols_ate, method_ols = method_ols., ols_se5 = ols_se5.,
                              myvcov = sandwich::vcovHC,
                              .progress = "text",
                              .parallel = para,
                              .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
ols <- as.numeric(map(est_ols, "ATE"))
ols_se <- as.numeric(map(est_ols, "se"))

#tbp
mean_ols <- mean(ols)
sd_ols <- sd(ols)
mean_se_ols <- mean(ols_se)


#KNN
system.time(est_knn <- dlply(test, .(id), knn_ate, knn_se5 = knn_se5., mymatch = Matching::Match, k.,
                             .progress = "text",
                             .parallel = para,
                             .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
knn <- as.numeric(map(est_knn, "ATE"))
knn_se <- as.numeric(map(est_knn, "se"))

#tbp#
mean_knn <- mean(knn)
sd_knn <- sd(knn)
mean_se_knn <- mean(knn_se)


#plot(density(knn))
#lines(density(knn_M),col="red")

# #Random Forests
system.time(est_rf <-  dlply(test, .(id), rf_ate, num_trees,
                             .progress = "text", 
                             .parallel = para,
                             .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
rf <- as.numeric(map(est_rf, "ATE"))
rf_se <- as.numeric(map(est_rf, "se"))
 
 #tbp#
mean_rf <- mean(rf)
sd_rf <- sd(rf)
mean_se_rf <- mean(rf_se)
# 
# #plot(density(rf),col="green")


#plot(density(knn),xlim=c(3,5))
#lines(density(ols),col="red")
#lines(density(naive), col="green")
#lines(density(knn), col="blue")
#lines(density(rf), col="purple")


############################################################################################################
########################################CATE Estimation#####################################################
############################################################################################################


system.time( est_C_knn <- dlply(test, .(id), knn_cate, C_knn_se5 = C_knn_se5.,
                                .progress = "text",
                                .parallel = para,
                                .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))

C_knn <- as.numeric(map(est_C_knn, "MSE"))
C_knn_se <- as.numeric(map(est_C_knn, "se"))

#tbp
C_mean_mse_knn <- mean(C_knn)
C_mean_se_knn <- mean(C_knn_se)
#plot(density(C_knn),xlim=c(0,10))

system.time( est_C_ols <- dlply(test, .(id), ols_cate, vcovmy = sandwich::vcovHC,
                                C_ols_se5 = C_ols_se5.,
                                .progress = "text",
                                .parallel = para,
                                .paropts = list(.packages = c("httr", "jsonlite", "dplyr"))))
C_ols <- as.numeric(map(est_C_ols, "MSE"))
C_ols_se <- as.numeric(map(est_C_ols, "se"))

#tbp
C_mean_mse_ols <- mean(C_ols)
C_mean_se_ols <- mean(C_ols_se)
#plot(density(C_ols_se),xlim=c(0,0.5))


system.time( est_C_rf <- dlply(test, .(id), rf_cate, num_trees, MSE_fun = MSE,
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


#plot(density(C_rf))
#lines(density(C_rf_se),col="green")


#plot(density(C_knn),xlim=c(0,100))
#lines(density(C_ols),col="red")
#lines(density(C_rf), col="green")


#report results
# res <- c(mean_naive, sd_naive, mean_se_naive,
#          mean_ols, sd_ols, mean_se_ols,
#          mean_psw_log_2, sd_psw_log_2, mean_se_psw_log_2,
#          mean_knn, sd_knn, mean_se_knn,
#          mean_rf, sd_rf, mean_se_rf,
#          C_mean_mse_ols, C_mean_se_ols,
#          C_mean_mse_knn, C_mean_se_knn,
#          C_mean_mse_rf, C_mean_se_rf)

rownames_ate <- c("avg_estimate", "sd_estimates", "avg_estimated_se")

res_naive_ate <- c(mean_naive, sd_naive, mean_se_naive)
res_ols_ate <- c(mean_ols, sd_ols, mean_se_ols) 
res_psw_log_ate <- c(mean_psw_log_2, sd_psw_log_2, mean_se_psw_log_2)
res_knn_ate <- c(mean_knn, sd_knn, mean_se_knn)
res_rf_ate <- c(mean_rf, sd_rf, mean_se_rf)

res_ols_cate <- c(C_mean_mse_ols, C_mean_se_ols)
res_knn_cate <- c(C_mean_mse_knn, C_mean_se_knn)
res_rf_cate <- c(C_mean_mse_rf, C_mean_se_rf)

res_df_ate <- data.frame(Description = rownames_ate, Naive = res_naive_ate, OLS = res_ols_ate,
                         PSW = res_psw_log_ate, k_NN = res_knn_ate, RF = res_rf_ate)
disp1 <- format(res_df_ate, justify = "right", digit = 4, width = 10, trim = TRUE)


rownames_cate <- c("avg_mse", "avg_mean_se")
res_df_cate <- data.frame(Description = rownames_cate, OLS = res_ols_cate,
                          k_NN = res_knn_cate, RF = res_rf_cate)
disp2 <- format(res_df_cate, justify = "right", digit = 4, width = 10, trim = TRUE)
return(list(ATE = disp1, CATE = disp2))
}