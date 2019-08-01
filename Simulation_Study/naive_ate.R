naive_ate <- function(data) {
  y1 <- data %>% dplyr::filter(D == 1) %>% dplyr::pull(Y_obs) # Outcome in treatment grp
  y0 <- data %>% dplyr::filter(D == 0) %>% dplyr::pull(Y_obs) # Outcome in control group
  
  n1 <- sum(data[,"D"])     # Number of obs in treatment
  n0 <- sum(1 - data[,"D"]) # Number of obs in control
  
  # Difference in means is ATE
  tauhat <- mean(y1) - mean(y0)
  
  # 95% Confidence intervals
  se_hat <- sqrt( var(y0)/(n0-1) + var(y1)/(n1-1) )
  #lower_ci <- tauhat - 1.96 * se_hat
  #upper_ci <- tauhat + 1.96 * se_hat
  
  #return(list(ATE = tauhat, lower_ci = lower_ci, upper_ci = upper_ci, se=se_hat))
  return(list(ATE = tauhat, se=se_hat))
  #return(tauhat)
}