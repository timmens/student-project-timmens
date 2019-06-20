# Simulation Study + (Data example)
## Please make sure to view the notebook via the nbviewer https://nbviewer.jupyter.org/
## Description
This simulation study will use different data generating processes to illustrate the importance of heterogeneous treatment effects. I'll start out with settings that can be tackled with off-the-shelf estimation techniques and will gradually increase the level of sophistication inherent to the settings to work out the benefits of using causal trees/forests. The latter exercise builds upon the first part of our joint project by Tim, and will be conducted using data from a randomized experiment as well as simulated data. For the (C)ATE, the estimation routines to be compared include:  The Naive estimator, Propensity Score Weighting, k-NN matching, OLS, local linear kernel regression, and causal forests.

The illustration will proceed in 3 steps.


1. One-dimensional feature space
   1. No Heterogeneity
      * Unconfoundedness vs. random treatment assignment
   2. Heterogeneity in one dimension only
      * develop intuition for the problem: CATE vs. ATE
      * departing from linearity in both Pr(D=1|X=x) and Y

2. d-dimensional feature space: Focus on CATE only
   1.  breakdown of nonparametric methods
   2.  k-NN Matching vs. causal forests

3.  Data from: Gerber, Green, and Larimer (2008)'s paper "Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment" (see article http://isps.yale.edu/sites/default/files/publication/2012/12/ISPS08-001.pdf) 
   1. compare all of the previously discussed methods
   2. visualize heterogeneity in treatment effects.

## To Do


