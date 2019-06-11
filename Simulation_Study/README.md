# Simulation Study + (Data example)
## Please make sure to view the notebook via the nbviewer https://nbviewer.jupyter.org/
## Description
This simulation study will use different data generating processes to illustrate the importance of heterogeneous treatment effects. I'll start out with settings that can be tackled with off-the-shelf estimation techniques such as OLS and/or propensity score matching and will gradually increase the level of sophistication inherent to the settings to work out the benefits of using causal trees/forests. The latter exercise builds upon the first part of our joint project by Tim, and may be conducted using data from a randomized experiment instead of simulated data. Whenever possible I'll try to use some of the estimation techniques studied within our course.
The simulation exercise can be divided into the following steps.

1. No Heterogeneity
   * Unconfoundedness vs. random treatment assignment
   * What does OLS recover?
2. Heterogeneity in one dimension only
   * develop intuition for the problem: CATE vs. ATE
   * departing from linearity
   * Estimation: direct conditional mean estimation (parametric vs. non-parametric)
3. Increasing the dimension of the feature matrix $X$
   * comparison of non-parametric regression and causal trees/forests
4. Data from a randomized experiment.   
   * compare all of the previously discussed methods
   * visualize heterogeneity

## To Do
1. Think about useful illustrations of the shortcomings of conventional estimation techniques in high dimensions
2. Nicely write-up existing results.
