# Simulation Study 
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-timmens/blob/master/Simulation_Study/simulation_study.ipynb" 
    target="_parent">
    <img align="center" 
   src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
       width="109" height="20">
</a> 
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-timmens/master?filepath=Simulation_Study%2Fsimulation_study.ipynb" 
     target="_parent">
     <img align="center" 
        src="https://mybinder.org/badge_logo.svg" 
        width="109" height="20">
</a> 



## Description
This simulation study will use different data generating processes to illustrate the importance of heterogeneous treatment effects. I'll start out with settings that can be tackled with off-the-shelf estimation techniques and will gradually increase the level of sophistication inherent to the settings to work out the benefits of using causal trees/forests. The latter exercise builds upon the first part of our joint project by Tim, and will be conducted using simulated data. For the (C)ATE, the estimation routines to be compared include:  The Naive estimator, Propensity Score Weighting, k-NN matching, OLS, and causal forests.

The illustration will proceed in 3 steps.


1. One-dimensional feature space
   1. No Heterogeneity
      * Unconfoundedness vs. random treatment assignment
   2. Heterogeneity in one dimension only
      * develop intuition for the problem: CATE vs. ATE
      * departing from linearity 

2. d-dimensional feature space: Focus on CATE only
   1.  k-NN Matching vs. causal forests

 3. Data from a randomized experiment <a name="myfootnote1">1</a>: Gerber, Green, and Larimer (2008)'s paper "Social Pressure and[//]: #  Voter Turnout: Evidence from a Large-Scale Field Experiment" (http://isps.yale.edu/sites/default/files/publication/2012/12/ISPS08-001.pdf)
  [//]: # (1. compare all of the previously discussed methods)
 [//]: #  2. visualize heterogeneity in treatment effects.




