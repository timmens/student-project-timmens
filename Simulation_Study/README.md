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



## Abstract
This part of the project is concerned with a simulation study that aims at illustrating the importance of heterogeneous treatment effects. The study will use different data generating processes and involve estimation methods such as Ordinary Least Squares, Propensity Score Weighting, k-Nearest-Neighbor Matching, and Causal Forests.
We'll start out with settings that are more favorable to off-the-shelf estimation techniques and will test causal forests on these more "traditional" setups.
Subsequently we will gradually increase the level of diffculty inherent to the settings to work out the benefits of using causal forests over the other methods.

The simulation study can proceeds in two steps:

1. One-dimensional feature space
   1. Unconfoundedness vs. random treatment assignment
   2. Average Treatment Effect vs. Conditional Average Treatment Effect - When does the distinction matter?

2. d-dimensional feature space
   1.  Smoothness as a parameter 
   2. Performance in high dimensions

## One-dimensional feature space
In the first part, we'll mainly rely on two functions. The first one outputs a table of results which lets you compare different estimators. The other function enables you to graphically explore the properties of the data generating process you have specified.


```R
create_output_table()
#generates the data according to the parameters you have specified, performs all the estimations, and returns two tables

plot_data()
#generates a single dataset from the input parameters, performs the estimations and returns a single ggplot
``` 
