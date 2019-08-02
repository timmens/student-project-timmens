# Student Project - Tim Mensinger and Florian Schoner


## Abstract

In this project we try to understand newly proposed methods for computing heterogenous treatment effects that leverage regression and classification approaches from the machine learning literature. In particular we are interested in the new method *Causal Trees*, first proposed in [*Recursive partitioning for heterogeneous causal effects*](https://www.pnas.org/content/113/27/7353),
by Susan Athey and Guido Imbens (2016); see reference below. Their method is heavily based upon the famous classification and regression tree algorithm, illustrated in the same-titled book by Breiman et al. (1984); see reference below. The project will be split in two parts. Part one, Causal Tree, will be written by Tim Mensinger, while part two will be done by Florian Schoner.

## Causal Tree
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-timmens/blob/master/causal_tree/causal_tree.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20">
</a> 
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-timmens/master?filepath=causal_tree%2Fcausal_tree.ipynb" 
    target="_parent">
    <img align="center" 
       src="https://mybinder.org/badge_logo.svg" 
       width="109" height="20">
</a> 

In this section we will work on a Python implementation of the algorithm (`CausalTree.py`) as well as on an illustrative notebook. The notebook will contain a concise explaination of the mathematics behind Causal Trees, along with an outlook on the use of more involved algorithms. 

## Simulation Study
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

Here we compare common approaches to estimating treatment effects in a homogenous and heterogenous settings. For simulated data we contrast the naive estimator, propensity score weighting, k-NN, OLS, local linear kernel regression and Causal Forests.<sup>1</sup> In Addition, the mentioned methods will be applied to data from a field experiment first considered in *Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment*, by Gerber et al. (2008). 

<font size="2">[1]: Causal Forests refer to the natural extension of Causal Trees, proposed in *Generalized Random Forests* by Susan Athey et al. (2019); see reference below.</font>

## Simpsons Paradox
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-timmens/blob/master/simpsons_paradox/simpsons_paradox.ipynb" 
    target="_parent">
    <img align="center" 
   src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
       width="109" height="20">
</a> 
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-timmens/master?filepath=simpsons_paradox%2Fsimpsons_paradox.ipynb" 
     target="_parent">
     <img align="center" 
        src="https://mybinder.org/badge_logo.svg" 
        width="109" height="20">
</a> 

This folder contains the submission to the Python challenge.


## References

* ***Recursive partitioning for heterogeneous causal effects***, Susan Athey and Guido Imbens; <font size="2">(PNAS July 5, 2016 113 (27) 7353-7360; first published July 5, 2016)</font>

* ***Classification and Regression Trees***, Breiman, L., Friedman, J., Olshen, R. and Stone, C.;  <font size="2">(Chapman and Hall, Wadsworth, New York; published 1984)</font>

* ***Generalized Random Forests***, Susan Athey, Julie Tibshirani and Stefan Wager; <font size="2">(Ann. Statist.; Volume 47, Number 2 (2019), 1148-1178.)</font>

* ***Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment***, Gerber, Green and Larimer;  <font size="2">(American Political Science Review (2008) 102(1): 33-48.)</font>

---
[//]: <> (Comment: Badges for Travis CI, MIT License and Black Code Style)

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](HumanCapitalAnalysis/student-project-timmens/blob/master/LICENSE) <a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 

