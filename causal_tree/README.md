# Causal Tree
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

## Abstract 

In this part of the project we will focus on implementing the causal tree algorithm formulated in *Recursive partitioning for heterogeneous causal effects*, published 2016 by Susan Athey and Guido Imbens; see reference below. Their algorithm structure is guided heavily by the classification and regression tree algorithm (see reference below). Hence, in step one we start our project by building a working implementation of the classical (regression) decision tree algorithm. Having verified the prediction capabilities of our implementation on the famous Iris data set, we devote our attention to the causal tree algorithm.<sup>1</sup> In the construction of `CausalTree.py` we closely follow Athey and Imbens 2016 in building our extension on the classical regression tree algorithm. Using the finished implementation we let our algorithm loose on simulated and real data comparing the results to an official library. Our findings and a concise mathematical explaination of the methods is presented in a small notebook &ndash;open this notebook by clicking one of the buttons above.


<font size=2>[1]: See notebook `decision_tree_testing.ipynb` for a visual check on estimation performance of the decision tree implementation by pressing buttons on the right.</font>
<a href="https://nbviewer.jupyter.org/github/HumanCapitalAnalysis/student-project-timmens/blob/master/causal_tree/causal_tree.ipynb"
   target="_parent">
   <img align="center" 
  src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="80" height="15">
</a>
<a href="https://mybinder.org/v2/gh/HumanCapitalAnalysis/student-project-timmens/master?filepath=causal_tree%2Fcausal_tree.ipynb" 
    target="_parent">
    <img align="center" 
       src="https://mybinder.org/badge_logo.svg" 
       width="80" height="15">
</a>

## `DecisionTree.py`
This class finalizes step one; It contains all the relevant functions to fit a regression tree on given data using an arbitrary loss function for split point evaluation. Furthermore, in contrast to the decision tree implemented in scikit-learn, this class also implements the important pruning algorithm guarding against overfitting. See [Hastie et.al. 2009](https://web.stanford.edu/~hastie/ElemStatLearn/) for the standard decision tree algorithm; see [PennState Stat508 Course](https://newonlinecourses.science.psu.edu/stat508/lesson/11/11.8) for an excellent introduction to minimal-cost complexity pruning (detailed references are found below).

#### Example Code

```python
X, y = simulate_data() # X and y are a numpy ndarray or pandas DataFrame / Series
tree = DecisionTree()
tree.fit(X, y) # fits tree (but also overfits)
  
optimal_tree = tree.apply_kFold_CV(X, y, k=5, fitted_tree=tree)

print(optimal_tree) # displays some relevant information on the tree
plot(optimal_tree)  # plots tree in a hierachical (upside-down) tree like structure 
``` 

## `CausalTree.py`
As described above, this class will build on the class `DecisionTree`; however, at relevant places changes will be made as illustrated in Athey
and Imbens 2016. In particular since individual treatment effects are not observed, to use standard supervised machine learning techniques we have to propose a pre-estimate. See main notebook for more detailed information. 

#### Example Code

```python
X, y, D = simulate_treatment_data() # X and y are a numpy ndarray or pandas DataFrame / Series
ctree = CausalTree()
ctree.fit(X, y, D) # fits causal tree (but also overfits)


ctree_sparse, ctree_opt = ctree.apply_kFold_CV(X, y, k=5, fitted_tree=ctree) # here an optimal 
# and a sparse tree are returned
 
print(ctree_sparse) # displays some relevant information on the tree
plot(ctree_sparse)  # plots tree in a hierachical (upside-down) tree like structure 
``` 

## References

* ***Recursive partitioning for heterogeneous causal effects***, Susan Athey and Guido Imbens; <font size="2">(PNAS July 5, 2016 113 (27) 7353-7360; first published July 5, 2016)</font>

* ***Classification and Regression Trees***, Breiman, L., Friedman, J., Olshen, R. and Stone, C.;  <font size="2">(Chapman and Hall, Wadsworth, New York; published 1984)</font>

* ***The Elements of Statistical Learning***, Hastie, Tibshirani and Friedman; <font size="2">(Springer Series in Statistics Springer New York Inc., New York, NY, USA, (2009))</font>

* ***Generalized Random Forests***, Susan Athey, Julie Tibshirani and Stefan Wager; <font size="2">(Ann. Statist.; Volume 47, Number 2 (2019), 1148-1178.)</font>

---
[//]: <> (Comment: Badges for Travis CI, MIT License and Black Code Style)

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](HumanCapitalAnalysis/student-project-timmens/blob/master/LICENSE) <a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

