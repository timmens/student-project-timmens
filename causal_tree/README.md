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

---
<font color="red", size=12>This README is under construction. The finished product will contain an abstract, list of relevant files, example code for `.py`-files and a detailed explaination of the project in general as well as references to the corresponding literature. </font>
---
## DecisionTree.py [DONE]
This class finalizes step one; It contains all the relevant functions to fit a regression tree on given data using an arbitrary loss function for split point evaluation. Furthermore, in contrast to the decision tree implemented in scikit-learn, this class also implements the important pruning algorithm guarding against overfitting. See [Hastie et.al. 2009](https://web.stanford.edu/~hastie/ElemStatLearn/) for the standard decision tree algorithm; see [PennState Stat508 Course](https://newonlinecourses.science.psu.edu/stat508/lesson/11/11.8) for an excellent introduction to minimal-cost complexity pruning.

#### Important Functions (introduced by an example)
  ```python
  X, y = get_data_somewhere() # X and y are a numpy ndarray or pandas DataFrame / Series
  tree = DecisionTree()
  tree.fit(X, y) # (pure over-)fits tree using given data; use code below 
  
  optimal_tree = get_optimal_subtree_via_k_fold_cv(X, y, k=5, fitted_tree=None)
  print(optimal_tree) # displays some relevant information on the tree
  plot(optimal_tree) # plots tree in a hierachical (upside-down) tree like structure 
  ``` 

## CausalTree.py [DONE]
As described above, this class will build on the class DecisionTree.py; however, at the relevant places changes will be made as illustrated in Athey (2015). 

## References

* ***Recursive partitioning for heterogeneous causal effects***, Susan Athey and Guido Imbens; <font size="2">(PNAS July 5, 2016 113 (27) 7353-7360; first published July 5, 2016)</font>

* ***Classification and Regression Trees***, Breiman, L., Friedman, J., Olshen, R. and Stone, C.;  <font size="2">(Chapman and Hall, Wadsworth, New York; published 1984)</font>

* ***Generalized Random Forests***, Susan Athey, Julie Tibshirani and Stefan Wager; <font size="2">(Ann. Statist.; Volume 47, Number 2 (2019), 1148-1178.)</font>

---
[//]: <> (Comment: Badges for Travis CI, MIT License and Black Code Style)

[![Build Status](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens.svg?branch=master)](https://travis-ci.org/HumanCapitalAnalysis/student-project-timmens) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](HumanCapitalAnalysis/student-project-timmens/blob/master/LICENSE) <a href="https://github.com/python/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

