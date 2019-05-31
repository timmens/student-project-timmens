# Causal Tree 
## Info 
This part of the project is written in python code only. We will use PyCharm locally and **not** use JupyterLab for this part. This is mainly due to the fact that we are implementing a machine learning algorithm which is class and function based (i.e. no need for visual feedback). The causal tree algorithm is very similar to the classical decision tree algorithm presented in Breiman (1994) and Tibshirani, Hastie (2001, 2013). Hence our strategy is characterized as follows. 

1. Implement the Regression Tree algorithm [DONE]
   * Build large tree from data 
   * Prune tree to avoid overfitting 
   * Use Cross Validation to select optimal hyper parameters  
2. Building on step one, change the tree in a way as described by Athey (2015) [In the making]
3. [Optimize Code] 

## DecisionTree.py [DONE]
This class finalizes step one; It contains all the relevant functions to fit a regression tree on given data using an arbitrary loss function for split point evaluation. Furthermore, in contrast to the decision tree implemented in scikit-learn, this class also implements the important pruning algorithm guarding against overfitting. See [Hastie et.al. 2009](https://web.stanford.edu/~hastie/ElemStatLearn/) for the standard decision tree algorithm; see [PennState Stat508 Course](https://newonlinecourses.science.psu.edu/stat508/lesson/11/11.8) for an excellent introduction to minimal-cost complexity pruning.

#### Important Functions
- get_optimal_subtree_via_k_fold_cv(X_learn, y_learn, k=5, fitted_tree=None) [computes the optimal tree via k-fold cross-validation using a learning dataset]
- plot(tree) [plots tree object in a hierachical (upside-down) tree-like structure]

## CausalTree.py [In the making..]
As described above, this class will build on the class DecisionTree.py; however, at the relevant places changes will be made as illustrated in Athey (2015). 


### TODO 
1. [Optimize Code] 


### TOCHECK
1. Check if I get any problems by not using deepcopy or np.copy 
2. Check if all loss functions that are being used are defined equally; in particular len(y) * var(y) vs. var(y) 

