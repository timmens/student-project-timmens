# Causal Tree 
## Info 
This part of the project is written in python code only. We will use PyCharm locally and **not** use JupyterLab for this part. This is mainly due to the fact that we are implementing a machine learning algorithm which is class and function based (i.e. no need for visual feedback). The causal tree algorithm is very similar to the classical decision tree algorithm presented in Breiman (1994) and Tibshirani, Hastie (2001, 2013). Hence our strategy is characterized as follows. 

1. Implement the Regression Tree algorithm 
   * Build large tree from data 
   * Prune tree to avoid overfitting 
   * Use Cross Validation to select optimal hyper parameters  
2. Building on step one, change the tree in a way as described by Athey (2015)
3. [Optimize Code] 

## DecisionTreePruned.py
Right now this is the most important file. It contains the class DecisionTreePruned which is an estimator that takes in training data and builds a regression tree from it. It can also predict and validate on new data and testing data, respectively. However, right now the only part of the algorithm which guards from overfitting is setting the hyper parameters min_leaf (minimum number of observations in each leaf) and max_depth (maximum number of depht of the tree). Unfortunately it can easily be shown that using these hyper parameters alone to avoid overfitting will lead to inefficient estimators. This is why I am currently implementing the pruning process as described by Breiman, Tibsharini, Hastie, etc. 

### TODO 

1. Implement Non Naive Pruning Function to get sequence of potentially optimally subtrees [DONE] 
2. Write Cross Validation Function that selects an optimal tree from a sequence of subtrees
3. [Optimize Code] 


### TOCHECK

1. Check if I get any problems by not using deepcopy or np.copy 
2. Check if all loss functions that are being used are defined equally; in particular len(y) * var(y) vs. var(y) 

### Notes 

- Cannot use Cross Validation from Sklearn since DecisionTree cross validation is inherently different compared to other cv strategies (or is it, maybe check that one again!)
