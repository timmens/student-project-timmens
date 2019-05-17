# Causal Tree 

This part of the project is written in pure python code. We will use PyCharm locally and **not** use JupyterLab for this part. This is mainly due to the fact that we are implementing a machine learning algorithm which is class and function based (i.e. no need for visual feedback). The causal tree algorithm is very similar to the classical decision tree algorithm presented in Breiman (1994) and Tibshirani, Hastie (2001, 2013). Hence our strategy is characterized as follows. 

1. Implement the Regression Tree algorithm 
   * Build large tree from data 
   * Prune tree to avoid overfitting 
   * Use Cross Validation to select optimal hyper parameters  
2. Building on step one, change the tree in a way as described by Athey (2015)
3. [Optimize Code] 
