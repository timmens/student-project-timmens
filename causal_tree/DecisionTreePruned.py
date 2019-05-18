import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class DecisionTreePruned(BaseEstimator, RegressorMixin):  # Inherits from sklearn to use their Cross Validation methods
    # 1. make sure that I do not change anything, because numpy uses copy by reference and not copy by value
    # maybe should use deepcopy more often
    # 2. implement method prune_non_naively

    def __init__(self, parent=None):  # , root=None):
        self._parent = parent
        # self._root = root
        self._left_child = None
        self._right_child = None
        self._value = None
        self._min_leaf = None
        self._split_var = None
        self._split_value = None
        self._loss = None
        self._node_loss = None
        self._branch_loss = None
        self._is_fitted = False
        self._feature_names = None
        self._num_features = None
        self._y = None

    def __str__(self):
        if self._is_fitted is False:
            string_dict = {"Loss": self._loss, "Estimate": self._value}
        elif self._feature_names is not None:
            string_dict = {"Loss": self._loss,
                           "Splitting Variable (First split)": self._feature_names[self._split_var],
                           "Splitting Value": self._split_value, "Tree Depth": self.depth, "Estimate": self._value}
        else:
            string_dict = {"Loss": self._loss, "Splitting Feature (First split)": self._split_var,
                           "Splitting Value": self._split_value, "Tree Depth": self.depth, "Estimate": self._value}
        return dict.__str__(string_dict)

    #  Property and Setter Functions
    ###################################################################################################################

    @property
    def y(self):
        return self._y

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child

    @right_child.setter
    def right_child(self, node):
        self._right_child = node

    @left_child.setter
    def left_child(self, node):
        self._left_child = node

    @property
    def depth(self):
        return DecisionTreePruned.detect_tree_depth(self)

    @property
    def is_leaf(self):
        if self._is_fitted:
            return self._left_child is None
        else:
            print("The tree has not been fitted yet, hence it is root and leaf at the same time.\n")
            return True

    @property
    def is_root(self):
        return self._parent is None

    @property
    def value(self):
        return self._value

    @property
    def node_loss(self):
        return self._node_loss

    @property
    def branch_loss(self):
        return self._branch_loss

    # Various Auxiliary Functions
    ###################################################################################################################

    def update_loss(self):
        leaf_list = DecisionTreePruned.get_leafs_in_list(self)
        loss_array = np.array([leaf.loss for leaf in leaf_list])
        self._loss = np.sum(loss_array)

    def output_partition_estimates(self):
        leaf_list = DecisionTreePruned.get_leafs_in_list(self)
        for i, leaf in enumerate(leaf_list):
            print("Leaf {:d}; Estimate: {:3.03f}".format(i, leaf.value))

    # Static Methods
    ###################################################################################################################

    @staticmethod
    def get_leafs_in_list(tree):
        leaf_list = []
        if tree.left_child is None:
            leaf_list.append(tree)
        else:
            leaf_list.extend(DecisionTreePruned.get_leafs_in_list(tree.left_child))
            leaf_list.extend(DecisionTreePruned.get_leafs_in_list(tree.right_child))
        return leaf_list

    @staticmethod
    def get_level_in_list(tree, level):
        #  level: 0 -> root, 1 -> first layer, 2 -> ...
        level_list = []
        if level == 0:
            level_list.append(tree)
        else:
            if tree.left_child is not None:
                level_list.extend(DecisionTreePruned.get_level_in_list(tree.left_child, level - 1))
                level_list.extend(DecisionTreePruned.get_level_in_list(tree.right_child, level - 1))
        return level_list

    @staticmethod
    def detect_tree_depth(tree):
        depth_left, depth_right = 0, 0
        if tree.left_child is not None:
            depth_left += 1 + DecisionTreePruned.detect_tree_depth(tree.left_child)
            depth_right += 1 + DecisionTreePruned.detect_tree_depth(tree.right_child)
        return max(depth_left, depth_right)

    @staticmethod
    def collapse_node_if(parent_node):
        parent_loss = compute_loss(parent_node.y)
        children_loss = compute_loss(parent_node.left_child.y)
        children_loss += compute_loss(parent_node.right_child.y)
        if parent_loss <= children_loss:
            parent_node.left_child = None
            parent_node.right_child = None
            parent_node.loss = compute_loss(parent_node.y)
        return parent_node

    @staticmethod
    def collapse_node_non_naively(parent_node):
        parent_loss = compute_loss(parent_node.y)
        children_loss = compute_loss(parent_node.left_child.y)
        children_loss += compute_loss(parent_node.right_child.y)
        if parent_loss <= children_loss:
            parent_node.left_child = None
            parent_node.right_child = None
            parent_node.loss = compute_loss(parent_node.y)
        return parent_node

    @staticmethod
    def prune_tree(tree, regularizer):
        depth = tree.depth
        if depth < 1:
            print("Nothing to prune here.")
            return
        for i in range(depth):
            for parent_node in DecisionTreePruned.get_level_in_list(tree, depth-i-1):
                if parent_node.left_child is not None:
                    DecisionTreePruned.collapse_node_if(parent_node, regularizer)

    @staticmethod
    def validate(tree, X_test, y_test):
        #  returns sum of squared residuals using the tree as estimator
        y_pred = tree.predict(X_test)
        return np.sum((y_test - y_pred)**2)

    @staticmethod
    def cost(tree):
       return 1 # check if this is equal to loss

    @staticmethod
    def cost_complexity(tree, alpha):
        assert alpha >= 0, "Cost-Complexity Parameter <<alpha>> has to be non-negative."
        DecisionTreePruned.cost(tree) + alpha * len(DecisionTreePruned.get_leafs_in_list(tree))

    @staticmethod
    def prune_tree_non_naively(tree, X_test, y_test):
        depth = tree.depth
        if depth < 1:
            print("Nothing to prune here.")
            return
        for i in range(depth):
            for parent_node in DecisionTreePruned.get_level_in_list(tree, depth-i-1):
                if parent_node.left_child is not None:
                    DecisionTreePruned.collapse_node_non_naively(parent_node, X_test, y_test)

    @staticmethod
    def get_pruned_tree_and_alpha_sequence(tree):
        #  1. Make sure that while fitting a tree every tree has the attributes R(t), R(T_t), |\tilde{T}_t|
        #     for each node t in the tree, as well as update functions for the last two attributes
        #  2. Compute the first Tree, i.e. T_1, set alpha_1 = 0
        #  3. Construct sequence of trees and alphas.
        pass

    # Algorithm Implementation and Fitting Function
    ###################################################################################################################

    def find_best_splitting_point(self, X, y):

        n, p = X.shape
        split_index = None
        split_value = None
        loss = float('Inf')

        for var_index in range(p):
            # loop through covariates

            x = X[:, var_index]
            sort_index = np.argsort(x)
            sorted_x, sorted_y = x[sort_index], y[sort_index]

            for i in range(self._min_leaf - 1, n - self._min_leaf):
                # loop through potential splitting points

                xi, yi = sorted_x[i], sorted_y[i]
                if xi == sorted_x[i + 1]:
                    continue

                lhs_count, lhs_loss = i + 1, compute_loss(sorted_y[:(i + 1)])
                rhs_count, rhs_loss = n - i - 1, compute_loss(sorted_y[(i + 1):])

                tmp_loss = lhs_count * lhs_loss + rhs_count * rhs_loss  # = SSE_left + SSE_right

                if tmp_loss < loss:
                    split_index, split_value, loss = var_index, xi, tmp_loss

        return split_index, split_value, loss

    def fit(self, X, y, min_leaf=5, max_depth=10):

        # Check Input Values
        if self.is_root:
            assert min_leaf >= 1, "Parameter <<min_leaf>> has to be bigger than one."
            assert max_depth >= 1, "Parameter <<max_depth>> has to be bigger than one."
            assert len(X) == len(y), "Data <<X>> and <<y>> must have the same number of observations."

        # Do Stuff for Root
        if self.is_root:
            try:
                self._feature_names = X.columns.values
            except:
                pass
            # Coerce Input
            X = coerce_to_ndarray(X)
            y = coerce_to_ndarray(y)
            self._num_features = X.shape[1]

        # Set Parameters
        self._min_leaf = min_leaf
        self._value = np.mean(y)
        self._y = y

        # Actual Fitting
        self._split_var, self._split_value, self._loss = self.find_best_splitting_point(X, y)
        self._is_fitted = True
        if self._split_var is None:
            self._loss = len(y) * np.var(y)

        if self._split_var is not None and max_depth > 0:
            index = X[:, self._split_var] <= self._split_value
            x_left, y_left = X[index], y[index]
            x_right, y_right = X[~index], y[~index]

            # root = self if self.is_root else self._root

            self._left_child = DecisionTreePruned(parent=self)  # , root=root)
            self._right_child = DecisionTreePruned(parent=self)  # , root=root)

            # do i need to call any other functions ???
            self._left_child.fit(x_left, y_left, min_leaf, max_depth-1)
            self._right_child.fit(x_right, y_right, min_leaf, max_depth-1)

        self.update_loss()
        return self

    #  Pruning and Post-Fit-Overfitting Avoidance
    ###################################################################################################################

    def return_subtrees(self):
        pass

    def prune(self, regularizer=0):
        #  see above for actual implementation
        DecisionTreePruned.prune_tree(self, regularizer=regularizer)
        self.update_loss()

    def non_naive_pruning(self, X_test, y_test):
        pass

    #  Cross Validation Functions and Co. FUCK IM ANGRY
    ###################################################################################################################

    def cost_complexity(self, X_train, y_train, alpha):
        num_leafs = len(self.get_leafs_in_list())
        return self.score(X_train, y_train) + alpha * num_leafs

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return np.mean((y - y_pred)**2)

    #  Prediction Functions
    ###################################################################################################################

    def predict_row(self, xi):
        if self.is_leaf:
            return self._value
        child = self._left_child if xi[self._split_var] <= self._split_value else self._right_child
        return child.predict_row(xi)

    def predict(self, x):
        x = coerce_to_ndarray(x)
        assert self._is_fitted, "The tree has not yet been fitted; no prediction is possible."
        assert x.shape[1] == self._num_features, "New Data must have the same dimension as the Data used for Fitting."

        return np.array([self.predict_row(xi) for xi in x])
    ###################################################################################################################


def coerce_to_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.values
    else:
        raise TypeError("Object was given with inappropriate type;"
                        "for matrices and vectors only use pandas Series, DataFrame or Numpy ndarrays")


def compute_loss(y, loss_func=None):
    if loss_func is None:
        loss = np.var(y)
    else:
        loss = loss_func(y)

    if loss is None:
        raise ValueError("Loss function cannot be computed on the outcome vector.")
    else:
        return loss
