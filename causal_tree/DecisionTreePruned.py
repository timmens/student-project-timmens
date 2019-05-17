import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class DecisionTreePruned(BaseEstimator, RegressorMixin):  # Inherits from sklearn to use their Cross Validation methods
    # make sure that I do not change anything, because numpy uses copy by reference and not copy by value
    # maybe should use deepcopy more often

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
        self._depth = 0
        self._is_fitted = False
        self._feature_names = None
        self._num_features = None

    def __str__(self):
        if self._is_fitted is False:
            string_dict = {"Loss": self._loss, "Estimate": self._value}
        elif self._feature_names is not None:
            string_dict = {"Loss": self._loss,
                           "Splitting Variable (First split)": self._feature_names[self._split_var],
                           "Splitting Value": self._split_value, "Tree Depth": self.tree_depth, "Estimate": self._value}
        else:
            string_dict = {"Loss": self._loss, "Splitting Feature (First split)": self._split_var,
                           "Splitting Value": self._split_value, "Tree Depth": self.tree_depth, "Estimate": self._value}
        return dict.__str__(string_dict)

    #  Property and Setter Functions
    ###################################################################################################################

    @property
    def loss(self):
        return self._loss

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child

    @property
    def tree_depth(self):
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
    def compute_loss(y, loss_func=None):
        if loss_func is None:
            loss = np.var(y)
        else:
            loss = loss_func(y)

        if loss is None:
            raise ValueError("Loss function cannot be computed on the outcome vector.")
        else:
            return loss

    @staticmethod
    def detect_tree_depth(tree):
        depth_left, depth_right = 0, 0
        if tree.left_child is not None:
            depth_left += 1 + DecisionTreePruned.detect_tree_depth(tree.left_child)
            depth_right += 1 + DecisionTreePruned.detect_tree_depth(tree.right_child)
        return max(depth_left, depth_right)

    @staticmethod
    def coerce_to_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.values
        else:
            raise TypeError("Object was given with inappropriate type;"
                            "for matrices and vectors only use pandas Series, DataFrame or Numpy ndarrays")

    @staticmethod
    def prune_tree(tree):
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

                lhs_count, lhs_loss = i + 1, DecisionTreePruned.compute_loss(sorted_y[:(i + 1)])
                rhs_count, rhs_loss = n - i - 1, DecisionTreePruned.compute_loss(sorted_y[(i + 1):])

                tmp_loss = lhs_count * lhs_loss + rhs_count * rhs_loss  # = SSE_left + SSE_right

                if tmp_loss < loss:
                    split_index, split_value, loss = var_index, xi, tmp_loss

        return split_index, split_value, loss

    def fit(self, X, y, min_leaf=5, max_depth=10):

        # Check Input Values
        if self.is_root:
            assert min_leaf >= 1, "Parameter <<min_leaf>> has to be bigger than one."
            assert max_depth >= 1, "Parameter <<max_depth>> has to be bigger than one."
            assert len(X) == len(y), "Data <<X>> and <<y>> must have to have the same number of observations."

        # Set Parameters
        self._min_leaf = min_leaf
        self._value = np.mean(y)

        # Do Stuff for Root
        if self.is_root:
            try:
                self._feature_names = X.columns.values
            except:
                pass
            # Coerce Input
            X = DecisionTreePruned.coerce_to_ndarray(X)
            y = DecisionTreePruned.coerce_to_ndarray(y)
            self._num_features = X.shape[1]

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

    #  Prediction Functions
    ###################################################################################################################

    def predict_row(self, xi):
        if self.is_leaf:
            return self._value
        child = self._left_child if xi[self._split_var] <= self._split_value else self._right_child
        return child.predict_row(xi)

    def predict(self, x):
        x = DecisionTreePruned.coerce_to_ndarray(x)
        assert self._is_fitted, "The tree has not yet been fitted; no prediction is possible."
        assert x.shape[1] == self._num_features, "New Data must have the same dimension as the Data used for Fitting."

        return np.array([self.predict_row(xi) for xi in x])
    ###################################################################################################################
