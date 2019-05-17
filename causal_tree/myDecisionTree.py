import numpy as np
import pandas as pd


class myDecisionTree:
    '''
        arguments:
                y: outcome vector (n x 0)
                x: training data  (n x k)
                index: stores indices of the rows it contains
                max_depth: (maximum) depth of decision tree
                min_leaf: (minimum) number of observations in a leaf
    '''
    # Should IMPLEMENT:
    # 1) pruning ??? cross validation ???

    def __init__(self, x, y, max_depth=10, min_leaf=5, feature_names=None):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.var_index = None  # variable on which (this) decision tree, or NODE will split
        self.split_point = None  # observed feature index where we split the sorted data
        self.n = len(y)
        self.estimate = y.mean()
        self._feature_names = myDecisionTree.get_columns_names_from_pandas_else(self.x, feature_names)
        self.x, self.y = self.coerce_to_ndarray(self.x), self.coerce_to_ndarray(self.y)
        self.k = x.shape[1]
        self.left_child, self.right_child = None, None
        self.loss_func = myDecisionTree.loss_function
        self.loss = self.n*self.loss_func(self.y)
        self.num_children = 0
        self.is_root = True
        self.is_fitted = False

    def __str__(self):
        if self._feature_names is None or self.var_index is None:
            string_dict = {"loss": self.loss, "node_estimate": self.estimate,
                           "split_var_index": self.var_index, "split_data_point": self.split_point,
                           "num_children": self.num_children}
        else:
            string_dict = {"loss": self.loss, "node_estimate": self.estimate,
                           "split_feature": self._feature_names[self.var_index],
                           "split_data_point": self.split_point, "num_children": self.num_children}

        return dict.__str__(string_dict)

    #  Static Methods
    ###################################################################################################################
    @staticmethod
    def get_columns_names_from_pandas_else(obj, _else):
        return list(obj) if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series) else _else

    @staticmethod
    def coerce_to_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.values
        else:
            raise TypeError("object was given with inappropriate type;"
                              "for matrices and vectors only use pandas Series, Dataframe or Numpy ndarrays")

    @staticmethod
    def loss_function(y):
        return ((y - y.mean())**2).sum()

    @staticmethod
    def count_num_children(tree):
        count_left = 0
        count_right = 0
        if tree.left_child is not None:
            count_left += 1 + myDecisionTree.count_num_children(tree.left_child)
        if tree.right_child is not None:
            count_right += 1 + myDecisionTree.count_num_children(tree.right_child)
        return max(count_left, count_right)

    @staticmethod
    def compute_weighted_loss(tree):
        loss = 0
        if tree.left_child is None:
            loss += tree.loss
        elif tree.left_child is not None:
            loss += myDecisionTree.compute_weighted_loss(tree.left_child)
            loss += myDecisionTree.compute_weighted_loss(tree.right_child)
        return loss

    #  Various Other Functions
    ###################################################################################################################
    def detect_num_children(self):
        self.num_children = myDecisionTree.count_num_children(self)

    def update_loss(self):
        self.loss = myDecisionTree.compute_weighted_loss(self)

    #  Algorithm Implementation and Fitting Function
    ###################################################################################################################
    def find_split_point_single_variable(self, var_index):
        x, y = self.x[:, var_index], self.y
        sort_index = np.argsort(x)
        sorted_x, sorted_y = x[sort_index], y[sort_index]

        for i in range(self.min_leaf, self.n - self.min_leaf):
            xi, yi = sorted_x[i], sorted_y[i]
            if xi == sorted_x[i + 1]:
                continue

            lhs_count, lhs_loss = i, self.loss_func(sorted_y[:(i + 1)])
            rhs_count, rhs_loss = self.n - i, self.loss_func(sorted_y[(i + 1):])

            tmp_overall_loss = lhs_count*lhs_loss + rhs_count*rhs_loss

            if tmp_overall_loss < self.loss:
                self.var_index, self.loss, self.split_point = var_index, tmp_overall_loss, xi

    def find_split_point_all_variables(self):
        for j in range(self.k):
            self.find_split_point_single_variable(j)

    def split_tree(self):
        if self.max_depth == 0:
            return

        self.find_split_point_all_variables()

        if self.split_point is None:
            return

        index_left = self.x[:, self.var_index] <= self.split_point

        self.left_child = myDecisionTree(self.x[index_left, :], self.y[index_left],
                                         max_depth=self.max_depth-1, min_leaf=self.min_leaf, feature_names=self._feature_names)
        self.right_child = myDecisionTree(self.x[~index_left], self.y[~index_left],
                                          max_depth=self.max_depth-1, min_leaf=self.min_leaf, feature_names=self._feature_names)
        self.left_child.is_root = False
        self.right_child.is_root = False

    def fit(self):
        if self is None:
            print("DecisionTree object is not initialized.")
            return

        self.is_fitted = True

        if self.n >= 2*self.min_leaf:
            self.split_tree()
            if self.left_child is not None:
                self.left_child.fit()
            if self.right_child is not None:
                self.right_child.fit()
        self.detect_num_children()

    #  Validate Fit
    ###################################################################################################################
    def validate_fit(self, x_test, y_test, validation_func):
        assert callable(validation_func)
        y_predict = self.predict(x_test)
        return validation_func(y_test, y_predict)

    #  Setter Functions
    ###################################################################################################################
    def set_feature_names(self, names):
        assert len(names) == self.k
        self._feature_names = names

    def set_loss_func(self, loss_func):
        assert callable(loss_func)
        self.loss_func = loss_func

    #  Property Functions
    ###################################################################################################################
    @property
    def split_col_index(self): return self.var_index

    @property
    def split_col(self): return self.x[:, self.var_index]

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    #  Prediction Functions
    ###################################################################################################################
    def predict(self, x):
        x = myDecisionTree.coerce_to_ndarray(x)
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.estimate
        child = self.left_child if xi[self.var_index] <= self.split_point else self.right_child
        return child.predict_row(xi)
