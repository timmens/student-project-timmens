import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class DecisionTree(BaseEstimator, RegressorMixin):
    # make sure that I do not change anything, because numpy uses copy by reference and not copy by value

    def __init__(self, parent=None, root=None):
        self._parent = parent
        self._root = root
        self._left_child = None
        self._right_child = None
        self._value = None
        self._min_leaf = None
        self._split_point = None
        self._split_value = None
        self._loss = None
        self._depth = 0
        self._is_fitted = False
        self._feature_names = None

    def __str__(self):
        if self._is_fitted is False:
            string_dict = {"Loss": self._loss, "Estimate": self._value}
        elif self._feature_names is not None:
            string_dict = {"Loss": self._loss, "Splitting Variable (First split)": self._feature_names[self._split_point],
                           "Splitting Value": self._split_value, "Tree Depth": self.tree_depth, "Estimate": self._value}
        else:
            string_dict = {"Loss": self._loss, "Splitting Feature (First split)": self._split_point,
                           "Splitting Value": self._split_value, "Tree Depth": self.tree_depth, "Estimate": self._value}
        return dict.__str__(string_dict)

    #  Property and Setter Functions
    ###################################################################################################################

    @property
    def left_child(self):
        return self._left_child

    @property
    def right_child(self):
        return self._right_child

    @property
    def tree_depth(self):
        return DecisionTree.detect_tree_depth(self)

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

    # Static Methods
    ###################################################################################################################

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
            depth_left += 1 + DecisionTree.detect_tree_depth(tree.left_child)
            depth_right += 1 + DecisionTree.detect_tree_depth(tree.right_child)
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

    # Algorithm Implementation and Fitting Function
    ###################################################################################################################

    def find_best_splitting_point(self, x_train, y_train):
        # for now lets assume x_train and y_train are numpy arrays and that dimensions work; x = [n x p] , y = [n x 1]

        n, p = x_train.shape
        split_index = None
        split_value = None
        loss = DecisionTree.compute_loss(y_train)  # loss = SSE = sum (y_i - y.mean)^2 / n

        for var_index in range(p):
            # loop through covariates

            x = x_train[:, var_index]
            sort_index = np.argsort(x)
            sorted_x, sorted_y = x[sort_index], y_train[sort_index]

            for i in range(self._min_leaf - 1, n - self._min_leaf):
                # loop through potential splitting points

                xi, yi = sorted_x[i], sorted_y[i]
                if xi == sorted_x[i + 1]:
                    continue

                lhs_count, lhs_loss = i+1, self.compute_loss(sorted_y[:(i + 1)])
                rhs_count, rhs_loss = n-i-1, self.compute_loss(sorted_y[(i + 1):])

                tmp_overall_loss = lhs_count * lhs_loss + rhs_count * rhs_loss  # = SSE_left + SSE_right

                if tmp_overall_loss < n * loss:
                    split_index, split_value, loss = var_index, xi, tmp_overall_loss

        return split_index, split_value, loss

    def fit(self, X, y, min_leaf=5, max_depth=10):

        # Check Input Values
        if self.is_root:
            assert min_leaf >= 1, "Parameter <<min_leaf>> has to be bigger than one."
            assert max_depth >= 1, "Parameter <<max_depth>> has to be bigger than one."
            assert len(X) == len(y), "Data and Outcome (X, y) have to have the same number of observations."

        # Set Parameters
        self._min_leaf = min_leaf
        self._value = np.mean(y)
        try:
            self._feature_names = X.columns.values
        except:
            pass

        # Coerce Input
        X = DecisionTree.coerce_to_ndarray(X)
        y = DecisionTree.coerce_to_ndarray(y)

        # Actual Fitting
        self._split_point, self._split_value, self._loss = self.find_best_splitting_point(X, y)
        self._is_fitted = True

        #  implement the max_depth criterion

        if self._split_point is not None and max_depth > 0:
            index = X[:, self._split_point] <= self._split_value
            x_left, y_left = X[index], y[index]
            x_right, y_right = X[~index], y[index]

            root = self if self.is_root else self._root

            self._left_child = DecisionTree(parent=self, root=root)
            self._right_child = DecisionTree(parent=self, root=root)

            # do i need to call any other functions ???
            self._left_child.fit(x_left, y_left, min_leaf, max_depth-1)
            self._right_child.fit(x_right, y_right, min_leaf, max_depth-1)

        return self

    #  Pruning and Post-Fit-Overfitting Avoidance
    ###################################################################################################################

    #  Prediction Functions
    ###################################################################################################################

    def predict_row(self, xi):
        if self.is_leaf:
            return self._value
        child = self._left_child if xi[self._split_point] <= self._split_value else self._right_child
        return child.predict_row(xi)

    def predict(self, x):
        #  Check if x has the right dtype and the right dimension
        if not self._is_fitted:
            print("The tree has not yet been fitted; no prediction is possible.")
            return None
        else:
            return np.array([self.predict_row(xi) for xi in x])

    ###################################################################################################################
