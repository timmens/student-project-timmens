import pandas as pd
import numpy as np
from copy import deepcopy
import sklearn


class DecisionTreePruned:
    # 1. make sure that I do not change anything, because numpy uses copy by reference and not copy by value
    # maybe should use deepcopy more often
    # 2. implement method prune_non_naively

    def __init__(self, parent=None):
        self._parent = parent
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
    def is_fitted(self):
        return self._is_fitted

    @property
    def number_of_leafs(self):
        # Because this is a property and not an attribute it gets updated automatically when it is called
        # however this might be a little inefficient, we will see
        return DecisionTreePruned.get_number_of_leafs(self)

    @property
    def value(self):
        return self._value

    @property
    def node_loss(self):
        return self._node_loss

    @property
    def branch_loss(self):
        self.update_branch_loss()
        return self._branch_loss

    # Various Auxiliary Functions
    ###################################################################################################################

    def update_branch_loss(self):
        leaf_list = DecisionTreePruned.get_leafs_in_list(self)
        loss_array = np.array([leaf.branch_loss for leaf in leaf_list])
        self._branch_loss = np.sum(loss_array)

    def output_partition_estimates(self):
        leaf_list = DecisionTreePruned.get_leafs_in_list(self)
        for i, leaf in enumerate(leaf_list):
            print("Leaf {:d}; Estimate: {:3.03f}".format(i, leaf.value))

    # Static Methods (Some of which should probably not be static methods)
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
    def get_number_of_leafs(tree):
        return len(DecisionTreePruned.get_leafs_in_list(tree))

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
    def validate(tree, X_test, y_test, metric=None):
        #  returns validation metric on predicted and true outcomes
        if metric is None:
            def metric(pred, true):
                return np.mean((pred - true)**2)
        y_pred = tree.predict(X_test)
        return np.mean(metric(y_pred, y_test))

    @staticmethod
    def prune_tree_non_naively(fitted_tree, X_test, y_test):
        depth = fitted_tree.depth
        if depth < 1:
            print("Nothing to prune here.")
            return
        for i in range(depth):
            for parent_node in DecisionTreePruned.get_level_in_list(fitted_tree, depth-i-1):
                if parent_node.left_child is not None:
                    DecisionTreePruned.collapse_node_non_naively(parent_node, X_test, y_test)

    @staticmethod
    def get_first_subtree(fitted_tree, thresh=None):
        subtree = deepcopy(fitted_tree)
        if thresh is None:
            thresh = np.sqrt(np.var(subtree.y)) / 50
        depth = subtree.depth
        if depth < 1:
            return subtree
        for i in range(depth):
            for parent_node in DecisionTreePruned.get_level_in_list(subtree, depth-i-1):
                if parent_node.left_child is not None:
                    if parent_node.node_loss <= parent_node.left_child.node_loss + \
                            parent_node.right_child.node_loss + thresh:
                        parent_node.left_child, parent_node.right_child = None, None
        return subtree

    @staticmethod
    def get_pruned_tree_and_alpha_sequence(fitted_tree):
        #  let tree be the node in question, i.e. tree = t
        #  then R(t)   = tree.node_loss
        #       R(T_t) = tree.branch_loss (gets updated automatically when called)
        #       |T_t|  = tree.number_of_leafs (gets updated automatically when called)
        #  2. Compute the first Tree, i.e. T_1, set alpha_1 = 0
        #  3. Construct sequence of trees and alphas.
        assert isinstance(fitted_tree, DecisionTreePruned), 'This method only works on Decision Trees'
        if not fitted_tree.is_fitted:
            raise ValueError('This method only works on fitted trees')

        alphas = [0]
        subtrees = [DecisionTreePruned.get_first_subtree(fitted_tree)]  # get_first_subtree() does deepcopy

        index = 0
        while not subtrees[index].is_root:
            tmp_argmin, tmp_min = g(subtrees[index])
            tmp_subtree = deepcopy(subtrees[index])
            alphas.append(tmp_min)
            for node in tmp_min:
                node.left_child = None
                node.right_child = None
            subtrees.insert(index, tmp_subtree)
            index += 1

        return {'alphas': alphas, 'subtrees': subtrees}

    @staticmethod
    def get_subtree_corresponding_to_arbitrary_alpha(tree, alpha):
        sequences = DecisionTreePruned.get_pruned_tree_and_alpha_sequence(tree)
        alphas = sequences['alphas']
        subtrees = sequences['subtrees']
        alphas = np.array(alphas)
        if alpha >= alphas[-1]:
            return subtrees[-1]
        else:
            index = np.where(alphas > alpha)[0][0]
            return subtrees[index]

    @staticmethod
    def get_optimal_subtree_via_k_fold_cv(fitted_tree, X_learn, y_learn, k):
        #  Here <<X_learn>> and <<y_learn>> were used to get <<fitted_tree>>

        kf = sklearn.model_selection.KFold(k)
        tree_max = fitted_tree  # complete maximal tree
        tree_max_sequences = DecisionTreePruned.get_pruned_tree_and_alpha_sequence(tree_max)
        tree_k_max_list = []  # list of maximal trees in each cross validation sample
        tree_k_subtree_dict = []
        test_X = []
        test_y = []

        for train_index, test_index in kf.split(X_learn, y_learn):
            tmp_tree = DecisionTreePruned()
            tmp_tree.fit(X_learn[train_index], y_learn[train_index])
            tree_k_max_list.append(tmp_tree)
            test_X.append(X_learn[test_index])
            test_y.append(y_learn[test_index])

        for tree in tree_k_max_list:
            tree_k_subtree_dict.append(DecisionTreePruned.get_pruned_tree_and_alpha_sequence(tree))

        alphas = tree_max_sequences['alphas']
        potential_subtrees = tree_max_sequences['subtrees']

        alpha_cv_errors = []
        for alpha in alphas:
            err_alpha = 0
            for k, cv_tree in enumerate(tree_k_max_list):
                cv_alpha_subtree = DecisionTreePruned.get_subtree_corresponding_to_arbitrary_alpha(cv_tree, alpha)
                err_alpha += DecisionTreePruned.validate(cv_alpha_subtree, test_X[k], test_y[k])

            alpha_cv_errors.append(err_alpha / k)

        alpha_cv_errors = np.array(alpha_cv_errors)
        optimal_index = int(np.where(alpha_cv_errors == alpha_cv_errors.min()))
        optimal_subtree = potential_subtrees[optimal_index]
        return optimal_subtree

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
            except AttributeError:
                pass
            # Coerce Input
            X = coerce_to_ndarray(X)
            y = coerce_to_ndarray(y)
            self._num_features = X.shape[1]

        # Set Parameters
        self._min_leaf = min_leaf
        self._value = np.mean(y)
        self._y = y
        self._node_loss = len(y) * np.var(y)

        # Actual Fitting
        self._split_var, self._split_value, tmp_loss = self.find_best_splitting_point(X, y)
        self._is_fitted = True

        if self._split_var is None:
            self._branch_loss = self._node_loss
        else:
            self._branch_loss = tmp_loss

        if self._split_var is not None and max_depth > 0:
            index = X[:, self._split_var] <= self._split_value
            x_left, y_left = X[index], y[index]
            x_right, y_right = X[~index], y[~index]

            self._left_child = DecisionTreePruned(parent=self)
            self._right_child = DecisionTreePruned(parent=self)

            # do i need to call any other functions ???
            self._left_child.fit(x_left, y_left, min_leaf, max_depth-1)
            self._right_child.fit(x_right, y_right, min_leaf, max_depth-1)

        self.update_branch_loss()
        return self

    #  Pruning Algorithm
    ###################################################################################################################

    def return_subtrees(self):
        pass

    def prune(self, regularizer=0):
        #  see above for actual implementation
        DecisionTreePruned.prune_tree(self, regularizer=regularizer)
        self.update_branch_loss()

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


#  General Function (Some of which might be static methods in a strict sense)
#######################################################################################################################

def pre_order_traverse_tree(root: DecisionTreePruned, func=None) -> list:
    values = []
    if func is None:
        if root is not None:
            values.append(root)
            values.extend(pre_order_traverse_tree(root.left_child))
            values.extend(pre_order_traverse_tree(root.right_child))
    else:
        if root is not None:
            values.append(func(root))
            values.extend(pre_order_traverse_tree(root.left_child, func=func))
            values.extend(pre_order_traverse_tree(root.right_child, func=func))
    return values


def g_value(node: DecisionTreePruned) -> float:
    return float('Inf') if node.is_leaf else (node.node_loss - node.branch_loss) / (node.number_of_leafs - 1)


def g(branch: DecisionTreePruned) -> tuple:
    nodes = pre_order_traverse_tree(branch)
    values = np.array(pre_order_traverse_tree(branch, g_value))
    minimum = values.min()
    argmin_index = np.where(values == minimum)[0]
    argmin = [nodes[ix] for ix in argmin_index]
    return argmin, minimum


def coerce_to_ndarray(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.values
    else:
        raise TypeError("Object was given with inappropriate type;"
                        "for matrices and vectors only use pandas Series, DataFrame or Numpy ndarrays")


def compute_loss(y, loss_func=None) -> float:
    if loss_func is None:
        loss = np.var(y)
    else:
        loss = loss_func(y)

    if loss is None:
        raise ValueError("Loss function cannot be computed on the outcome vector.")
    else:
        return loss
