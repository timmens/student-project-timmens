import pandas as pd
import numpy as np
from graphviz import Graph
from copy import deepcopy
from sklearn.model_selection import KFold


class DecisionTree:
    def __init__(self, parent=None):
        self._parent = parent
        self._left_child = None
        self._right_child = None
        self._value = None
        self._min_leaf = None
        self._split_var = None
        self._split_value = None
        self._node_loss = None
        self._branch_loss = None
        self._is_fitted = False
        self._feature_names = None
        self._num_features = None
        self._y = None

    def __str__(self):
        if self._is_fitted is False:
            string_dict = {"Loss": self.branch_loss, "Estimate": self.value}
        elif self._feature_names is not None:
            string_dict = {
                "Loss": self.branch_loss,
                "Splitting Variable (First split)": self._feature_names[
                    self._split_var
                ],
                "Splitting Value": self._split_value,
                "Tree Depth": self.depth,
                "Estimate": self.value,
            }
        else:
            string_dict = {
                "Loss": self.branch_loss,
                "Splitting Feature (First split)": self._split_var,
                "Splitting Value": self._split_value,
                "Tree Depth": self.depth,
                "Estimate": self.value,
            }
        return dict.__str__(string_dict)

    #  Property and Setter Functions

    @property
    def y(self):
        return self._y

    @property
    def parent(self):
        return self._parent

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
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        if self.feature_names is None:
            self._feature_names = value

    @property
    def depth(self):
        return DecisionTree.detect_tree_depth(self)

    @property
    def is_leaf(self):
        if self._is_fitted:
            return self._left_child is None
        else:
            print(
                "The tree has not been fitted yet, hence it is root and leaf at the "
                "same time.\n"
            )
            return True

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def number_of_leafs(self):
        return DecisionTree.get_number_of_leafs(self)

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

    def update_branch_loss(self):
        leaf_list = DecisionTree.get_leafs_in_list(self)
        loss_array = np.array([leaf.node_loss for leaf in leaf_list])
        self._branch_loss = np.sum(loss_array)

    def output_partition_estimates(self):
        leaf_list = DecisionTree.get_leafs_in_list(self)
        for i, leaf in enumerate(leaf_list):
            print("Leaf {:d}; Estimate: {:3.03f}".format(i, leaf.value))

    def splitting_info_to_string(self):
        if self.left_child is None:
            return ""
        else:
            if self._feature_names is None:
                return "Variable %d <= %3.3f" % (self._split_var, self._split_value)
            else:
                return "%s <= %3.3f" % (
                    self._feature_names[self._split_var],
                    self._split_value,
                )

    # Static Methods (Some of which should probably not be static methods)

    @staticmethod
    def get_leafs_in_list(tree):
        leaf_list = []
        if tree.left_child is None:
            leaf_list.append(tree)
        else:
            leaf_list.extend(DecisionTree.get_leafs_in_list(tree.left_child))
            leaf_list.extend(DecisionTree.get_leafs_in_list(tree.right_child))
        return leaf_list

    @staticmethod
    def get_number_of_leafs(tree):
        return len(DecisionTree.get_leafs_in_list(tree))

    @staticmethod
    def get_level_in_list(tree, level):
        #  level: 0 -> root, 1 -> first layer, 2 -> ...
        level_list = []
        if level == 0:
            level_list.append(tree)
        else:
            if tree.left_child is not None:
                level_list.extend(
                    DecisionTree.get_level_in_list(tree.left_child, level - 1)
                )
                level_list.extend(
                    DecisionTree.get_level_in_list(tree.right_child, level - 1)
                )
        return level_list

    @staticmethod
    def detect_tree_depth(tree):
        depth_left, depth_right = 0, 0
        if tree.left_child is not None:
            depth_left += 1 + DecisionTree.detect_tree_depth(tree.left_child)
            depth_right += 1 + DecisionTree.detect_tree_depth(tree.right_child)
        return max(depth_left, depth_right)

    @staticmethod
    def validate(tree, X_test, y_test, metric=None):
        #  returns assumed validation metric on predicted and true outcomes
        if metric is None:

            def metric(pred, true):
                return np.mean(
                    (pred - true) ** 2
                )  # if no metric is given Mean Squared Error is used (MSE)

        y_pred = tree.predict(X_test)
        return np.mean(metric(y_pred, y_test))

    @staticmethod
    def get_first_subtree(fitted_tree, thresh=None):
        subtree = deepcopy(fitted_tree)
        if thresh is None:
            thresh = np.sqrt(np.var(subtree.y)) / 50
        depth = subtree.depth
        if depth < 1:
            return subtree
        for i in range(depth):
            for parent_node in DecisionTree.get_level_in_list(subtree, depth - i - 1):
                if parent_node.left_child is not None:
                    if (
                        parent_node.node_loss
                        <= parent_node.left_child.node_loss
                        + parent_node.right_child.node_loss
                        + thresh
                    ):
                        parent_node.left_child, parent_node.right_child = None, None
        return subtree

    @staticmethod
    def get_pruned_tree_and_alpha_sequence(fitted_tree, thresh):
        assert isinstance(
            fitted_tree, DecisionTree
        ), "This method only works on Decision Trees"
        if not fitted_tree.is_fitted:
            raise ValueError("This method only works on fitted trees")

        alphas = [0]
        subtrees = [
            DecisionTree.get_first_subtree(fitted_tree, thresh)
        ]  # get_first_subtree() does deepcopy

        index = 0
        while subtrees[index].left_child is not None:
            tmp_argmin, tmp_min = g(subtrees[index])
            tmp_subtree = deepcopy(subtrees[index])
            alphas.append(tmp_min)
            for node in tmp_argmin:
                node.left_child = None
                node.right_child = None
            subtrees.insert(index, tmp_subtree)
            index += 1

        if not test_monotonicity_list(alphas) and not test_monotonicity_list(
            alphas, strictly=False
        ):
            raise RuntimeError("Sequence of alphas is not increasing")
        if not test_monotonicity_list(alphas):
            print("Sequence of alphas is only weakly increasing.")
        return {
            "alphas": alphas,
            "subtrees": subtrees,
        }  # i think i want: return alphas, subtrees

    @staticmethod
    def get_subtree_crspnd_to_abtry_alpha(tree, alpha, thresh):
        sequences = DecisionTree.get_pruned_tree_and_alpha_sequence(tree, thresh)
        alphas = sequences["alphas"]
        subtrees = sequences["subtrees"]
        alphas = np.array(alphas)
        if alpha >= alphas[-1]:
            return subtrees[-1]
        else:
            index = np.where(alphas > alpha)[0][0]
            return subtrees[index]

    @staticmethod
    def apply_kFold_CV(X_learn, y_learn, k=5, thresh=0, fitted_tree=None):
        try:
            feature_names = X_learn.columns.values
        except AttributeError:
            feature_names = None

        if fitted_tree is None:
            fitted_tree = DecisionTree()
            fitted_tree.fit(X_learn, y_learn)
        assert len(y_learn) == len(X_learn), (
            "Argument <<X_learn>> and <<y_learn>> must have the same number "
            "of observations."
        )
        X_learn = coerce_to_ndarray(X_learn)
        y_learn = coerce_to_ndarray(y_learn)

        kf = KFold(k)
        tree_max = fitted_tree  # complete maximal tree
        tree_max_sequences = DecisionTree.get_pruned_tree_and_alpha_sequence(
            tree_max, thresh
        )
        tree_k_max_list = []  # list of maximal trees in each cross validation sample
        tree_k_subtree_dict = []
        test_X = []
        test_y = []

        for train_index, test_index in kf.split(X_learn, y_learn):
            tmp_tree = DecisionTree()
            tmp_tree.fit(X_learn[train_index], y_learn[train_index])
            tree_k_max_list.append(tmp_tree)
            test_X.append(X_learn[test_index])
            test_y.append(y_learn[test_index])

        for tree in tree_k_max_list:
            tree_k_subtree_dict.append(
                DecisionTree.get_pruned_tree_and_alpha_sequence(tree, thresh)
            )

        alphas = tree_max_sequences["alphas"]
        potential_subtrees = tree_max_sequences["subtrees"]

        alpha_cv_errors = []
        for alpha in alphas:
            err_alpha = 0
            for k, cv_tree in enumerate(tree_k_max_list):
                cv_alpha_subtree = DecisionTree.get_subtree_crspnd_to_abtry_alpha(
                    cv_tree, alpha, thresh
                )
                err_alpha += DecisionTree.validate(
                    cv_alpha_subtree, test_X[k], test_y[k]
                )

            alpha_cv_errors.append(err_alpha / k)

        alpha_cv_errors = np.array(alpha_cv_errors)
        optimal_index = int(
            np.where(alpha_cv_errors == alpha_cv_errors.min())[0]
        )  # if multiple trees achieve the
        optimal_subtree = potential_subtrees[optimal_index]
        optimal_subtree.feature_names = feature_names

        return optimal_subtree

    # Algorithm Implementation and Fitting Functions

    def find_best_splitting_point(self, X, y):

        n, p = X.shape
        split_index = None
        split_value = None
        loss = float("Inf")

        for var_index in range(p):
            # loop through covariates

            x = X[:, var_index]
            sort_index = np.argsort(x)
            sorted_x, sorted_y = x[sort_index], y[sort_index]

            for i in range(self._min_leaf - 1, n - self._min_leaf):
                # loop through potential splitting points

                xi = sorted_x[i]
                if xi == sorted_x[i + 1]:
                    continue

                lhs_count, lhs_loss = i + 1, compute_loss(sorted_y[: (i + 1)])
                rhs_count, rhs_loss = n - i - 1, compute_loss(sorted_y[(i + 1) :])

                tmp_loss = (
                    lhs_count * lhs_loss + rhs_count * rhs_loss
                )  # = SSE_left + SSE_right

                if tmp_loss < loss:
                    split_index, split_value, loss = var_index, xi, tmp_loss

        return split_index, split_value, loss

    def fit(self, X, y, min_leaf=5, max_depth=10):
        # Check Input Values
        if self.is_root:
            assert min_leaf >= 1, "Parameter <<min_leaf>> has to be bigger than one."
            assert (
                max_depth >= 1
            ), "Parameter <<max_depth>> has to be bigger or equal than one."
            assert len(X) == len(
                y
            ), "Data <<X>> and <<y>> must have the same number of observations."
            assert len(y) >= 2 * min_leaf, (
                "Data has not enough observations for a single split to occur "
                "given value of <<min_leaf>>."
            )

        # Do Stuff for Root
        if self.is_root:
            try:
                self._feature_names = X.columns.values
            except AttributeError:
                pass
            X = coerce_to_ndarray(
                X
            )  # coerce to ndarray (since all following functions expect numpy arrays)
            y = coerce_to_ndarray(y)
            self._num_features = X.shape[1]

        # Set Parameters
        self._min_leaf = min_leaf
        self._value = np.mean(y)
        self._y = y
        self._node_loss = np.sum(
            (self._y - self._value) ** 2
        )  # = sum of squared residuals when predicting the mean

        # Actual Fitting
        self._split_var, self._split_value, tmp_loss = self.find_best_splitting_point(
            X, y
        )
        self._is_fitted = True

        if self._split_var is None:
            self._branch_loss = self._node_loss
        else:
            self._branch_loss = tmp_loss

        if self._split_var is not None and max_depth >= 1:
            index = X[:, self._split_var] <= self._split_value

            self._left_child = DecisionTree(parent=self)
            self._right_child = DecisionTree(parent=self)
            self._left_child.feature_names = self.feature_names
            self._right_child.feature_names = self.feature_names

            self._left_child.fit(X[index], y[index], min_leaf, max_depth - 1)
            self._right_child.fit(X[~index], y[~index], min_leaf, max_depth - 1)

        self.update_branch_loss()
        return self

    def predict_row(self, xi):
        if self.is_leaf:
            return self._value
        child = (
            self._left_child
            if xi[self._split_var] <= self._split_value
            else self._right_child
        )
        return child.predict_row(xi)

    def predict(self, x):
        x = coerce_to_ndarray(x)
        assert (
            self._is_fitted
        ), "The tree has not yet been fitted; no prediction is possible."
        assert (
            x.shape[1] == self._num_features
        ), "New Data must have the same dimension as the Data used for Fitting."

        return np.array([self.predict_row(xi) for xi in x])

    def plot(self, render=False, save=False, filename=None):
        if filename is None:
            filename = "decision_tree.svg"
        dot = Graph(name="decision_tree", filename=filename, format="svg")
        dot.node(
            str(id(self)),
            self.splitting_info_to_string()
            + "\nestimate:"
            + str(round(float(self.value), 3)),
        )
        for i in range(self.depth):
            nodes = DecisionTree.get_level_in_list(self, i + 1)
            for node in nodes:
                if node.left_child is None:
                    dot.node(
                        str(id(node)),
                        "This node is not split"
                        + "\nestimate:"
                        + str(round(float(node.value), 3)),
                    )
                    dot.edge(str(id(node.parent)), str(id(node)))
                else:
                    dot.node(
                        str(id(node)),
                        node.splitting_info_to_string()
                        + "\nestimate:"
                        + str(round(float(node.value), 3)),
                    )
                    dot.edge(str(id(node.parent)), str(id(node)))
        if render:
            dot.render(view=True)
        if save:
            dot.save()
        return dot


#  General Function (Some of which might be static methods in a strict sense)


def pre_order_traverse_tree(root: DecisionTree, func=None) -> list:
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


def g_value(node: DecisionTree) -> float:
    return (
        float("Inf")
        if node.is_leaf
        else (node.node_loss - node.branch_loss) / (node.number_of_leafs - 1)
    )


def g(branch: DecisionTree) -> tuple:
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
        raise TypeError(
            "Object was given with inappropriate type;"
            "for matrices and vectors only use pandas Series, DataFrame or "
            "Numpy ndarrays"
        )


def compute_loss(y, loss_func=None) -> float:
    if loss_func is None:
        loss = np.var(y)
    else:
        loss = loss_func(y)

    if loss is None:
        raise ValueError("Loss function cannot be computed on the outcome vector.")
    else:
        return float(loss)


def test_monotonicity_list(lst: list, strictly=True) -> bool:
    if strictly:
        return all(x < y for x, y in zip(lst, lst[1:]))
    else:
        return all(x <= y for x, y in zip(lst, lst[1:]))
