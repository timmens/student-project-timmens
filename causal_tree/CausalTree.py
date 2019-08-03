import pandas as pd
import numpy as np
from graphviz import Graph
from copy import deepcopy
from sklearn.model_selection import KFold


class CausalTree:
    """ The CausalTree class fits a Causal Tree to given data and computes (potentially)
     heterogeneous treatment effects for new data.
    """

    def __init__(self, parent=None):
        """
        :param parent: an object of class CausalTree
        """
        self._parent = parent
        self._left_child = None
        self._right_child = None
        self._value = None
        self._min_leaf = None
        self._max_distance = None
        self._crit_num_obs = None
        self._split_var = None
        self._split_value = None
        self._node_loss = None
        self._branch_loss = None
        self._is_fitted = False
        self._feature_names = None
        self._num_features = None
        self._y = None
        self._y_transformed = None
        self._treatment_status = None
        self._cv_error = None

    def __str__(self):
        """ method for representing CausalTree object when called by print / str method

        :return: readable string representation of CausalTree
        """
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

    def __repr__(self):
        """ method for representing CausalTree object when called by evaluating / repr

        :return: unambiguous string representation of CausalTree
        """
        return f"Causal Tree; fitted = {str(self.is_fitted)}; id = {id(self)}"

    #  Property and Setter Functions

    @property
    def y(self):
        """ndarray: 1D array containing outcomes."""
        return self._y

    @property
    def y_transformed(self):
        """ndarray: 1D array containing transformed outcomes"""
        return self._y_transformed

    @property
    def treatment_status(self):
        """ndarray: 1D array containing treatment status of observations"""
        return self._treatment_status

    @property
    def parent(self):
        """CausalTree: Parent node."""
        return self._parent

    @property
    def left_child(self):
        """CausalTree: Left child."""
        return self._left_child

    @property
    def right_child(self):
        """CausalTree: Right child."""
        return self._right_child

    @right_child.setter
    def right_child(self, node):
        self._right_child = node

    @left_child.setter
    def left_child(self, node):
        self._left_child = node

    @property
    def cv_error(self):
        """float: Cross-validation error of tree with this node as root."""
        return self._cv_error

    @property
    def feature_names(self):
        """ndarray: 1D array containing feature names"""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        if self.feature_names is None:
            self._feature_names = value

    @property
    def depth(self):
        """int: Tree depth starting from this node."""
        return CausalTree.detect_tree_depth(self)

    @property
    def is_leaf(self):
        """bool: True if given node is a leaf and false otherwise."""
        if self._is_fitted:
            return self._left_child is None
        else:
            print(
                "The tree has not been fitted yet, "
                "hence it is root and leaf at the same time.\n"
            )
            return True

    @property
    def is_root(self):
        """bool: True if node is root of tree and false otherwise."""
        return self._parent is None

    @property
    def is_fitted(self):
        """bool: True if method fit was already called without errors, false else."""
        return self._is_fitted

    @property
    def number_of_leafs(self):
        """int: Number of leafs originating from this node."""
        return CausalTree.get_number_of_leafs(self)

    @property
    def value(self):
        """float: Estimated treatment effect using observations contained in this
         node."""
        return self._value

    @property
    def node_loss(self):
        """float: Estimated loss arising from estimating treatment effects with
        attribute value on observations contained in this node."""
        return self._node_loss

    @property
    def branch_loss(self):
        """float: Estimated loss arising from summing attribute node_loss of all
        leafs originating from this node."""
        self.update_branch_loss()
        return self._branch_loss

    @property
    def min_leaf(self):
        """int: Minimum number of observations required in each leaf."""
        return self._min_leaf

    @property
    def max_distance(self):
        """int: Maximum number of absolute difference in observations with treatment
        and observations without treatment in each leaf."""
        return self._max_distance

    @property
    def crit_num_obs(self):
        """int: Critical value of observations in each leaf: When reached assertion
        using attribute max_distance is binding."""
        return self._crit_num_obs

    # Various Auxiliary Functions

    def update_branch_loss(self):
        """Computes loss in leafs originating from self and updates attribute
        branch_loss with sum of losses.
        """
        leaf_list = CausalTree.get_leafs_in_list(self)
        loss_array = np.array([leaf.node_loss for leaf in leaf_list])
        self._branch_loss = np.sum(loss_array)

    def output_partition_estimates(self):
        """Prints treatment effect estimate of each partition (leaf) --Mostly useful for
        debugging."""
        leaf_list = CausalTree.get_leafs_in_list(self)
        for i, leaf in enumerate(leaf_list):
            print("Leaf {:d}; Estimate: {:3.03f}".format(i, leaf.value))

    def splitting_info_to_string(self):
        """Constructs string encoding information about splitting feature and splitting
        point of given node.

        :return: str: Splitting information of given node.
        """
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
    def transform_outcome(y, treatment_status, p=None):
        """Transforms outcomes using propensity scores and treatment status, such that
        expected value of individual outcomes is equal to individual treatment effects.
        If p=None we implicitly set p=1/2 for all observations.

        :param y: ndarray: 1D array containing outcomes.
        :param treatment_status: ndarray: 1D array containing treatment status.
        :param p: ndarray: 1D array containing individual propensity scores.
        :return: ndarray: 1D array containing transformed outcomes.
        """
        if treatment_status.dtype != "int":
            treatment_status = np.array(treatment_status, dtype="int")
        if p is None:
            y_star = 2 * y * treatment_status - 2 * y * (1 - treatment_status)
        else:
            y_star = y * (treatment_status - p) / (p * (1 - p))
        return y_star

    @staticmethod
    def get_leafs_in_list(tree):
        """
        :param tree: CausalTree.
        :return: list: Leafs originating from tree.
        """
        leaf_list = []
        if tree.left_child is None:
            leaf_list.append(tree)
        else:
            leaf_list.extend(CausalTree.get_leafs_in_list(tree.left_child))
            leaf_list.extend(CausalTree.get_leafs_in_list(tree.right_child))
        return leaf_list

    @staticmethod
    def get_number_of_leafs(tree):
        """
        :param tree: CausalTree
        :return: int: Number of leafs originating from tree.
        """
        return len(CausalTree.get_leafs_in_list(tree))

    @staticmethod
    def detect_tree_depth(tree):
        """
        :param tree: CausalTree.
        :return: int: Maximum number of nodes to cross when traveling from tree to any
        leaf originating from tree, i.e. depth of CausalTree starting at node tree.
        """
        depth_left, depth_right = 0, 0
        if tree.left_child is not None:
            depth_left += 1 + CausalTree.detect_tree_depth(tree.left_child)
            depth_right += 1 + CausalTree.detect_tree_depth(tree.right_child)
        return max(depth_left, depth_right)

    @staticmethod
    def validate(tree, X_test, y_transformed_test, metric=None):
        """Evaluates tree model on testing data given some user specific metric. If
        metric is None we use standard l2 distance.

        :param tree: CausalTree.
        :param X_test: ndarray: 2D array containing testing data on features.
        :param y_transformed_test: ndarray: 1D array containing testing data on
        transformed outcomes.
        :param metric: function: User specific metric.
        :return: float: loss obtained from predicting treatment effects in testing data
        using tree.
        """
        #  returns assumed validation metric on predicted and true outcomes
        tau_pred = tree.predict(X_test)
        if metric is None:
            return np.mean((tau_pred - y_transformed_test) ** 2)
        else:
            return metric(tau_pred, y_transformed_test)

    @staticmethod
    def get_first_subtree(fitted_tree, thresh=None):
        """This function executes the first step of the pruning process, that is,
        computing the smallest subtree which has equivalent predictive power as
        fitted_tree. Starting from the leafs, in-sample loss of parent and leafs
        are compared, where leafs are cut-off if parent loss is smaller than sum of leaf
        loss minus thresh.

        :param fitted_tree: CausalTree object which already had method fit called.
        :param thresh: float: Penalizer on larger trees.
        :return: CausalTree: Subtree of fitted_tree with equivalent in-sample
        predictive power.
        """
        subtree = deepcopy(fitted_tree)
        if thresh is None:
            thresh = 0  # np.sqrt(np.var(subtree.y)) / 50
        depth = subtree.depth
        if depth < 1:
            return subtree
        for i in range(depth):
            for parent_node in subtree.get_level_in_list(depth - i - 1):
                if parent_node.left_child is not None:
                    if (
                        parent_node.node_loss
                        <= parent_node.left_child.branch_loss  # gets updated automatic
                        + parent_node.right_child.branch_loss
                        + thresh
                    ):
                        parent_node.left_child, parent_node.right_child = None, None
        return subtree

    @staticmethod
    def get_pruned_tree_and_alpha_sequence(fitted_tree, thresh):
        """Implementation of the pruning process. Sequence of increasing cost-complexity
        parameters and corresponding optimal subtrees of fitted_tree are computed.

        :param fitted_tree: CausalTree object which already had method fit called.
        :param thresh: float: Penalizer on larger trees.
        :return: dict: alphas: Increasing list of cost-complexity paramaters.
                        subtrees: List of subtrees corresponding to alphas.
        """
        assert isinstance(
            fitted_tree, CausalTree
        ), "This method only works on CausalTree Trees"
        if not fitted_tree.is_fitted:
            raise ValueError("This method only works on fitted trees")

        alphas = [0]
        subtrees = [
            CausalTree.get_first_subtree(fitted_tree, thresh)
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
    def get_subtree_given_alpha(tree, alpha, thresh):
        """Given a tree and cost-complexity parameter alpha, this function computes
        the unique optimal subtree of tree corresponding to alpha.

        :param tree: CausalTree.
        :param alpha: float: Positive cost-complexity parameter.
        :param thresh: float: Positive penalizer on larger trees.
        :return: CausalTree: Subtree of tree corresponding to alpha.
        """
        sequences = CausalTree.get_pruned_tree_and_alpha_sequence(tree, thresh)
        alphas = sequences["alphas"]
        subtrees = sequences["subtrees"]
        alphas = np.array(alphas)
        if alpha >= alphas[-1]:
            return subtrees[-1]
        else:
            index = np.where(alphas > alpha)[0][0]
            return subtrees[index]

    @staticmethod
    def apply_kFold_CV(
        X_learn,
        y_learn,
        treatment_status_learn=None,
        k=5,
        thresh=0,
        sparsity_bias=0.5,
        fitted_tree=None,
    ):
        """This static method computes the optimal causal tree given training data.
        The tree will be optimal in the sense that it minimizes out-of-sample loss
        estimated by k-fold cross validation. Since we often prefer sparser models, the
        function returns the optimal tree as well as an optimal sparse tree, which is
        computed by the one standard error rule of thumb.

        :param X_learn: ndarray: 2D array of training data on features.
        :param y_learn: ndarray: 1D array of training data on outcomes.
        :param treatment_status_learn: ndarray: 1D array of training data on treatment,
        if not given it is assumed that treatment status is a feature in X_learn.
        :param k: int: Indicating the number of folds for CV.
        :param thresh: float: Penalizer for larger trees.
        :param sparsity_bias: float between 0 and 1: larger number leading to sparser
        tree
        :param fitted_tree: CausalTree: Tree fitted using X_learn, y_learn and
        treatment_status_learn. If equal to none, a new tree is fitted on given data.
        :return: tuple: optimal_subtree computed via k-fold CV and
        optimal_sparse_subtree which uses the one standard error rule of thumb of
        selecting a sparser model.
        """
        try:
            feature_names = X_learn.columns.values
        except AttributeError:
            feature_names = None

        if fitted_tree is None:
            fitted_tree = CausalTree()
            fitted_tree.fit(X_learn, y_learn, treatment_status_learn)
        assert len(y_learn) == len(X_learn), (
            "Argument <<X_learn>> and <<y_learn>> must have the same number "
            "of observations."
        )
        X_learn = coerce_to_ndarray(X_learn)
        y_learn = coerce_to_ndarray(y_learn)

        kf = KFold(k)
        tree_max = fitted_tree  # complete maximal tree
        tree_max_sequences = CausalTree.get_pruned_tree_and_alpha_sequence(
            tree_max, thresh
        )
        tree_k_max_list = []  # list of maximal trees in each cross validation sample
        test_X = []
        test_y = []
        test_treatment_status = []

        for train_index, test_index in kf.split(X_learn, y_learn):
            tmp_tree = CausalTree()
            if treatment_status_learn is None:
                train_treatment_status = None
            else:
                train_treatment_status = treatment_status_learn[train_index]
            tmp_tree.fit(
                X_learn[train_index], y_learn[train_index], train_treatment_status
            )
            tree_k_max_list.append(tmp_tree)
            test_X.append(X_learn[test_index])
            test_y.append(y_learn[test_index])
            test_treatment_status.append(treatment_status_learn[test_index])

        alphas = tree_max_sequences["alphas"]
        potential_subtrees = tree_max_sequences["subtrees"]

        alpha_cv_errors = []
        for alpha in alphas:
            err_alpha = 0
            for k, cv_tree in enumerate(tree_k_max_list):
                cv_alpha_subtree = CausalTree.get_subtree_given_alpha(
                    cv_tree, alpha, thresh
                )
                test_y_transformed = CausalTree.transform_outcome(
                    test_y[k], test_treatment_status[k]
                )
                err_alpha += CausalTree.validate(
                    cv_alpha_subtree, test_X[k], test_y_transformed
                )
            alpha_cv_errors.append(err_alpha)

        alpha_cv_errors = np.array(alpha_cv_errors)
        optimal_index = int(np.where(alpha_cv_errors == alpha_cv_errors.min())[0][-1])
        sparsity_adjustment = sparsity_bias * np.std(alpha_cv_errors)
        optimal_sparse_index = int(
            np.where(
                alpha_cv_errors <= alpha_cv_errors[optimal_index] + sparsity_adjustment
            )[0][-1]
        )
        optimal_subtree = potential_subtrees[optimal_index]
        optimal_subtree.feature_names = feature_names
        optimal_subtree._cv_error = alpha_cv_errors[optimal_index]
        optimal_sparse_subtree = potential_subtrees[optimal_sparse_index]
        optimal_sparse_subtree.feature_names = feature_names
        optimal_sparse_subtree._cv_error = alpha_cv_errors[optimal_sparse_index]
        return optimal_sparse_subtree, optimal_subtree

    # Algorithm Implementation and Fitting Functions

    @staticmethod
    def estimate_treatment_in_leaf(y, treatment_status):
        """Computes treatment effect for given data y via simple mean differences.

        :param y: ndarray: 1D array containing outcomes of observations
        :param treatment_status: ndarray: 1D array containing treatment_status
        :return: float: Treatment effect estimate for given array.
        """
        if treatment_status.dtype != "bool":
            treatment_status = np.array(treatment_status, dtype="bool")
        y_treat = y[treatment_status]
        y_untreat = y[~treatment_status]
        assert len(y_treat) > 0 and len(y_untreat) > 0, "Error: Empty Outcome Vector"
        return y_treat.mean() - y_untreat.mean()

    def is_valid_split(self, sorted_treatment, index):
        """Checks if a split of sorted_treatment at index index would violate that each
        new region must not contain too little treated or untreated observations.

        :param sorted_treatment: ndarray: Treatment status sorted beforehand.
        :param index: int: Splitting index.
        :return: bool: True if split is valid, False else.
        """
        sorted_treat_left = sorted_treatment[: (index + 1)]
        sorted_treat_right = sorted_treatment[(index + 1) :]
        if np.sum(sorted_treat_right) + np.sum(sorted_treat_left) == 0:
            return False
        if np.sum(1 - sorted_treat_right) + np.sum(1 - sorted_treat_left) == 0:
            return False
        valid_left = (
            len(sorted_treat_left) > self.crit_num_obs
            or np.abs(np.sum(sorted_treat_left) - np.sum(1 - sorted_treat_left))
            <= self.max_distance
        )
        valid_right = (
            len(sorted_treat_right) > self.crit_num_obs
            or np.abs(np.sum(sorted_treat_right) - np.sum(1 - sorted_treat_right))
            <= self.max_distance
        )
        return valid_left and valid_right

    def find_best_splitting_point(self, X):
        """Standard implementation of the recursive binary splitting algorithm.

        :param X: ndarray: 2D array containing data on features used to find splits.
        :return: tuple: Feature at which to split the data (split_index), data point of
        given feature at which to split (split_value) and resulting loss of split.
        """
        n, p = X.shape
        split_index = None
        split_value = None
        loss = float("Inf")

        for var_index in range(p):
            # loop through covariates
            x = X[:, var_index]
            sort_index = np.argsort(x)
            sorted_x, sorted_y, sorted_treatment = (
                x[sort_index],
                self.y[sort_index],
                self.treatment_status[sort_index],
            )

            for i in range(self._min_leaf - 1, n - self._min_leaf - 1):
                # loop through potential splitting points

                xi = sorted_x[i]
                if xi == sorted_x[i + 1]:
                    continue

                if self.is_valid_split(sorted_treatment, i):
                    lhs_treat_effect = self.estimate_treatment_in_leaf(
                        sorted_y[: (i + 1)], sorted_treatment[: (i + 1)]
                    )

                    rhs_treat_effect = self.estimate_treatment_in_leaf(
                        sorted_y[(i + 1) :], sorted_treatment[(i + 1) :]
                    )

                    sorted_transformed_y = CausalTree.transform_outcome(
                        sorted_y, sorted_treatment
                    )
                    lhs_loss = np.sum(
                        (lhs_treat_effect - sorted_transformed_y[: (i + 1)]) ** 2
                    )
                    rhs_loss = np.sum(
                        (rhs_treat_effect - sorted_transformed_y[(i + 1) :]) ** 2
                    )

                    tmp_loss = lhs_loss + rhs_loss
                    if tmp_loss < loss:
                        split_index, split_value, loss = var_index, xi, tmp_loss

        return split_index, split_value, loss

    def fit(
        self, X, y, treatment_status=None, min_leaf=8, max_distance=4, crit_num_obs=None
    ):
        """Uses data X, y and treatment_status to fit a Causal Tree, only restricted by
        arguments min_leaf, max_distance and crit_num_obs. If treatment_status is None
        we assume that X is a pandas DataFrame containing a column named
        "treatment_status".

        :param X: ndarray: 2D array containing data on features.
        :param y: ndarray: 1D array containing data on outcomes.
        :param treatment_status: ndarray: 1D array containg data on treatment status.
        :param min_leaf: int: Minimum number of observations required in each leaf.
        :param max_distance: int: Maximum number of absolute difference in observations
        with treatment and observations without treatment in each leaf.
        :param crit_num_obs: int: Critical value of observations in each leaf: When
        reached assertion using attribute max_distance is binding.
        :return: CausalTree: Fitted tree object.
        """
        # Check Input Values and do stuff for root
        if self.is_root:
            assert min_leaf >= 1, "Parameter <<min_leaf>> has to be bigger than one."
            assert len(X) == len(
                y
            ), "Data <<X>> and <<y>> must have the same number of observations."
            assert len(y) >= 2 * min_leaf, (
                "Data has not enough observations for a single split to occur "
                "given value of <<min_leaf>>."
            )
            if treatment_status is None:
                assert "treatment_status" in X.columns
                treatment_status = X[["treatment_status"]]
                X = X.drop(["treatment_status"], axis=1)
            else:
                assert len(y) == len(treatment_status)

            try:
                self._feature_names = X.columns.values
            except AttributeError:
                pass
            X = coerce_to_ndarray(X)  # coerce to ndarray
            y = coerce_to_ndarray(y)
            treatment_status = coerce_to_ndarray(treatment_status)
            assert np.array_equiv(np.unique(treatment_status), np.array([0, 1]))
            self._num_features = X.shape[1]
            if crit_num_obs is None:
                crit_num_obs = 4 * max_distance

        # Set Parameters
        self._min_leaf = min_leaf
        self._max_distance = max_distance
        self._crit_num_obs = crit_num_obs
        self._value = self.estimate_treatment_in_leaf(y, treatment_status)
        self._y = y
        self._y_transformed = CausalTree.transform_outcome(y, treatment_status)
        self._treatment_status = treatment_status
        self._node_loss = np.sum((self.value - self.y_transformed) ** 2)

        # Actual Fitting
        self._split_var, self._split_value, tmp_loss = self.find_best_splitting_point(X)
        self._is_fitted = True

        if self._split_var is None:
            self._branch_loss = self._node_loss
        else:
            self._branch_loss = tmp_loss

        if self._split_var is not None:
            index = X[:, self._split_var] <= self._split_value

            self._left_child = CausalTree(parent=self)
            self._right_child = CausalTree(parent=self)
            self._left_child.feature_names = self.feature_names
            self._right_child.feature_names = self.feature_names

            self._left_child.fit(
                X[index],
                y[index],
                treatment_status[index],
                min_leaf,
                max_distance,
                crit_num_obs,
            )
            self._right_child.fit(
                X[~index],
                y[~index],
                treatment_status[~index],
                min_leaf,
                max_distance,
                crit_num_obs,
            )

        self.update_branch_loss()
        return self

    def get_level_in_list(self, level):
        """Traverses tree and saves all nodes in a given level in a list.
        Example:
            level:
                0: tree
                1: [tree.left_child, tree.right_child] (first level)
        :param level: int: Number representing desired depth.
        :return: list: Nodes in given level.
        """
        level_list = []
        if level == 0:
            level_list.append(self)
        else:
            if self.left_child is not None:
                level_list.extend(self.left_child.get_level_in_list(level - 1))
                level_list.extend(self.right_child.get_level_in_list(level - 1))
        return level_list

    def predict_row(self, xi):
        """
        :param xi: ndarray: 1D array of new data on features of a single observation.
        :return: float: Treatment prediction conditional on xi.
        """
        if self.is_leaf:
            return self.value
        child = (
            self._left_child
            if xi[self._split_var] <= self._split_value
            else self._right_child
        )
        return child.predict_row(xi)

    def predict(self, x):
        """
        :param x: ndarray: 2D array containing data on features.
        :return: ndarray: 1D array containing estimated treatment effects.
        """
        x = coerce_to_ndarray(x)
        assert (
            self._is_fitted
        ), "The tree has not yet been fitted; no prediction is possible."
        assert (
            x.shape[1] == self._num_features
        ), "New Data must have the same dimension as the Data used for Fitting."

        return np.array([self.predict_row(xi) for xi in x])

    def plot(self, render=False, save=False, filename=None):
        """Plotting function that nicely visualizes the information contained in the
        class by creating an (upside-down) tree-like structure plot. Inner nodes contain
        information on the splitting process of the feature space while leafs contain
        information on treatment predictions in regions of the partition.

        :param tree: CausalTree: Fitted tree.
        :param filename: str: Name of file that should be created.
        :param save: bool: Should the file be saved to disk.
        """
        if not self.is_fitted:
            print("The tree must be fitted in order to be plotted")
            return
        if filename is None:
            filename = "causal_tree.svg"
        dot = Graph(name="causal_tree", filename=filename, format="svg")
        dot.node(
            str(id(self)),
            self.splitting_info_to_string()
            + "\nestimate:"
            + str(round(float(self.value), 3)),
        )
        for i in range(self.depth):
            nodes = self.get_level_in_list(i + 1)
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


#  General Functions (Some of which might be static methods in a strict sense)


def pre_order_traverse_tree(root: CausalTree, func=None) -> list:
    """For many steps of multiple algorithms we need to traverse the tree and apply
    a function to each node one after the other. This function traverses the tree in a
    standard pre-order fashion. If func is None we simply append each node to a list and
    return that, otherwise func is applied before appending.

    :param root: CausalTree: Root node of a tree.
    :param func: Function that should be applied to nodes while traversing the tree.
    :return: list: Elements are either nodes given in pre-order or nodes in pre-order
    with func applied to each one.
    """
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


def g_value(node: CausalTree) -> float:
    return (
        float("Inf")
        if node.is_leaf
        else (node.node_loss - node.branch_loss) / (node.number_of_leafs - 1)
    )


def g(branch: CausalTree) -> tuple:
    nodes = pre_order_traverse_tree(branch)
    values = np.array(pre_order_traverse_tree(branch, g_value))
    minimum = values.min()
    argmin_index = np.where(values == minimum)[0]
    argmin = [nodes[ix] for ix in argmin_index]
    return argmin, minimum


def coerce_to_ndarray(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, pd.Series):
        return obj.to_frame().values
    elif isinstance(obj, pd.DataFrame):
        return obj.values
    else:
        raise TypeError(
            "Object was given with inappropriate type;"
            "for matrices and vectors only use pandas Series, "
            "DataFrame or Numpy ndarrays"
        )


def test_monotonicity_list(lst: list, strictly=True) -> bool:
    """Test for (strict) monotonicity. In particular we expect the complexity-parameter
    in the pruning process to be an increasing sequence, which will be tested by this
    function.

    :param lst: Arbitrary list containing floats.
    :param strictly: bool: Should strict monotonicity be tested.
    :return: bool: True if lst is (strict) monotone increasing, False else.
    """
    if strictly:
        return all(x < y for x, y in zip(lst, lst[1:]))
    else:
        return all(x <= y for x, y in zip(lst, lst[1:]))
