# Random Forest Classifier from Scratch

## Overview

A custom random forest implementation. It uses Scikit-learn's [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
as a benchmark. The Scikit-learn implementation is much faster, because it is written in Cython and also parallelises work across trees. 

## RandomForestClassifier
The class is created with:
`forest = RandomForestClassifier(n_trees=100, random_state=None, max_features=None, max_depth=None, bootstrap=True, sample_size=None,  replacement=True)`.

The parameters are:
- `n_trees [int]`: number of trees (estimators). Sklearn equivalent: n_estimators_.
- `random_state [None int or numpy.random.RandomState]`: a Numpy RandomState entity. Sklearn equivalent: random_state
- `max_features [None or int or 'sqrt']`: maximum number of features to randomly select from per split. If 'sqrt', then sqrt(n_features) is used. Sklearn equivalent: max_features.
- `max_depth [None or int]`: stop splitting after reaching this depth. Sklearn equivalent: max_depth.
- `min_samples_leaf [int]`: the mininum number of samples allowed in a leaf. Sklearn equivalent: min_samples_leaf.
- `bootstrap [bool]`: use a random subset of samples per tree. Sklearn equivalent: max_depth.
- `sample_size [int]`: number of samples to use per tree. Sklearn equivalent: N/A.
- `oob_score [bool]` (only used if bootstrap=True or sample_size<1.0): calculate the out-of-bag score. This is the mean accuracy of the predictions made for each sample using
only the trees that were _not_ trained on that sample. 

Let the _k=sample_size_ and _n=total_samples_.
Then on average with replacement, n(1-1/n)^k ~ exp(-n/k) samples will not be used in each tree. This is 36.8% of samples per tree if _k=n_. 
This is a significant portion of samples. Therefore the out-of-bag samples form a useful proxy validation set.

It has the following external methods:
- `fit(X,Y)`: fit the data to the random forest classifier. Y can have multiple classes. Sklearn equivalent: fit().
- `predict(X) [array]`: returns the predicted classes y for the independent variable X. Predictions are made using majority voting between trees. Sklearn equivalent: predict().
- `score(X, y) [float]`: returns the fraction of correct predictions of X for y. Sklearn equivalent: score().
- `perm_feature_importance(X, y, n_repeats=10) [dict]`: calculates the feature importance as the change in accuracy when this feature column is randomly shuffled (permuted). 
Runs it n_repeats times. Keys are 'means' and 'stds' for the mean and standard deviations per feature over the trials. Sklearn equivalent: N/A.

It has the following attributes:
- `trees [array]`: list of DecisionTree. Sklearn equivalent: estimators_. 
- `feature_importances_ [array]`: the feature importance calculated per feature as the sum of the change in impurity per node where that feature splits the node, 
weighted by the fraction of samples used in that node. 
Unlike the permutation importance, this is independent of the input data. Sklearn equivalent: feature_importances_.
- `oob_score_ [float]`: the mean out-of-bag score.

## DecisionTree
A binary decision tree. It is based off Scikit-learn's [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
 The tree is encoded as a set of parallel lists for children_left and children_right.

The class is created with:
`tree = DecisionTree(X, Y, random_state=None, max_depth=None, max_features=None)`.

Unlike Scikit-learn, a fit function is not used.

The parameters are:
- `X, Y [arrays]`: independent and dependent variables respectively. 
- `random_state [None int or numpy.random.RandomState]`: a Numpy RandomState entity. Sklearn equivalent: random_state
- `max_depth [None or int]`: stop splitting after reaching this depth. Sklearn equivalent: max_depth.
- `max_features [None or int or 'sqrt']`: maximum number of features to randomly select from per split. If 'sqrt', then sqrt(n_features) is used. Sklearn equivalent: max_features.
- `min_samples_leaf [int]`: the mininum number of samples allowed in a leaf. Sklearn equivalent: min_samples_leaf.
 
It has the following external methods:
- `get_n_splits() [int]`: returns the number of splits in the tree. Sklearn equivalent: get_n_leaves() (might return a slightly different result).
- `get_max_depth() [int]`: returns the maximum depth in the tree. Sklearn equivalent: get_depth().
- `is_leaf(node_id) [bool]`: returns if this node is a leaf or not (no children?). Sklearn equivalent: N/A.
- `split_name(node_id) [str]`: returns the feature name which is used for the split at this node. Sklearn equivalent: N/A.
- `predict_prob(X) [array]`: returns the class probability for each sample in X, calculated as the fraction of samples per class in the final node. 
Sklearn equivalent: predict_proba(). 
- `predict_count(X) [array]`: returns the class probability for each sample in X, calculated as the sample number per class in the final node. Less accuracte than predict_prob().
Sklearn equivalent: N/A. 
- `predict(X) [array]`: returns the class predictions for X based on the maximum class probability from predict_prob. Sklearn equivalent: predict().
- `get_info(node_id) [array]`: returns values associated with this node in the tree. Sklearn equivalent: N/A.
- `leaf_to_string(node_id) [str]`: returns a formatted string of the get_info(node_id) data. Sklearn equivalent: N/A.
 
It has the following attributes:
- `tree_ [BinaryTree]`: A binary tree encoded as a set of parallel lists for children_left and children_right. Sklearn equivalent: tree_. 
- `n_samples [array]`: number of samples in this node. Sklearn equivalent: weighted_n_node_samples.
- `scores [array]`: weighted Gini impurities in children nodes after splitting, optimised during training. Sklearn equivalent: N/A.
- `values [array]`: count per class in each node. Sklearn equivalent: value.
- `impurities [array]`: Gini impurity of each node. Sklearn equivalent: impurity.
- `split_features [array]`: feature used to split each node. Sklearn equivalent: feature.
- `split_values [array]`: value used to split each node. Sklearn equivalent: threhold.
- `size [int]`: total number of nodes in the tree. Sklearn equivalent: N/A.
- `depths [array]`: depths for each node. Sklearn equivalent: N/A.

## Test data sets

Two test sets are used from Kaggle:
- [Bank_Loan_Classification](https://www.kaggle.com/sriharipramod/bank-loan-classification/): Binary classification problem of approved loads. 5000 entries and 14 features.
This is an easy problem. The feature Income is a high valued predictor. Using Income>100 as a benchmark model achieves a 83.52% accuracy over the whole data set.
The random forest achieves up to 99.10% accuracy on test datasets.
- [Iris Species](https://www.kaggle.com/uciml/iris): 3-class classification problem of iris species based on plant dimensions. 150 entries and 3 features.
The random forest achieves 96%-100% accuracy on test datasets.

## Dependencies

[Numpy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/).