"""
14 September 2020

Random Forests from scratch
https://course18.fast.ai/lessonsml1/lesson5.html

"""

import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.inspection import permutation_importance

from utilities import load_data

class RandomForestClassifier:
    def __init__(self, 
                n_trees=100, 
                random_state=None, 
                sample_size=None, 
                max_depth=None, 
                max_features=None, 
                bootstrap=True, 
                replacement=True):
        self.RandomState = np.random.RandomState(random_state)
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.replacement= replacement

        self.n_features = None
        self.feature_importances_ = None
        
    def fit(self, X, y):
        if self.bootstrap and (not self.replacement) and (self.sample_size is None): 
                print("Warning: TreeEnsemble.fit() sample size without replacement should be less than n_samples else"\
                      " every tree uses all samples")
                
        self.trees = [self.create_tree(X, y) for i in range(self.n_trees)]
        
        # set attributes
        self.n_features = X.shape[1]
        self.feature_importances_ = self.gini_feature_importance()

    def create_tree(self, X, y, categories=None):
        assert len(X) == len(y), ""
        n_samples = X.shape[0]

        if self.bootstrap:
            sample_size = n_samples if self.sample_size is None else min(self.sample_size, n_samples)
            if self.replacement: # with replacement
                rand_idxs = self.RandomState.randint(0, n_samples, sample_size)
            else: # without replacement
                rand_idxs = self.RandomState.permutation(np.arange(n_samples))[:sample_size] 
            X_, y_ = X.iloc[rand_idxs, :], y.iloc[rand_idxs]
        else:
            X_, y_ = X.copy(), y.copy() # do nothing to the data

        return DecisionTree(X_, y_, categories=categories,
                            max_depth=self.max_depth, 
                            max_features=self.max_features,
                            random_state=self.RandomState
                            )

    def predict(self, X):
        probs = np.mean([t.predict(X) for t in self.trees], axis=0)
        return (probs > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    def gini_feature_importance(self):
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point.
           Independent of input or output data
        """
        feature_importances = np.zeros((self.n_trees, self.n_features))
        total_samples = self.trees[0].n_samples

        for i, tree in enumerate(self.trees):
            for leaf in tree.preorder():
                if leaf.is_leaf:
                    continue 
                j = leaf.var_idx
                # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
                feature_importances[i, j] += (leaf.impurity-leaf.score) * (leaf.n_samples/total_samples)

        # normalise per tree
        feature_importances = feature_importances/feature_importances.sum(axis=1)[:, None]

        return np.mean(feature_importances, axis=0)

    def perm_feature_importance(self, X, y, n_repeats=10):
        """Calculate feature importance based on random permutations of each feature column.
        The larger the drop in accuracy from shuffling each column, the higher the feature importance.
        """
        y_pred = self.predict(X)
        acc_full = np.mean(y_pred == y)

        feature_importances = np.zeros((n_repeats, self.n_features))
        for j, col in enumerate(X.columns):
            X_sub = X.copy()
            for i in range(n_repeats):
                X_sub[col] = self.RandomState.permutation(X_sub[col].values)
                y_pred = self.predict(X_sub)
                feature_importances[i, j] = acc_full - np.mean(y_pred == y)

        return np.mean(feature_importances, axis=0)


if __name__ == "__main__":
    # load data
    file_name = 'tests/UniversalBank_cleaned.csv'
    target = "Personal Loan_1"
    X_train, X_test, y_train, y_test = load_data(file_name, target, test_size=0.2, seed=42)

    forest = RandomForestClassifier(n_trees=10, 
                                    bootstrap=True,
                                    replacement=True,
                                    sample_size = None,
                                    max_features = 'sqrt',
                                    max_depth = 15,
                                    random_state=42)

    forest.fit(X_train, y_train)

    # display descriptors
    depths =[t.get_max_depth() for t in forest.trees]
    n_leaves = [t.get_n_splits() for t in forest.trees]
    acc_test = forest.score(X_test, y_test)
    acc_train = forest.score(X_train, y_train)
    print("depth range, average:    %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    fi1 = forest.perm_feature_importance(X_train, y_train)
    fi2 = forest.feature_importances_

    for fi in (fi1, fi2):
        order = np.argsort(fi)[::-1] # descending order
        print("Feature importances")
        for col, val in zip(X_train.columns[order], fi[order]):
            print('%s: %.4f' % (col, val)) 