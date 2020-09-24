"""
14 September 2020

Random Forests from scratch
https://course18.fast.ai/lessonsml1/lesson5.html

"""

import numpy as np
import pandas as pd
import warnings
import time
from typing import List, Tuple, Dict

from DecisionTree import DecisionTree
from utilities import split_data, check_RandomState, check_sample_size, encode_one_hot

class RandomForestClassifier:
    def __init__(self, 
                n_trees=100, 
                random_state=None, 
                max_depth=None, 
                max_features=None, 
                min_samples_leaf=1,
                sample_size=None, 
                bootstrap=True, 
                oob_score=False):
        self.n_trees = n_trees
        self.RandomState = check_RandomState(random_state)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf=min_samples_leaf
        self.sample_size = sample_size
        self.bootstrap = bootstrap
        self.oob_score = oob_score

        self.n_features = None
        self.n_classes = None
        self.feature_importances_ = None
        
    def fit(self, X, Y):
        "fit the random tree to the independent variable X, to determine the dependent variable Y"
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        n_samples = X.shape[0]
        self.sample_size_ = check_sample_size(self.sample_size, n_samples)

        self.trees = []
        rng_states = [] # save the random states to regenerate the random indices for the oob_score
        for i in range(self.n_trees):
            rng_states.append(self.RandomState.get_state())
            self.trees.append(self._create_tree(X, Y))

        # set attributes
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.feature_importances_ = self.gini_feature_importance()

        if self.oob_score:
            if not (self.bootstrap or (self.sample_size_<n_samples)):
                warnings.warn("out-of-bag score will not be calculated because bootstrap=False")
            else:
                self.oob_score_ = self.calculate_oob_score(X, Y, rng_states)
        
    def calculate_oob_score(self, X, Y, rng_states):
        n_samples = X.shape[0]
        oob_prob = np.zeros(Y.shape)
        oob_count = np.zeros(n_samples)
        all_samples = np.arange(n_samples)
        rng = np.random.RandomState()
        for i, state in enumerate(rng_states):
            rng.set_state(state)
            if self.bootstrap:
                rand_idxs = rng.randint(0, n_samples, self.sample_size_)
            else: #self.sample_size_ < n_samples
                rand_idxs = rng.permutation(all_samples)[:self.sample_size_]
            row_oob = np.setxor1d(all_samples, rand_idxs)
            oob_prob[row_oob, :] += self.trees[i].predict_prob(X.iloc[row_oob])
            oob_count[row_oob] += 1
        # remove nan-values
        valid = oob_count > 0 
        oob_prob = oob_prob[valid, :]
        oob_count = oob_count[valid][:, np.newaxis] # transform to column vector for broadcasting during the division
        y_test    =  np.argmax(Y[valid], axis=1)
        # predict out-of-bag score
        y_pred = np.argmax(oob_prob/oob_count, axis=1)
        return np.mean(y_pred==y_test)
                
    def _create_tree(self, X, Y):
        assert len(X) == len(Y), ""
        n_samples = X.shape[0]

        # get sub-sample 
        if self.bootstrap:
            rand_idxs = self.RandomState.randint(0, n_samples, self.sample_size_) # with replacement
            X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs] # approximate unique values =n*(1-np.exp(-sample_size_/n_samples))
        elif self.sample_size_ < n_samples:
            rand_idxs = self.RandomState.permutation(np.arange(n_samples))[:self.sample_size_]  # without replacement
            X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs]
        else:
            X_, Y_ = X.copy(), Y.copy() # do nothing to the data

        return DecisionTree(X_, Y_,
                            max_depth=self.max_depth, 
                            max_features=self.max_features,
                            random_state=self.RandomState,
                            min_samples_leaf=self.min_samples_leaf
                            )
                
    def predict(self, X):
        "Predict the class for each sample in X"
        probs = np.sum([t.predict_prob(X) for t in self.trees], axis=0)
        #probs = np.sum([t.predict_count(X) for t in self.trees], axis=0)
        return np.nanargmax(probs, axis=1)

    def score(self, X, y):
        "The accuracy score of random forest predictions for X to the true classes y"
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    def gini_feature_importance(self) -> np.ndarray:
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point.
           Independent of input or output data
        """
        feature_importances = np.zeros((self.n_trees, self.n_features))
        total_samples = self.trees[0].n_samples[0]

        for i, tree in enumerate(self.trees):
            for node in range(len(tree.impurities)):
                if tree.is_leaf(node):
                    continue 
                j = tree.split_features[node]
                impurity = tree.impurities[node]
                n_samples = tree.n_samples[node]
                # calculate score
                left = tree.tree_.children_left[node]
                right = tree.tree_.children_right[node]
                lhs_gini = tree.impurities[left]
                rhs_gini = tree.impurities[right]
                lhs_count = tree.n_samples[left]
                rhs_count = tree.n_samples[right]
                score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
                # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
                feature_importances[i, j] += (impurity-score) * (n_samples/total_samples)

        # normalise per tree
        feature_importances = feature_importances/feature_importances.sum(axis=1)[:, None]

        return np.mean(feature_importances, axis=0)

    def perm_feature_importance(self, X, y, n_repeats=10) -> Dict:
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

        return {'means': np.mean(feature_importances, axis=0), 'stds': np.std(feature_importances, axis=0)}


if __name__ == "__main__":
    # load data
    # Binary class test with 5000 samples
    file_name = 'tests/UniversalBank_cleaned.csv'
    target = "Personal Loan"
    # 3-class test with 1000 samples
    # file_name = 'tests/Iris_cleaned.csv'  
    # target = "Species"

    data = pd.read_csv(file_name)
    X = data.drop(columns=[target])
    y = data[target]
    n_samples, n_features = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)

    forest = RandomForestClassifier(n_trees=20, 
                                    bootstrap=True,
                                    sample_size=1.0, # default is None
                                    max_features=5, # default is None
                                    #max_depth = 5, # default is None
                                    oob_score=True,
                                    min_samples_leaf=3,
                                    random_state=42)

    start_time = time.time()
    forest.fit(X_train, y_train)
    end_time = time.time()
    print('Fitting time: %.3fs' % ((end_time-start_time)))

    # display descriptors
    depths =[t.get_max_depth() for t in forest.trees]
    n_splits = [t.get_n_splits() for t in forest.trees]
    n_leaves = [t.get_n_leaves() for t in forest.trees]
    acc_test = forest.score(X_test, y_test)
    acc_train = forest.score(X_train, y_train)
    print("depth range, average:    %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("n_splits range, average: %d-%d, %.2f" % (np.min(n_splits), np.max(n_splits), np.mean(n_splits)))
    print("n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))
    print("train accuracy: %.2f%%" % (acc_train*100))
    if hasattr(forest, 'oob_score_'):
        print("oob accuracy:   %.2f%%" % (forest.oob_score_*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    ### ----------- Feaure importance ----------- ###### 
    start_time = time.time()
    fi_perm   = forest.perm_feature_importance(X_train, y_train) # very slow
    end_time = time.time()
    print('Permutation importance time: %.3fs' % ((end_time-start_time)))
    fi1 = fi_perm['means']
    fi2 = forest.feature_importances_
    for fi in (fi1, fi2):
        order = np.argsort(fi)[::-1] # descending order
        print("Feature importances")
        for col, val in zip(X_train.columns[order], fi[order]):
            print('%-15s %.4f' % (col+':', val)) 

    #import matplotlib.pyplot as plt # comment out to avoid dependency 
    # order = np.argsort(fi_perm['means'])

    # fig, ax = plt.subplots()
    # inds = np.arange(n_features)
    # width = 0.4
    # fi = fi_perm['means'][order]/fi_perm['means'].sum()
    # ax.barh(inds+width/2, fi, width, xerr=fi_perm['stds'][order], label='permutation')
    # fi = forest.feature_importances_[order]
    # ax.barh(inds-width/2, fi, width, label='weighted impurity')
    # #ax.grid(True)
    # ax.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    # ax.set_yticks(inds)
    # ax.set_yticklabels(X.columns[order])
    # ax.set_ylabel('feature')
    # ax.set_xlabel('relative feature importance score')
    # ax.set_title("Feature importances")

    # ### ----------- Fitting acuracy per number of trees ----------- ###### 
    # fig, ax = plt.subplots()
    # preds = np.stack([t.predict_prob(X_test) for t in forest.trees])
    # n_trees = forest.n_trees
    # n = len(y_test)
    # acc = np.zeros(n_trees)
    # for i in range(0, n_trees):
    #     y_pred = np.argmax(np.sum(preds[:i+1, :, :], axis=0), axis=1)
    #     acc[i] = np.mean(y_pred == y_test)
    # ax.plot(acc)
    # ax.set_xlabel("number of trees")
    # ax.set_ylabel("accuracy")

    # plt.show()

