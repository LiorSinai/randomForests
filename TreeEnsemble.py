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
from utilities import *


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

        self.features = None
        self.n_features = None
        self.n_classes = None
        self.feature_importances_ = None
        
    def fit(self, X, Y):
        "fit the random tree to the independent variable X, to determine the dependent variable Y"
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns.values
        n_samples = X.shape[0]
        self.sample_size_ = check_sample_size(self.sample_size, n_samples)

        # create decision trees
        self.trees = []
        rng_states = [] # save the random states to regenerate the random indices for the oob_score
        for i in range(self.n_trees):
            rng_states.append(self.RandomState.get_state())
            self.trees.append(self._create_tree(X, Y))

        # set attributes
        self.feature_importances_ = self.impurity_feature_importances()
        if self.oob_score:
            if not (self.bootstrap or (self.sample_size_<n_samples)):
                warnings.warn("out-of-bag score will not be calculated because bootstrap=False")
            else:
                self.oob_score_ = self.calculate_oob_score(X, Y, rng_states)
    
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

        new_tree =  DecisionTree(max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState,
                                 min_samples_leaf=self.min_samples_leaf
                                )
        new_tree.fit(X_, Y_)
        return new_tree
                
    def predict(self, X):
        "Predict the class for each sample in X"
        probs = np.sum([t.predict_prob(X) for t in self.trees], axis=0)
        #probs = np.sum([t.predict_count(X) for t in self.trees], axis=0)
        return np.nanargmax(probs, axis=1)

    def score(self, X, y):
        "The accuracy score of random forest predictions for X to the true classes y"
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

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

    def impurity_feature_importances(self) -> np.ndarray:
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point. """
        feature_importances = np.zeros((self.n_trees, self.n_features))

        for i, tree in enumerate(self.trees):
            feature_importances[i, :] = tree.feature_importances_

        return np.mean(feature_importances, axis=0)
