"""
14 September 2020

Random Forests from scratch
https://course18.fast.ai/lessonsml1/lesson5.html

"""

import numpy as np
import pandas as pd

from DecisionTree import DecisionTree
from utilities import split_data, split_data, check_RandomState, encode_one_hot

class RandomForestClassifier:
    def __init__(self, 
                n_trees=100, 
                random_state=None, 
                max_depth=None, 
                max_features=None, 
                sample_size=None, 
                bootstrap=True, 
                replacement=True,
                oob_score=False):
        self.RandomState = check_RandomState(random_state)
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.replacement= replacement
        self.oob_score = oob_score

        self.n_features = None
        self.n_classes = None
        self.feature_importances_ = None
        
    def fit(self, X, Y):
        if self.bootstrap and (not self.replacement) and (self.sample_size is None): 
                print("Warning: TreeEnsemble.fit() sample size without replacement should be less than n_samples else"\
                      " every tree uses all samples")

        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        self.trees = []
        if self.oob_score:
            if not self.bootstrap:
                print("Warning: out-of-bag score will not be calculated because bootstrap=False")
            else:
                oob_prob = np.zeros(Y.shape)
                n_samples = Y.shape[0]
                oob_count = np.zeros((n_samples))
                all_samples = set(range(n_samples))
        for i in range(self.n_trees):
            new_tree, rows = self._create_tree(X, Y) 
            self.trees.append(new_tree)
            if self.oob_score and self.bootstrap:
                row_oob = all_samples.difference(rows) # approximately length=n*(1-np.exp(-sample_size/n_samples))
                row_oob = np.array(list(row_oob))
                oob_prob[row_oob, :] += new_tree.predict_prob(X.iloc[row_oob])
                oob_count[row_oob] += 1

        # set attributes
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.feature_importances_ = self.gini_feature_importance()
        if self.oob_score and self.bootstrap:
            y_test = np.argmax(Y, axis=1)
            # remove nan-values
            valid = oob_count > 0 
            oob_prob = oob_prob[valid, :]
            oob_count = oob_count[valid][:, np.newaxis]
            y_test    = y_test[valid]
            # predict out-of-bag score
            y_pred = np.argmax(oob_prob/oob_count, axis=1)
            self.oob_score_ = np.mean(y_pred==y_test)

    def _create_tree(self, X, Y):
        assert len(X) == len(Y), ""
        n_samples = X.shape[0]

        if self.bootstrap:
            sample_size = n_samples if self.sample_size is None else min(self.sample_size, n_samples)
            if self.replacement: # with replacement
                rand_idxs = self.RandomState.randint(0, n_samples, sample_size)
            else: # without replacement
                rand_idxs = self.RandomState.permutation(np.arange(n_samples))[:sample_size] 
            X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs]
        else:
            X_, Y_ = X.copy(), Y.copy() # do nothing to the data
            rand_idxs = None # this data isn't used

        return DecisionTree(X_, Y_,
                            max_depth=self.max_depth, 
                            max_features=self.max_features,
                            random_state=self.RandomState
                            ), rand_idxs
                

    def predict(self, X):
        probs = np.sum([t.predict_prob(X) for t in self.trees], axis=0)
        #probs = np.sum([t.predict_count(X) for t in self.trees], axis=0)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    def gini_feature_importance(self):
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
                score = tree.scores[node]
                n_samples = tree.n_samples[node]
                # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
                feature_importances[i, j] += (impurity-score) * (n_samples/total_samples)

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
                                    bootstrap=False,
                                    replacement=True,
                                    #sample_size=round(0.2*X_train.shape[0]), # default is None
                                    #max_features='sqrt', # default is None
                                    #max_depth = 5, # default is None
                                    oob_score=True,
                                    random_state=42)

    forest.fit(X_train, y_train)

    # display descriptors
    depths =[t.get_max_depth() for t in forest.trees]
    n_leaves = [t.get_n_splits() for t in forest.trees]
    acc_test = forest.score(X_test, y_test)
    acc_train = forest.score(X_train, y_train)
    print("depth range, average:    %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))
    if hasattr(forest, 'oob_score_'):
        print("oob accuracy:   %.2f%%" % (forest.oob_score_*100))
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    ### ----------- Feaure importance ----------- ###### 
    fi_perm   = forest.perm_feature_importance(X_train, y_train) # very slow
    fi1 = fi_perm['means']
    fi2 = forest.feature_importances_
    for fi in (fi1, fi2):
        order = np.argsort(fi)[::-1] # descending order
        print("Feature importances")
        for col, val in zip(X_train.columns[order], fi[order]):
            print('%s: %.4f' % (col, val)) 

    # import matplotlib.pyplot as plt # comment out to avoid dependency 
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

