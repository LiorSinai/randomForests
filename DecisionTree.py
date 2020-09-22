"""
14 September 2020

Random Forests from scratch

Decision tree v1
Tree is represent as a nested object

TODO: so far only works for binary classes. Extened to many classes

"""

import numpy as np
import pandas as pd
from utilities import load_data

def gini_score(count, s1): s0 = count-s1; return 1 - (s0/count)**2 - (s1/count)**2
def std_agg(count, s1, s2): return np.sqrt((s2/count) - (s1/count)**2)


class DecisionTree:
    def __init__(self, X, y, categories=None, max_depth=None, depth=0, max_features=None, random_state=None):
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        self.RandomState = np.random.RandomState() if random_state is None else  random_state
        self.max_features = max_features
        self.max_depth = max_depth
        self.depth = depth

        self.features = X.columns

        # defaults before split
        self.score = float('inf')
        self.impurity = gini_score(self.n_samples, sum(y))
        self.val = np.mean(y)
        self.var_idx = None
        self.left = None
        self.right = None
        
        # split
        max_depth_ = float('inf') if max_depth is None else max_depth
        if (self.depth < max_depth_) and (self.impurity > 0):
            self.find_varsplit(X, y) 

    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    @property
    def split_name(self):
        return self.features[self.var_idx]

    def get_max_depth(self):
        if self.is_leaf:
            return 0 
        max_left = 0 if self.left is None else self.left.get_max_depth()
        max_right = 0 if self.right is None else self.right.get_max_depth()
        return max(max_left, max_right) + 1

    def find_varsplit(self, X, y):
        # choose n features for possible splits
        if self.max_features is not None:
            n_features = self.n_features
            if self.max_features is 'sqrt':
                n = np.ceil(np.sqrt(n_features)).astype(int)
            elif isinstance(self.max_features, int):
                n = min(self.max_features, n_features)
            else:
                raise Exception('Unknown parameter \'%s\' for max_features' % self.max_features)
        else:
            n = self.n_features 
            # the random shuffling ensures a random variable is used if 2 splits are equal
            # this is most noticeable with small sample sizes
            # else there will be a slight bias towards a variable based on its position in the array
        features = self.RandomState.permutation(self.n_features)[:n] # randomly shuffle features

        # make the split
        for i in features:
            self.find_bettersplit(i, X, y)
        if self.is_leaf: 
            return

        # make children
        x_split = X.values[:, self.var_idx]
        lhs = np.nonzero(x_split<=self.split_val)
        rhs = np.nonzero(x_split>self.split_val)
        self.left =  DecisionTree(X.iloc[lhs], 
                                 y.iloc[lhs], 
                                 depth=self.depth+1, 
                                 max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState)
        self.right = DecisionTree(X.iloc[rhs], 
                                 y.iloc[rhs], 
                                 depth=self.depth+1, 
                                 max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState)

    # def find_bettersplit(self, var_idx, x, y):
    #     x, y = x.values[self.idxs, var_idx], y.values[self.idxs]
    #     #O(n**2) complexity
    #     for i in range(1, self.n_samples-1):
    #         lhs = x <= x[i]
    #         n_lhs = lhs.sum()
    #         rhs = x > x[i]
    #         n_rhs = rhs.sum()
    #         if n_rhs == 0:
    #             continue
    #         # compare std deviations
    #         lhs_std = y[lhs].std()
    #         rhs_std = y[rhs].std()
    #         curr_score = (lhs_std * n_lhs + rhs_std * n_rhs)/(n_lhs + n_rhs)
    #         if curr_score < self.score:
    #             self.var_idx, self.score, self.split_val = var_idx, curr_score, x[i]

    def find_bettersplit(self, var_idx, X, y, method='gini'):
        X, y = X.values[:, var_idx], y.values 

        # sort the variables. Start with all on the right. Then move one sample to left one at a time
        order = np.argsort(X)
        X_sort, y_sort = X[order], y[order]
        lhs_count, lhs_sum, lhs_sum2 = 0., 0., 0.
        rhs_count, rhs_sum, rhs_sum2 = self.n_samples, y_sort.sum(), np.square(y_sort).sum()

        # O(n) complexity
        for i in range(0, self.n_samples-1):
            xi, yi = X_sort[i], y_sort[i]
            lhs_count += 1;  rhs_count -= 1
            lhs_sum += yi;   rhs_sum -= yi
            lhs_sum2 += yi**2;  rhs_sum2 -= yi**2
            if xi == X_sort[i+1]:
                continue
            # compare std deviations
            if method == 'std':
                lhs_std = std_agg(lhs_count, lhs_sum, lhs_sum2)
                rhs_std = std_agg(rhs_count, rhs_sum, rhs_sum2)
                curr_score = (lhs_std * lhs_count + rhs_std * rhs_count)/self.n_samples
            elif method == 'gini':
                # Gini Impurity
                lhs_gini = gini_score(lhs_count, lhs_sum)
                rhs_gini = gini_score(rhs_count, rhs_sum)
                curr_score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/self.n_samples
            else:
                raise Exception("Unknown parameter \"%s\" for method" % method)
            if curr_score < self.score:
                thres = (xi + X_sort[i+1])/2
                self.var_idx, self.score, self.split_val = var_idx, curr_score, thres

    def predict(self, X):
        if X.values.ndim == 1:
            return np.array([self.predict_row(X.values)])
        return np.array([self.predict_row(xi) for xi in X.values])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        t = self.left if xi[self.var_idx] <= self.split_val else self.right
        return t.predict_row(xi)

    def __repr__(self):
        s = 'n_samples: {:d}; val: {:.5f}'.format(self.n_samples, self.val)
        if not self.is_leaf:
            s += ' score: {:.5f}; split: {}<={:.3f}'.format(self.impurity, self.split_name, self.split_val)
        return s
    
    def get_info(self):
        if self.is_leaf:
            return self.n_samples, self.val
        else:
            return self.n_samples, self.val, self.var_idx, self.split_val, self.impurity

    def get_n_splits(self):
        n_leaves = 0 if self.is_leaf else 1
        if self.left is not None:
            n_leaves += self.left.get_n_splits()
        if self.right is not None:
            n_leaves += self.right.get_n_splits()
        return n_leaves  

    def preorder(self, depth=0):
        "Pre-order tree traversal"
        if self is not None:
            yield self
        if self.left is not None:
            for leaf in self.left.preorder(depth=depth+1):
                yield leaf
        if self.right is not None:
            for leaf in self.right.preorder(depth=depth+1):
                yield leaf


if __name__ == '__main__':
    # load Data
    file_name = 'tests/UniversalBank_cleaned.csv'
    target = "Personal Loan_1"
    X_train, X_test, y_train, y_test = load_data(file_name, target, test_size=0.2, seed=42)

    tree = DecisionTree(X_train, y_train)

    # descriptors
    print("max depth: %d" % tree.get_max_depth())
    print("n_splits:  %d" % tree.get_n_splits())

    # accuracy
    y_pred = tree.predict(X_test)
    acc_test = np.mean(y_pred == y_test)
    y_pred = tree.predict(X_train)
    acc_train = np.mean(y_pred == y_train)
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    for i, leaf in enumerate(tree.preorder()):
            d = leaf.depth
            print('%03d'%i,'-'*d, leaf)