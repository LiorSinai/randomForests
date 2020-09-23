"""
22 September 2020

Random Forests from scratch

Decision tree v3
- Tree is represent with 2 parallel arrays. This is more compact and requires much recusion than a linked list.
    - left_child_id = tree_.children_left[parent_id]
    - right_child_id = tree_.children_left[parent_id]
    - if id = -1, this node does not exist
- Works for multi-class problems 

"""

import numpy as np
import pandas as pd
from utilities import split_data, check_RandomState, encode_one_hot

def gini_score(counts): 
    score = 1
    n = sum(counts)
    for c in counts:
        score -= (c/n)**2
    return score

class DecisionTree:
    def __init__(self, X, Y, max_depth=None, depth=0, max_features=None, random_state=None):
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        self.RandomState = check_RandomState(random_state)
        self.max_features = max_features
        self.max_depth = max_depth
        self.depth = depth

        self.features = X.columns
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        self.n_classes = Y.shape[1]

        # defaults before split
        self.score = float('inf')
        self.val = Y.sum(axis=0)
        self.impurity = gini_score(self.val)
        self.var_idx = None
        self.left = None
        self.right = None
        
        # split
        max_depth_ = float('inf') if max_depth is None else max_depth
        if (self.depth < max_depth_) and (self.impurity > 0):
            self._find_varsplit(X, Y) 

    @property
    def is_leaf(self):
        return self.score == float('inf')
    
    @property
    def split_name(self):
        return self.features[self.var_idx]

    def get_max_depth(self):
        "Calculate the maximum depth of the tree"
        if self.is_leaf:
            return 0 
        max_left = 0 if self.left is None else self.left.get_max_depth()
        max_right = 0 if self.right is None else self.right.get_max_depth()
        return max(max_left, max_right) + 1

    def _find_varsplit(self, X, Y):
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
            self._find_bettersplit(i, X, Y)
        if self.is_leaf: 
            return

        # make children
        x_split = X.values[:, self.var_idx]
        lhs = np.nonzero(x_split<=self.split_val)
        rhs = np.nonzero(x_split>self.split_val)
        self.left =  DecisionTree(X.iloc[lhs], 
                                 Y[lhs[0], :], 
                                 depth=self.depth+1, 
                                 max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState)
        self.right = DecisionTree(X.iloc[rhs], 
                                 Y[rhs[0], :], 
                                 depth=self.depth+1, 
                                 max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState)

    def _find_bettersplit(self, var_idx, X, Y):
        X = X.values[:, var_idx] 

        # sort the variables. Start with all on the right. Then move one sample to left one at a time
        order = np.argsort(X)
        X_sort, Y_sort = X[order], Y[order, :]
        rhs_count = Y.sum(axis=0)
        lhs_count = np.zeros(rhs_count.shape)

        # O(n) complexity
        for i in range(0, self.n_samples-1):
            xi, yi = X_sort[i], np.argmax(Y_sort[i, :])
            lhs_count[yi] += 1;  rhs_count[yi] -= 1
            if xi == X_sort[i+1]:
                continue
            # Gini Impurity
            lhs_gini = gini_score(lhs_count)
            rhs_gini = gini_score(rhs_count)
            curr_score = (lhs_gini * lhs_count.sum() + rhs_gini * rhs_count.sum())/self.n_samples
            if curr_score < self.score:
                thres = (xi + X_sort[i+1])/2
                self.var_idx, self.score, self.split_val = var_idx, curr_score, thres

    def _predict_row(self, xi):
        if self.is_leaf:
            return self.val
        t = self.left if xi[self.var_idx] <= self.split_val else self.right
        return t._predict_row(xi)

    def predict_prob(self, X):
        "Return the probability in the final leaf for each class, given as the fraction of each class in that leaf"
        if X.values.ndim == 1:
            probs = np.array([self._predict_row(X)])
        else:
            probs = np.zeros((X.shape[0], self.n_classes))
            for i, xi in enumerate(X.values):
                vals = self._predict_row(xi)
                probs[i, :] = vals/vals.sum()
        return probs

    def predict(self, X):
        "Return the most likely class in the final leaf"
        probs = self.predict_prob(X)
        return np.argmax(probs, axis=1)

    def predict_count(self, X):
        "Return the sample count in the final leaf for each class"
        if X.values.ndim == 1:
            return np.array([self._predict_row(X.values)])
        return np.array([self._predict_row(xi) for xi in X.values])

    def __repr__(self):
        s = 'n_samples: {:d}; val: {}'.format(self.n_samples, self.val)
        if not self.is_leaf:
            s += ' score: {:.5f}; split: {}<={:.3f}'.format(self.impurity, self.split_name, self.split_val)
        return s
    
    def get_info(self):
        if self.is_leaf:
            return self.n_samples, self.val
        else:
            return self.n_samples, self.val, self.var_idx, self.split_val, self.impurity

    def get_n_splits(self):
        "Return the number leaves (number of parameters/2) not counting the final leaves in the tree"
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
    # Binary class test with 5000 samples
    file_name = 'tests/UniversalBank_cleaned.csv'
    target = "Personal Loan"
    # 3-class test with 1000 samples
    # file_name = 'tests/Iris_cleaned.csv'  
    # target = "Species"

    # Load Data
    data = pd.read_csv(file_name)
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)

    tree = DecisionTree(X_train, y_train)
    # from sklearn.tree import DecisionTreeClassifier
    # tree = DecisionTreeClassifier()
    # tree.fit(X_train, y_train)

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
        