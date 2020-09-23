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
        p = c/n
        score -= p*p
    return score

class DecisionTree:
    def __init__(self, X, Y, max_depth=None, max_features=None, random_state=None):
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        self.RandomState = check_RandomState(random_state)
        self.max_features = max_features
        self.max_depth = max_depth

        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        # initialise arrays
        self.tree_ = BinaryTree() # tree_.children_left and tree_.children_right
        self.n_samples = []
        self.scores = []
        self.values = []
        self.impurities = []
        self.split_features = []
        self.split_values = []

        # set internal variables
        self.n_classes = Y.shape[1]
        self.features = X.columns
        self.max_depth_ = float('inf') if max_depth is None else max_depth
        if self.max_features is not None:
            if self.max_features is 'sqrt':
                n = np.ceil(np.sqrt(self.n_features)).astype(int)
            elif isinstance(self.max_features, int):
                n = min(self.max_features, self.n_features)
            else:
                raise Exception('Unknown parameter \'%s\' for max_features' % self.max_features)
        else:
            n = self.n_features 
        self.n_features_split = n
        self.size = 0 # current node = size - 1

        # initial split which recursively calls itself
        self._find_varsplit(X, Y, 0) 

        # set attributes
        self.depths = self.tree_.find_depths()

    def get_n_splits(self):
        return self.tree_.get_n_splits(0)

    def get_max_depth(self):
        return self.tree_.get_max_depth(0)

    def is_leaf(self, node_id):
        return self.scores[node_id] == float('inf')
    
    def split_name(self, node_id):
        return self.features[self.split_features[node_id]]

    def _set_defaults(self, node_id, Y):
        self.scores.append(float('inf'))
        val = Y.sum(axis=0)
        self.values.append(val)
        self.impurities.append(gini_score(val))
        self.split_features.append(None)
        self.split_values.append(None)
        self.n_samples.append(Y.shape[0])
        self.tree_.children_left.append(-1)
        self.tree_.children_right.append(-1)
    
    def _find_varsplit(self, X, Y, depth):
        node_id = self.size
        self.size += 1
        self._set_defaults(node_id, Y)
        if self.impurities[node_id] == 0: # only one class in this node
            return
        
        # random shuffling ensures a random variable is used if 2 splits are equal or if all features are used
        features = self.RandomState.permutation(self.n_features)[:self.n_features_split]

        # make the split
        for i in features:
            self._find_bettersplit(i, X, Y, node_id)
        if self.is_leaf(node_id):
            # a split was not made, most likely because all X values are the same
            features2 = set(range(self.n_features)).difference(features) #try all other features
            features2 = self.RandomState.permutation(list(features2))
            for i in features2:
                self._find_bettersplit(i, X, Y, node_id)
            if self.is_leaf(node_id):
                return # give up

        # make children
        if depth < self.max_depth_: 
            x_split = X.values[:, self.split_features[node_id]]
            lhs = np.nonzero(x_split<=self.split_values[node_id])
            rhs = np.nonzero(x_split> self.split_values[node_id])
            self.tree_.children_left[node_id] = self.size
            self._find_varsplit(X.iloc[lhs], Y[lhs[0], :], depth+1)
            self.tree_.children_right[node_id] = self.size
            self._find_varsplit(X.iloc[rhs], Y[rhs[0], :], depth+1)
    
    def _find_bettersplit(self, var_idx, X, Y, node_id):
        X = X.values[:, var_idx] 
        n_samples = self.n_samples[node_id]

        # sort the variables. Start with all on the right. Then move one sample to left one at a time
        order = np.argsort(X)
        X_sort, Y_sort = X[order], Y[order, :]
        rhs_count = Y.sum(axis=0)
        lhs_count = np.zeros(rhs_count.shape)

        for i in range(0, n_samples-1):
            xi, yi = X_sort[i], np.argmax(Y_sort[i, :])
            lhs_count[yi] += 1;  rhs_count[yi] -= 1
            if xi == X_sort[i+1]:
                continue
            # Gini Impurity
            lhs_gini = gini_score(lhs_count)
            rhs_gini = gini_score(rhs_count)
            curr_score = (lhs_gini * lhs_count.sum() + rhs_gini * rhs_count.sum())/n_samples
            if curr_score < self.scores[node_id]:
                self.split_features[node_id] = var_idx
                self.scores[node_id] = curr_score
                self.split_values[node_id]= (xi + X_sort[i+1])/2

    def _predict_row(self, xi):
        next_node = 0
        while not self.is_leaf(next_node):
            left = self.tree_.children_left[next_node]
            right = self.tree_.children_right[next_node]
            next_node = left if xi[self.split_features[next_node]] <= self.split_values[next_node] else right
        return self.values[next_node]

    def predict_prob(self, X):
        "Return the probability in the final leaf for each class, given as the fraction of each class in that leaf"
        if X.values.ndim == 1:
            probs = np.array([self._predict_row(X)])
        else:
            probs = np.array([self._predict_row(xi) for xi in X.values])
            probs /= np.sum(probs, axis=1)[:, None]
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

    def get_info(self, node_id):
        n_samples =  self.n_samples[node_id]
        val =        self.values[node_id]
        impurity =   self.impurities[node_id]
        var_idx    = self.split_features[node_id]
        split_val  = self.split_values[node_id]
        if self.is_leaf(node_id):
            return n_samples, val
        else:
            return n_samples, val, var_idx, split_val, impurity

    def leaf_to_string(self, node_id):
        if self.is_leaf(node_id):
            n_samples, val = self.get_info(node_id)
            s = 'n_samples: {:d}; val: {}'.format(n_samples, val)
        else:
            n_samples, val, var_idx, split_val, impurity = self.get_info(node_id)
            split_name = self.split_name(node_id)
            s =  'n_samples: {:d}; val: {}'.format(n_samples, val)
            s += ' score: {:.5f}; split: {}<={:.3f}'.format(impurity, split_name, split_val)
        return s


class BinaryTree():
    def __init__(self):
        self.children_left = []
        self.children_right = []

    @property
    def size(self):
        return max(len(self.children_left), len(self.children_right))

    def find_depths(self):
        depths = np.zeros(self.size, dtype=int)
        depths[0] = -1
        stack = [(0, 0)] # (parent, node_id)
        while stack:
            parent, node_id = stack.pop()
            if node_id == -1:
                continue
            depths[node_id] = depths[parent] + 1
            left = self.children_left[node_id]
            right = self.children_right[node_id]
            stack.extend([(node_id, left), (node_id, right)])
        return depths

    def is_leaf(self, node_id):
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        return right == left #(left == -1) and (right == -1)

    def get_max_depth(self, node_id=0):
        "Calculate the maximum depth of the tree"
        if self.is_leaf(node_id):
            return 0 
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        return max(self.get_max_depth(left), self.get_max_depth(right)) + 1

    def get_n_splits(self, node_id=0):
        "Return the number of leaves (number of parameters/2) not counting the final leaves in the tree"
        n_leaves = 0 
        stack = [0]
        while stack:
            node_id = stack.pop()
            if node_id == -1:
                continue
            if not self.is_leaf(node_id):
                n_leaves += 1
            left = self.children_left[node_id]
            right = self.children_right[node_id]
            stack.extend([left, right])
        return n_leaves  

    def preorder(self, node_id=0):
        "Pre-order tree traversal"
        # Note: the parallel arrays are already in pre-order
        # Therefore can just return np.arange(self.size)
        if node_id != -1:
            yield node_id
        left = self.children_left[node_id]
        right = self.children_right[node_id]
        if left != -1:
            for leaf in self.preorder(left):
                yield leaf
        if right != -1:
            for leaf in self.preorder(right):
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

    tree = DecisionTree(X_train, y_train, random_state=0, max_depth=None)
    # from sklearn.tree import DecisionTreeClassifier
    # tree = DecisionTreeClassifier()
    # tree.fit(X_train, y_train, random_state=42)

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

    depths = tree.depths
    for i, leaf in enumerate(tree.tree_.preorder()):
        d = depths[leaf]
        print('%03d'%i,'-'*d, tree.leaf_to_string(leaf))
        #print('%03d'%i,'-'*d, leaf)
        