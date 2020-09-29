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
from typing import List, Tuple, Dict

import time

from utilities import *

def gini_score(counts: List[int]) -> float: 
    score = 1
    n = sum(counts)
    for c in counts:
        p = c/n
        score -= p*p
    return score

class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.RandomState = check_RandomState(random_state)

        # initialise internal variables
        self.tree_ = BinaryTree() 
        self.n_samples = []
        self.values = []
        self.impurities = []
        self.split_features = []
        self.split_values = []
        self.n_features = None
        self.n_classes = None
        self.features = None
        self.size = 0 # current node = size - 1
        
    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
        elif Y.shape[1] == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable

        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns
        self.max_depth_ = float('inf') if self.max_depth is None else self.max_depth
        if self.max_features is not None:
            if self.max_features is 'sqrt':
                n = int(np.sqrt(self.n_features))
            elif isinstance(self.max_features, int):
                n = min(self.max_features, self.n_features)
            else:
                raise Exception('Unknown parameter \'%s\' for max_features' % self.max_features)
        else:
            n = self.n_features 
        self.n_features_split = n

        # initial split which recursively calls itself
        self._split_node(X, Y, 0) 

        # set attributes
        self.feature_importances_ = self.impurity_feature_importance()
        self.depths = self.tree_.find_depths()

    def get_n_splits(self) -> int:
        "The number of nodes (number of parameters/2) not counting the leaves in the tree"
        return self.tree_.n_splits
    
    def get_n_leaves(self) -> int:
        "The number of leaves (nodes without children) in the tree"
        return self.tree_.n_leaves

    def get_max_depth(self) -> int:
        "The maximum depth in the tree"
        return self.tree_.get_max_depth(0)

    def is_leaf(self, node_id: int) -> bool:
        return self.tree_.is_leaf(node_id)
    
    def split_name(self, node_id: int) -> str:
        return self.features[self.split_features[node_id]]

    def _set_defaults(self, node_id: int, Y):
        val = Y.sum(axis=0)
        self.values.append(val)
        self.impurities.append(gini_score(val))
        self.split_features.append(None)
        self.split_values.append(None)
        self.n_samples.append(Y.shape[0])
        self.tree_.add_node()
    
    def _split_node(self, X, Y, depth: int):
        node_id = self.size
        self.size += 1
        self._set_defaults(node_id, Y)
        if self.impurities[node_id] == 0: # only one class in this node
            return
        
         # random shuffling removes any bias due to the feature order
        features = self.RandomState.permutation(self.n_features)[:self.n_features_split]

        # make the split
        best_score = float('inf')
        for i in features:
            best_score= self._find_bettersplit(i, X, Y, node_id, best_score)
        if best_score == float('inf'):
            return
            # if X.shape[0] <= self.min_samples_leaf:
            #     return
            # ## a split was not made, either because all X values are the same or because min_samples_leaf was not satisfied
            # ## try all other features to force a split -> Can increases overfitting
            # features2 = np.setxor1d(np.arange(self.n_features), features)
            # features2 = self.RandomState.permutation(list(features2))
            # for i in features2:
            #     best_score = self._find_bettersplit(i, X, Y, node_id, best_score)
            # if best_score == float('inf'):
            #     return # give up

        # make children
        if depth < self.max_depth_: 
            x_split = X.values[:, self.split_features[node_id]]
            lhs = np.nonzero(x_split<=self.split_values[node_id])
            rhs = np.nonzero(x_split> self.split_values[node_id])
            self.tree_.set_left_child(node_id, self.size)
            self._split_node(X.iloc[lhs], Y[lhs[0], :], depth+1)
            self.tree_.set_right_child(node_id, self.size)
            self._split_node(X.iloc[rhs], Y[rhs[0], :], depth+1)
    
    def _find_bettersplit(self, var_idx: int, X, Y, node_id: int, best_score:float) -> float:
        X = X.values[:, var_idx] 
        n_samples = self.n_samples[node_id]

        # sort the variables. 
        order = np.argsort(X)
        X_sort, Y_sort = X[order], Y[order, :]

        #Start with all on the right. Then move one sample to left one at a time
        rhs_count = Y.sum(axis=0)
        lhs_count = np.zeros(rhs_count.shape)
        for i in range(0, n_samples-1):
            xi, yi = X_sort[i], Y_sort[i, :]
            lhs_count += yi;  rhs_count -= yi
            if (xi == X_sort[i+1]) or (sum(lhs_count) < self.min_samples_leaf):
                continue
            if sum(rhs_count) < self.min_samples_leaf:
                break
            # Gini Impurity
            lhs_gini = gini_score(lhs_count)
            rhs_gini = gini_score(rhs_count)
            curr_score = (lhs_gini * lhs_count.sum() + rhs_gini * rhs_count.sum())/n_samples
            if curr_score < best_score:
                self.split_features[node_id] = var_idx
                best_score = curr_score
                self.split_values[node_id]= (xi + X_sort[i+1])/2
        return best_score

    def _predict_row(self, xi):
        next_node = 0
        while not self.is_leaf(next_node):
            left, right = self.tree_.get_children(next_node)
            next_node = left if xi[self.split_features[next_node]] <= self.split_values[next_node] else right
        return self.values[next_node]

    def _predict_batch(self, X, node=0):
        # Helper function for predict_prob(). Predicts multiple batches of a row at time. Faster than _predict_row(self, xi)
        if self.is_leaf(node):
            return self.values[node]
        if len(X) == 0:
            return np.empty((0, self.n_classes))
        left, right = self.tree_.get_children(node)

        lhs = X[:, self.split_features[node]] <= self.split_values[node]
        rhs = X[:, self.split_features[node]] >  self.split_values[node]

        probs = np.zeros((X.shape[0], self.n_classes))
        probs[lhs] = self._predict_batch(X[lhs], node=left)
        probs[rhs] = self._predict_batch(X[rhs], node=right)
        return probs

    def predict_prob(self, X):
        "Return the probability in the final leaf for each class, given as the fraction of each class in that leaf"
        if X.values.ndim == 1:
            probs = np.array([self._predict_row(X)])
        else:
            #start_time = time.time()
            #probs = np.apply_along_axis(self._predict_row, 1, X.values) # slow because this is a for loop
            probs = self._predict_batch(X.values)
            #end_time = time.time()
            #print('%.1fms' % ((end_time-start_time)*1000))
            probs /= np.sum(probs, axis=1)[:, None]
        return probs

    def predict(self, X):
        "Return the most likely class in the final leaf"
        probs = self.predict_prob(X)
        return np.nanargmax(probs, axis=1)

    def predict_count(self, X):
        "Return the sample count in the final leaf for each class"
        if X.values.ndim == 1:
            return np.array([self._predict_row(X.values)])
        return np.apply_along_axis(self._predict_row, 1, X.values)

    def impurity_feature_importance(self):
        """Calculate feature importance weighted by the number of samples affected by this feature at each split point. """
        feature_importances = np.zeros(self.n_features)
        total_samples = self.n_samples[0]
        for node in range(len(self.impurities)):
            if self.is_leaf(node):
                continue 
            spit_feature = self.split_features[node]
            impurity = self.impurities[node]
            n_samples = self.n_samples[node]
            # calculate score
            left, right = self.tree_.get_children(node)
            lhs_gini = self.impurities[left]
            rhs_gini = self.impurities[right]
            lhs_count = self.n_samples[left]
            rhs_count = self.n_samples[right]
            score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
            # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
            feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)

        feature_importances = feature_importances/feature_importances.sum()
        return feature_importances

    def get_info(self, node_id: int):
        n_samples =  self.n_samples[node_id]
        val =        self.values[node_id]
        impurity =   self.impurities[node_id]
        var_idx    = self.split_features[node_id]
        split_val  = self.split_values[node_id]
        if self.is_leaf(node_id):
            return n_samples, val, impurity
        else:
            return n_samples, val, var_idx, split_val, impurity

    def node_to_string(self, node_id: int) -> str:
        if self.is_leaf(node_id):
            n_samples, val, impurity = self.get_info(node_id)
            s = 'n_samples: {:d}; value: {}; impurity: {:.4f}'.format(n_samples, val.astype(int), impurity)
        else:
            n_samples, val, var_idx, split_val, impurity = self.get_info(node_id)
            split_name = self.split_name(node_id)
            s =  'n_samples: {:d}; value: {}; impurity: {:.4f}'.format(n_samples, val.astype(int), impurity)
            s += '; split: {}<={:.3f}'.format(split_name, split_val)
        return s


class BinaryTree():
    def __init__(self):
        self.children_left = []
        self.children_right = []

    @property
    def size(self):
        "The number of nodes in the tree"
        return len(self.children_left)

    @property
    def n_leaves(self):
        "The number of leaves (nodes without children) in the tree"
        return self.children_left.count(-1) 

    @property
    def n_splits(self):
        "The number of nodes (number of parameters/2) not counting the leaves in the tree"
        return self.size - self.n_leaves

    def add_node(self):
        self.children_left.append(-1)
        self.children_right.append(-1)
    
    def set_left_child(self, node_id: int, child_id: int):
        self.children_left[node_id] = child_id

    def set_right_child(self, node_id: int, child_id: int):
        self.children_right[node_id] = child_id

    def get_children(self, node_id: int): return self.children_left[node_id], self.children_right[node_id]

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

    def is_leaf(self, node_id: int):
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

    tree = DecisionTree(random_state=0,max_depth=None, min_samples_leaf=3,max_features='sqrt')
    # from sklearn.tree import DecisionTreeClassifier
    # tree = DecisionTreeClassifier()

    tree.fit(X_train, y_train,)

    # descriptors
    print("max depth: %d" % tree.get_max_depth())
    print("n_splits:  %d" % tree.get_n_splits())
    print("n_leaves:  %d" % tree.get_n_leaves())

    # accuracy
    y_pred = tree.predict(X_test)
    acc_test = np.mean(y_pred == y_test)
    y_pred = tree.predict(X_train)
    acc_train = np.mean(y_pred == y_train)
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))
    y_pred = tree.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    print(C)
    if C.shape[0] == 2:
        precision, recall, f1 = calc_f1_score(y_test, y_pred)
        print("precision, recall, F1: {:.2f}%, {:.2f}%, {:.4f}".format(precision*100, recall*100, f1))
    print("")
    
    # feature importance
    fi_perm = perm_feature_importance(tree, X_train, y_train, random_state=1) 
    fi1 = fi_perm['means']
    fi2 = tree.feature_importances_
    for fi in (fi1, fi2):
        order = np.argsort(fi)[::-1] # descending order
        print("Feature importances")
        for col, val in zip(X_train.columns[order], fi[order]):
            print('%-15s %.4f' % (col+':', val)) 
    print("")

    depths = tree.depths
    for i, leaf in enumerate(tree.tree_.preorder()):
        d = depths[leaf]
        print('%03d'%i,'-'*d, tree.node_to_string(leaf))
        #print('%03d'%i,'-'*d, leaf)
    
