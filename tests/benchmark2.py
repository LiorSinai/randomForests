"""
17 September 2020

Random Forests from scratch

Scratchpad

"""

from sklearn.ensemble import RandomForestClassifier as RFC_sk
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import sys
sys.path.append(".") # hack to add level above to the system path

from TreeEnsemble import RandomForestClassifier as RFC
from utilities import perm_feature_importance

def preorder_sk(tree, node_id=0):
    "Pre-order tree traversal"
    if node_id != -1:
        yield node_id
        for leaf in preorder_sk(tree, tree.tree_.children_left[node_id]):
            yield leaf
        for leaf in preorder_sk(tree, tree.tree_.children_right[node_id]):
            yield leaf


def traverse_tree_sk(tree):
    #https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    tree_ = tree.tree_
    n_nodes = tree_.node_count
    n_features = tree_.n_features
    children_left = tree_.children_left
    children_right = tree_.children_right

    n_samples = tree_.weighted_n_node_samples[0]

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    feature_importances = np.zeros(shape=n_features, dtype=float)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

        # gini feature importance # note already present in tree.feature_importances_
        if not is_leaves[node_id]:
            left = children_left[node_id]
            right = children_right[node_id]
            var_idx = tree.tree_.feature[node_id]

            n_node = tree_.weighted_n_node_samples[node_id]
            n_left = tree_.weighted_n_node_samples[left]
            n_right = tree_.weighted_n_node_samples[right]

            score = tree_.impurity[node_id]
            score_left =  tree_.impurity[left]
            score_right = tree_.impurity[right] 
            score_w = (score_left * n_left + score_right * n_right)/n_node        

            feature_importances[var_idx] += (score - score_w) * (n_node/n_samples)

    feature_importances = feature_importances/sum(feature_importances)
    
    return node_depth, is_leaves, feature_importances


def sklearn_leaf_to_string(tree, node_id):
    var_idx = tree.tree_.feature[node_id]

    n_samples = tree.tree_.n_node_samples[node_id]
    values = tree.tree_.value[node_id][0]
    val = values[1]/sum(values)

    is_leaf = (tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id])

    s = 'n_samples: {:d}; val: {:.5f}'.format(n_samples, val)
    if not is_leaf:
        split_val = tree.tree_.threshold[node_id]
        score = tree.tree_.impurity[node_id]
        split_name = 'Variable%d' %(var_idx+1)
        s += ' score: {:.5f}; split: {}<={:.3f}'.format(score, split_name, split_val)
    return s


def get_info_sklearn_leaf(tree, node_id):
    n_samples = tree.tree_.n_node_samples[node_id]
    values = tree.tree_.value[node_id][0]
    val = values[1]/sum(values)
    is_leaf = (tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id])
    if is_leaf:
        return n_samples, val
    else:
        var_idx = tree.tree_.feature[node_id]
        split_val = tree.tree_.threshold[node_id]
        score = tree.tree_.impurity[node_id]
        return n_samples, val, var_idx, split_val, score


def almostEqual(a, b):
    return round(a-b, 7) == 0

if __name__ == '__main__':
    # common parameters
    ## for 1 tree: (remove randomness)
    n_trees = 1
    max_features = None
    bootstrap = False
    min_samples_leaf = 1
    compare_trees = True
    ## for multiple trees:
    n_trees = 20
    max_features = 'sqrt'
    bootstrap = True
    min_samples_leaf = 3
    compare_trees = False

    # TreeEnsemble only parameters
    sample_size = None

    forest = RFC(n_trees=n_trees, 
                 bootstrap=bootstrap,
                 max_features=max_features,
                 sample_size=sample_size,
                 min_samples_leaf= min_samples_leaf,
                 random_state=42)
    skForest = RFC_sk(n_estimators=n_trees, 
                      random_state=42, 
                      bootstrap=bootstrap, 
                      max_features=max_features,
                      min_samples_leaf = min_samples_leaf,
                      criterion='gini')

    # load data
    file_name = "tests/UniversalBank_cleaned.csv"
    target = "Personal Loan"
    data = pd.read_csv(file_name)
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    forest.fit(X_train, y_train)
    skForest.fit(X_train, y_train)

    # Number of splits in the forest
    n_sk_leaves = [e.get_n_leaves() for e in skForest.estimators_]
    n_leaves = [t.get_n_leaves() for t in forest.trees]
    print("TE n_leaves range, average: %d-%d, %.2f" % (np.min(n_sk_leaves), np.max(n_sk_leaves), np.mean(n_sk_leaves)))
    print("SK n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))

    # Maximum depth of trees
    depths_sk = [e.get_depth() for e in skForest.estimators_]
    depths = [t.get_max_depth() for t in forest.trees]
    print("TE depth range, average: %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("SK depth range, average: %d-%d, %.2f" % (np.min(depths_sk), np.max(depths_sk), np.mean(depths_sk)))

    # Accuracy
    ySk = skForest.predict(X_test)
    y0 = forest.predict(X_test)
    print("SK Correct predictions: %.3f" % (np.mean(ySk==y_test)))
    print("TE Correct predictions: %.3f" % (np.mean(y0==y_test)))
    print("")

    # feature importance
    if 1==1:
        fi_sk = []
        for tree in skForest.estimators_:
            _, _, fi= traverse_tree_sk(tree) 
            fi_sk.append(fi)
        fi_sk = np.mean(fi_sk, axis=0)
        
        fi1 = perm_feature_importance(forest, X_train, y_train, random_state=1)['means']
        fi_sk = permutation_importance(skForest, X_train, y_train).importances_mean

        #fi2 = forest.gini_feature_importance()  # gini impurity feature importance
        #fi_sk = skForest.feature_importances_   # gini impurity feature importance

        for fi in (fi1, fi_sk):
            order = np.argsort(fi)[::-1] # desceding order
            print("Feature importances")
            for col, val in zip(X_train.columns[order], fi[order]):
                print('%s: %.4f' % (col, val)) 
        print("")
    
    skTree0 = skForest.estimators_[0]
    myTree0 = forest.trees[0]

    if 1==0:
        # print Sklearn tree
        depths, _, _ = traverse_tree_sk(skTree0) 
        for i, node_id in enumerate(preorder_sk(skTree0)):
            d = depths[node_id]
            s = sklearn_leaf_to_string(skTree0, node_id)
            print('%03d'%i, '-'*d, '%s' % s)
        print("")
            
        # print myTree
        for i, node in enumerate(myTree0.tree_.preorder()):
            d = myTree0.depths[node]
            print('%03d'%i,'-'*d, myTree0.node_to_string(node))
        print("")

    if compare_trees:
        # compare all leaves
        skLeaves = []
        for node_id in (preorder_sk(skTree0)):
            skLeaves.append(get_info_sklearn_leaf(skTree0, node_id))
        myLeaves = []
        for j, node in enumerate(myTree0.tree_.preorder()):
            myLeaves.append(myTree0.get_info(node))
  
        # print which values are the same
        # note: leaves may be printed in different orders. When sample number is low, the choice for splitting is almost random
        # Therefore it is highly unlikely these will be the exact same
        # Also if there is slight difference in the number of splits, than that offsets everything else and makes the rest look wrong
        num_same = 0
        total_parameters = 0
        for i, info in enumerate(zip(skLeaves, myLeaves)):
            info_sk_leaf, info_my_leaf = info
            print(i, end = ' ')
            for (var1, var2) in zip(info_sk_leaf, info_my_leaf):
                if isinstance(var2, list) or isinstance(var2, np.ndarray):
                    var2 = var2[1]/sum(var2)
                same = almostEqual(var1, var2)
                total_parameters += 1
                num_same += same
                print(same, end=', ')
            print('')
        print("number of same variables: %d, %.2f%%" % (num_same, num_same/total_parameters*100))

        # compare only split parameters, 2 per node: split_idx and split_val
        # compare all leaves
        skLeaves = []
        for node_id in (preorder_sk(skTree0)):
            is_leaf = (skTree0.tree_.children_left[node_id] == -1)
            if not is_leaf:
                n_samples, val, split_idx, split_val, score = get_info_sklearn_leaf(skTree0, node_id)
                skLeaves.append([split_idx, split_val])
        myLeaves = []
        for node_id in myTree0.tree_.preorder():
            if not myTree0.is_leaf(node_id):
                n_samples, val, split_idx, split_val, impurity = myTree0.get_info(node_id)
                myLeaves.append([split_idx, split_val])
        num_same = 0
        total_parameters = 0
        for i, info in enumerate(zip(skLeaves, myLeaves)):
            for (var1, var2) in zip(info_sk_leaf, info_my_leaf):
                if isinstance(var2, list) or isinstance(var2, np.ndarray):
                    var2 = var2[1]/sum(var2)
                same = almostEqual(var1, var2)
                total_parameters += 1
                num_same += same
        print("number of same parameters: %d, %.2f%%" % (num_same, num_same/total_parameters*100))


