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
import time
import matplotlib.pyplot as plt

from DecisionTree import DecisionTree
from utilities import perm_feature_importance, confusion_matrix, calc_f1_score, split_data


if __name__ == '__main__':
    # load Data
    # Binary class test with 5000 samples
    file_name = 'tests/UniversalBank_cleaned.csv'
    #file_name = 'tests/UniversalBank_one_hot.csv'
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

    tree.fit(X_train, y_train)
    
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