"""
14 September 2020

Random Forests from scratch
https://course18.fast.ai/lessonsml1/lesson5.html

Test the tree ensemble

"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from TreeEnsemble import RandomForestClassifier 
from utilities import perm_feature_importance, confusion_matrix, calc_f1_score, split_data


def run_rfc(file_name: str, target: str, n_trees: int):
    data = pd.read_csv(file_name)
    X = data.drop(columns=[target])
    y = data[target]
    n_samples, n_features = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, seed=42)

    forest = RandomForestClassifier(n_trees=n_trees, 
                                    bootstrap=True,
                                    sample_size=1.0, # default is None
                                    max_features='sqrt', # default is None
                                    #max_depth = 5, # default is None
                                    oob_score=True,
                                    min_samples_leaf=3,
                                    random_state=42)

    start_time = time.time()
    forest.fit(X_train, y_train)
    end_time = time.time()
    print('Fitting time: %.3fs' % ((end_time-start_time)))
    print("")

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
    y_pred = forest.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    print(C)
    if C.shape[0] == 2:
        precision, recall, f1 = calc_f1_score(y_test, y_pred)
        print("precision, recall, f1: {:.2f}%, {:.2f}%, {:.4f}".format(precision*100, recall*100, f1))
        print("")

    ### ----------- Feaure importance ----------- ###### 
    start_time = time.time()
    fi_perm   =  perm_feature_importance(forest, X_train, y_train, random_state=forest.RandomState) # very slow
    end_time = time.time()
    print('Permutation importance time: %.3fs' % ((end_time-start_time)))
    fi1 = fi_perm['means']
    fi2 = forest.feature_importances_
    for fi in (fi1, fi2):
        order = np.argsort(fi)[::-1] # descending order
        print("Feature importances")
        for col, val in zip(X_train.columns[order], fi[order]):
            print('%-15s %.4f' % (col+':', val)) 
    print("")

    target_corrs = data.corr()[target].drop(target, axis= 0) # correlations
    target_corrs = abs(target_corrs)/sum(abs(target_corrs))

    return {
        'X_train': X_train,
        'X_test':  X_test,
        'y_train': y_train,
        'y_test':  y_test,
        'forest':  forest,
        'fi_perm': fi_perm,
        'n_samples': n_samples,
        'n_features': n_features,
        'target_corrs': target_corrs
    }

def print_tree(tree):
    depths = tree.depths
    for i, leaf in enumerate(tree.tree_.preorder()):
        d = depths[leaf]
        print('%03d'%i,'-'*d, tree.node_to_string(leaf))
        #print('%03d'%i,'-'*d, leaf)


def plot_feature_importance(forest, fi_perm, target_corrs):
    fi_means = fi_perm['means']/(fi_perm['means'].sum())
    fi_std = fi_perm['stds']/(fi_perm['means'].sum())
    order = np.argsort(fi_means) # order by magnitude of the permutation importances
    n_features = len(fi_means)

    fig, ax = plt.subplots()
    inds = np.arange(n_features)
    width = 0.4
    ax.barh(inds+width/2, fi_means[order], width, xerr=fi_std[order], label='permutation')
    ax.barh(inds-width/2, forest.feature_importances_[order], width, label='weighted impurity')
    #ax.barh(inds-width/2, target_corrs[order], width, label='correlation')
    #ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    ax.set_yticks(inds)
    features = forest.features
    features[7] ='Securities\nAccount' # make it easier to plot on axis
    ax.set_yticklabels(features[order])
    ax.set_ylabel('feature')
    ax.set_xlabel('relative feature importance score')
    ax.set_title("Feature importances")


def plot_fitting_accuracy_per_trees(forest, X, y):
    ### ----------- Fitting acuracy per number of trees ----------- ###### 
    fig, ax = plt.subplots()
    preds = np.stack([t.predict_prob(X) for t in forest.trees])
    n_trees = forest.n_trees
    n = len(y)
    acc = np.zeros(n_trees)
    for i in range(0, n_trees):
        y_pred = np.argmax(np.sum(preds[:i+1, :, :], axis=0), axis=1)
        acc[i] = np.mean(y_pred == y)
    ax.plot(acc)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("accuracy")


if __name__ == "__main__":
    # load data
    # Binary class test with 5000 samples
    file_name = 'tests/UniversalBank_cleaned.csv'
    target = "Personal Loan"
    n_trees = 20
    # 3-class test with 1000 samples
    #file_name = 'tests/Iris_cleaned.csv'  
    #target = "Species"
    #n_trees = 10

    plt.rcParams.update({'font.size': 14})

    info = run_rfc(file_name, target, n_trees)
    forest = info['forest']

    print_tree(forest.trees[0])

    fi_perm = info['fi_perm']
    target_corrs = info['target_corrs']
    plot_feature_importance(forest, fi_perm, target_corrs)
 
    X_test = info['X_test']
    y_test = info['y_test']
    plot_fitting_accuracy_per_trees(forest, X_test, y_test)

    plt.show()
