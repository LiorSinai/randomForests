"""
14 September 2020

Random Forests from scratch

Benchmark using Sci-kit Learn
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import sklearn.tree
import graphviz 
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/sinai/Anaconda3/envs/MachineLearning/Library/bin/' # else graphviz doesn't work

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import sys
sys.path.append(".") # hack to add level above to the system path
from utilities import calc_f1_score

if __name__ == '__main__':
    #### -------------- load data  -------------- ###
    file_name = "tests/UniversalBank_cleaned.csv"
    #file_name = "tests/UniversalBank_one_hot.csv"
    target = "Personal Loan"

    data = pd.read_csv(file_name)
    X = data.drop(columns=[target])
    y = data[target]
    n_samples, n_features = X.shape[0], X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #### -------------- simplest strategy -------------- ###
    print("Based on Income>100 only:")

    def baseline_model(X):
        pred = np.full(X.shape[0], 1, dtype=bool)
        pred = np.logical_and(pred, (X['Income'] >= 100))
        #pred = np.logical_and(pred, (X['Securities Account'] == 0))
        #pred = np.logical_and(pred, (X['CCAvg'] > 3))
        return pred
    y_full = baseline_model(X)
    acc_full = np.mean(y_full == y)
    print("full accuracy:  %.2f%%" % (acc_full*100))
    y_pred = baseline_model(X_train)
    acc_train = np.mean(y_pred == y_train)
    print("train accuracy: %.2f%%" % (acc_train*100))
    y_pred = baseline_model(X_test)
    acc_test = np.mean(y_pred == y_test)
    print("test accuracy:  %.2f%%" % (acc_test*100))
    print(confusion_matrix(y, y_full))
    precision, recall, f1 = calc_f1_score(y, y_full)
    print("precision, recall, f1: {:.2f}%, {:.2f}%, {:.4f}".format(precision*100, recall*100, f1))
    print("")
    
    #### -------------- random forest classifier  -------------- ###
    warm_start=False
    start_time = time.time()
    if warm_start:
        rfc = RandomForestClassifier(random_state=42, bootstrap=True, min_samples_leaf=3, warm_start=warm_start, n_estimators=0)
        sample_size = 3500
        random_instance = np.random.RandomState(100)
        for i in range(100):
            rfc.n_estimators += 1
            rand_idxs = random_instance.permutation(np.arange(X_train.shape[0]))[:sample_size]
            rfc.fit(X_train.iloc[rand_idxs,:], y_train.iloc[rand_idxs])
    else:
        rfc = RandomForestClassifier(random_state=42, oob_score=True, min_samples_leaf=3)
        rfc.fit(X_train, y_train)
    end_time = time.time()
    print("fitting time: {:4f}s".format(end_time-start_time))


    # display descriptors
    depths =[e.get_depth() for e in rfc.estimators_]
    n_leaves = [e.get_n_leaves() for e in rfc.estimators_]
    acc_test = rfc.score(X_test, y_test)
    acc_train = rfc.score(X_train, y_train)
    print("depth range, average:    %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))
    print("train accuracy: %.2f%%" % (acc_train*100))
    if hasattr(rfc, 'oob_score_'):
        print("oob accuracy:   %.2f%%" % (rfc.oob_score_*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))
    y_pred = rfc.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    precision, recall, f1 = calc_f1_score(y_test, y_pred)
    print("precision, recall, f1: {:.2f}%, {:.2f}%, {:.4f}".format(precision*100, recall*100, f1))
    print("")

    # feature importance
    start_time = time.time()
    fi = permutation_importance(rfc, X_train, y_train)
    end_time = time.time()
    print("permutation_importance time: {:4f}s".format(end_time-start_time))
    fi_means = fi.importances_mean/(fi.importances_mean.sum())
    fi_std = fi.importances_std/(fi.importances_mean.sum())
    order = np.argsort(fi_means) # order by magnitude of the permutation importances

    target_corrs = data.corr()[target].drop(target, axis= 0) # correlations
    target_corrs = abs(target_corrs)/sum(abs(target_corrs))

    fig, ax = plt.subplots()
    inds = np.arange(n_features)
    width = 0.4
    ax.barh(inds+width/2, fi_means[order], width, xerr=fi_std[order], label='permutation')
    ax.barh(inds-width/2, rfc.feature_importances_[order], width, label='weighted impurity')
    #ax.barh(inds-width/2, target_corrs[order], width, label='correlation')
    #ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    ax.set_yticks(inds)
    ax.set_yticklabels(X.columns[order])
    ax.set_ylabel('feature')
    ax.set_xlabel('relative feature importance score')
    ax.set_title("Feature importances")

    ### ----------- Fitting acuracy per number of trees ----------- ###### 
    fig, ax = plt.subplots()
    preds = np.stack([e.predict(X_test) for e in rfc.estimators_])
    n_trees = rfc.n_estimators
    n = len(y_test)
    acc = np.zeros(n_trees)
    for i in range(0, n_trees):
        y_pred = np.mean(preds[:i+1, :], axis=0) > 0.5
        acc[i] = np.mean(y_pred == y_test)
    ax.plot(acc)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("accuracy")


    ### ----------- Plot tree ----------- ######   
    k = 0
    if 1==1:
        fig, ax = plt.subplots()
        sklearn.tree.plot_tree(rfc.estimators_[k], ax=ax)
        ax.set_title("tree %d"%k)

    if 1==1:
        dot_data = sklearn.tree.export_graphviz(rfc.estimators_[k], out_file=None) 
        graph = graphviz.Source(dot_data) 
        graph.render("tree_%d" % k) 

    plt.show()