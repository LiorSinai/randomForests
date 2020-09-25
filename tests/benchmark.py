"""
14 September 2020

Random Forests from scratch

Benchmark using Sci-kit Learn
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

if __name__ == '__main__':
    #### -------------- load data  -------------- ###
    file_name = "tests/UniversalBank_cleaned.csv"
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
        pred = np.logical_and(pred, (X['CCAvg'] > 3))
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
    print("")
    
    #### -------------- random forest classifier  -------------- ###

    rfc = RandomForestClassifier(random_state=42, oob_score=True, min_samples_leaf=3)

    start_time = time.time()
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
    if hasattr(rfc, 'oob_score_'):
        print("oob accuracy:   %.2f%%" % (rfc.oob_score_*100))
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    # feature importance
    fi = permutation_importance(rfc, X_train, y_train)
    fi_means = fi.importances_mean
    fi_std = fi.importances_std
    order = np.argsort(fi_means) # order by magnitude

    fig, ax = plt.subplots()
    inds = np.arange(n_features)
    width = 0.4
    fi = fi_means[order]/fi_means.sum()
    ax.barh(inds+width/2, fi, width, xerr=fi_std[order], label='permutation')
    fi = rfc.feature_importances_[order]
    ax.barh(inds-width/2, fi, width, label='weighted impurity')
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

    plt.show()