"""
14 September 2020

Random Forests from scratch

Benchmark using Sci-kit Learn
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    y_pred = X_train['Income'] >= 100
    acc_train = np.mean(y_pred == y_train)
    print("train accuracy: %.2f%%" % (acc_train*100))
    y_pred = X_test['Income'] >= 100
    acc_test = np.mean(y_pred == y_test)
    print("test accuracy:  %.2f%%" % (acc_test*100))
    print("")

    #### -------------- random forest classifier  -------------- ###

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    # display descriptors
    depths =[e.get_depth() for e in rfc.estimators_]
    n_leaves = [e.get_n_leaves() for e in rfc.estimators_]
    acc_test = rfc.score(X_test, y_test)
    acc_train = rfc.score(X_train, y_train)
    print("depth range, average:    %d-%d, %.2f" % (np.min(depths), np.max(depths), np.mean(depths)))
    print("n_leaves range, average: %d-%d, %.2f" % (np.min(n_leaves), np.max(n_leaves), np.mean(n_leaves)))
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    # feature importance
    fi = permutation_importance(rfc, X_train, y_train)
    fi_means = fi.importances_mean
    fi_std = fi.importances_std
    order = np.argsort(fi_means) # order by magnitude

    fig, ax = plt.subplots()
    ax.barh(range(n_features), fi_means[order], xerr=fi_std[order])
    #ax.grid(True)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(X.columns[order])
    ax.set_ylabel('feature')
    ax.set_xlabel('feature importance score')
    ax.set_title("Importance for the random forest classifier")

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