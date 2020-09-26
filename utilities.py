"""
17 September 2020

Random Forests from scratch

Scratchpad

"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict


def split_data(X, y, test_size=0.1, seed=1):
    # shuffle data
    np.random.seed(seed)
    perm = np.random.permutation(X.index)
    X = X.loc[perm]
    y = y.loc[perm]
    
    # split into training and test sets
    n_samples = X.shape[0]
    if isinstance(test_size, float):
        if test_size <= 0 or test_size >= 1:
            raise ValueError("The test size should fall in the range (0,1)")
        n_train = n_samples - round(test_size*n_samples)
    elif isinstance(test_size, int):
        n_train = n_samples - test_size
    else:
        raise ValueError("Improper type \'%s\' for test_size" % type(test_size))

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

def encode_one_hot(data): # note: pd.get_dummies(df) does the same
    # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
    one_hot = np.zeros((data.size, data.max()+1))
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot

def check_RandomState(random_state):
    """ Parse different input types for the random state"""
    if  random_state is None: 
        rng = np.random.RandomState() 
    elif isinstance(random_state, int): 
        # seed the random state with this integer
        rng = np.random.RandomState(random_state) 
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise ValueError ("improper type \'%s\' for random_state parameter" % type(random_state))
    return rng

def check_sample_size(sample_size, n_samples: int):
    if sample_size is None:
        n = n_samples
    elif isinstance(sample_size, int):
        if sample_size == 1:
            warnings.warn("Interpreting sample_size as 1 sample. Use sample_size=1.0 for 100% of the data")
        n = min(sample_size, n_samples)
    elif isinstance(sample_size, float):
        frac = min(sample_size, 1)
        n = int(frac*n_samples)
    else:
        raise ValueError("Improper type \'%s\' for sample_size" %type(sample_size))
    return n

def confusion_matrix(y_actual, y_pred):
    """ Returns a confusion matrix where the rows are the actual classes, and the columns are the predicted classes"""
    if y_actual.shape != y_pred.shape:
        raise ValueError ("input arrays must have the same shape, {}!={}".format(y_actual.shape, y_pred.shape))
    n = max(max(y_actual), max(y_pred)) + 1
    C = np.zeros((n, n), dtype=int)
    for label_actual in range(n):
        idxs_true = (y_actual == label_actual)
        for label_pred in range(n):
            C[label_actual, label_pred] = sum(y_pred[idxs_true] == label_pred)
    return C

def calc_f1_score(y_actual, y_pred) -> Tuple[float]:
    C = confusion_matrix(y_actual, y_pred)
    if C.shape[0] != 2:
        raise ValueError ("input arrays must only have binary values")
    recall    = C[1][1]/(C[1][0]+C[1][1])
    precision = C[1][1]/(C[0][1]+C[1][1]) 
    if (recall == 0) or (precision == 0):
        f1 = 0
    else:
        f1 = 2 * recall*precision/(recall + precision) # = 2/((1/recall)+(1/precision))
    return recall, precision, f1


def perm_feature_importance(model, X, y, n_repeats=10, random_state=None) -> Dict:
    """Calculate feature importance based on random permutations of each feature column.
    The larger the drop in accuracy from shuffling each column, the higher the feature importance.

    """
    if getattr(model, 'predict', None) is None:
        raise Exception("model does not have a predict method")

    y_pred = model.predict(X)
    acc_full = np.mean(y_pred == y)
    n_features = X.shape[1]

    random_instance = check_RandomState(random_state)

    feature_importances = np.zeros((n_repeats, n_features))
    for j, col in enumerate(X.columns):
        X_sub = X.copy()
        for i in range(n_repeats):
            X_sub[col] = random_instance.permutation(X_sub[col].values)
            y_pred = model.predict(X_sub)
            feature_importances[i, j] = acc_full - np.mean(y_pred == y)

    return {'means': np.mean(feature_importances, axis=0), 'stds': np.std(feature_importances, axis=0)}
