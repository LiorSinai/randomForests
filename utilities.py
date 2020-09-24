"""
17 September 2020

Random Forests from scratch

Scratchpad

"""

import numpy as np
import pandas as pd
import warnings


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
        n_train = round(1 - test_size*n_samples)
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

def check_sample_size(sample_size, n_samples):
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