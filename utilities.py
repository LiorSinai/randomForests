"""
17 September 2020

Random Forests from scratch

Scratchpad

"""

import numpy as np
import pandas as pd

def load_data(file_name:str, target_name: str, test_size=0.1, seed=1):
    # Load Data
    data = pd.read_csv(file_name)
    X = data.drop(columns=[target_name])
    y = data[target_name]

    # shuffle data
    np.random.seed(seed)
    perm = np.random.permutation(data.index)
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