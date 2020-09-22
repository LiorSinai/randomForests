"""
14 September 2020

Random Forests from scratch

Testing

"""

import unittest
from sklearn.ensemble import RandomForestClassifier as RFC_sk
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sys
sys.path.append(".") # hack to add level above to the system path

from TreeEnsemble import RandomForestClassifier as RFC
from benchmark2 import preorder_sk, sklearn_leaf_to_string, get_info_sklearn_leaf



class TestRandomForest(unittest.TestCase):
    def setUp(self):
        file_name = "tests/UniversalBank_cleaned.csv"
        target = 'Personal Loan'
        data = pd.read_csv(file_name)
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.data = X_train, X_test, y_train, y_test


    def first_split(self):
        [X_train, X_test, y_train, y_test] = self.data
        myForest = RFC(n_trees=1, 
                       bootstrap=False,
                       random_state=42,
                       max_depth=1)
        skForest = RFC_sk(n_estimators=1, 
                          random_state=42, 
                          bootstrap=False, 
                          max_features=None, 
                          max_depth=1)

        skForest.fit(X_train, y_train)
        myForest.fit(X_train, y_train)

        #print("max depth of sklearn tree:", skForest.estimators_[0].tree_.max_depth)
        #print("max depth of my tree:     ", myForest.trees[0].max_depth_)
        d1 = skForest.estimators_[0].tree_.max_depth
        d2 = myForest.trees[0].get_max_depth()
        self.assertEqual(d1, d2)

        print("")
        sklearn_leaf_to_string(skForest.estimators_[0], 0)
        sklearn_leaf_to_string(skForest.estimators_[0], 1)
        sklearn_leaf_to_string(skForest.estimators_[0], 2)
        print("")
        print(myForest.trees[0])
        print(myForest.trees[0].left)
        print(myForest.trees[0].right)

        skLeaves = []
        skLeaves.append(get_info_sklearn_leaf(skForest.estimators_[0], 0))
        skLeaves.append(get_info_sklearn_leaf(skForest.estimators_[0], 1))
        skLeaves.append(get_info_sklearn_leaf(skForest.estimators_[0], 2))
        myLeaves = []
        tree0 = myForest.trees[0]
        for leaf in (tree0, tree0.left, tree0.right):
            info = list(leaf.get_info())
            info[1] = info[1][1]/sum(info[1])
            myLeaves.append(info)

        for info_sk_leaf, info_my_leaf in zip (skLeaves, myLeaves):
            for (var1, var2) in zip(info_sk_leaf, info_my_leaf):
                self.assertAlmostEqual(var1, var2)
        

    def full_tree(self):
        # common parameters
        bootstrap = False   # True or False. If True, introduces randomness -> test results may vary
        max_features = 'sqrt' # None or 'sqrt' If sqrt, introduces randomness -> test results may vary

        myForest = RFC(n_trees=1, 
                       bootstrap=bootstrap,
                       max_features=max_features,
                       random_state=42)
        skForest = RFC_sk(n_estimators=1, 
                          random_state=42, 
                          bootstrap=bootstrap, 
                          max_features=max_features,
                          criterion='gini')

        [X_train, X_test, y_train, y_test] = self.data
        skForest.fit(X_train, y_train)
        myForest.fit(X_train, y_train)

        print("")
        skTree0 = skForest.estimators_[0]
        sklearn_leaf_to_string(skTree0, 0)
        print('-', end=' '); sklearn_leaf_to_string(skTree0, skTree0.tree_.children_left[0])
        print('-', end=' '); sklearn_leaf_to_string(skTree0, skTree0.tree_.children_right[0])
        print("")
        myTree0 = myForest.trees[0]
        print(myTree0)
        print('-', myTree0.left)
        print('-', myTree0.right)

        # print("")
        # print("max depth of sklearn tree:", skTree0.tree_.max_depth)
        # print("max depth of my tree:     ", myTree0.max_depth_)
        d1 = skTree0.tree_.max_depth
        d2 = myTree0.get_max_depth()
        self.assertEqual(d1, d2, "TreeEnsemble and Sklearn trees have different depths")

        n_sk_leaves = sum([e.get_n_leaves() for e in skForest.estimators_])
        n_leaves = sum([t.get_n_splits() for t in myForest.trees])
        #self.assertEqual(n_leaves, n_sk_leaves, "TreeEnsemble and Sklearn trees have different number of leaves")
        # different by 1 ???

        # should be 100% correct on the training data
        y0 = myForest.predict(X_train)
        self.assertEqual(sum(y0==y_train), len(y0), "TreeEnsemble predictions are different to training")

        # should have the same values on the testing data
        #y1 = skForest.predict(X_test.iloc[3].values.reshape(1,-1))
        #y2 = myForest.predict(X_test.iloc[3])
        
        y1 = skForest.predict(X_test)
        y2 = myForest.predict(X_test)
        self.assertEqual(sum(y1==y2), len(y1), "TreeEnsemble predictions are different to Sklearn Trees")
    
    
    def full_forest(self):
        [X_train, X_test, y_train, y_test] = self.data
        # common parameters
        bootstrap = False     # True or False. If True, introduces randomness -> test results may vary
        max_features = 'sqrt' # None or 'sqrt' If sqrt, introduces randomness -> test results may vary
        n_trees = 20

        myForest = RFC(n_trees=n_trees, 
                       bootstrap=bootstrap,
                       random_state=42,
                       max_features=max_features
                       )
        skForest = RFC_sk(n_estimators=n_trees, 
                          random_state=42, 
                          bootstrap=bootstrap, 
                          max_features=max_features,
                          criterion='gini')
        skForest.fit(X_train, y_train)
        myForest.fit(X_train, y_train)

        # should be mostly correct on the training data
        #y0 = myForest.predict(X_train)
        #self.assertEqual(sum(y0==y_train), len(y0), "TreeEnsemble predictions are different to training")

        ySk = skForest.predict(X_test)
        y0 = myForest.predict(X_test)

        print("")
        print("Correct sklearn predictions:      %.3f" % (np.mean(ySk==y_test)))
        print("Correct TreeEnsemble predictions: %.3f" % (np.mean(y0==y_test)))

        self.assertEqual(sum(ySk==y0), len(y0), "TreeEnsemble predictions are different to Sklearn Trees")



def suite():
    "Set order of tests in ascending order of complexity and code required"
    suite = unittest.TestSuite()
    # functions test
    suite.addTest(TestRandomForest('first_split'))
    suite.addTest(TestRandomForest('full_tree'))
    suite.addTest(TestRandomForest('full_forest'))
    return suite



if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

    

