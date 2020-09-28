"""
14 September 2020

Random Forests from scratch

Testing to ensure reproducible results

"""

import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append(".") # hack to add level above to the system path

from TreeEnsemble import RandomForestClassifier 
from utilities import split_data, perm_feature_importance


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        file_name = "tests/UniversalBank_cleaned.csv"
        target = 'Personal Loan'
        data = pd.read_csv(file_name)
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=999, seed=42)

        self.data = X_train, X_test, y_train, y_test


    def first_split(self):
        X_train, X_test, y_train, y_test = self.data
        forest = RandomForestClassifier(n_trees=1, bootstrap=False, random_state=42, max_depth=1)

        forest.fit(X_train, y_train)

        depth = forest.trees[0].get_max_depth()
        self.assertEqual(1, depth)

        leaves = []
        tree0 = forest.trees[0]
        for node in (0, 1, 2):
            info = list(tree0.get_info(node))
            info[1] = info[1][1]/sum(info[1])
            leaves.append(info)

        info = [
            [4001, 0.09722569357660585, 2, 114.5, 0.17554571617030754], 
            [3204, 0.02122347066167291], 
            [797, 0.4027603513174404]
            ]

        for info_sk_leaf, info_my_leaf in zip (leaves, info):
            for (var1, var2) in zip(info_sk_leaf, info_my_leaf):
                self.assertAlmostEqual(var1, var2)
        

    def full_tree(self):
        X_train, X_test, y_train, y_test = self.data
        
        forest = RandomForestClassifier(n_trees=1, bootstrap=True, max_features=4, random_state=42)
        forest.fit(X_train, y_train)

        tree0 = forest.trees[0]
        depth = tree0.get_max_depth()
        self.assertEqual(depth, 13)

        n_leaves = sum([t.get_n_leaves() for t in forest.trees])
        self.assertEqual(n_leaves, 83)

        # should be 100% correct on the training data
        y0 = forest.predict(X_train)
        self.assertEqual(sum(y0==y_train), 3971)
        
        y1 = forest.predict(X_test)
        self.assertEqual(sum(y1==y_test), 979)
    
    
    def full_forest(self):
        X_train, X_test, y_train, y_test = self.data

        forest = RandomForestClassifier(n_trees=20, bootstrap=True, max_features=4, random_state=42, oob_score=True)
        forest.fit(X_train, y_train)

        depths =[t.get_max_depth() for t in forest.trees]
        n_leaves = [t.get_n_leaves() for t in forest.trees]
        self.assertEqual(sum(depths), 265)
        self.assertEqual(sum(n_leaves), 1736)

        self.assertAlmostEqual(0.9847538115471132, forest.oob_score_)

        # feature importances
        fi = [0.03290998, 0.02696517, 0.32195696, 0.12680437, 0.12968169,
              0.25954914, 0.03072707, 0.00458653, 0.04989645, 0.00683082,
              0.01009182]
        for (var1, var2) in zip(forest.feature_importances_, fi):
            self.assertAlmostEqual(var1, var2) 

        fi_perm = perm_feature_importance(forest, X_train, y_train, random_state=forest.RandomState)
        fi = [0.00289928, 0.00272432, 0.19155211, 0.06698325, 0.02099475,
              0.08990252, 0.00104974, 0.00027493, 0.00289928, 0.00062484,
              0.00052487]
        for (var1, var2) in zip(fi_perm['means'], fi):
            self.assertAlmostEqual(var1, var2) 
            
        # should be mostly correct on the training data
        y0 = forest.predict(X_train)
        self.assertEqual(sum(y0==y_train), 4000)

        # should have a high accuracy on the test data
        y1 = forest.predict(X_test)
        self.assertEqual(sum(y1==y_test), 984)


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

    

