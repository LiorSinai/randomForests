"""

Clean iris data


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    #### -------------- Load data  -------------- ###
    file_name = 'tests/Iris.csv'
    iris = pd.read_csv(file_name)

    iris.info()
    print(iris.head())

    #### -------------- process data  -------------- ###
    ## Drop columns which are not significant
    iris.drop(["Id"], axis=1,inplace=True)
    ## Convert Categorical Columns to Dummies
    iris['Species'] = iris.Species.astype('category')
    iris['Species'] = iris['Species'].cat.codes

    print("\nafter processing:")
    print(iris.head())
    
    # #### -------------- inspect data  -------------- ###
    n_samples, n_features = iris.shape
    n_features -= 1 # exclude the target column
    fig, axes = plt.subplots(1, n_features, sharey=True)

    target = 'Species'

    groups = iris.groupby([target])
    i = -1
    for col in iris.columns:
        if col == target:
            continue
        i += 1   

        n_unqiue = len(iris[col].unique())

        x0 = groups[col].get_group(0)
        x1 = groups[col].get_group(1)
        x2 = groups[col].get_group(2)
        axes[i].hist([x0, x1, x2], label=['0', '1', '2'], align='mid')
        axes[i].set_xlabel(col)
    
    axes[0].set_ylabel('frequency')
    axes[0].legend(title="Target", loc="upper left")
    fig.suptitle("Distribution of variables")


    fig, ax = plt.subplots()
    ax.hist(iris[target], label=['0', '1', '2'], align='mid')
    ax.set_ylabel('frequency')
    ax.set_xticks(iris[target].unique())
    ax.set_title("Target variable: %s" % target)


    print(iris.corr()[target])

    #### --------------  save data  -------------- ###
    #save data
    iris.to_csv(file_name[:-4]+'_cleaned.csv', index=False)



    #### --------------  random forest model  -------------- ###
    X = iris.drop(columns=[target])
    y = iris[target]
    n_samples, n_features = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    rfc = RandomForestClassifier(random_state=42, n_estimators=20)
    rfc.fit(X_train, y_train)

    acc_test = rfc.score(X_test, y_test)
    acc_train = rfc.score(X_train, y_train)
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))

    ### ----------- Fitting acuracy per number of trees ----------- ###### 
    fig, ax = plt.subplots()
    preds = np.stack([e.predict_proba(X_test) for e in rfc.estimators_])
    n_trees = rfc.n_estimators
    n = len(y_test)
    acc = np.zeros(n_trees)
    for i in range(0, n_trees):
        y_pred = np.argmax(np.sum(preds[:i+1, :, :], axis=0), axis=1)
        acc[i] = np.mean(y_pred == y_test)
    ax.plot(acc)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("accuracy")

    plt.show()


    plt.show()
