"""

Clean iris data

From https://www.kaggle.com/uciml/iris

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    ### Settings
    plt.rcParams.update({'font.size': 14})

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
    species_names = iris['Species'].cat.categories
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

        x0 = groups[col].get_group(0)
        x1 = groups[col].get_group(1)
        x2 = groups[col].get_group(2)
        axes[i].hist([x0, x1, x2], label=species_names, align='mid')
        axes[i].set_xlabel(col)

    axes[0].set_ylabel('frequency')
    axes[0].legend(title="Target", loc="upper left")
    fig.suptitle("Distribution of variables")

    # Pie chart of target variable
    # see https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py
    fig, ax = plt.subplots()
    freqs = [len(iris[iris[target]==c]) for c in [0, 1, 2]]
    def label_fmt(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "  {:.1f}%\n({:d})".format(pct, absolute) 
    ax.pie(freqs, labels=species_names, labeldistance = None, startangle=-30,
           autopct=lambda pct: label_fmt(pct, freqs), pctdistance=0.5, textprops=dict(color="w"))
    ax.legend(loc='upper left')#, bbox_to_anchor=(0.8, 0.9))
    ax.axis('equal')
    ax.set_title("Target variable: %s" % target)
    # ax.hist(iris[target], label=['0', '1', '2'], align='mid')
    # ax.set_ylabel('frequency')
    # ax.set_xticks(iris[target].unique())
    # ax.set_title("Target variable: %s" % target)


    print(iris.corr()[target].drop(target, axis= 0).sort_values(ascending=False))

    #### --------------  save data  -------------- ###
    #save data
    #iris.to_csv(file_name[:-4]+'_cleaned.csv', index=False)

    #### --------------  split data into train and test sets  -------------- ###

    X = iris.drop(columns=[target])
    y = iris[target]
    n_samples, n_features = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #### -------------- baseline model -------------- ###
    print("\nBaseline model")
    def baseline_model(X):
        scores = np.zeros((X.shape[0], 3))
        # class 0
        scores[X['PetalLengthCm'] < 2.5, 0] += 5 # give away indicator
        # class 1
        #scores[:, 1] += np.logical_and(X['SepalLengthCm'] > 5.0, X['SepalLengthCm'] < 7.0)
        #scores[:, 1] += X['SepalWidthCm'] < 3.5
        scores[:, 1] += np.logical_and(X['PetalLengthCm'] >= 3.0, X['PetalLengthCm'] <= 5.0) 
        scores[:, 1] += np.logical_and(X['PetalWidthCm'] >= 1.0, X['PetalWidthCm'] < 1.5) 
        # class 2
        scores[:, 2] += X['SepalLengthCm'] >= 7.0
        scores[:, 2] += X['SepalWidthCm'] >= 3.5
        scores[:, 2] += (X['PetalLengthCm'] >= 5.0) 
        scores[:, 2] += (X['PetalWidthCm'] > 1.7  ) 

        preds = np.argmax(scores, axis=1)
        return preds
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

    #### --------------  random forest model  -------------- ###
    rfc = RandomForestClassifier(random_state=42, n_estimators=20, min_samples_leaf=1)
    rfc.fit(X_train, y_train)

    acc_test = rfc.score(X_test, y_test)
    acc_train = rfc.score(X_train, y_train)
    print("Random Forest model")
    print("train accuracy: %.2f%%" % (acc_train*100))
    print("test accuracy:  %.2f%%" % (acc_test*100))
    # print(confusion_matrix(y, y_full))
    # print("")

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
