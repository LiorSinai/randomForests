"""

Lior Sinai, 22 September 2020

Bank loan classification from
https://www.kaggle.com/sriharipramod/bank-loan-classification/

See notebook at 
https://www.kaggle.com/sriharipramod/bank-loan-classification-boosting-technique

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #### -------------- Load data  -------------- ###
    file_name = 'tests/UniversalBank.csv'
    data = pd.read_csv(file_name)

    data.info()
    print(data.head())

    #### -------------- process data  -------------- ###
    ## Drop columns which are not significant
    data.drop(["ID","ZIP Code"], axis=1,inplace=True)
    ## Convert Categorical Columns to Dummies
    cat_cols = ["Family","Education","Personal Loan","Securities Account","CD Account","Online","CreditCard"]
    #data = pd.get_dummies(data,columns=cat_cols,drop_first=True)  # one_hot encode
    for col in cat_cols:
        data[col] = data[col].astype('category').cat.codes

    print("\nafter processing:")
    print(data.head())
    
    #### -------------- inspect data  -------------- ###
    n_samples, n_features = data.shape
    n_features -= 1 # exclude the target column
    mod = n_features//2 if n_features % 2 == 0 else (n_features+1)//2
    fig, axes = plt.subplots(2, mod, sharey=True)

    target = 'Personal Loan'

    groups = data.groupby([target])
    discrete_thres = 10
    i = -1
    for col in data.columns:
        if col == target:
            continue
        i += 1   
        r, c = i//mod, i % mod 

        n_unqiue = len(data[col].unique())
        nbins = n_unqiue if (n_unqiue < discrete_thres) else None

        x0 = groups[col].get_group(0)
        x1 = groups[col].get_group(1)
        axes[r][c].hist([x0, x1], bins=nbins, label=['0', '1'], align='mid')

        if n_unqiue < discrete_thres:
           axes[r][c].set_xticks(data[col].unique())
        axes[r][c].set_xlabel(col)
    if n_features % 2 != 0:
        fig.delaxes(axes[1, mod-1])
    
    axes[0][0].set_ylabel('frequency')
    axes[1][0].set_ylabel('frequency')
    axes[0][0].legend(title="Target", loc="upper left")
    fig.suptitle("Distribution of variables")

    fig, ax = plt.subplots()
    ax.hist(data[target], bins=2, label=['0', '1'], align='mid')
    ax.set_ylabel('frequency')
    ax.set_xticks(data[target].unique())
    ax.set_title("Target variable: %s" % target)

    print(data.corr()[target])

    # zero in on income data
    
    approved = groups.get_group(1) # approved loans
    n_approved = approved.shape[0]
    freqs = np.histogram(approved['Income'], bins=np.arange(60, 221, 20))

    for f, bin in zip(*freqs):
        print('%d %.2f%%' % (bin, f/n_approved*100))

    #### --------------  save data  -------------- ###

    # save data
    data.to_csv(file_name[:-4]+'_cleaned.csv', index=False)

    plt.show()
