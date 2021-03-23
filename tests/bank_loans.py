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
    ### Settings
    plt.rcParams.update({'font.size': 14})
    saveData = False

    #### -------------- Load data  -------------- ###
    file_name = 'tests/UniversalBank.csv'
    data = pd.read_csv(file_name)

    data.info()
    print(data.head())

    #### -------------- process data  -------------- ###
    ## Drop columns which are not significant
    data.drop(["ID","ZIP Code"], axis=1,inplace=True)
    ## Convert Categorical Columns to Dummies
    cat_cols = ["Family","Education"]##,"Personal Loan","Securities Account","CD Account","Online","CreditCard"]
    for col in cat_cols:
        data[col] = data[col].astype('category').cat.codes

    print("\nafter processing:")
    print(data.head())
    
    #### -------------- inspect data  -------------- ###
    n_samples, n_features = data.shape
    n_features -= 1 # exclude the target column
    target = 'Personal Loan'

    mod = n_features//2 if n_features % 2 == 0 else (n_features+1)//2
    fig, axes = plt.subplots(2, mod, sharey=True, figsize=(12, 8))
    groups = data.groupby([target])
    discrete_thres = 10
    i = -1
    for col in data.columns:
        if col == target:
            continue
        i += 1   
        r, c = i//mod, i % mod 

        x0 = groups[col].get_group(0)
        x1 = groups[col].get_group(1)

        n_unique = len(data[col].unique())
        if n_unique < discrete_thres:
            # classes = data[col].unique()
            # classes.sort()
            # w=0.4
            # freqs0, bins = np.histogram(x0, bins=n_unique)
            # freqs1, bins = np.histogram(x1, bins=n_unique)
            # axes[r][c].bar(classes-w/2, freqs0, width=w)
            # axes[r][c].bar(classes+w/2, freqs1, width=w)
            # or histogram
            axes[r][c].hist([x0, x1], bins=n_unique, label=['0', '1'], align='mid')
            axes[r][c].set_xticks(data[col].unique())
        else:
            axes[r][c].hist([x0, x1], label=['0', '1'], align='mid')

        axes[r][c].set_xlabel(col)
    if n_features % 2 != 0:
        fig.delaxes(axes[1, mod-1])
    
    axes[0][0].set_ylabel('frequency')
    axes[1][0].set_ylabel('frequency')
    axes[0][0].legend(title="Target", loc="upper left")
    fig.suptitle("Distribution of variables")

    # Pie chart of target variable
    # see https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py
    fig, ax = plt.subplots()
    freqs = [len(data[data[target]==0]), len(data[data[target]==1])]
    def label_fmt(pct, allvals):
        absolute = round(pct/100.*np.sum(allvals))
        return "  {:.1f}%\n({:d})".format(pct, absolute) 
    ax.pie(freqs, labels=['0','1'], labeldistance = None, startangle=70,
           autopct=lambda pct: label_fmt(pct, freqs), pctdistance=0.5, textprops=dict(color="w"))
    ax.legend(loc='right', bbox_to_anchor=(1, 0.8))
    ax.axis('equal')
    ax.set_title("Target variable: %s" % target)
    # ax.hist(data[target], bins=2, label=['0', '1'], align='mid')
    # ax.set_ylabel('frequency')
    # ax.set_xticks(data[target].unique())
    

    target_corrs = data.corr()[target].drop(target, axis= 0)
    order = np.argsort(-abs(target_corrs)) # negative for descending order sort by magnitude
    print(target_corrs[order])

    # zero in on income data
    
    approved = groups.get_group(1) # approved loans
    n_approved = approved.shape[0]
    freqs = np.histogram(approved['Income'], bins=np.arange(60, 221, 20))

    for f, bin in zip(*freqs):
        print('%d %.2f%%' % (bin, f/n_approved*100))

    
    #### --------------  save data  -------------- ###

    if saveData:
        data.to_csv(file_name[:-4]+'_cleaned.csv', index=False)

        cat_cols = ["Family","Education"]##,"Personal Loan","Securities Account","CD Account","Online","CreditCard"]
        data = pd.get_dummies(data,columns=cat_cols,drop_first=False)  # one_hot encode
        data.to_csv(file_name[:-4]+'_one_hot.csv', index=False)

    plt.show()