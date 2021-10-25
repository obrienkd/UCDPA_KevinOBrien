import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, f1_score, \
    recall_score

# read csv file in
nba = pd.read_csv(r'c:\users\hp\nba_draft_combine_all_years2.csv')

# predict if nba player is drafted or not based on values from the draft competition
nba.columns

# add drafted or no column = 0 or 1
nba['drafted or no'] = nba.apply(lambda x: 1 if x['Draft pick'] > 0 else 0, axis=1)
nba['drafted_top20'] = nba.apply(lambda x: 2 if (x['Draft pick'] >0 and x['Draft pick'] <21) else (1 if x['Draft pick'] >=21 else 0), axis=1)

# class imbalance
print(nba['drafted or no'].value_counts())
print(nba['drafted_top20'].value_counts())

# Any n/a's?
print(nba.isna().sum())

# median values assigned to 67 missing verticals, weight, body fat and 233 bench press values
nba['Vertical (Max)'] = nba['Vertical (Max)'].replace(np.nan, nba['Vertical (Max)'].median())
nba['Vertical (Max Reach)'] = nba['Vertical (Max Reach)'].replace(np.nan, nba['Vertical (Max Reach)'].median())
nba['Vertical (No Step Reach)'] = nba['Vertical (No Step Reach)'].replace(np.nan, nba['Vertical (No Step Reach)'].median())
nba['Vertical (No Step)'] = nba['Vertical (No Step)'].replace(np.nan, nba['Vertical (No Step)'].median())
nba['Weight'] = nba['Weight'].replace(np.nan, nba['Weight'].median())
nba['Body Fat'] = nba['Body Fat'].replace(np.nan, nba['Body Fat'].median())
nba['Agility'] = nba['Agility'].replace(np.nan, nba['Agility'].median())
nba['Sprint'] = nba['Sprint'].replace(np.nan, nba['Sprint'].median())
nba['Bench'] = nba['Bench'].replace(np.nan, nba['Bench'].median())

# box plots -
sns.set_theme(style="whitegrid")

# convert all inches to cm
inch_cols = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)',
             'Vertical (Max Reach)', 'Vertical (No Step)',
             'Vertical (No Step Reach)']

for x in inch_cols:
    nba[x] = nba[x] * 2.54

# convert weight in lbs to kilos
nba['Weight'] = nba['Weight'] * 0.453592

# assess the distribution of the variables in the columns in comparison with the 2 categorical columns created - drafted yes or no, drafted top 20
numeric_figs = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)',
                'Vertical (Max Reach)', 'Vertical (No Step)',
                'Vertical (No Step Reach)', 'Weight', 'Body Fat', 'Bench', 'Agility',
                'Sprint']

for m in numeric_figs:
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Distribution of: {}'.format(m), fontsize=16)
    sns.histplot(x=nba[m], ax=ax[0])
    sns.boxplot(y=m, data=nba, ax=ax[1])
    fig.savefig("./figures/distributions/{}.png".format(m))

for n in numeric_figs:
    fig2 = sns.catplot(data=nba, kind="swarm", x="drafted or no", y=n, palette="dark")
    plt.title('Group Differences in: {}'.format(n), fontsize=16)
    fig2.savefig("./figures//group_differences/{}.png".format(n))

    for n in numeric_figs:
        fig2 = sns.catplot(data=nba, kind="bar", x="drafted_top20", y=n, palette="dark")
        plt.title('Group Differences in: {}'.format(n), fontsize=16)
        fig2.savefig("./figures//group_differences_top20/{}.png".format(n))

# figure out if there is much correlation between variables as some are similar
cor_mat = nba.corr()
f, ax = plt.subplots(figsize=(15, 8))
g = sns.heatmap(cor_mat, linewidths=1)
plt.show()
plt.savefig("./figures/correlation/correlation.png")

# transform variables due to outliers
for col in numeric_figs:
    nba["{}_zscore".format(col)] = (nba[col] - nba[col].mean()) / nba[col].std(ddof=0)

# remove all non z-score columns
nba.drop(columns=['Draft pick', 'Player', 'Year', 'Height (No Shoes)', 'Height (With Shoes)',
                  'Wingspan', 'Standing reach', 'Vertical (Max)', 'Vertical (Max Reach)',
                  'Vertical (No Step)', 'Vertical (No Step Reach)', 'Weight', 'Body Fat',
                  'Bench', 'Agility', 'Sprint', 'drafted_top10', 'drafted_top20'], inplace=True)
nba.columns

## evaluate a minority, SMOTE, majority of sampling
# from imblearn import under_sampling, over_sampling
# from imblearn.over_sampling import SMOTE

xcols = [x for x in list(nba.columns) if x != 'drafted or no' not in x]
undersample = RandomOverSampler(strategy='minority')
X = nba[xcols]
Y = nba[['drafted or no']]
x_over, y_over = undersample.fit_resample(X, Y)
print((x_over.shape, y_over.shape))

# due to some error in my pycharm importing imblearn, i've ran it in jupyter, exported as csv, and uploaded to pycharm
x_over = pd.read_csv('x_over.csv')
y_over = pd.read_csv('y_over.csv')

# With the new variables from SMOTE - enter PCA
pca = decomposition.PCA().fit(x_over)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

# evaluate the variance in the PCA
print(np.cumsum(pca.explained_variance_ratio_))

# PCA
x_pca = decomposition.PCA(n_components=10).fit(x_over)
x_pca.singular_values_
components = x_pca.transform(x_over)
x_projected = x_pca.inverse_transform(components)
print(x_projected)

# Support vector machine
x_train, x_test, y_train, y_test = train_test_split(x_projected,y_over, test_size=0.30) #x_projected, x_over
print((x_train.shape, x_test.shape, y_train.shape, y_test.shape))

grid_params = {'C':[0.001, 0.01, 0.1, 1,10,100,1000],'gamma':[100,10,1,0.1,0.001,0.0001], 'kernel':['linear','rbf', 'polynomial']}
gs = GridSearchCV(SVC(), grid_params, verbose=2, cv=3, n_jobs=2)
gs_results = gs.fit(x_train, y_train.values.ravel())
print("Best parameters set found on development set:")
print(gs_results.best_params_)

from sklearn.svm import SVC
f1 = []
precisions = []
recalls = []
for i in range(0,100):
    clf =SVC(**gs_results.best_params_)
    clf.fit(x_train, y_train.values.ravel())
    y_pred = clf.predict(x_test)
    f1.append(f1_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
print("F1 {}+- {}".format(np.mean(f1), np.std(f1)))
print("Precision {}+- {}".format(np.mean(precisions), np.std(precisions)))
print("Recall {}+- {}".format(np.mean(recalls), np.std(recalls)))


# F1 0.91
# Precision = 0.83
# Recall = 1.0

# Create confusion matrix 2x2
conf_mat = confusion_matrix(y_test, y_pred)

true_pos = conf_mat[0][0]
false_pos = conf_mat[0][1]
false_neg = conf_mat[1][0]
true_neg = conf_mat[1][1]
print(conf_mat)

Accuracy = (true_pos + true_neg) / (true_pos + false_pos + false_neg + true_neg)
Recall = true_pos/(true_pos+false_neg)
Precision = true_pos/(true_pos+false_pos)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)

# AUC
auc = roc_auc_score(y_test, y_pred)
# ROC
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='black', label='ROC (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Reference Line')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
plt.savefig("./figures/ROC/roc_curve.png")