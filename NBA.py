import numpy
import pandas as pd
import matplotlib.pyplot as plt
import regex
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import decomposition, datasets
from sklearn.model_selection import KFold

# read csv file in
nba = pd.read_csv(r'c:\users\hp\nba_draft_combine_all_years2.csv')

# predict if nba player is drafted or not based on values from the draft competition

# drop draft pick column & height-no shoes

# class imbalance
print(nba['drafted or no'].value_counts())

# over sample / SMOTE / under
nba.isna().sum()

# Any n/a's?
print(nba.isna().sum())

# median values assigned to 67 missing verticals, weight, body fat and 233 bench press values

nba['Vertical (Max)'] = nba['Vertical (Max)'].replace(np.nan, nba['Vertical (Max)'].median())
nba['Vertical (Max Reach)'] = nba['Vertical (Max Reach)'].replace(np.nan, nba['Vertical (Max Reach)'].median())
nba['Vertical (No Step Reach)'] = nba['Vertical (No Step Reach)'].replace(np.nan,
                                                                          nba['Vertical (No Step Reach)'].median())
nba['Vertical (No Step)'] = nba['Vertical (No Step)'].replace(np.nan, nba['Vertical (No Step)'].median())
nba['Weight'] = nba['Weight'].replace(np.nan, nba['Weight'].median())
nba['Body Fat'] = nba['Body Fat'].replace(np.nan, nba['Body Fat'].median())
nba['Agility'] = nba['Agility'].replace(np.nan, nba['Agility'].median())
nba['Sprint'] = nba['Sprint'].replace(np.nan, nba['Sprint'].median())
nba['Bench'] = nba['Bench'].replace(np.nan, nba['Bench'].median())

print(nba.isna().sum())

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

# assess the distribution of the variables in the columns
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

# figure out if there is much correlation between variables as some are similar
cor_mat = nba.corr()
g = sns.heatmap(cor_mat, linewidths=1)
plt.show()
plt.savefig("./figures/correlation/correlation.png")
# large correlation between ht variables

# transform variables due to outliers

for col in numeric_figs:
    nba["{}_zscore".format(col)] = (nba[col] - nba[col].mean()) / nba[col].std(ddof=0)

# 384 drafted, 133 not drafted

# convert to 'drafted or no' to 1 or 0

# nba['draft'] = [1 if x == 'Drafted' else 0 for x in nba['drafted or no']]

# drop columns

# PCA - a lot of correlation between variables
nba.columns
# the column trying to preditct
y = nba['drafted or no']

# remove all non z-score columns
nba.drop(columns=['Draft pick', 'Player', 'Year', 'Height (No Shoes)', 'Height (With Shoes)',
                  'Wingspan', 'Standing reach', 'Vertical (Max)', 'Vertical (Max Reach)',
                  'Vertical (No Step)', 'Vertical (No Step Reach)', 'Weight', 'Body Fat',
                  'Bench', 'Agility', 'Sprint', 'drafted or no', 'drafted_top10'], inplace=True)
nba.columns

# one hot encode years


# pca

print(nba.isna().sum())
pca = decomposition.PCA().fit(nba)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
print(np.cumsum(pca.explained_variance_ratio_))
x_pca = decomposition.PCA(n_components=8).fit(nba)
x_pca.singular_values_
components = x_pca.transform(nba)
x_projected = x_pca.inverse_transform(components)
print(x_projected)

principalDF = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
principalDF

x_train, x_test, y_train, y_test = train_test_split(principalDF, y, test_size=0.3, random_state=666)
### KNN

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

print("The training accuracy score is : ", knn.score(x_train, y_train))
y_pred = knn.predict(x_test)
print("The test accuracy score is : ", accuracy_score(y_test, y_pred))

param_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute'],
              'leaf_size': [18, 20, 25, 27, 30, 32, 34],
              'n_neighbors': [3, 5, 7, 9, 10, 11, 12, 13]
              }

gridsearch = GridSearchCV(knn, param_grid, verbose=3)
gridsearch.fit(x_train, y_train)

gridsearch.best_params_

knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
knn.fit(x_train, y_train)

# Lets check our training accuracy
print("The training accuracy after hyperparameter tuning is : ", knn.score(x_train, y_train))

y_pred = knn.predict(x_test)
print("The testing accuracy after hyperparameter tuning is : ", accuracy_score(y_test, y_pred))

kfold = KFold(n_splits=12, random_state=666, shuffle=True)
kfold.get_n_splits(nba)

from statistics import mean

knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
cnt = 0
count = []
train_score = []
test_score = []

principal_scale = principalDF.to_numpy()

for train_index, test_index in kfold.split(principal_scale):
    X_train, X_test = principal_scale[train_index], principal_scale[
        test_index]  # our scaled data is an array so it can work on x[value]
    y_train, y_test = y.iloc[train_index], y.iloc[
        test_index]  # y is a dataframe so we have to use "iloc" to retreive data
    knn.fit(X_train, y_train)
    train_score_ = knn.score(X_train, y_train)
    test_score_ = knn.score(X_test, y_test)
    cnt = np.array()
    cnt += 1
    count.append(cnt)
    train_score.append(train_score_)
    test_score.append(test_score_)

    print("for k = ", cnt)
    print("train_score is :  ", train_score_, "and test score is :  ", test_score_)
print("************************************************")
print("************************************************")
print("Average train score is :  ", mean(train_score))
print("Average test score is :  ", mean(test_score))
