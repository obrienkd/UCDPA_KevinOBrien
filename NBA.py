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


# read csv file in
nba = pd.read_csv(r'c:\users\hp\nba_draft_combine_all_years.csv')

#predict if nba player is drafted or not based on values from the draft competition

# drop draft pick column & height-no shoes

#class imbalance
print(nba['drafted or no'].value_counts())

# over sample / SMOTE / under
nba.isna().sum()

# Any n/a's?
print(nba.isna().sum())

# median values assigned to 67 missing verticals, weight, body fat and 233 bench press values

nba['Vertical (Max)'] = nba['Vertical (Max)'].replace(np.nan,nba['Vertical (Max)'].median())
nba['Vertical (Max Reach)'] = nba['Vertical (Max Reach)'].replace(np.nan,nba['Vertical (Max Reach)'].median())
nba['Vertical (No Step Reach)'] = nba['Vertical (No Step Reach)'].replace(np.nan,nba['Vertical (No Step Reach)'].median())
nba['Vertical (No Step)'] = nba['Vertical (No Step)'].replace(np.nan,nba['Vertical (No Step)'].median())
nba['Weight'] = nba['Weight'].replace(np.nan,nba['Weight'].median())
nba['Body Fat'] = nba['Body Fat'].replace(np.nan,nba['Body Fat'].median())
nba['Agility'] = nba['Agility'].replace(np.nan,nba['Agility'].median())
nba['Sprint'] = nba['Sprint'].replace(np.nan,nba['Sprint'].median())
nba['Bench'] = nba['Bench'].replace(np.nan,nba['Bench'].median())
# only 1 value missing height with shoes


print(nba.isna().sum())

# box plots -

sns.set_theme(style="whitegrid")

# convert all inches to cm
inch_cols = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)',
       'Vertical (Max Reach)', 'Vertical (No Step)',
       'Vertical (No Step Reach)']

for x in inch_cols:
    nba[x] = nba[x]*2.54

# convert weight in lbs to kilos
nba['Weight'] = nba['Weight'] * 0.453592


# assess the distribution of the variables in the columns
numeric_figs = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)',
       'Vertical (Max Reach)', 'Vertical (No Step)',
       'Vertical (No Step Reach)', 'Weight', 'Body Fat', 'Bench', 'Agility',
       'Sprint']

for m in numeric_figs:
    fig, ax =plt.subplots(1,2, figsize=(15,8))
    fig.suptitle('Distribution of: {}'.format(m), fontsize=16)
    sns.histplot(x=nba[m], ax=ax[0])
    sns.boxplot(y=m, data=nba, ax=ax[1])
    fig.savefig("./figures/distributions/{}.png".format(m))



for n in numeric_figs:
    fig2 = sns.catplot(data=nba, kind="swarm", x="drafted or no", y=n, palette="dark")
    plt.title('Group Differences in: {}'.format(n), fontsize=16)
    fig2.savefig("./figures//group_differences/{}.png".format(n))

#drop draft


# figure out if there is much correlation between variables as some are similar
cor_mat = nba.corr()
g = sns.heatmap(cor_mat,linewidths=1)
plt.show()
plt.savefig("./figures/correlation/correlation.png")
# large correlation between ht variables

# transform variables due to outliers

for col in numeric_figs:
    nba["{}_zscore".format(col)] = (nba[col] - nba[col].mean())/nba[col].std(ddof=0)

# 384 drafted, 133 not drafted

#convert to 'drafted or no' to 1 or 0

nba['draft'] = [1 if x == 'Drafted' else 0 for x in nba['drafted or no']]

# drop drafted or no column

# PCA - a lot of correlation between variables

#plot
#









