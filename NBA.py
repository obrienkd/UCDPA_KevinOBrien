import pandas as pd
import matplotlib.pyplot as plt
nba = pd.read_csv(r'c:\users\hp\nba_draft_combine_all_years.csv')
print(nba.head())


#predict if nba player is drafted or not based on values from the draft competition

# drop draft pick column & height-no shoes
print(nba.columns)
    nba = nba.drop(columns = 'Draft pick', 'Height (No Shoes)')

#class imbalance
print(nba['drafted or no'].value_counts())

# over sample / SMOTE / under

# Any n/a's?
print(nba.isna().sum())


# plot drafted & non drafted players

plt.interactive(False)
# plot histograms of all columns:
nba.hist()
plt.show()

# box plots -
import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x="drafted or no", y="Body Fat", data=nba)
print(ax)
plt.show()

# convert all inches to cm
inch_cols = ['Height (No Shoes)', 'Height (With Shoes)', 'Wingspan', 'Standing reach', 'Vertical (Max)',
       'Vertical (Max Reach)', 'Vertical (No Step)',
       'Vertical (No Step Reach)']

for n in inch_cols:
    nba[n] = nba[n] * 2.54

# convert weight in lbs to kilos
nba['Weight'] = nba['Weight'] * 0.453592
print(nba['Weight'].describe())

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
    plt.show()


# create new variable - vertical no step : height... e.g. what is the height someone can jump in relation to their height
print(nba['Vertical (Max)'].describe())
nba['Vertical_to_height'] = nba['Vertical (N']

#drop draft
nba.columns
nba = nba.drop(columns = 'Draft pick')
nba.columns

# figure out if there is much correlation between variables as some are similar
cor_mat = nba.corr()
sns.heatmap(cor_mat)
plt.show()

large correlation between ht variables

# transform variables due to outliers
for m in numeric_figs:
    nba["{}_zscore".format(m)] = (nba[m] - nba[m].mean())/nba[m].std(ddof=0)
nba.columns

nba['drafted or no'].value_counts()
# 384 drafted, 133 not drafted


#convert to 'drafted or no' to boolean


# PCA - a lot of correlation between variables

#plot
#


#kNN as can handle missing variables







