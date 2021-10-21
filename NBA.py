import pandas as pd
import matplotlib.pyplot as plt
nba = pd.read_csv(r'c:\users\hp\nba_draft_combine_all_years.csv')

print(nba.head())

#predict if nba player is drafted or not based on values from the draft competition

#class imbalance
print(nba.dtypes)

# plot drafted & non drafted players

plt.bar(nba['drafted or not'])

#drop draft pick number, height_with_shoes

#add new variable e.g. wingspan:ht

#add new variable bench x 185 pounds : body weight

#add new variable sprint:agility ratio

#add new variable fat free muscle mass & fat mass

#add new variable vertical max : height

#add new variable vertical max - vertical no step (the impact of the step)

#plot
#


#normalise

#cluster

#log reg





