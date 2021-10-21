import matplotlib.pyplot as plt
import seaborn


# add match winner to each line
print(pbp.head())



# split the file based on break point or not



bp1 = pbp['P1BreakPoint'] == 1
dfbp1 = pbp[bp1]
bp2 = pbp['P1BreakPoint'] == 2
dfbp2 = pbp[bp2]
dfnp1 = pbp[~bp1]
dfnp2 = pbp[~bp2]

print(dfbp1.describe())
print(dfnp1.describe())

