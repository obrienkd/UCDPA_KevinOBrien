# install required libraries

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import re


pd.options.display.min_rows = 999
pd.options.display.max_columns = 999

# i need to merge a list of matches (both male & female) and create the one df for them
# create a new csv file to save all them together

path = r'G:\My Drive\College G Drive\Tennis_ML\tennis_slam_pointbypoint-master\tennis_slam_pointbypoint-master\list_of_matches'
all_files = glob.glob(path + "/*.csv")
df_files = (pd.read_csv(f) for f in all_files)
df_list_of_matches   = pd.concat(df_files, ignore_index=True)
df_list_of_matches.to_csv(path + '/all_matches.csv', index = False)

# check the file
print(df_list_of_matches)

# Similar approach to each csv file for all the point by point data for each grandslam
# create a new csv file from the compiled df
path = r'G:\My Drive\College G Drive\Tennis_ML\tennis_slam_pointbypoint-master\tennis_slam_pointbypoint-master\pbp_all'
all_files = glob.glob(path + "/*.csv")
df_files = (pd.read_csv(f) for f in all_files)
df_point_by_point   = pd.concat(df_files, ignore_index=True)
df_point_by_point.to_csv(path + '/point_by_point_all.csv', index = False)

# rename df_point_by_point to df as it will be the primary df for the next bit of cleaning
df = df_point_by_point

# in tennis the scoring system at times as letters in it. e.g. AD - replace this with 50 to keep the column as INT

df['P1Score'] = df['P1Score'].replace('AD', '50')
df['P2Score'] = df['P2Score'].replace('AD', '50')
df['P1Score'].astype(int)
df['P2Score'].astype(int)




