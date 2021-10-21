# install required libraries

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import regex as re


pd.options.display.min_rows = 999
pd.options.display.max_columns = 999

# i need to merge a list of matches (both male & female) and create the one df for them
# create a new csv file to save all them together

path = r'G:\My Drive\College G Drive\Tennis_ML\tennis_slam_pointbypoint-master\tennis_slam_pointbypoint-master\list_of_matches'
all_files = glob.glob(path + "/*.csv")
df_files = (pd.read_csv(f) for f in all_files)
df_list_of_matches   = pd.concat(df_files, ignore_index=True)
df_list_of_matches.to_csv(path + '/all_matches.csv', index = False)


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

# remove the x from 0X a time where a point was repeated e.g. Hawkeye
# don't think this worked....

point_num = pd.DataFrame()
point_num = df['PointNumber'].astype(str)
pn_out = ''.join(c for c in point_num if c.isnumeric())

# remove columns that have NaN values
df = df.drop(columns=['Speed_MPH', 'History', 'Rally', 'P1Momentum', 'P2Momentum','P2FirstSrvIn',
                      'P1SecondSrvIn', 'P2SecondSrvIn', 'P2SecondSrvWon', 'P1ForcedError', 'P2ForcedError',
                      'P1FirstSrvIn', 'P1FirstSrvWon', 'P2FirstSrvWon', 'P1SecondSrvWon', 'Serve_Direction',
                      'Winner_FH', 'Winner_BH', 'ServingTo', 'P1TurningPoint', 'P2TurningPoint'])

# drop columns with multiple n/a
df_list_of_matches = df_list_of_matches.drop(columns=['status', 'event_name', 'round', 'court_name', 'court_id', 'nation1', 'nation2'])

df_merged = pd.merge(df, df_list_of_matches, on='match_id')
print(df_merged.head())


df_a = df_merged
print(head.df_a())

# adding surface to each grandslam match

def map_values(row, values_dict):
    return values_dict[row]

values_dict = {'usopen': 'hard', 'wimbledon': 'grass', 'frenchopen': 'clay', 'ausopen': 'hard'}
df_a['surface'] = df_a['slam'].apply(map_values, args = (values_dict,))

print(df_a.head())

df_a.to_csv(r'G:\My Drive\College G Drive\Tennis_ML\df_merged.csv')
# df not defined?? print(type(df['PointNumber']))
# working?? re.sub('\D', '',  df['PointNumber'])

# notworking pattern = r'[0-9]'
# not
# working df['PointNumber'] = re.sub(pattern, '', df['PointNumber'])

playerid = pd.read_csv(r'G:\My Drive\College G Drive\Tennis_ML\player_list.csv')


# function to perform a lookup across two files & identify the player's id

def xlookup(lookup_value, lookup_array, return_array, if_not_found: str = ''):
    match_value = return_array.loc[lookup_array == lookup_value]
    if match_value.empty:
        return f'"{lookup_value}" not found!' if if_not_found == '' else if_not_found

    else:
        return match_value.tolist()[0]

# check it works
print(xlookup('Novak Djokovic', playerid['full_name'], playerid['atp_id'])

# need to add in wta

print(df.head())

# next

# apply across all player 1 and player 2 columns

df_a['player1ID'] = df_a['player1'].apply(xlookup, args=(playerid['full_name'], playerid['atp_id']))
print(df_a.head())
