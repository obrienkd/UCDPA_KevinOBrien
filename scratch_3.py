# install required libraries

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import regex as re


pbp = pd.read_csv(r'G:\My Drive\College G Drive\Tennis_ML\tennis_slam_pointbypoint-master\tennis_slam_pointbypoint-master\pbp_all\point_by_point_all.csv')

print(pbp.head())
print(pbp.dtypes)

pbp = pbp.drop(columns=['Speed_MPH', 'History', 'Rally', 'P1Momentum', 'P2Momentum','P2FirstSrvIn',
                      'P1SecondSrvIn', 'P2SecondSrvIn', 'P2SecondSrvWon', 'P1ForcedError', 'P2ForcedError',
                      'P1FirstSrvIn', 'P1FirstSrvWon', 'P2FirstSrvWon', 'P1SecondSrvWon', 'Serve_Direction',
                      'Winner_FH', 'Winner_BH', 'ServingTo', 'P1TurningPoint', 'P2TurningPoint'])

print(pbp['PointNumber'].unique())

pbp['PointNumber'] = re.sub("[^0-9]","", pbp['PointNumber'])

print(pbp[#{}])