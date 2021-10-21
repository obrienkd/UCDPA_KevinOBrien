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

pbp = pbp.drop(columns = 'PointNumber')

print(pbp.dtypes)

print(pbp['ServeNumber'].unique())

pbp['ServeNumber'].dropna()
print(pbp.describe())

# create running totals for the various point values
pbp['p1acecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1Ace'].transform(lambda x: x.cumsum())
pbp['p2acecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2Ace'].transform(lambda x: x.cumsum())
pbp['p1winnercumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1Winner'].transform(lambda x: x.cumsum())
pbp['p2winnercumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2Winner'].transform(lambda x: x.cumsum())
pbp['p1dfcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1DoubleFault'].transform(lambda x: x.cumsum())
pbp['p2dfcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2DoubleFault'].transform(lambda x: x.cumsum())
pbp['p1uecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1UnfErr'].transform(lambda x: x.cumsum())
pbp['p2uecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2UnfErr'].transform(lambda x: x.cumsum())
pbp['p1netpointcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1NetPoint'].transform(lambda x: x.cumsum())
pbp['p2netpointcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2NetPoint'].transform(lambda x: x.cumsum())
pbp['p1netpointwoncumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1NetPointWon'].transform(lambda x: x.cumsum())
pbp['p2netpointwoncumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2NetPointWon'].transform(lambda x: x.cumsum())
pbp['p1bpcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1BreakPoint'].transform(lambda x: x.cumsum())
pbp['p2bpcumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2BreakPoint'].transform(lambda x: x.cumsum())
pbp['p1bpwoncumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1BreakPointWon'].transform(lambda x: x.cumsum())
pbp['p2bpwoncumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2BreakPointWon'].transform(lambda x: x.cumsum())
pbp['p1bplosecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P1BreakPointMissed'].transform(lambda x: x.cumsum())
pbp['p2bplosecumsum'] = pbp.groupby(['match_id', 'ServeNumber'])['P2BreakPointMissed'].transform(lambda x: x.cumsum())

print(pbp)