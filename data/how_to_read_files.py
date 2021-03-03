import numpy as np
import pandas as pd
import scipy as sc
import uproot as up

#we'll use uproot to read the root trees stored in the files
#this way of working may be clunky, but should be good for now
#note: next time, instead of reading out by tower, we'll read out by run so this should be fine since data will
#be grouper closer time-wise.
#if time-sorting is an issue, I'll pull the data that way instead

#load up all the files using uproot; note that all the root files have a TTree named "tree"
events1 = up.open('path_to_file/ds3564Tower1.root')['tree']
events2 = up.open('path_to_file/ds3564Tower2.root')['tree']
events3 = up.open('path_to_file/ds3564Tower3.root')['tree']
#.
#.
#.
#cast all the data as pandas dataframes
data1 = events1.pandas.df()
data2 = events2.pandas.df()
data3 = events3.pandas.df()
#.
#.
#.

frames = [data1, data2, data3]
#concatenate the frames!
alldata = pd.concat(frames) 
