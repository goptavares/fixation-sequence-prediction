#!/usr/bin/python

"""
split_data.py
Author: Gabriela Tavares, gtavares@caltech.edu
"""

import numpy as np
import pandas as pd


df1 = pd.DataFrame.from_csv("model_1_data.csv", header=0,
                            sep=",", index_col=None)
df2 = pd.DataFrame.from_csv("model_2_data.csv", header=0,
                            sep=",", index_col=None)

output1_train = pd.DataFrame()
output1_valid = pd.DataFrame()
output1_test = pd.DataFrame()

dataPoints = df1.data_point.unique()
endTrain = len(dataPoints) // 3
endValid = 2 * (len(dataPoints) // 3)

for count, dataPoint in enumerate(dataPoints[0:endTrain]):
    currData = np.array(
        df1.loc[df1["data_point"]==dataPoint,
        ["value_left", "value_right", "choice", "rt", "location"]])

    for d in xrange(currData.shape[0]):
        output1_train = output1_train.append(
                    {"data_point": count, "value_left": currData[d,0],
                     "value_right": currData[d,1], "choice": currData[d,2],
                     "rt": currData[d,3], "location": currData[d,4]},
                    ignore_index=True)

for count, dataPoint in enumerate(dataPoints[endTrain:endValid]):
    currData = np.array(
        df1.loc[df1["data_point"]==dataPoint,
        ["value_left", "value_right", "choice", "rt", "location"]])

    for d in xrange(currData.shape[0]):
        output1_valid = output1_valid.append(
                    {"data_point": count, "value_left": currData[d,0],
                     "value_right": currData[d,1], "choice": currData[d,2],
                     "rt": currData[d,3], "location": currData[d,4]},
                    ignore_index=True)

for count, dataPoint in enumerate(dataPoints[endValid:]):
    currData = np.array(
        df1.loc[df1["data_point"]==dataPoint,
        ["value_left", "value_right", "choice", "rt", "location"]])

    for d in xrange(currData.shape[0]):
        output1_test = output1_test.append(
                    {"data_point": count, "value_left": currData[d,0],
                     "value_right": currData[d,1], "choice": currData[d,2],
                     "rt": currData[d,3], "location": currData[d,4]},
                    ignore_index=True)

output1_train.to_csv(
    "model_1_train.csv", sep=",", index=False, float_format="%d",
    columns=["data_point", "value_left", "value_right", "choice", "rt",
             "location"])
output1_valid.to_csv(
    "model_1_valid.csv", sep=",", index=False, float_format="%d",
    columns=["data_point", "value_left", "value_right", "choice", "rt",
             "location"])
output1_test.to_csv(
    "model_1_test.csv", sep=",", index=False, float_format="%d",
    columns=["data_point", "value_left", "value_right", "choice", "rt",
             "location"])
