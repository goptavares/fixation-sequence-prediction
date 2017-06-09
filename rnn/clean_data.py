#!/usr/bin/python

"""
clean_data.py
Author: Gabriela Tavares, gtavares@caltech.edu
"""

import numpy as np
import pandas as pd


def convert_item_values(value):
    return np.absolute((np.absolute(value) - 15) / 5)


timeStepSize = 50

df1 = pd.DataFrame.from_csv("expdata.csv", header=0, sep=",", index_col=None)
df2 = pd.DataFrame.from_csv("fixations.csv", header=0, sep=",", index_col=None)

output1 = pd.DataFrame()
output2 = pd.DataFrame()

dataCount = 0

subjectIds = df1.parcode.unique()

for subjectId in subjectIds:
    dataSubject = np.array(
        df1.loc[df1["parcode"]==subjectId,
        ["trial", "rt", "choice", "item_left", "item_right"]])
    fixationsSubject = np.array(
        df2.loc[df2["parcode"]==subjectId,
        ["trial", "fix_item", "fix_time"]])

    trialIds = np.unique(dataSubject[:,0]).tolist()
    for trialId in trialIds:
        dataTrial = np.array(
            df1.loc[(df1["trial"]==trialId) & (df1["parcode"]==subjectId),
            ["rt", "choice", "item_left", "item_right"]])
        fixationsTrial = np.array(
            df2.loc[(df2["trial"]==trialId) & (df2["parcode"]==subjectId),
            ["fix_item", "fix_time"]])

        valueLeft = convert_item_values(dataTrial[0,2])
        valueRight = convert_item_values(dataTrial[0,3])

        if dataTrial[0,1] == -1:
            choice = 1
        elif dataTrial[0,1] == 1:
            choice = 2

        fixCount = 1
        for fixation in fixationsTrial:
            if fixation[0] != 1 and fixation[0] != 2:
                fixLocation = 0
            else:
                fixLocation = fixation[0]

            fixDuration = fixation[1]

            output2 = output2.append(
                {"data_point": dataCount, "value_left": valueLeft,
                 "value_right": valueRight, "choice": choice,
                 "rt": dataTrial[0,0], "fix_type": fixCount,
                 "location": fixLocation, "duration": fixDuration},
                ignore_index=True)

            numTimeSteps = int(fixDuration / timeStepSize) + 1
            for timeStep in xrange(numTimeSteps):
                output1 = output1.append(
                {"data_point": dataCount, "value_left": valueLeft,
                 "value_right": valueRight, "choice": choice,
                 "rt": dataTrial[0,0], "location": fixLocation},
                ignore_index=True)

            fixCount += 1

        dataCount += 1

output2.to_csv(
    "model_2_data.csv", sep=",", index=False, float_format="%d",
    columns=["data_point", "value_left", "value_right", "choice", "rt",
             "fix_type", "location", "duration"])

output1.to_csv(
    "model_1_data.csv", sep=",", index=False, float_format="%d",
    columns=["data_point", "value_left", "value_right", "choice", "rt",
             "location"])
