#!/usr/bin/python

"""
clean_data.py
Author: Gabriela Tavares, gtavares@caltech.edu
"""

import numpy as np
import pandas as pd


def clean_data():

    df = pd.DataFrame.from_csv("juri_data.csv", header=0, sep=",",
                               index_col=None)

    output = pd.DataFrame()

    dataCount = 0

    subjectIds = df.parcode.unique()

    for subjectId in subjectIds:
        dataSubject = np.array(
            df.loc[df["parcode"]==subjectId,
            ["trial", "location", "category", "novelty"]])

        trialIds = np.unique(dataSubject[:,0]).tolist()
        for trialId in trialIds:
            dataTrial = np.array(
                df.loc[(df["trial"]==trialId) & (df["parcode"]==subjectId),
                ["location", "category", "novelty"]])

            for sample in dataTrial:
                output = output.append(
                        {"data_point": dataCount, "location": sample[0]},
                        ignore_index=True)

            dataCount += 1

    output.to_csv(
        "juri_data_clean.csv", sep=",", index=False, float_format="%d",
        columns=["data_point", "location"])


def split_data():
    df = pd.DataFrame.from_csv("juri_data_clean.csv", header=0, sep=",",
                               index_col=None)

    output_train = pd.DataFrame()
    output_test = pd.DataFrame()

    dataPoints = df.data_point.unique()
    endTrain = (len(dataPoints) // 3) * 2

    for count, dataPoint in enumerate(dataPoints[0:endTrain]):
        currData = np.array(
            df.loc[df["data_point"]==dataPoint, ["location"]])

        for sample in currData:
            output_train = output_train.append(
                        {"data_point": count, "location": sample[0]},
                        ignore_index=True)

    for count, dataPoint in enumerate(dataPoints[endTrain:]):
        currData = np.array(
            df.loc[df["data_point"]==dataPoint, ["location"]])

        for sample in currData:
            output_test = output_test.append(
                        {"data_point": count, "location": sample[0]},
                        ignore_index=True)

    output_train.to_csv(
        "juri_train.csv", sep=",", index=False, float_format="%d",
        columns=["data_point", "location"])
    output_test.to_csv(
        "juri_test.csv", sep=",", index=False, float_format="%d",
        columns=["data_point", "location"])


print("Cleaning data...")
clean_data()
print("Splitting data...")
split_data()

