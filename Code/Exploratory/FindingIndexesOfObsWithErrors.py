#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:59:56 2021

@author: postd
"""
import pandas as pd
import numpy as np
import os

path =  "/home/postd/Documents/gitRepos/Technical-Test-Sorcero/"
os.chdir(path)
# Need to figure out where the original shift happened and shift
# the rest of the values in the rows
# Going to need the possible labels to find rows with their labels missing
labels = ["true","false","mixture","unproven"]

train = pd.read_csv("Data/RawData/train.tsv",sep="\t")

tr_idx = []
for i in range(len(train)):
    if pd.isna(train.loc[i,"label"]):
        tr_idx.append(i)
    else:
        try:
            train.loc[i,"claim_id"] = int(train.loc[i,"claim_id"])
        except:
            tr_idx.append(i)
tr_err = train.loc[tr_idx,:]
# Dev Set
dev = pd.read_csv("Data/RawData/dev.tsv",sep="\t")

dev_idx = []
for i in range(len(dev)):
    if pd.isna(dev.loc[i,"label"]):
        dev_idx.append(i)
    else:
        try:
            dev.loc[i,"claim_id"] = int(dev.loc[i,"claim_id"])
        except:
            dev_idx.append(i)
dev_err = dev.loc[dev_idx,:]
# Test Set
test = pd.read_csv("Data/RawData/test.tsv",sep="\t")
test = test.iloc[:,1:]
test = test.reset_index()
test.loc[:,"claim_id"] = test.loc[:,"claim_id"].astype("float")

test_idx = []
for i in range(len(test)):
    if pd.isna(test.loc[i,"label"]):
        test_idx.append(i)
    else:
        try:
            test.loc[i,"claim_id"] = int(test.loc[i,"claim_id"])
        except:
            test_idx.append(i)
test_err = test.loc[test_idx,:]

# All test set rows are okay, so no changes needed for it
print(tr_idx)
print(dev_idx)
print(test_idx)