#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:59:56 2021

@author: postd
"""
import pandas as pd
import numpy as np

# Need to figure out where the original shift happened and shift
# the rest of the values in the rows
train = pd.read_csv("~/Downloads/PUBHEALTH/train.tsv",sep="\t")
train = train.reset_index()
tr_idx = []
for i in range(len(train)):
    try:
        train.loc[i,"claim_id"] = int(train.loc[i,"claim_id"])
    except:
        tr_idx.append(i)
tr_idx

dev = pd.read_csv("~/Downloads/PUBHEALTH/dev.tsv",sep="\t")
dev = dev.reset_index()

test = pd.read_csv("~/Downloads/PUBHEALTH/test.tsv",sep="\t")
test = test.iloc[:,1:]
test = test.reset_index()
test.loc[:,"claim_id"] = test.loc[:,"claim_id"].astype("float")




