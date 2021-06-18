#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 21:31:00 2021

@author: postd
"""

import pandas as pd
import numpy as np

path = "/home/postd/Documents/gitRepos/Technical-Test-Sorcero/Data/RawDataCsvFormat/"
train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path + "test.csv",index_col=0)
dev = pd.read_csv(path + "dev.csv")

test.loc[:,"label"].unique()
dev.loc[:,"label"].unique()
train.loc[:,"label"].unique()
