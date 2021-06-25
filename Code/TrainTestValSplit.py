# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:34:15 2021

@author: postd
"""

import pandas as pd
import random

def split(path):
    tr = pd.read_csv(path+"claimLabel_train.csv")
    val =  pd.read_csv(path+"claimLabel_dev.csv")
    te =  pd.read_csv(path+"claimLabel_test.csv")
    
    
    full = pd.concat([tr,val,te]).reset_index(drop=True)
    
    idx_list = [x for x in full.index]
    
    random.seed(183)
    random.shuffle(idx_list)
    
    split_1 = int(0.8 * len(idx_list))
    split_2 = int(0.9 * len(idx_list))
    
    tr_new = full.loc[idx_list[:split_1],:]
    val_new = full.loc[idx_list[split_1:split_2],:]
    te_new = full.loc[idx_list[split_2:],:]
    
    return tr_new, val_new, te_new

def main():
    data_path = "../Data/RawDataCsvFormat/"
    
    train, dev, test = split(data_path)
    
    train.to_csv(data_path+"train.csv",index=False)
    dev.to_csv(data_path+"dev.csv",index=False)
    test.to_csv(data_path+"test.csv",index=False)
    
if __name__=="__main__":
    main()