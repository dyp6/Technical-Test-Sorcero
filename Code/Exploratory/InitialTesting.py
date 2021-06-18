import pandas as pd
import numpy as np

from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer

path = "/home/postd/Documents/gitRepos/Technical-Test-Sorcero/Data/RawDataCsvFormat/"
train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path + "test.csv",index_col=0)
dev = pd.read_csv(path + "dev.csv")

datasets_list = list_datasets()
data = load_dataset('health_fact')
