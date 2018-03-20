##script used to combine multiple files into a matrix
import os, sys
import pandas as pd
import numpy as np


sum_matrix = sys.argv[1]+"_binary.matrix.txt"
# open with pandas
df = pd.read_csv(sys.argv[1], sep='\t', index_col = 0)

df1 = df.dropna(axis=0, how='any')
df1 = pd.get_dummies(df1)
df1.to_csv(path_or_buf=str(sum_matrix), sep="\t", header=True)