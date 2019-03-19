import sys, os
import pandas as pd

for i in range (1,len(sys.argv),2):
            if sys.argv[i] == "-df":
                DF1 = sys.argv[i+1]
            if sys.argv[i] == "-col_list":
                COLLIST = sys.argv[i+1]
            if sys.argv[i] == "-save":
                SAVE = sys.argv[i+1]
                
df1 = pd.read_csv(DF1, sep='\t', index_col = 0)
col_list = pd.read_csv(COLLIST, sep='\t')

col_list1= list(col_list.iloc[:,0])
x= len(col_list1)
print(col_list1, x)

df= df1.filter(col_list1, axis=1)
print(df)
df.to_csv(path_or_buf=str(DF1)+str(SAVE)+"_redRF_200.txt", sep="\t", header=True)