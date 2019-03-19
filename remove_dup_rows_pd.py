'''remove duplicate lines from data frame

input= matrix with gene:data
any rows that are the same will be removed'''

import sys
import pandas as pd

DF1= sys.argv[1]
df1 = pd.read_csv(DF1, sep='\t', index_col = 0)
#df= df1.drop_duplicates() #drop entire duplicate rows

dfind = df1.index.drop_duplicates() #indices that are still duplicated
print(dfind) 

df= df1[~df1.index.duplicated(keep='first')] #keep only the first index
print(df)

df.to_csv(path_or_buf=str(DF1)+"_nodups.txt", sep="\t", header=True)