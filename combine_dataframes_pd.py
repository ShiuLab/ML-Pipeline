#This script joins 2 data frames together. -df1= first dataframe; -df2 second dataframe; 
#-type {i= inner/intersection, o= outer/union, df1= exact join from first dataframe
#for type inner (i), assumes 2 data frames with class in column 1 for each. for o OR df1, assumes class column1 for df1, but not for df2
import sys, os
import pandas as pd

DF3 = "none"
for i in range (1,len(sys.argv),2):
            if sys.argv[i] == "-df1":
                DF1 = sys.argv[i+1]
            if sys.argv[i] == "-df2":
                DF2 = sys.argv[i+1]
            if sys.argv[i] == "-df3":
                DF3 = sys.argv[i+1]
            if sys.argv[i] == "-type":
                TYPE = str(sys.argv[i+1])
df1 = pd.read_csv(DF1, sep='\t', index_col = 0)
df2 = pd.read_csv(DF2, sep='\t', index_col = 0)

col_list1= list(df1.columns.values)
col_list2= list(df2.columns.values)
x= len(col_list1)
y= len(col_list2)
df1x= df1[df1.columns[0]]
df1x = df1x.dropna(axis=0)
df1y = df1[df1.columns[1:x]]
df1= pd.concat([df1x, df1y], axis=1, join='inner') #join intersection after getting rid of NAs for class

if DF3 != "none":
    df3 = pd.read_csv(DF3, sep='\t', index_col = 0)
    col_list3= list(df3.columns.values)
    z= len(col_list3)
    if TYPE == "i":
        df2x = df2[df2.columns[1:y]]
        df3x = df3[df3.columns[1:z]]   
        df= pd.concat([df1, df2x, df3x], axis=1, join='inner') #join intersection    
        #added for combining binary and continuous data but same classes
        df1name = str(DF1).split(".")[0]
        df.to_csv(path_or_buf=str(df1name)+".combined_matrix4.0_NAimputed.txt", sep="\t", header=True)
    elif TYPE == "o":
        result = pd.concat([df1, df2, df3], axis=1, join='outer') # join union
        #for traditional combining of both names, use the following
        result.to_csv(path_or_buf=str(DF1)+"_"+ str(DF2)+ str(DF3)+".txt", sep="\t", header=True)
    elif TYPE == "df1":
        result = pd.concat([df1, df2, df3], axis=1, join_axes=[df1.index]) # exact join from first dataframe
        result.to_csv(path_or_buf=str(DF1)+"_"+ str(DF2)+ str(DF3)+".txt", sep="\t", header=True)
    else:
        print ("Need TYPE {i= inner/intersection, o= outer/union, df1= exact join from first dataframe")

else:
    if TYPE == "i":
        df2x = df2[df2.columns[1:y]]    
        df= pd.concat([df1, df2x], axis=1, join='inner') #join intersection    
        #added for combining binary and continuous data but same classes
        df1name = str(DF1).split(".")[0]
        df.to_csv(path_or_buf=str(df1name)+".combined_matrix4.0_NAimputed.txt", sep="\t", header=True)
    elif TYPE == "o":
        result = pd.concat([df1, df2], axis=1, join='outer') # join union
        #for traditional combining of both names, use the following
        result.to_csv(path_or_buf=str(DF1)+"_"+ str(DF2)+".txt", sep="\t", header=True)
    elif TYPE == "df1":
        result = pd.concat([df1, df2], axis=1, join_axes=[df1.index]) # exact join from first dataframe
        result.to_csv(path_or_buf=str(DF1)+"_"+ str(DF2)+".txt", sep="\t", header=True)
    else:
        print ("Need TYPE {i= inner/intersection, o= outer/union, df1= exact join from first dataframe")