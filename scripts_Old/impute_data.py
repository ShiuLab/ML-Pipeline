"""
PURPOSE: impute missing data
Must set path to Miniconda in HPC:  export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH or have scipy and sklearn installed

INPUT:
    -df       ML Dataframe with gene|Class|feature data
    -dtype1   Data type in matrix: n=numeric,c=categorical,b=binary
    -mv       type of missing value calculation: 0 = drop all NAs, 1 = get feature distribution then choose random value from
              this distribution, 2 = value for numeric is the median, value for binary or categorical is the mode

Optional INPUT:
    -df2      A second dataframe in same format as first
    -dtype2   Data type of second matrix
    -df3      A third data frame in same format as first
    -dtype3   Data type of third matrix
    -drop     percent NA data to drop, default is 50%, enter in decimal format (ie. 50% = 0.5)
    
OUTPUT:
      -df.NAimputed.txt    A new dataframe with values imputed     

"""

#DF= dataframe, dtype= datatype(n=numeric,c=categorical,b=binary) mv= type of missing value calculation
import sys, os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

def main():
    
    # Default code parameters
    mv, DF2, dtype2, DF3, dtype3, drop = 0, 'NA', 'NA', 'NA', 'NA', 0.5
	
    for i in range (1,len(sys.argv),2):
            if sys.argv[i] == "-df":
                DF1 = sys.argv[i+1]
            if sys.argv[i] == "-dtype1":
                dtype1 = sys.argv[i+1]
            if sys.argv[i] == "-mv":
                mv = int(sys.argv[i+1])
            if sys.argv[i] == "-df2":
                DF2 = sys.argv[i+1]
            if sys.argv[i] == "-dtype2":
                dtype2 = sys.argv[i+1]
            if sys.argv[i] == "-df3":
                DF3 = sys.argv[i+1]
            if sys.argv[i] == "-dtype3":
                dtype3 = sys.argv[i+1]
            if sys.argv[i] == "-drop":
                drop = sys.argv[i+1]

    if len(sys.argv) <= 1:
	    print(__doc__)
	    exit()
	    
    df = pd.read_csv(DF1, sep='\t', index_col = 0)
    if DF2 == 'NA':
        df2 = 'NA'
    else:
        df2 = pd.read_csv(DF2, sep='\t', index_col = 0)
    if DF3 == 'NA':
        df3 = 'NA'
    else:
        df3 = pd.read_csv(DF3, sep='\t', index_col = 0)
    #missing values
    # turn all NAs or ? to NaN (NA recognized by numpy)
    def replaceNAs (df):
        df = df.replace("?",np.nan)
        df = df.replace("NA",np.nan)
        df = df.replace("",np.nan)
        df = df.replace("NaN",np.nan)
        return (df)
	
    # find the percent of missing feature, if greater than 50%, then drop!
    def drop_missing50 (df, drop):
        col_length=len(df.columns)-3
        print (col_length)
        drop_list = []
        for i in range(1,col_length):
            #print (i)
            missing=  df.iloc[:,[i]].isnull().sum()
            miss_pct = missing/len(df)
            if  miss_pct.iloc[0] > float(drop):
                print (missing, miss_pct, df.columns[i])
                print ("dropping",df.columns[i])
                drop_list.append(df.columns[i])
        for i in range(1,len(drop_list)):
            df = df.drop(drop_list[i], 1)

	
        return (df)
        
    #print(df0)
    def get_percent(counts):#function to get proportion
        L=[]
        z = len(counts)
        print(z)
        for j in range(0,z):
            if float(counts.sum()) == 0.0:
                L.append(float(0))
            else:
                p =counts[j] / float(counts.sum())
                L.append(p)
        return (L)
        
    def get_percent1(counts):#function to get proportion
        L=[]
        z = len(counts)+1
        print(z)
        for j in range(1,z):
            if float(counts.sum()) == 0.0:
                L.append(float(0))
            else:
                p =counts[j] / float(counts.sum())
               
                L.append(p)
        return (L)
        
    def get_cat1(df,y):
        df1 = df[df.columns[1:y]]
        col_list2= list(df1.columns.values)
        for i in range(0,len(df1.columns)):
            x= col_list2[i] #column name
            x1 = df1.iloc[:,[i]].dropna(axis=0) #get values of column name, drop NAs
            cats= x1[x].unique() #get unique categories 
            
            counts= x1[x].value_counts() #get counts of each category
            
            #print (counts)
            if 0 in cats:
                pcts = get_percent(counts) #get proportion of each category
            else:
                pcts= get_percent1(counts)
                
            #print (pcts)
            df1.loc[:,[x]] = df.loc[:,[x]].fillna(value=np.random.choice(cats, p=pcts),axis=1)
            #randomly choose from unique categories\
            #based on their proportion in the dataset, replace NAs with this category
    
        return df1
        
    def get_num1(df, y):
        df2= df[df.columns[1:y]]#get numeric
        
        col_list= list(df2.columns.values)
        for i in range(0,len(df2.columns)):
             x= col_list[i]
             x1 = df2.iloc[:,[i]].dropna(axis=0)
             if x1.empty == True:
                 pass
             else:
                 df2.loc[:,[x]] = df2.loc[:,[x]].fillna(value=np.random.choice(x1[x]), axis=1)#replace NAs with random choice from actual distibution
        return df2
    
    def get_cat2(df, y):
        dfx = df[df.columns[1:y]]
        
        col_list= list(dfx.columns.values)
        for i in range(0,len(dfx.columns)):
            x= col_list[i] #column name
            x1 = dfx.iloc[:,[i]].dropna(axis=0)
            m= stats.mode(x1) #get mode without NAs
            p = m[0].tolist()
            p= p[0]
            p= p[0]
            p = ''.join(str(p))
            dfx.loc[:,[x]] = dfx.loc[:,[x]].fillna(value=p,axis=1) #replace NAs with mode
        return dfx
        
    def get_num2(df, y):
        dfx = df[df.columns[1:y]]#get numeric
        col_list= list(dfx.columns.values)
        for i in range(0,len(dfx.columns)):
            x= col_list[i]
            x1 = dfx.iloc[:,[i]].dropna(axis=0)
            med= np.median(x1)
            dfx.loc[:,[x]] = df.loc[:,[x]].fillna(value=med) #replace NAs with median
        return (dfx)
    
    def convert_cat2bin(df):
        dfx = df[df.columns[1:y]]
        dfx= OneHotEncoder.fit_transform(dfx).toarray()
        print (dfx)
        return (dfx)
        
    df0 = df.iloc[:,[0]]
    df=replaceNAs(df)
    
    if mv == 0: #choice to drop NAs
        if DF2 and DF3 == 'NA':
            df= df.dropna(axis=0)
        elif DF2 != 'NA' and DF3 == 'NA':
            df2=replaceNAs(df2)
            df2=drop_missing50(df2, drop)
            frames  = [df, df2]
            df= pd.concat(frames, axis=1)
            df= df.dropna(axis=0)
        else:
            df2=replaceNAs(df2)
            df2=drop_missing50(df2, drop)
            df3=replaceNAs(df3)
            df3=drop_missing50(df3, drop)
            frames  = [df, df2, df3]
            df= pd.concat(frames, axis=1)
            df= df.dropna(axis=0)
    
    elif mv == 1: #choice 1, impute missing values with random choice from a distribution
        df=drop_missing50(df, drop) #drop column if over 50% is missing
        col_list= list(df.columns.values)
        y= len(col_list)
        print("Replacing NA's for ", y, " columns")
        if dtype1 == 'c':
            df1= convert_cat2bin(df, y)
            df1= get_cat1(df, y)
        elif dtype1 == 'b':
            df1= get_cat1(df, y)
        elif dtype1 == 'n':
            df1= get_num1(df, y)
        else:
            print ("need -dtype : n=numeric, c=categorical, b=binary")

        frames  = [df0, df1] #put all frames back together ##need to add back class
        df= pd.concat(frames, axis=1)
    
    elif mv == 2: #this option imputes median or mode for either numeric or categorical data respectively
        df=drop_missing50(df, drop) #drop column if over 50% is missing
        col_list= list(df.columns.values)
        y= len(col_list)
        print("Replacing NA's for ", y, " columns")
        if dtype1 == 'c':
            df1= convert_cat2bin(df)
            df1= get_cat2(df, y)
        elif dtype1 == 'b':
            df1= get_cat2(df, y)
        elif dtype1 == 'n':
            df1= get_num2(df, y)
        else:
            print ("need -dtype : n=numeric, c=categorical, b=binary")
	            
        frames  = [df0, df1] #put all frames back together
        df= pd.concat(frames, axis=1)
        
    if DF2 == 'NA':
        pass
    else:
        df0 = df2.iloc[:,[0]]
        df2=replaceNAs(df2)
        df2=drop_missing50(df2, drop)
        if mv == 1: #choice 1, impute missing values with random choice from a distribution
            col_list= list(df2.columns.values)
            y= len(col_list)
            print("Replacing NA's for ", y, " columns")
            if dtype2 == 'c':
                df2= get_cat1(df2, y)
            elif dtype2 == 'b':
                df2= get_cat1(df2, y)
            elif dtype2 == 'n':
                df2= get_num1(df2, y)
            else:
                print ("need -dtype2 : n=numeric, c=categorical, b=binary")
    
        elif mv == 2: #this option imputes median or mode for either numeric or categorical data respectively
            col_list= list(df.columns.values)
            y= len(col_list)
            print("Replacing NA's for ", y, " columns")
            if dtype2 == 'c':
                df2= get_cat2(df2, y)
            elif dtype2 == 'b':
                df2= get_cat2(df2, y)
            elif dtype2 == 'n':
                df2= get_num2(df2, y)
            else:
                print ("need -dtype2 : n=numeric, c=categorical, b=binary")
	            

    if DF3 == 'NA':
        pass
    else:
        df0 = df3.iloc[:,[0]]
        df3=replaceNAs(df3)
        df3=drop_missing50(df3, drop)
        if mv == 1: #choice 1, impute missing values with random choice from a distribution
            col_list= list(df3.columns.values)
            #print(col_list)
            y= len(col_list)
            print("Replacing NA's for ", y, " columns")
            if dtype3 == 'c':
                df3= get_cat1(df3, y)
            elif dtype3 == 'b':
                df3= get_cat1(df3, y)
            elif dtype3 == 'n':
                df3= get_num1(df3, y)
            else:
                print ("need -dtype2 : n=numeric, c=categorical, b=binary")
    
        elif mv == 2: #this option imputes median or mode for either numeric or categorical data respectively
            col_list= list(df3.columns.values)
            y= len(col_list)
            print("Replacing NA's for ", y, " columns")
            if dtype3 == 'c':
                df3= get_cat2(df3, y)
            elif dtype3 == 'b':
                df3= get_cat2(df3, y)
            elif dtype3 == 'n':
                df3= get_num2(df3, y)
            else:
                print ("need -dtype2 : n=numeric, c=categorical, b=binary")
	            

    if mv == 0: 
        df.to_csv(path_or_buf=str(DF1)+".NAimputed.txt", sep="\t", header=True)
    elif DF2 and DF3 == 'NA':
        if mv != 0:
            df.to_csv(path_or_buf=str(DF1)+".NAimputed.txt", sep="\t", header=True)
    elif DF2 != 'NA' and DF3 == 'NA':
        print ("concatenating dataframes 1 and 2")
        df= pd.concat([df, df2], axis=1)
        df.to_csv(path_or_buf=str(DF1)+".NAimputed.txt", sep="\t", header=True)
    else:
        print ("concatenating dataframes 1, 2, and 3")
        df= pd.concat([df, df2, df3], axis=1)
        df.to_csv(path_or_buf=str(DF1)+".NAimputed.txt", sep="\t", header=True)
    
if __name__ == '__main__':
	main()