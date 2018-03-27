"""
PURPOSE:
Pre-process feature-instance matrix for machine learning. 
Options include:
    Imput NAs: Drops features with >50% NAs
               Then impute from a random distribution (-m 1) or 
                    from the median or mode (-m2 -dtype n/c/b)

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
    
    REQUIRED:
    -df       Feature & class dataframe for ML
    -f        Function (imp, 1hot)
    
    OPTIONAL:
    -m        Mode for inputation (1=from random distribution; 2=for median or mode)
    -dtype    Data type: n=numeric, c=categorical, b=binary
     
OUTPUT:
    -df_f     Modified dataframe 

AUTHOR: Beth Moore

REVISIONS:   Add one-hot encoding (CA)
"""


#DF= dataframe, dtype= datatype(n=numeric,c=categorical,b=binary) mv= type of missing value calculation
import sys, os
import pandas as pd
import numpy as np
from scipy import stats
#from statistics import mode

def main():
    
    # Default code parameters
    mv = 0
	
    for i in range (1,len(sys.argv),2):
            if sys.argv[i] == "-df":
                DF = sys.argv[i+1]
            if sys.argv[i] == "-dtype":
                dtype = sys.argv[i+1]
            if sys.argv[i] == "-f":
                f = sys.argv[i+1]
            if sys.argv[i] == "-mv":
                mv = int(sys.argv[i+1])

    if len(sys.argv) <= 1:
	    print(__doc__)
	    exit()
	    
    df = pd.read_csv(DF, sep='\t', index_col = 0)
        
    #missing values
    # turn all NAs or ? to NaN (NA recognized by numpy)
    df = df.replace("?",np.nan)
    df = df.replace("NA",np.nan)
    df = df.replace("",np.nan)
	
    # find the percent of missing feature, if greater than 50%, then drop!
    #col_list1= list(df.columns.values)
    col_length=len(df.columns)-3
    print (col_length)
    for i in range(1,col_length):
        print (i)
        missing=  df.iloc[:,[i]].isnull().sum()
        miss_pct = missing/len(df)
        print (missing, miss_pct, df.columns[i])
        if  miss_pct.iloc[0] > float(0.5):
            df = df.drop(df.columns[i], 1) 
            #print (df.columns[i], 1)
	
    df0 = df.iloc[:,[0]]
    #print(df0)
    def get_percent(counts):#function to get proportion
        L=[]
        z = len(counts)
        print(z)
        for j in range(0,z):
            if float(counts.sum()) == 0.0:
                L.append(float(0))
            else:
                #print(counts[j], counts.sum())
                p =counts[j] / float(counts.sum())
                #print(p)
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
                #print(counts[j], counts.sum())
                p =counts[j] / float(counts.sum())
                #print(p)
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
            #print(float(counts.sum()))
            print (counts)
            if 0 in cats:
                pcts = get_percent(counts) #get proportion of each category
            else:
                pcts= get_percent1(counts)
                
            print (pcts)
            df1.loc[:,[x]] = df.loc[:,[x]].fillna(value=np.random.choice(cats, p=pcts),axis=1)
            #randomly choose from unique categories\
            #based on their proportion in the dataset, replace NAs with this category
            #print (df1.loc[:,[x]])
        return df1
        
    def get_num1(df, y):
        df2= df[df.columns[1:y]]#get numeric
        #print(df2)
        col_list= list(df2.columns.values)
        for i in range(1,len(df2.columns)):
             x= col_list[i]
             x1 = df2.iloc[:,[i]].dropna(axis=0)
             df2.loc[:,[x]] = df2.loc[:,[x]].fillna(value=np.random.choice(x1[x]), axis=1)#replace NAs with random choice from actual distibution
        return df2
    
    def get_cat2(df, y):
        dfx = df[df.columns[1:y]]
        #df1= dfx.select_dtypes(include=['object'])#categorical
        col_list= list(dfx.columns.values)
        for i in range(0,len(dfx.columns)):
            x= col_list[i] #column name
            x1 = dfx.iloc[:,[i]].dropna(axis=0)
            m= stats.mode(x1) #get mode without NAs
            p = m[0].tolist()
            p= p[0]
            p= p[0]
            p = ''.join(str(p))
            #p= mode(x1)
            #p = pd.Series(m[0]) #put in series
            dfx.loc[:,[x]] = dfx.loc[:,[x]].fillna(value=p,axis=1) #replace NAs with mode
        return dfx
        
    def get_num2(df, y):
        dfx = df[df.columns[1:y]]#get numeric
        #print (df3)
        col_list= list(dfx.columns.values)
        for i in range(0,len(dfx.columns)):
	        x= col_list[i]
	        x1 = dfx.iloc[:,[i]].dropna(axis=0)
	        med= np.median(x1)
	        dfx.loc[:,[x]] = df.loc[:,[x]].fillna(value=med) #replace NAs with median
    
    if mv == 1: #choice 1, impute missing values with random choice from a distribution
        #print (mv)
        #df1= df.select_dtypes(include=['object'])#get categorical
        col_list= list(df.columns.values)
        #print(col_list)
        y= len(col_list)
        print("Replacing NA's for ", y, " columns")
        if dtype == 'c':
            df1= get_cat1(df, y)
        elif dtype == 'b':
            df1= get_cat1(df, y)
        elif dtype == 'c':
            df1= get_num1(df, y)
        else:
            print ("need -dtype : n=numeric, c=categorical, b=binary")

        frames  = [df0, df1] #put all frames back together ##need to add back class
        df= pd.concat(frames, axis=1)
    
    elif mv == 2: #this option imputes median or mode for either numeric or categorical data respectively
        col_list= list(df.columns.values)
        y= len(col_list)
        print("Replacing NA's for ", y, " columns")
        if dtype == 'c':
            df1= get_cat2(df, y)
        elif dtype == 'b':
            df1= get_cat2(df, y)
        elif dtype == 'c':
            df1= get_num2(df, y)
        else:
            print ("need -dtype : n=numeric, c=categorical, b=binary")
	            
        frames  = [df0, df1] #put all frames back together
        df= pd.concat(frames, axis=1)
        
    elif mv == 0:
        df = df.dropna(axis=0)   #third option, also default, remove all rows with NA, thereby leaving out missing values

    df.to_csv(path_or_buf=str(DF)+".NAimputed.txt", sep="\t", header=True)
    
if __name__ == '__main__':
	main()