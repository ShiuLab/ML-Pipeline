'''This script parses importance file (-imp) and input matrix (-df) to get 
scaled directional importance. Since imp files are only obtained from binary models,
this script is set up to use a positive and negative class only.

Author: Bethany Moore

NEED:
    -imp importance file from ML output
    -df  input dataframe to ML with genes in first column and Class in second column
    -pos name of positive class
    -neg name of negative class
    -alg algorithm used to get importance: RF,GB,SVM,LogReg
'''

import sys, os
import pandas as pd
import numpy as np


## analyze if feature data is binary or continuous
    
def binorcont(df):
        df0 = df.iloc[:, 0:1] #get classes for bin matrix
        df1 = df.iloc[:, 0:1] #get classes for continuous matrix
        #print(df0, df1)
        #get features
        col_list= list(df.columns.values)
        y= len(col_list)
        df2= df[df.columns[2:y]]
        #print(df2.columns.values)
        #loop through feature columns to asses whether cont or binary
        onevalue = []
        for i in range(0,len(df2.columns)):
             x= df2.columns.values[i]
             #print(x)
             x1 = df2.iloc[:,[i]].dropna(axis=0)
             if x1.empty == True:
                 pass
             else:
                 lista = x1[x].unique().tolist()
                 #print(x1)
                 if len(lista) == 2:
                     if 0 and 1 in lista:
                         df0= pd.concat([df0, x1], axis=1, sort=True)
                     else:
                         df1= pd.concat([df1, x1], axis=1, sort=True)
             
                 elif len(lista) > 2:
                     df1= pd.concat([df1, x1], axis=1, sort=True)
                     #df1=df1.append(x1[x])
                 else:
                     onevalue.append(x)

        #return binary and cont matrices            
        print ("Number of features with only one value: ", len(onevalue), "out of ", len(list(df2.columns.values)))
        return(df0, df1)
                     
                     
def get_enrich(df, pos, neg):
    col_list= list(df.columns.values)
    y= len(col_list)
    #df= pd.DataFrame(df.fillna(value=pd.np.nan, inplace=True))
    #print(df)
    #df2= df[df.columns[1:y]]
    #print(df2.columns.values)
    newlist= []
    
    for i in range(1,len(df.columns)):
        
        #get counts for each class
        df2 = df.loc[df['Class'] == pos]
        x= df2.columns.values[i]
        #print(x)
        x1 = df2.iloc[:,[i]].dropna(axis=0) 
        counts= x1[x].value_counts()
        pos1= counts.get(1)
        pos2= counts.get(0)
        if pd.isnull(pos1) == True:
            pos1= 0
        elif pd.isnull(pos2) == True:
            pos2 = 0
        elif pos1 == 'None':
            pos1= 0
        elif pos2 == 'None':
            pos2 = 0
        #print(pos1, pos2)
        totalpos= float(pos1+pos2)
        df3 = df.loc[df['Class'] == neg]
        x2= df3.columns.values[i]
        x12 = df3.iloc[:,[i]].dropna(axis=0) 
        counts2= x12[x2].value_counts()
        neg1= counts2.get(1)
        neg2= counts2.get(0)
        if pd.isnull(neg1) == True:
            neg1= 0
        elif pd.isnull(neg2) == True:
            neg2 = 0
        elif neg1 == 'None':
            neg1= 0
        elif neg2 == 'None':
            neg2 = 0
        totalneg= float(neg1+neg2)
        #print (neg1, neg2)
        
        #calculate enrichment
        
        pos_en= (float(pos1))/totalpos
        neg_en= (float(neg1))/totalneg
        if pos_en > neg_en:
            enrich = '+'
        elif pos_en < neg_en:
            enrich = '-'
        else:
            enrich = 'None'

        newlist.append([x, pos_en, neg_en, enrich])
        
    newdf = pd.DataFrame(newlist, columns=['feature','pos','neg', 'enrichment'])
    return(newdf)
                          
def get_median(df, pos, neg): ##use correlation?
    col_list= list(df.columns.values)
    y= len(col_list)
    #df2= df[df.columns[1:y]]
    #print(df2.columns.values)
    newlist= []
    
    for i in range(1,len(df.columns)):
        
        #get median for each class
        df2 = df.loc[df['Class'] == pos]
        x= df2.columns.values[i]
        #print(x)
        x1 = df2.iloc[:,[i]].dropna(axis=0)
        meanpos= np.mean(x1[x])
        medpos= np.median(x1[x])
        df3 = df.loc[df['Class'] == neg]
        x2= df3.columns.values[i]
        #print(x2)
        x3 = df3.iloc[:,[i]].dropna(axis=0)
        meanneg= np.mean(x3[x2])
        medneg= np.median(x3[x2])
        if float(meanpos) > float(meanneg):
            if float(medpos) > float(medneg):
                enrich = '+'
            elif float(medpos) == float(medneg):
                enrich = '+'
            elif float(medpos) < float(medneg):
                enrich = '-'
            else:
                print(medpos, medneg)
        elif float(meanpos) < float(meanneg):
            if float(medpos) < float(medneg):
                enrich = '-'
            elif float(medpos) == float(medneg):
                enrich = '-'
            elif float(medpos) > float(medneg):
                enrich = '+'
            else:
                print(medpos, medneg)
        else:
            if float(medpos) == float(medneg):
                enrich = 'None'
            elif float(medpos) > float(medneg):
                enrich = '+'
            elif float(medpos) < float(medneg):
                enrich = '-'
            else:
                enrich = 'None'
            
        newlist.append([x, meanpos, meanneg, enrich]) 
    
    newdf = pd.DataFrame(newlist, columns=['feature','pos','neg', 'enrichment']) #index=['feature']       
    return(newdf)
        

def normalize_values(lista):
    #gene = lista[0]
    newlist=[]
    data= lista
    #print(data)
    datafl=[]
    datafl = data[np.logical_not(pd.isnull(data))] # remove NA's to get floats and min/max
    if len(datafl) > 0: #check list is not empty
        datafl=np.array(datafl, dtype=np.float32) #convert to float in numpy
        mindata= np.amin(datafl)
        maxdata= np.amax(datafl)
        print(mindata, maxdata)
        dem= float(maxdata)-float(mindata)
        if dem != float(0):
            for i in data: #have to go back to original data because have removed NAs which can be placeholders
                    if i != np.nan:
                            norm= (float(i)-float(mindata))/(dem)
                            newlist.append(float(norm))
                    else:
                        newlist.append(np.nan)
            #print(newlist)
            return(newlist)
        else: #replace with NAs if denominator is = 0
            dem=float(0.00001)
            for i in data: #have to go back to original data because have removed NAs which can be placeholders
                    if i != np.nan:
                            norm= (float(i)-float(mindata))/(dem)
                            newlist.append(float(norm))
                    else:
                        newlist.append(np.nan)
            #print(newlist)
            return(newlist)
    else: #replace with NAs if all NAs
        for i in data:
            newlist.append(np.nan)
        #print(newlist)
        return(newlist)

def get_percentrank(lista):
    newlist=[]
    data= lista
    datafl=[]
    datafl = data[np.logical_not(pd.isnull(data))] # remove NA's to get floats and min/max
    if len(datafl) > 0: #check list is not empty
        l= len(datafl)
        datafl=np.array(datafl, dtype=np.float32) #convert to float in numpy
        for i in data: #have to go back to original data because have removed NAs which can be placeholders
                    if i != np.nan:
                        percent= (float(i))/(float(l))
                        newlist.append(float(percent))
                    else:
                        newlist.append(np.nan)
            #print(newlist)
        return(newlist)

    else: #replace with NAs if all NAs
        for i in data:
            newlist.append(np.nan)
        #print(newlist)
        return(newlist)

def get_scale(df, alg):
    # get importance score
    df2= df.loc[:,'imp_score']
    #take absolute value
    df2= pd.DataFrame(df2.abs())
    #get rank
    dfrank= df2.rank(axis=0,method='average')
    print("ranking importance scores")
    #get row names in list
    rows_to_norm = df2.index[1:].values.tolist()
    #print(df2, rows_to_norm)
    #normalize values
    print('normalizing importance scores')
    df3= df2.loc[rows_to_norm,:].apply(normalize_values, axis=0, result_type= 'expand')
    #rename columns
    df3= df3.rename(index=str, columns={"imp_score": "imp_score_scaled"})
    #get percentile
    print("getting percentile rank")
    dfr= dfrank.loc[rows_to_norm,:].apply(get_percentrank, axis=0, result_type= 'expand')
    #rename columns
    dfr= dfr.rename(index=str, columns={"imp_score": "imp_score_rank"})
    #join dataframes
    df4= pd.DataFrame(pd.concat([df, df3, dfr], axis=1, join='inner'))
    print(df4)
    #get enrichment score
    if alg == 'RF' or 'GB':
        print('scaling importance score based on enrichment')
        df5 = df4.loc[df['enrichment'] == '-']
        df5= df5[df5.imp_score != float(0)]
        #print(df5.loc[:,'imp_score_scaled'])
        x= df5.loc[:,'imp_score_scaled'].multiply(float(-1))
        z= df5.loc[:,'imp_score_rank'].multiply(float(-1))
        #print(x)
        df6 = df4.loc[df['enrichment'] == '+']
        df6= df6[df2.imp_score != float(0)]
        y= df6.loc[:,'imp_score_scaled']
        a= df6.loc[:,'imp_score_rank']
        #print(y)
        xy= pd.DataFrame(pd.concat([x, y], axis=0, join='inner'))
        za= pd.DataFrame(pd.concat([z, a], axis=0, join='inner'))
        #print(za)
        xy= xy.rename(index=str, columns={"imp_score": "imp_score_scaled"})
        newdf= pd.DataFrame(pd.concat([xy, za], axis=1, join='inner'))
        
    else:
        print('scaling importance score based on model sign')
        df5 = df4.loc[:,'imp_score']
        sign_array= np.sign(df5)
        #print(sign_array)
        result = df4.loc[:,'imp_score_scaled'].mul(sign_array, axis=0)
        result2= dfr.loc[:,'imp_score_rank'].mul(sign_array, axis=0)
        #print (result)
        df4['imp_score_scaled']= result
        df4['imp_score_rank']=result2
        #print(df4)
        df4= pd.DataFrame(df4.loc[:,'imp_score_scaled':'imp_score_rank'])
        #print(df4)
        newdf= pd.DataFrame(df4[df4.loc[:,'imp_score_scaled'] != float(0)])
    return(newdf)

def main():
    
    for i in range (1,len(sys.argv),2):
            if sys.argv[i] == "-df":
                DF = sys.argv[i+1]
            if sys.argv[i] == "-imp":
                imp = sys.argv[i+1]
            if sys.argv[i] == "-pos":
                pos = sys.argv[i+1]
            if sys.argv[i] == "-neg":
                neg = sys.argv[i+1]
            if sys.argv[i] == "-alg":
                alg = sys.argv[i+1]


    if len(sys.argv) <= 1:
	    print(__doc__)
	    exit()

    df = pd.read_csv(DF, sep='\t', index_col = 0)
    #print(df)
    # get binary and continous matrices
    print('getting binary and continuous matrices from full feature matrix')
    bin_df, con_df = binorcont(df)
    #print (bin_df, con_df)
    # for binary, get enrichment to determine feature direction
    print('getting binary enrichment')
    enrich_df= get_enrich(bin_df, pos, neg)
    #print(enrich_df)
    # for continuous, get median/mean to determine feature direction
    print('getting mean for continous data')
    med_df= get_median(con_df, pos, neg)
    #print(med_df)
    
    #print('getting correlation for continous data')
    #med_df= get_corr(con_df, pos, neg)
    # concatenate cont and bin directions
    dirdf= pd.concat([enrich_df, med_df], axis=0, join='inner')
    
    dirdf= dirdf.set_index('feature')
    #print(dirdf)
    
    # add direction to importance score
    impdf = pd.read_csv(imp, sep='\t', index_col = 0, names = ['feature', 'imp_score'])
    #print(impdf)
    #impdf= impdf.set_index('feature')
    
    dfx= pd.concat([impdf, dirdf], axis=1, join_axes=[impdf.index]) # exact join from first dataframe
    #dfx= impdf.set_index('feature').join(dirdf.set_index('feature'))
    #print(dfx)
    #dfx= dfx.set_index('feature')
    # scale importance score
    findf= get_scale(dfx, alg)
    #print(findf)
    df_final= pd.concat([dfx, findf], axis=1, join='inner')
    df_final= df_final.sort_values(by=['imp_score_scaled'],  ascending=False)
    print("writing final matrix: ", df_final.head(n=10))
    df_final.to_csv(path_or_buf=str(imp)+'_scaled.txt', sep="\t", header=True)
    
if __name__ == '__main__':
	main()    
	    
