##script used to combine multiple files into a matrix
import os, sys
import pandas as pd
import numpy as np


sum_matrix = open(sys.argv[1]+"_binary.matrix.txt","w")
# open with pandas
df = pd.read_csv(sys.argv[1], sep='\t', index_col = 0)

#get first line as title list
col_list= list(df.columns.values)
print (col_list)
#title_list = start_inp.readline()

#turn column to array, get each column and add their unique categorical value to make a title list
final_list = []
for i in range(0,len(col_list)):
    print (i)
    colname= col_list[i]
    print(colname)
    dfx= df.as_matrix([df.columns[i]])
    dfx_un= np.unique(dfx)
    for j in dfx_un:
        if str(j) == 'nan':
            pass
        else:
            string= str(df.columns[i]) + "."+str(j)
            if string not in final_list:
                final_list.append(string)
print (final_list)
final_str = "\t".join(str(j) for j in final_list)
sum_matrix.write("gene\t%s\n" % (final_str))

start_inp = open(sys.argv[1], "r") #categorical matrix
#loop through directory for each file to add input
D={}
def add_data_to_dict(inp,D):
    for line in inp:
        line.strip().replace('"','')
        if line.startswith("AT"):
            L = line.strip().split("\t")
            gene = L[0]
            clust = L[1:]
            if gene not in D:
                D[gene] = clust
            else:
                print(gene)

add_data_to_dict(start_inp,D)
print(D)
y = len(final_list)
for gene in D:
    feature_list= []
    for i in range(y):
        feature_list.append(0)
    #feature_list= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #print(feature_list)
    data_list= D[gene]
    for data in data_list:
        for xx in final_list:
            ind= final_list.index(xx)
            x1= xx.split(".")[0]
            #print (x1)
            x2= xx.split(".")[1]
            #print (x2)
            for x in col_list:
                if x1 == x:
                    if x2 == data:
                        feature_list[ind] = 1
                    elif data == 'NA':
                        feature_list[ind] = 'NA'
    #print (feature_list)
    feat_str= "\t".join(str(k) for k in feature_list)
    sum_matrix.write("%s\t%s\n" % (gene, feat_str))

sum_matrix.close()