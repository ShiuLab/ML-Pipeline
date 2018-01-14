print("""
inp1 = -df = pred file
inp2 = -pos = positive class name
inp3 = -neg = negative class name
""")

import sys, os
import pandas as pd
import numpy as np
from sklearn import metrics

for i in range (1,len(sys.argv),2):
        if sys.argv[i] == "-df":
            DF = sys.argv[i+1]
        if sys.argv[i] == "-pos":
            pos_name = int(sys.argv[i+1])
        if sys.argv[i] == "-neg":
            neg_name = int(sys.argv[i+1])
            
def get_score_list(inp):
    #cs_l = []
    col_list= list(inp.columns.values)
    actual_name= col_list[0]
    score_name= col_list[1]
    inp[score_name] = inp[score_name].astype(float)
    inp=inp.sort_values(by=score_name, ascending=False)
    #print (inp)
    #actual = inp.iloc[:,[0]]
    #score = inp.iloc[:,[1]]
    inp['tup_col'] = inp[[actual_name, score_name]].apply(tuple, axis=1)
    dfList = inp['tup_col'].tolist()
    #print (dfList)
    return (dfList)

def list_split(list):
	i0_l = []
	i1_l = []
	for pair in list:
		i0 = pair[0];i0_l.append(int(i0))
		i1 = pair[1];i1_l.append(i1)
	return i0_l,i1_l

def get_split_list(yp_l):
	sp_l = []
	for i in range(0,len(yp_l)-1):
	       #print (yp_l[i], yp_l[i+1])
	       if yp_l[i] != yp_l[i+1]:
	           sp_l.append(yp_l[i])
	return sp_l

def populate_counts(list):
	cnt1 = 0.0
	cnt0 = 0.0
	for item in list:
		if item == pos_name:
			cnt1 += 1.0
		elif item == neg_name:
			cnt0 += 1.0
	return cnt1,cnt0

def get_tpfnfptn(class_lst,split_index):
	pred_pos = class_lst[0:split_index]
	#print (pred_pos)
	pred_neg = class_lst[split_index:]
	tp,fp = populate_counts(pred_pos)
	fn,tn = populate_counts(pred_neg)
	return tp,fn,fp,tn

def get_tpfnfptn2(class_lst,split_index):
	pred_pos = class_lst[split_index:]
	#print (pred_pos)
	pred_neg = class_lst[0:split_index]
	tp,fp = populate_counts(pred_pos)
	fn,tn = populate_counts(pred_neg)
	return tp,fn,fp,tn

def calcKappa(tp,fn,fp,tn):
	predictedCorrect = tp+tn
	allPredictions = tp+fn+fp+tn
	if allPredictions == 0:
	    kappa = "NC"
	else:
	   class1freq = (tp+fn)/allPredictions
	   class2freq = (fp+tn)/allPredictions
	   numPredC1 = tp+fp
	   numPredC2 = fn+tn
	   ranC1cor = numPredC1*class1freq
	   ranC2cor = numPredC2*class2freq
	   randomCorrect = ranC1cor+ranC2cor
	   extraSuccesses = predictedCorrect-randomCorrect
	   kappa = extraSuccesses/(allPredictions-randomCorrect)
	   return kappa

def calc_aucroc(label_list,score_list,pos_nm=1): # requires scikit, NumPy, and SciPy modules! | pos = 1, neg = 0 in label list
	y = np.array(label_list)
	scores = np.array(score_list)
	fpr,tpr,thresholds = metrics.roc_curve(y,scores,pos_label=pos_nm)
	roc_auc = metrics.auc(fpr,tpr)
	return roc_auc

def perf_at_thresh(out,split_l,score_l,class_l,name):
	D={}
	D2={}
	#print(split_l)
	#print(score_l)
	for split_ind in split_l:
		threshold = split_ind
		index1= score_l.index(split_ind)
		tp,fn,fp,tn = get_tpfnfptn(class_l,index1)
		#print (tp, fn, fp, tn)
		kappa = calcKappa(tp,fn,fp,tn)
		if tp+fp == 0:
		    prec= 0
		else:
	           prec = tp/(tp+fp)
		if tp+fn == 0:
		    rec = 0
		else:
		    rec = tp/(tp+fn)
		if fp+tn == 0:
		    FPR = 0
		else:
		    FPR = fp/(fp+tn)
		if prec == 0 and rec == 0:
			fm = float(0)
		else:
			fm = float((2*prec*rec)/(prec+rec))
		# print threshold,kappa,fm,prec,rec,tp,fn,fp,tn
		
		D[fm]=threshold
	print (D)
	try:
	    maxFM = max(D.keys())
	    maxthresh = D[maxFM]
	except ValueError:
	    maxFM = float(0)
	    if np.mean(score_l) == float(0):
	        maxthresh = float(0)
	    else:
	        maxthresh = float(1)

	print(name,maxFM,maxthresh)
	
	for split_ind in split_l:
		threshold = split_ind
		index1= score_l.index(split_ind)
		tp,fn,fp,tn = get_tpfnfptn2(class_l,index1)
		#print (tp, fn, fp, tn)
		kappa = calcKappa(tp,fn,fp,tn)
		if tp+fp == 0:
		    prec= 0
		else:
	           prec = tp/(tp+fp)
		if tp+fn == 0:
		    rec = 0
		else:
		    rec = tp/(tp+fn)
		if fp+tn == 0:
		    FPR = 0
		else:
		    FPR = fp/(fp+tn)
		if prec == 0 and rec == 0:
			fm = float(0)
		else:
			fm = float((2*prec*rec)/(prec+rec))
		# print threshold,kappa,fm,prec,rec,tp,fn,fp,tn
		
		D2[fm]=threshold
	print (D2)
	try:
	    maxFM2 = max(D2.keys())
	    maxthresh2 = D2[maxFM2]
	except ValueError:
	    maxFM2 = float(0)
	    if np.mean(score_l) == float(0):
	        maxthresh2 = float(0)
	    else:
	        maxthresh2 = float(1)
	       
	
	print(name,maxFM2,maxthresh2)
	if maxFM2 >= maxFM:
	    split_ind2a = score_l.index(maxthresh2)
	    tp,fn,fp,tn = get_tpfnfptn2(class_l,split_ind2a)
	    kappa = calcKappa(tp,fn,fp,tn)
	    if tp+fp == 0:
	       prec= 0
	    else:
	       prec = tp/(tp+fp)
	    if tp+fn == 0:
	       rec = 0
	    else:
	       rec = tp/(tp+fn)
	    if fp+tn == 0:
	       FPR = 0
	    else:
	       FPR = fp/(fp+tn)
	    roc_auc = calc_aucroc(class_l,score_l)
	    out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(name,kappa,maxFM2,roc_auc,prec,rec,FPR))
	else:
	    split_ind2 = score_l.index(maxthresh)
	    tp,fn,fp,tn = get_tpfnfptn(class_l,split_ind2)
	    kappa = calcKappa(tp,fn,fp,tn)
	    if tp+fp == 0:
	       prec= 0
	    else:
	       prec = tp/(tp+fp)
	    if tp+fn == 0:
	       rec = 0
	    else:
	       rec = tp/(tp+fn)
	    if fp+tn == 0:
	       FPR = 0
	    else:
	       FPR = fp/(fp+tn)
	    roc_auc = calc_aucroc(class_l,score_l)
	    out.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(name,kappa,maxFM,roc_auc,prec,rec,FPR))
			
def main():
    df = pd.read_csv(DF, sep='\t', index_col = 0)
    df0 = df.iloc[:,[0]]
    #print(df0)
    col_list= list(df.columns.values)
    output= open(DF+".thresh_perf","w")
    output.write("feature\tkappa\tfm\taucroc\tprec\trec\tfpr\n")
    for i in range(1,len(df.columns)):#loop through columns
        x= col_list[i] #column name
        print(x)
        #output.write("%s\t" % (x))
        x1 = df.iloc[:,[i]] #get each column
        frames = [df0, x1]
        newdf= pd.concat(frames, axis=1)
        newdf= newdf.dropna(axis=0) #drop NAs
        #print (newdf.head())
        class_score_list = get_score_list(newdf)
        #print (class_score_list)
        class_list,score_list = list_split(class_score_list)
        #print (class_list, score_list)
        split_list = get_split_list(score_list)
        #print(class_list,score_list,split_list)
        perf_at_thresh(output,split_list,score_list,class_list,x)

if __name__ == '__main__':
	main()