"""
PURPOSE:
Machine learning classifications implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
	
	REQUIRED ALL:
	-df       Feature & class dataframe for ML. See "" for an example dataframe
	-alg      Available: RF, SVM (linear), SVMpoly, SVMrbf
	
	OPTIONAL:
	
	-pos      name of positive class (default = 1 or first class provided with -cl_train)
	-cl_train List of classes to include in the training set. Default = all classes. If binary, first label = positive class.
	-save     Save name. Default = [df]_[alg] (caution - will overwrite!)
	-feat     Import file with list of features to keep if desired. Default: keep all.
	-cv       # of cross-validation folds. Default = 10
	-gs       Set to True if parameter sweep is desired. Default = False
	-n        # of random balanced datasets to run. Default = 50
	-class    String for what column has the class. Default = Class
	-apply    To which non-training class labels should the models be applied? Enter 'all' or a list (comma-delimit if >1)
	-cm       T/F - Do you want to output the confusion matrix & confusion matrix figure? (Default = False)
	-plots    T/F - Do you want to output ROC and PR curve plots for each model? (Default = False)
	-tag      String for the TAG column in the RESULTS.txt output.
	-mv       missing value type, default= 0 (rows with NAs are removed), 1= NAs drawn from sample distribution, 2= NAs drawn from median (numeric) or mode (categorical)

OUTPUT:
	-SAVE_imp           Importance scores for each feature
	-SAVE_GridSearch    Results from parameter sweep sorted by F1
	-RESULTS.txt     Accumulates results from all ML runs done in a specific folder - use unique save names! XX = RF or SVC

"""

import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
import time

import ML_functions as ML
from scipy import stats

def main():
	
	# Default code parameters
	n, FEAT, CL_TRAIN, apply, n_jobs, class_col, CM, POS, plots, cv_num, TAG = 50, 'all', 'all','none', 14, 'Class', 'False', 1, 'False', 10, ''
	
	# Default parameters for Grid search
	GS, gs_score = 'F', 'roc_auc'
	
	# Default Random Forest parameters
	n_estimators, max_depth, max_features = 500, 10, "sqrt"
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'linear', 1, 2, 1, 'hinge', "500"
	
	#default for missing values (-mv)
	mv = 0

	for i in range (1,len(sys.argv),2):
		if sys.argv[i] == "-df":
			DF = sys.argv[i+1]
			SAVE = DF
		if sys.argv[i] == '-save':
			SAVE = sys.argv[i+1]
		if sys.argv[i] == '-feat':
			FEAT = sys.argv[i+1]
		if sys.argv[i] == "-gs":
			GS = sys.argv[i+1]
		if sys.argv[i] == "-gs_score":
			gs_score = sys.argv[i+1]
		if sys.argv[i] == '-cl_train':
			CL_TRAIN = sys.argv[i+1].split(',')
			POS = CL_TRAIN[0]
		if sys.argv[i] == '-apply':
			apply = sys.argv[i+1].lower()
			if apply != "all":
				apply = sys.argv[i+1].split(',')
		if sys.argv[i] == "-class":
			class_col = sys.argv[i+1]
		if sys.argv[i] == "-n":
			n = int(sys.argv[i+1])
		if sys.argv[i] == "-alg":
			ALG = sys.argv[i+1]
		if sys.argv[i] == "-cv":
			cv_num = int(sys.argv[i+1])
		if sys.argv[i] == "-n_jobs":
			n_jobs = int(sys.argv[i+1])
		if sys.argv[i] == "-cm":
			CM = sys.argv[i+1]
		if sys.argv[i] == "-plots":
			plots = sys.argv[i+1]
		if sys.argv[i] == "-pos":
			POS = sys.argv[i+1]
		if sys.argv[i] == "-tag":
			TAG = sys.argv[i+1]
		if sys.argv[i] == "-mv":
		        mv = sys.argv[i+1]

	if len(sys.argv) <= 1:
		print(__doc__)
		exit()
	
	####### Load Dataframe & Pre-process #######
	
	df = pd.read_csv(DF, sep='\t', index_col = 0)
	
	# Specify class column - default = Class
	if class_col != 'Class':
		df = df.rename(columns = {class_col:'Class'})
	
	# Filter out features not in feat file given - default: keep all
	if FEAT != 'all':
		with open(FEAT) as f:
			features = f.read().splitlines()
			features = ['Class'] + features
		df = df.loc[:,features]
	
	
	#missing values
	# turn all NAs or ? to NaN (NA recognized by numpy)
	df = df.replace("?",np.nan)
	df = df.replace("NA",np.nan)
	
	# find the percent of missing feature, if greater than 50%, then drop!
	#col_list1= list(df.columns.values)
	for i in range(2,len(df.columns)):
	    missing=  df.iloc[:,[i]].isnull().sum()
	    miss_pct = missing/len(df)
	    if  miss_pct.iloc[0] > float(0.5):
	        df = df.drop(df.columns[i], 1) ##### need to test
	
	df0 = df.iloc[:,[0,1]]
	print(df0)
	def get_percent(counts):#function to get proportion
	    L=[]
	    for j in range(0,len(counts)):
	        p =counts[j] / float(counts.sum())
	        L.append(p)
	    return (L)
	#choice 1, impute missing values with random choice from a distribution
	if mv == 1:
	    df1= df.select_dtypes(include=['object'])#get categorical
	    print(df1)
	    col_list= list(df1.columns.values)
	    for i in range(2,len(df1.columns)):
	        x= col_list[i] #column name
	        x1 = df1.iloc[:,[i]].dropna(axis=0) #get values of column name, drop NAs
	        cats= x1.unique() #get unique categories 
	        counts= x1.value_counts() #get counts of each category
	        pcts = get_percent(counts) #get proportion of each category
	        df1.iloc[:,[i]] = df.iloc[:,[i]].replace(np.nan,np.random.choice(cats, p=pcts)) #randomly choose from unique categories\
	        #based on their proportion in the dataset, replace NAs with this category
	    
	    #get integers, may be categorical or numeric
	    df2= df.select_dtypes(include=['int'])
	    col_list= list(df2.columns.values)
	    print(df2)
	    for i in range(2,len(df2.columns)):
	        x= col_list[i]
	        x1 = df2.iloc[:,[i]].dropna(axis=0)
	        cats= x1.unique()
	        if cats.sum() == 1: #this is the case for binary- sum of unique categories should be 1
	            counts= x1.value_counts() #get counts of each catgory
	            pcts = get_percent(counts) #get proportion
	            df2.iloc[:,[i]] = df2.iloc[:,[i]].replace(np.nan,np.random.choice(cats, p=pcts))#replace NAs with random choice from proportion of categories
	        else:
	            df2.iloc[:,[i]] = df2.iloc[:,[i]].replace(np.nan,np.random.choice(x1))#replace NAs with random choice from actual distibution
	            
	    df3= df.select_dtypes(include=['float'])#get numeric
	    print(df3)
	    col_list= list(df3.columns.values)
	    for i in range(2,len(df3.columns)):
	        x= col_list[i]
	        x1 = df3.iloc[:,[i]].dropna(axis=0)
	        df3.iloc[:,[i]] = df3.iloc[:,[i]].replace(np.nan,np.random.choice(x1))#replace NAs with random choice from actual distibution
	    frames  = [df0, df1, df2, df3] #put all frames back together ##need to add back class
	    df= pd.concat(frames, axis=1)
	    
	elif mv == 2: #this option imputes median or mode for either numeric or categorical data respectively
	    df1= df.select_dtypes(include=['object'])#categorical
	    col_list= list(df1.columns.values)
	    for i in range(2,len(df1.columns)):
	        x= col_list[i] #column name
	        x1 = df1.iloc[:,[i]].dropna(axis=0)
	        m= stats.mode(x1) #get mode without NAs
	        p = pd.Series(m[0]) #put in series
	        df1.iloc[:,[i]] = df1.iloc[:,[i]].replace(np.nan,  p.loc[0]) #replace NAs with mode
	        
	    df2= df.select_dtypes(include=['int'])#get integers, may be categorical or numeric
	    col_list= list(df2.columns.values)
	    for i in range(2,len(df2.columns)):
	        x= col_list[i]
	        x1 = df2.iloc[:,[i]].dropna(axis=0)
	        cats= x1.unique()
	        if cats.sum() == 1: ##for binary data
	            m= stats.mode(x1)
	            p = pd.Series(m[0])
	            df2.iloc[:,[i]] = df2.iloc[:,[i]].replace(np.nan,  p.loc[0])
	        else:
	            med= np.median(x1) #get median for numeric data
	            df2.iloc[:,[i]] = df2.iloc[:,[i]].replace(np.nan, med) #replace NAs with median
	            
	    df3= df.select_dtypes(include=['float']) #get numeric
	    col_list= list(df3.columns.values)
	    for i in range(2,len(df3.columns)):
	        x= col_list[i]
	        x1 = df3.iloc[:,[i]].dropna(axis=0)
	        med= np.median(x1)
	        df3.iloc[:,[i]] = df3.iloc[:,[i]].replace(np.nan, med) #replace NAs with median
	    frames  = [df0, df1, df2, df3] #put all frames back together
	    df= pd.concat(frames, axis=1)
	else:
	    df = df.dropna(axis=0)   #third option, also default, remove all rows with NA, thereby leaving out missing values

	
	# Set up dataframe of unknown instances that the final models will be applied to
	if CL_TRAIN != 'all' and apply != 'none':
		apply_unk = True
		if apply.lower() == 'all':
			df_unknowns = df[(~df['Class'].isin(CL_TRAIN))]
		else:
			df_unknowns = df[(df['Class'].isin(apply))]
	else:
		apply_unk = False
		df_unknowns = ''
	
	# Remove classes that won't be included in the training (e.g. unknowns)
	if CL_TRAIN != 'all':
		df = df[(df['Class'].isin(CL_TRAIN))]

	# Generate training classes list. If binary, establish POS and NEG classes. Set grid search scoring: roc_auc for binary, f1_macro for multiclass
	if CL_TRAIN == 'all':
		classes = df['Class'].unique()
		if len(classes) == 2:
			gs_score = 'roc_auc'
			for clss in classes:
				if clss != POS:
					NEG = clss
					try:
						NEG = int(NEG)
					except:
						pass
					break
		else:
			NEG = 'multiclass_no_NEG'
			gs_score = 'f1_macro'
	else:
		if len(CL_TRAIN) == 2:
			NEG = CL_TRAIN[1]
			gs_score = 'roc_auc'
		else:
			NEG = 'multiclass_no_NEG'
			gs_score = 'f1_macro'
		classes = np.array(CL_TRAIN)

	classes.sort()
	
	print("Snapshot of data being used:")
	print(df.head())
	print("CLASSES:",classes)
	print("POS:",POS,type(POS))
	print("NEG:",NEG,type(NEG))
	n_features = len(list(df)) - 1
	
	# Determine minimum class size (for making balanced datasets)
	min_size = (df.groupby('Class').size()).min() - 1
	print('Balanced dataset will include %i instances of each class' % min_size)
	
	SAVE = SAVE + "_" + ALG
	
	####### Run parameter sweep using a grid search #######
	
	if GS.lower() == 'true' or GS.lower() == 't':
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use,balanced_ids, param_names = ML.fun.GridSearch(df, SAVE, ALG, classes, min_size, gs_score, n, cv_num, n_jobs, POS, NEG)
		
		# Print results from grid search
		if ALG == 'RF':
			max_depth, max_features = params2use
			print("Parameters selected: max_depth=%s, max_features=%s" % (
				str(max_depth), str(max_features)))
	
		elif ALG == 'SVM':
			C, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s" % (str(kernel), str(C)))
		
		elif ALG == "SVMpoly":
			C, degree, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(C), str(degree), str(gamma)))
		
		elif ALG == "SVMrbf":
			C, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, gamma=%s" % (str(kernel), str(C), str(gamma)))
		
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	
	else:
		balanced_ids = ML.fun.EstablishBalanced(df,classes,min_size,n)
	
	bal_id = pd.DataFrame(balanced_ids)
	bal_id.to_csv(SAVE + '_BalancedIDs.csv', index=False, header=False)

	 
	####### Run ML models #######
	start_time = time.time()
	print("\n\n===>  ML Pipeline started  <===")

	results = []
	df_proba = pd.DataFrame(data=df['Class'], index=df.index, columns=['Class'])
	if apply_unk == True:
		df_proba2 = pd.DataFrame(data=df_unknowns['Class'], index=df_unknowns.index, columns=['Class'])
		df_proba = pd.concat([df_proba,df_proba2],axis=0)
	
	for j in range(len(balanced_ids)):
		
		print("  Round %s of %s"%(j+1,len(balanced_ids)))
		
		#Make balanced datasets
		df1 = df[df.index.isin(balanced_ids[j])]
		df_notSel = df[~df.index.isin(balanced_ids[j])]
		
		# Remove non-training classes from not-selected dataframe
		if CL_TRAIN != 'all':
			df_notSel = df_notSel[(df_notSel['Class'].isin(CL_TRAIN))]
		
		# Prime classifier object based on chosen algorithm
		if ALG == "RF":
			parameters_used = [n_estimators, max_depth, max_features]
			clf = ML.fun.DefineClf_RandomForest(n_estimators,max_depth,max_features,j,n_jobs)
		elif ALG == "SVM" or ALG == 'SVMrbf' or ALG == 'SVMpoly':
			parameters_used = [C, degree, gamma, kernel]
			clf = ML.fun.DefineClf_SVM(kernel,C,degree,gamma,j)
		
		# Run ML algorithm on balanced datasets.
		result,current_scores = ML.fun.BuildModel_Apply_Performance(df1, clf, cv_num, df_notSel, apply_unk, df_unknowns, classes, POS, NEG, j)
		results.append(result)
		df_proba = pd.concat([df_proba,current_scores], axis = 1)

	print("ML Pipeline time: %f seconds" % (time.time() - start_time))


	
	####### Unpack ML results #######
	
	## Make empty dataframes
	conf_matrices = pd.DataFrame(columns = np.insert(arr = classes.astype(np.str), obj = 0, values = 'Class'))
	imp = pd.DataFrame(index = list(df.drop(['Class'], axis=1)))
	threshold_array = []
	AucRoc_array = []
	AucPRc_array = []
	accuracies = []
	f1_array = np.array([np.insert(arr = classes.astype(np.str), obj = 0, values = 'M')])
	
	count = 0
	for r in results:
		count += 1
		if 'cm' in r:
			cmatrix = pd.DataFrame(r['cm'], columns = classes)
			cmatrix['Class'] = classes
			conf_matrices = pd.concat([conf_matrices, cmatrix])
		
		# For binary predictions
		if 'importances' in r:
			imp[count] = r['importances'][0]
		if 'AucRoc' in r:
			AucRoc_array.append(r['AucRoc'])
		if 'AucPRc' in r:
			AucPRc_array.append(r['AucPRc'])
		if 'threshold' in r:
			threshold_array.append(r['threshold'])

		# For Multi-class predictions
		if 'accuracy' in r:
			accuracies.append(r['accuracy'])
		if 'macro_f1' in r:
			f1_temp_array = np.insert(arr = r['f1'], obj = 0, values = r['macro_f1'])
			f1_array = np.append(f1_array, [f1_temp_array], axis=0)

	cm_mean = conf_matrices.groupby('Class').mean()

	# Multiclass Output
	if len(classes) > 2:
		f1 = pd.DataFrame(f1_array)
		f1.columns = f1.iloc[0]
		f1 = f1[1:]
		f1.columns = [str(col) + '_F1' for col in f1.columns]
		f1 = f1.astype(float)		
		
		# Calculate accuracy and f1 stats
		AC = np.mean(accuracies)
		AC_std = np.std(accuracies)
		MacF1 = f1['M_F1'].mean()
		MacF1_std = f1['M_F1'].std()

		print("\nML Results: \nAccuracy: %03f (+/- stdev %03f)\nF1 (macro): %03f (+/- stdev %03f)\nAUC-ROC (macro): %03f (+/- stdev %03f)" % (
		AC, AC_std, MacF1, MacF1_std, MacAUC, MacAUC_std))


	# Binary Prediction Output
	else: 
		# Get AUC for ROC and PR curve
		ROC = [np.mean(AucRoc_array), np.std(AucRoc_array), np.std(AucRoc_array)/np.sqrt(len(AucRoc_array))]
		PRc = [np.mean(AucPRc_array), np.std(AucPRc_array), np.std(AucPRc_array)/np.sqrt(len(AucPRc_array))]
		
		# Find mean threshold
		final_threshold = np.mean(threshold_array)

		# Determine final prediction call - using the final_threshold on the mean predicted probability.
		proba_columns = [c for c in df_proba.columns if c.startswith('score_')]
		df_proba.insert(loc=1, column = 'Mean', value = df_proba[proba_columns].mean(axis=1)) 
		df_proba.insert(loc=2, column = 'stdev', value = df_proba[proba_columns].std(axis=1))
		Pred_name =  'Predicted_' + str(final_threshold)
		df_proba.insert(loc=3, column = Pred_name, value = df_proba['Class'])
		df_proba[Pred_name][df_proba.Mean >= final_threshold] = POS
		df_proba[Pred_name][df_proba.Mean < final_threshold] = NEG
		df_proba.to_csv(SAVE + "_scores.txt", sep="\t")
	

		# Get model preformance scores using final_threshold
		TP,TN,FP,FN,TPR,FPR,FNR,Pr,Ac,F1 = ML.fun.Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, POS, NEG)
		

		# Plot confusion matrix (% class predicted as each class) based on balanced dataframes
		if CM.lower() == 'true' or CM.lower() == 't':
			cm_mean.to_csv(SAVE + "_cm.csv")
			done = ML.fun.Plot_ConMatrix(cm_mean, SAVE)

		# Plot ROC & PR curves
		if plots.lower() == 'true' or plots.lower() == 't':
			print("\nGenerating ROC & PR curves")
			pr = ML.fun.Plots(df_proba, balanced_ids, ROC, PRc, POS, NEG, n, SAVE)
		
		# Export importance scores
		try:
			imp['mean_imp'] = imp.mean(axis=1)
			imp = imp.sort_values('mean_imp', 0, ascending = False)
			imp_out = SAVE + "_imp.csv"
			imp['mean_imp'].to_csv(imp_out, sep = ",", index=True)
		except:
			pass

		timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		
		# Save to summary RESULTS file with all models run from the same directory
		
		if not os.path.isfile('RESULTS.txt'):
			out2 = open('RESULTS.txt', 'a')
			out2.write('DateTime\tID\tTag\tAlg\tClasses\tFeatureNum\tBalancedSize\tCVfold\tBalancedRuns\tAUCROC\tAUCROC_sd\ttAUCROC_se\t')
			out2.write('AUCPRc\tAUCPRc_sd\tAUCPRc_se\tAc\tAc_sd\tAc_se\tF1\tF1_sd\tF1_se\tPr\tPr_sd\tPr_se\tTPR\tTPR_sd\tTPR_se\t')
			out2.write('FPR\tFPR_sd\tFPR_se\tFNR\tFNR_sd\tFNR_se\tTP\tTP_sd\tTP_se\tTN\tTN_sd\tTN_se\tFP\tFP_sd\tFP_se\t')
			out2.write('FN\tFN_sd\tFN_se\n')
			out2.close()
		out2 = open('RESULTS.txt', 'a')
		out2.write('%s\t%s\t%s\t%s\t%s\t%i\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
			timestamp, SAVE, TAG, ALG, [POS,NEG], n_features, min_size, cv_num , n, 
			'\t'.join(str(x) for x in ROC), '\t'.join(str(x) for x in PRc), '\t'.join(str(x) for x in Ac), '\t'.join(str(x) for x in F1),
			'\t'.join(str(x) for x in Pr), '\t'.join(str(x) for x in TPR), '\t'.join(str(x) for x in FPR),
			'\t'.join(str(x) for x in FNR), '\t'.join(str(x) for x in TP), '\t'.join(str(x) for x in TN),
			'\t'.join(str(x) for x in FP), '\t'.join(str(x) for x in FN)))

		# Save detailed results file 
		out = open(SAVE + "_results.txt", 'w')
		out.write('%s\nID: %s\nTag: %s\nAlgorithm: %s\nTrained on classes: %s\nApplied to: %s\nNumber of features: %i\n' % (
			timestamp, SAVE, TAG, ALG, classes, apply, n_features))
		out.write('Min class size: %i\nCV folds: %i\nNumber of models: %i\nGrid Search Used: %s\nParameters used:%s\n' % (
			min_size, cv_num, n, GS, parameters_used))
		out.write('\nMetric\tMean\tSD\tSE\n')
		out.write('AucROC\t%s\nAucPRc\t%s\nAccuracy\t%s\nF1\t%s\nPrecision\t%s\nTPR\t%s\nFPR\t%s\nFNR\t%s\n' % (
			'\t'.join(str(x) for x in ROC),'\t'.join(str(x) for x in PRc), '\t'.join(str(x) for x in Ac), '\t'.join(str(x) for x in F1),
			'\t'.join(str(x) for x in Pr), '\t'.join(str(x) for x in TPR), '\t'.join(str(x) for x in FPR), '\t'.join(str(x) for x in FNR)))
		out.write('TP\t%s\nTN\t%s\nFP\t%s\nFN\t%s\n' % (
			'\t'.join(str(x) for x in TP), '\t'.join(str(x) for x in TN), '\t'.join(str(x) for x in FP), '\t'.join(str(x) for x in FN)))
		out.write('\nMean Balanced Confusion Matrix:\n')
		out.close()
		cm_mean.to_csv(SAVE + "_results.txt", mode='a', sep='\t')

		print("\n\n===>  ML Results  <===")
		print("Accuracy: %03f (+/- stdev %03f)\nF1: %03f (+/- stdev %03f)\nAUC-ROC: %03f (+/- stdev %03f)\nAUC-PRC: %03f (+/- stdev %03f)" % (
			Ac[0], Ac[1], F1[0], F1[1], ROC[0], ROC[1], PRc[0], PRc[1]))
	
	
	

if __name__ == '__main__':
	main()
