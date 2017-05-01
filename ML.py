"""
PURPOSE:
Bianary classifications using RandomForestClassifier and LinearSVC implemented in sci-kit learn. 

Optional grid seach function (-gs True) to run a parameter sweep on a subset of the balanced datasets

To access pandas, numpy, and sklearn packages on HPS first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH


INPUTS:
	
	REQUIRED:
	-df       Feature & class dataframe for ML. See "" for an example dataframe
	-save     Unique save name (caution - will overwrite!)
	
	OPTIONAL:
	-feat     Import file with list of features to keep if desired. Default: keep all.
	-gs       Set to True if parameter sweep is desired. Default = False
	-alg      Algorithm to use. Currently available: RandomForest (RF)(Default), LinearSVC (SVC)
	-n        Number of random balanced datasets to run. Default = 50
	-class    String for what column has the class. Default = Class
	-cl_train    List of classes to include in the training set. Default = all classes (if >2 default to multiclass pred)
	-cl_test     List of classes to apply the trained model to (i.e. Unknowns). Default = na
	-cm      T/F - Do you want to output the confusion matrix & confusion matrix figure

OUTPUT:
	-SAVE_imp           Importance scores for each feature
	-SAVE_GridSearch    Results from parameter sweep sorted by F1
	-RESULTS_XX.txt     Accumulates results from all ML runs done in a specific folder - use unique save names! XX = RF or SVC

"""
import sys
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import (precision_recall_curve, f1_score)
import time

import ML_functions as ML

def main():
	
	# Default code parameters
	n, FEAT, CL_TRAIN, CL_APPLY, SAVE, ALG, n_jobs, class_col, CM = 50, 'all', 'all','none','test', 'RF', 50, 'Class', 'False'

	# Default parameters for Grid search
	GS, gs_score, gs_n = 'F', None, 50

	# Default Random Forest parameters
	n_estimators, max_depth, max_features = 100, 10, "sqrt"

	# Default Linear SVC parameters
	C, loss, max_iter = 1, 'hinge', "500"

	for i in range (1,len(sys.argv),2):
		if sys.argv[i] == "-df":
			DF = sys.argv[i+1]
		if sys.argv[i] == '-save':
			SAVE = sys.argv[i+1]
		if sys.argv[i] == '-feat':
			FEAT = sys.argv[i+1]
		if sys.argv[i] == "-gs":
			GS = sys.argv[i+1]
		if sys.argv[i] == "-gs_score":
			gs_score = sys.argv[i+1]
		if sys.argv[i] == "-gs_n":
			gs_n = int(sys.argv[i+1])
		if sys.argv[i] == '-cl_train':
			CL_TRAIN = sys.argv[i+1].split(',')
		if sys.argv[i] == '-cl_apply':
			CL_APPLY = sys.arg[i+1]
		if sys.argv[i] == "-class":
			class_col = sys.argv[i+1]
		if sys.argv[i] == "-n":
			n = int(sys.argv[i+1])
		if sys.argv[i] == "-alg":
			ALG = sys.argv[i+1]
		if sys.argv[i] == "-criterion":
			criterion = sys.argv[i+1]
		if sys.argv[i] == "-n_jobs":
			n_jobs = int(sys.argv[i+1])
		if sys.argv[i] == "-cm":
			CM = sys.argv[i+1]

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

	# Set up dataframe of unknown instances that the final model will be applied to
	if CL_APPLY != 'none':
		df_unknowns = df[(df['Class'].isin(CL_APPLY))]

	# Remove classes that won't be included in the training (i.e. to ignore or unknowns)
	if CL_TRAIN != 'all':
		df = df[(df['Class'].isin(CL_TRAIN))]

	# Determine minimum class size (for making balanced datasets)
	min_size = (df.groupby('Class').size()).min()
	print('Balanced dataset will include %i instances of each class' % min_size)

	# Remove instances with NaN values
	df = df.dropna(axis=0)
	classes = df['Class'].unique()
	n_features = len(list(df)) - 1


	SAVE = SAVE + "_" + ALG



	####### Run parameter sweep using a grid search #######

	if GS == 'True' or GS == 'T' or GS == 'true' or GS == 't':
		params2use = ML.fun.GridSearch(df, SAVE, ALG, classes, min_size, gs_score, gs_n, n_jobs)
		
		if ALG == 'RF':
			max_depth, max_features, n_estimators = params2use
			print("Parameters selected: max_depth=%s, max_features=%s, n_estimators(trees)=%s" % (
				str(max_depth), str(max_features), str(n_estimators)))

		elif ALG == "SVC":
			C, loss, max_iter = params2use
			print("Parameters selected: C=%s, loss=%s, max_iter=%s" % (str(C), str(loss), str(max_iter)))


	

	####### Run ML models #######

	start_time = time.time()
	results = []

	for j in range(n):
		
		#Make balanced datasets
		df1 = pd.DataFrame(columns=list(df))
		for cl in classes:
			temp = df[df['Class'] == cl].sample(min_size, random_state=j)
			df1 = pd.concat([df1, temp])


		# Run ML algorithm on balanced datasets. Default = RF
		if ALG == "RF":
			parameters_used = [n_estimators, max_depth, max_features, criterion]
			result = ML.fun.RandomForest(df1, classes, n_estimators, max_depth, max_features, criterion, n_jobs, j)
		elif ALG == "SVC":
			parameters_used = [C, loss, max_iter]
			result = ML.fun.LinearSVC(df1, classes, C, loss, max_iter, n_jobs, j)
	
		results.append(result)

	print("ML Pipeline time: %f seconds" % (time.time() - start_time))



	####### Unpack ML results #######

	## Make empty dataframes
	conf_matrices = pd.DataFrame(columns = np.insert(arr = classes, obj = 0, values = 'Class'))
	accuracies = []
	f1_array = np.array([np.insert(arr = classes, obj = 0, values = 'Macro')])
	
	# For binary classifications also gather importance scores
	if len(df['Class'].unique()) == 2:
		imp = pd.DataFrame(index = names(df.drop(['Class'], axis=1) ))
		print(imp)

	for r in results:  # 0: confusion matrix, 1: accuracy, 2: macro_F1, 3: F1s, 4: Imp scores
		# Pull confusion matricies from each model
		cmatrix = pd.DataFrame(r[0], columns = classes) #, index = classes)
		cmatrix['Class'] = classes
		conf_matrices = pd.concat([conf_matrices, cmatrix])
		
		# Pull accuracies from each model
		accuracies.append(r[1])

		# Pull F-measures from each model
		f1_temp_array = np.insert(arr = r[3], obj = 0, values = r[2])
		f1_array = np.append(f1_array, [f1_temp_array], axis=0)

		# If binary classification pull importance scores from each model
		if len(df['Class'].unique()) == 2:
			imp = pd.concat([imp, pd.DataFrame(r[3])], axis=1, join_axes = [imp.index])
	
	cm_mean = conf_matrices.groupby('Class').mean()
	cm_mean.to_csv(SAVE + "_cm.csv")

	f1 = pd.DataFrame(f1_array)
	f1.columns = f1.iloc[0]
	f1 = f1[1:]
	f1.columns = [str(col) + '_F1' for col in f1.columns]
	
	# Plot confusion matrix (% class predicted as each class)
	cm_mean = conf_matrices.groupby('Class').mean()
	if CM == 'T' or CM == 'True' or CM == 'true' or CM == 't':
		cm_mean.to_csv(SAVE + "_cm.csv")
		done = ML.fun.Plot_ConMatrix(cm_mean, SAVE)
		print(done)
	

	# Calculate mean importance scores and sort
	#imp_df = pd.DataFrame(imp_array[1:,:], columns=imp_array[0,:], dtype = float)  
	#imp_mean_df = pd.DataFrame(imp_df.mean(axis=0, numeric_only=True), columns = ['importance'], index = imp_df.columns.values)
	#imp_mean_df = imp_mean_df.sort_values('importance', 0, ascending = False)
	#print("Top five most important features:")
	#print(imp_mean_df.head(5))
	
	#imp_out = SAVE + "_imp"
	#imp_mean_df.to_csv(imp_out, sep = "\t", index=True)

	# Calculate accuracy and f1 stats

	AC = np.mean(accuracies)
	AC_std = np.std(accuracies)
	MacF1 = f1['Macro_F1'].mean()
	MacF1_std = f1['Macro_F1'].std()
	print("\nML Results: \nAccuracy: %03f (+/- stdev %03f)\nF1 (macro): %03f (+/- stdev %03f)" % (
		AC, AC_std, MacF1, MacF1_std))

	# Write summary results to main RESULTS.txt file
	open("RESULTS.txt",'a').write('%s\t%i\t%i\t%i\t%.5f\t%.5f\t%.5f\t%.5f\n' % (
		SAVE, n_features, min_size, n, AC, AC_std, MacF1, MacF1_std))

	# Write detailed results file
	out = open(SAVE + "_results.csv", 'w')
	out.write('ID: %s\nAlgorithm: %s\nClasses trained on: %s\nNumber of features: %i\nMin class size: %i\nNumber of models: %i\n' % (
		SAVE, ALG, classes, n_features, min_size, n)) #, AC, AC_std, MacF1, MacF1_std))
	out.write('Grid Search Used: %s\nParameters used:%s\n' % (GS, parameters_used))
	out.write('\nModel Performance Measures\n\nAccuracy (std): %.5f\t%.5f\nMacro_F1 (std): %.5f\t%.5f\n' % (
		AC, AC_std, MacF1, MacF1_std))
	#str(n_estimators), str(max_depth), str(max_features), str(criterion), np.mean(acc), acc_stdv, acc_SE, np.mean(f1), f1_stdv, f1_SE))
	#print ('\nColumn Names in RF_RESULTS.txt output: Run_Name, #Pos_Examples, #Neg_Examples, #Features, #Random_df_Reps, n_estimators, max_depth, max_features, criterion, Accuracy, Accuracy_Stdev, Accuracy_SE, F1, F1_Stdev, F1_SE')
	for cl in classes:
		out.write('F1_%s (std): %.5f\t%.5f\n' % (cl, f1[cl+'_F1'].mean(), f1[cl+'_F1'].std()))

	out.write('\nConfusion Matrix:\n')
	cm_mean.to_csv(SAVE + "_results.csv", mode='a')

	
if __name__ == '__main__':
	main()


	