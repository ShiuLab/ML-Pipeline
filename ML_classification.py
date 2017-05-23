"""
PURPOSE:
Machine learning classifications implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
	
	REQUIRED ALL:
	-df       Feature & class dataframe for ML. See "" for an example dataframe
	-alg      Available: RF, SVM (LinearSVC)
	
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
	-cm       T/F - Do you want to output the confusion matrix & confusion matrix figure?

OUTPUT:
	-SAVE_imp           Importance scores for each feature
	-SAVE_GridSearch    Results from parameter sweep sorted by F1
	-RESULTS_XX.txt     Accumulates results from all ML runs done in a specific folder - use unique save names! XX = RF or SVC
"""
import sys
import pandas as pd
import numpy as np
import time

import ML_functions as ML

def main():
	
	# Default code parameters
	n, FEAT, CL_TRAIN, apply, n_jobs, class_col, CM, POS = 50, 'all', 'all','none', 28, 'Class', 'False', 1
	
	# Default parameters for Grid search
	GS, gs_score, gs_n, cv_num = 'F', 'roc_auc', 25, 10
	
	# Default Random Forest parameters
	n_estimators, max_depth, max_features = 100, 10, "sqrt"
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'linear', 1, 2, 1, 'hinge', "500"
<<<<<<< HEAD

=======
	# 'gamma': np.logspace(-9,3,13)
	
>>>>>>> 1c2cb72d65621ac34999b2ed437b4e0c56eb4848
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
		if sys.argv[i] == "-pos":
			POS = sys.argv[i+1]
	
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
	
	print("Snapshot of data being used:")
	print(df.head())
	
	# Remove instances with NaN or NA values
	df = df.replace("?",np.nan)
	df = df.dropna(axis=0)
	
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
	
	print("CLASSES:",classes)
	print("POS:",POS,type(POS))
	print("NEG:",NEG,type(NEG))
	
	n_features = len(df['Class'].unique()) - 1
	
	# Determine minimum class size (for making balanced datasets)
	min_size = (df.groupby('Class').size()).min() - 1
	print('Balanced dataset will include %i instances of each class' % min_size)
	
	SAVE = SAVE + "_" + ALG
	
	####### Run parameter sweep using a grid search #######
	
	if GS.lower() == 'true' or GS.lower() == 't':
		
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use,balanced_ids = ML.fun.GridSearch(df, SAVE, ALG, classes, min_size, gs_score, n, cv_num, n_jobs, POS, NEG)
		
		if ALG == 'RF':
			max_depth, max_features, n_estimators = params2use
			print("Parameters selected: max_depth=%s, max_features=%s, n_estimators(trees)=%s" % (
				str(max_depth), str(max_features), str(n_estimators)))
	
		elif ALG == "SVM":
			C, degree, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(C), str(degree), str(gamma)))
		
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	else:
		balanced_ids = ML.fun.EstablishBalanced(df,classes,min_size,n)
	
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
			# result = ML.fun.RandomForest_ManCV(df1, classes, POS, cv_num, n_estimators, max_depth, max_features, n_jobs, j, df_notSel, apply_unk, df_unknowns)
		elif ALG == "SVM":
			parameters_used = [C, degree, gamma, kernel]
			clf = ML.fun.DefineClf_LinearSVC(kernel,C,degree,gamma,j)
			# result = ML.fun.LinearSVC(df1, classes, POS, C, degree, gamma, kernel, n_jobs, j)
		
		# Run ML algorithm on balanced datasets.
		result,current_scores = ML.fun.BuildModel_Apply_Performance(df1, clf, cv_num, df_notSel, apply_unk, df_unknowns, classes, POS, NEG, j)
		results.append(result)
		df_proba = pd.concat([df_proba,current_scores],axis = 1)
	
	print("ML Pipeline time: %f seconds" % (time.time() - start_time))
	
	print(df_proba.head())
	# sys.exit()
	
	####### Unpack ML results #######
	
	## Make empty dataframes
	conf_matrices = pd.DataFrame(columns = np.insert(arr = classes.astype(np.str), obj = 0, values = 'Class'))
	imp = pd.DataFrame(index = list(df.drop(['Class'], axis=1)))
	accuracies = []
	auc_array = []
	f1_array = np.array([np.insert(arr = classes.astype(np.str), obj = 0, values = 'M')])
	
	count = 0
	for r in results:
		count += 1
		if 'cm' in r:
			cmatrix = pd.DataFrame(r['cm'], columns = classes) #, index = classes)
			cmatrix['Class'] = classes
			conf_matrices = pd.concat([conf_matrices, cmatrix])
		
		if 'accuracy' in r:
			accuracies.append(r['accuracy'])
		
		if 'macro_f1' in r:
			f1_temp_array = np.insert(arr = r['f1'], obj = 0, values = r['macro_f1'])
			f1_array = np.append(f1_array, [f1_temp_array], axis=0)
		
		if 'AucRoc' in r:
			auc_array.append(r['AucRoc'])
		
		if 'importances' in r:
			imp[count] = r['importances'][0]
	
	cm_mean = conf_matrices.groupby('Class').mean()
	cm_mean.to_csv(SAVE + "_cm.csv")
	
	f1 = pd.DataFrame(f1_array)
	f1.columns = f1.iloc[0]
	f1 = f1[1:]
	f1.columns = [str(col) + '_F1' for col in f1.columns]
	f1 = f1.astype(float)
	
	
	# Plot confusion matrix (% class predicted as each class)
	cm_mean = conf_matrices.groupby('Class').mean()
	if CM == 'T' or CM == 'True' or CM == 'true' or CM == 't':
		cm_mean.to_csv(SAVE + "_cm.csv")
		done = ML.fun.Plot_ConMatrix(cm_mean, SAVE)
	
	if len(classes) == 2:
		try:
			imp['mean_imp'] = imp.mean(axis=1)
			imp = imp.sort_values('mean_imp', 0, ascending = False)
			imp_out = SAVE + "_imp.csv"
			imp['mean_imp'].to_csv(imp_out, sep = ",", index=True)
		except:
			pass
	
	# Calculate accuracy and f1 stats
	AC = np.mean(accuracies)
	AC_std = np.std(accuracies)
	MacF1 = f1['M_F1'].mean()
	MacF1_std = f1['M_F1'].std()
	try:
		MacAUC = np.mean(auc_array)
		MacAUC_std = np.std(auc_array)
	except:
		MacAUC = MacAUC_std = 'Na'
	
	print("\nML Results: \nAccuracy: %03f (+/- stdev %03f)\nF1 (macro): %03f (+/- stdev %03f)\nAUC-ROC (macro): %03f (+/- stdev %03f)" % (
		AC, AC_std, MacF1, MacF1_std, MacAUC, MacAUC_std))
	
	# Write summary results to main RESULTS.txt file
	open("RESULTS.txt",'a').write('%s\t%i\t%i\t%i\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n' % (
		SAVE, n_features, min_size, n, AC, AC_std, MacAUC, MacAUC_std, MacF1, MacF1_std))
	
	# Write detailed results file
	out = open(SAVE + "_results.txt", 'w')
	out.write('ID: %s\nAlgorithm: %s\nClasses trained on: %s\nNumber of features: %i\nMin class size: %i\nNumber of models: %i\n' % (
		SAVE, ALG, classes, n_features, min_size, n)) #, AC, AC_std, MacF1, MacF1_std))
	out.write('Grid Search Used: %s\nParameters used:%s\n' % (GS, parameters_used))
	out.write('\nModel Performance Measures\n\nAccuracy (std): %.5f\t%.5f\nAUC-ROC (std): %.5f\t%.5f\nMacro_F1 (std): %.5f\t%.5f\n' % (
		AC, AC_std, MacAUC, MacAUC_std, MacF1, MacF1_std))
	
	for cl in classes:
		out.write('F1_%s (std): %.5f\t%.5f\n' % (cl, f1[str(cl)+'_F1'].mean(), f1[str(cl)+'_F1'].std()))
	out.write('\nConfusion Matrix:\n')
	out.close()
	out = open(SAVE + "_results.txt", 'a')
	cm_mean.to_csv(out, sep='\t')
	
	out_scores = open(SAVE + "_scores.txt","w")
	out_scores.write("ID\t"+pd.DataFrame.to_csv(df_proba,sep="\t"))
	out_scores.close()

if __name__ == '__main__':
	main()
