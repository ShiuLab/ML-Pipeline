"""
PURPOSE:
Machine learning classifications implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
	
	REQUIRED:
	-df       Feature & class dataframe for ML. 
						  See "example_binary.txt" for an example dataframe
	-alg      Available: RF, SVM (linear), SVMpoly, SVMrbf, GB, and Logistic Regression (LogReg)
	
	OPTIONAL:
	-y_name   Name of the column to predict (Default = Class)
	-cl_train List of classes to include in the training set. (Default = all)		
							If binary, first label = positive class.
	-apply    List of non-training class labels that the models should be applied to.
							Enter 'all' or a list (comma-delimit if >1)
	-ho       File with list of intances to holdout from feature selection
	-sep 			Set seperator for input data (Default = '\t')
	-pos      name of positive class for binary classifier. (Default = 1 or see -cl_train)
	-drop_na  T/F to drop rows with NAs
	-gs       T/F if grid search over parameter space is desired. (Default = True)
	-cv       # of cross-validation folds. (Default = 10)
	-n/-b     # of random balanced datasets to run (i.e. model replicates). (Default = 100)
	-min_size Number of instances to draw from each class. Default = size of smallest class
	-p        # of processors. (Default = 1, max for HPCC = 14)
	-tag      String for SAVE name and TAG column in RESULTS.txt output.
	-feat     Import file with subset of features to use. If invoked,-tag arg is recommended. Default: keep all features.
	-threshold_test   What model score to use for setting the optimal threshold (Default = F1. Also avilable: accuracy)
	-save     Adjust save name prefix. Default = [df]_[alg]_[tag (if used)]
							CAUTION: will overwrite!
	-short    Set to True to output only the median and std dev of prediction scores, default = full prediction scores
	-gs_full 	T/F Output full results from the grid search. (Default = F)
	-gs_reps  Number of replicates of the grid search (Default = 10)
	-gs_type  Full grid search or randomized search (Default = full, alt = random)
	-df2      File with class information. Use only if df contains the features but not the classes 
							* Need to specifiy what column in df2 is y using -y_name 

	PARAMETER OPTIONS
	* If you are not using the grid search, you can run default parameters or define your own
	RF & GB:
	-n_estimators		Grid Search [100, 500, 1000]
	-max_depth			Grid Search [3, 5, 10]
	-max_features		Grid Search [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None (i.e. all)]
	-learning_rate	(GB only!) Grid Search [0.001, 0.01, 0.1, 0.5, 1]
	SVM:
	-kernel  				Not in the grid search, to try different kernels, run with -alg SVM / SVMrbf / or SVMpoly
	-C 							Grid Search [0.001, 0.01, 0.1, 0.5, 1, 10, 50]
	-gamma 					(poly & rbf only!) Grid Search [np.logspace(-5,1,7)]
	-degree 				(poly only!) Grid Search [2,3,4]
	LogReg:
	-C 							Grid Search [0.001, 0.01, 0.1, 0.5, 1, 10, 50]
	-intercept_scaling Grid Search  [0.1, 0.5, 1, 2, 5, 10]
	-penalty 				Grid Search [l1, l2]
	
	PLOT OPTIONS:
	-cm       T/F Output the confusion matrix & confusion matrix figure (Default = False)
	-plots    T/F Output the ROC and PR curve plots for each model? (Default = False)
							These plots can be generated after for 1+ model using ML_plots.py

OUTPUT:
	-SAVE_imp           Importance scores for each feature
	-SAVE_GridSearch    Results from parameter sweep sorted by F1
	-RESULTS.txt     		Accumulates results from all ML runs done in a specific folder - use unique save names! XX = RF or SVC

"""

import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import ML_functions as ML

start_total_time = time.time()
def main():
	
	# Default code parameters
	n, FEAT, CL_TRAIN, apply_model, n_jobs =  100, 'all', 'all','none', 1,
	y_name, CM, POS, plots, cv_num = 'Class', 'False', 1, 'False', 10,
	TAG, SAVE, MIN_SIZE, short_scores =   '', '', '', False
	SEP, THRSHD_test, DF2 = '\t','F1', 'None'
	drop_na, ho = 'False', ''

	# Default parameters for Grid search
	GS, gs_score, GS_REPS, GS_TYPE, gs_full = 'F', 'roc_auc', 10, 'full', 'f'
	
	# Default Random Forest and Gradient Boosting parameters
	n_estimators, max_depth, max_features, learning_rate = 500, 10, "sqrt", 0.1
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'linear', 1, 2, 1, 'hinge', "500"
	
	# Default Logistic Regression paramemter
	penalty, C, intercept_scaling = 'l2', 1.0, 1.0
	
	for i in range (1,len(sys.argv),2):
		if sys.argv[i].lower() == "-df":
			DF = sys.argv[i+1]
		if sys.argv[i].lower() == "-sep":
			SEP = sys.argv[i+1]
		elif sys.argv[i].lower() == '-save':
			SAVE = sys.argv[i+1]
		elif sys.argv[i].lower() == '-feat':
			FEAT = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs":
			GS = sys.argv[i+1]
		elif sys.argv[i].lower() == "-kernel":
			kernel = sys.argv[i+1]
		elif sys.argv[i].lower() == "-C":
			C = sys.argv[i+1]
		elif sys.argv[i].lower() == "-degree":
			degree = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gamma":
			gamma = sys.argv[i+1]
		elif sys.argv[i].lower() == "-loss":
			loss = sys.argv[i+1]
		elif sys.argv[i].lower() == "-penalty":
			penalty = sys.argv[i+1]
		elif sys.argv[i].lower() == "-intercept_scaling":
			intercept_scaling = sys.argv[i+1]
		elif sys.argv[i].lower() == "-n_estimators":
			n_estimators = sys.argv[i+1]
		elif sys.argv[i].lower() == "-max_depth":
			max_depth = sys.argv[i+1]
		elif sys.argv[i].lower() == "-max_features":
			max_features = sys.argv[i+1]
		elif sys.argv[i].lower() == "-learning_rate":
			learning_rate = sys.argv[i+1]
		elif sys.argv[i].lower() == "-max_iter":
			max_iter = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_score":
			gs_score = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_reps":
			GS_REPS = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-gs_type":
			GS_TYPE = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_full":
			gs_full = sys.argv[i+1]
		elif sys.argv[i].lower() == '-cl_train':
			CL_TRAIN = sys.argv[i+1].strip().split(',')
			POS = CL_TRAIN[0]
		elif sys.argv[i].lower() == '-apply':
			apply_model = sys.argv[i+1]
			if apply_model != "all":
				apply_model = sys.argv[i+1].split(',')
		elif sys.argv[i].lower() == "-y_name":
			y_name = sys.argv[i+1]
		elif sys.argv[i].lower() == "-n" or sys.argv[i].lower() == "-b":
			n = int(sys.argv[i+1])
		elif sys.argv[i].lower() == '-ho':
			ho = sys.argv[i+1]
		elif sys.argv[i].lower() == "-drop_na":
			drop_na = sys.argv[i+1]
		elif sys.argv[i].lower() == "-min_size":
			MIN_SIZE = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-alg":
			ALG = sys.argv[i+1]
		elif sys.argv[i].lower() == "-cv":
			cv_num = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-cm":
			CM = sys.argv[i+1]
		elif sys.argv[i].lower() == "-plots":
			plots = sys.argv[i+1]
		elif sys.argv[i].lower() == "-pos":
			POS = sys.argv[i+1]
		elif sys.argv[i].lower() == "-tag":
			TAG = sys.argv[i+1]
		elif sys.argv[i].lower() == "-threshold_test":
			THRSHD_test = sys.argv[i+1]
		elif sys.argv[i].lower() == "-df2":
			DF2 = sys.argv[i+1]
		elif sys.argv[i].lower() == "-n_jobs" or sys.argv[i].lower() == "-p":
			n_jobs = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-short":
			scores_len = sys.argv[i+1]
			if scores_len.lower() == "true" or scores_len.lower() == "t":
				short_scores = True

	if len(sys.argv) <= 1:
		print(__doc__)
		exit()
	
	####### Load Dataframe & Pre-process #######
	
	df = pd.read_csv(DF, sep=SEP, index_col = 0)
	
	# If features  and class info are in separate files, merge them: 
	if DF2 != 'None':
		start_dim = df.shape
		df_class = pd.read_csv(DF2, sep=SEP, index_col = 0)
		df = pd.concat([df_class[y_name], df], axis=1, join='inner')
		print('Merging the feature & class dataframes changed the dimensions from %s to %s (instance, features).' 
			% (str(start_dim), str(df.shape)))

	# Specify class column - default = Class
	if y_name != 'Class':
		df = df.rename(columns = {y_name:'Class'})
	
	# Filter out features not in feat file given - default: keep all
	if FEAT != 'all':
		print('Using subset of features from: %s' % FEAT)
		with open(FEAT) as f:
			features = f.read().strip().splitlines()
			features = ['Class'] + features
		df = df.loc[:,features]

	# Check for Nas
	if df.isnull().values.any() == True:
		if drop_na.lower() == 't' or drop_na.lower() == 'true':
			start_dim = df.shape
			df = df.dropna(axis=0)
			print('Dropping rows with NA values changed the dimensions from %s to %s.' 
				% (str(start_dim), str(df.shape)))
		else:
			print(df.columns[df.isna().any()].tolist())
			print('There are Na values in your dataframe.\n Impute them or add -drop_na True to remove rows with nas' )
			quit()
	
	
	# Set up dataframe of unknown instances that the final models will be applied to
	if CL_TRAIN != 'all' and apply_model != 'none':
		apply_unk = True
		# if apply to all, select all instances with a class not in CL_TRAIN
		if apply_model == 'all':
			df_unknowns = df[(~df['Class'].isin(CL_TRAIN))]
		else: # apply to specified classes
			df_unknowns = df[(df['Class'].isin(apply_model))]
	else:
		apply_unk = False
		df_unknowns = ''
	
	# Remove classes that won't be included in the training (e.g. unknowns)
	if CL_TRAIN != 'all':
		df = df[(df['Class'].isin(CL_TRAIN))]

	# Set up dataframe of holdout instances that the final models will be applied to
	if ho !='':
		df_all = df.copy()
		print('Removing holdout instances to apply model on later...')
		with open(ho) as ho_file:
			ho_instances = ho_file.read().splitlines()
		try:
			ho_df = df.loc[ho_instances, :]
			df = df.drop(ho_instances)
		except:
			ho_instances = [int(x) for x in ho_instances]
			ho_df = df.loc[ho_instances, :]
			df = df.drop(ho_instances)

	else:
		ho_df = 'None'
		ho_instances = 'None'
		df_all = df.copy()
	


	# Generate training classes list. If binary, establish POS and NEG classes. 
	# Set grid search scoring: roc_auc for binary, f1_macro for multiclass
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
	
	
	# Determine minimum class size (for making balanced datasets)
	if MIN_SIZE == '':
		min_size = (df.groupby('Class').size()).min() - 1
	else:
		min_size = int(MIN_SIZE)
	
	# Define save name if not specified using -save
	if SAVE == "":
		if TAG == "":
			SAVE = DF + "_" + ALG
		else:
			SAVE = DF + "_" + ALG + "_" + TAG
	
	# Normalize data frame for SVM algorithms
	if ALG == "SVM" or ALG == "SVMpoly" or ALG == "SVMrbf":
		from sklearn import preprocessing
		y = df['Class']
		X = df.drop(['Class'], axis=1)
		min_max_scaler = preprocessing.MinMaxScaler()
		X_scaled = min_max_scaler.fit_transform(X)
		df = pd.DataFrame(X_scaled, columns = X.columns, index = X.index)
		df.insert(loc=0, column = 'Class', value = y)
	
	
	print("Snapshot of data being used:")
	print(df.head())
	print("CLASSES:",classes)
	print("POS:",POS,type(POS))
	print("NEG:",NEG,type(NEG))
	print('Balanced dataset will include %i instances of each class' % min_size)
	n_features = len(list(df)) - 1
	
	####### Run parameter sweep using a grid search #######
	
	if GS.lower() == 'true' or GS.lower() == 't':
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use, balanced_ids, param_names = ML.fun.GridSearch(df, SAVE, ALG, classes, min_size, gs_score, n, cv_num, n_jobs, GS_REPS, GS_TYPE, POS, NEG, gs_full)
		
		# Print results from grid search
		if ALG.lower() == 'rf':
			max_depth, max_features, n_estimators = params2use
			print("Parameters selected: max_depth=%s, max_features=%s, n_estimators=%s" % (str(max_depth), str(max_features), str(n_estimators)))
	
		elif ALG.lower() == 'svm':
			C, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s" % (str(kernel), str(C)))
		
		elif ALG.lower() == "svmpoly":
			C, degree, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(C), str(degree), str(gamma)))
		
		elif ALG.lower() == "svmrbf":
			C, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, gamma=%s" % (str(kernel), str(C), str(gamma)))
		
		elif ALG.lower() == "logreg":
			C, intercept_scaling, penalty = params2use
			print("Parameters selected: penalty=%s, C=%s, intercept_scaling=%s" % (str(penalty), str(C), str(intercept_scaling)))

		elif ALG.lower() == "gb":
			learning_rate, max_depth, max_features, n_estimators = params2use
			print("Parameters selected: learning rate=%s, max_features=%s, max_depth=%s, n_estimators=%s" % (str(learning_rate), str(max_features), str(max_depth), str(n_estimators)))
	
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	
	else:
		print('Not running grid search. Using default or given parameters instead')
		balanced_ids = ML.fun.EstablishBalanced(df,classes,int(min_size),n)
	
	bal_id = pd.DataFrame(balanced_ids)
	bal_id.to_csv(SAVE + '_BalancedIDs', index=False, header=False,sep="\t")

	 
	####### Run ML models #######
	start_time = time.time()
	print("\n\n===>  ML Pipeline started  <===")

	results = []
	results_ho = []
	df_proba = pd.DataFrame(data=df_all['Class'], index=df_all.index, columns=['Class'])
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
		if ALG.lower() == "rf":
			parameters_used = [n_estimators, max_depth, max_features]
			clf = ML.fun.DefineClf_RandomForest(n_estimators,max_depth,max_features,j,n_jobs)
		elif ALG.lower() == "svm" or ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
			parameters_used = [C, degree, gamma, kernel]
			clf = ML.fun.DefineClf_SVM(kernel,C,degree,gamma,j)
		elif ALG.lower() == "logreg":
			parameters_used = [C, intercept_scaling, penalty]
			clf = ML.fun.DefineClf_LogReg(penalty, C, intercept_scaling)
		elif ALG.lower() == "gb":
			parameters_used = [learning_rate, max_features, max_depth]
			clf = ML.fun.DefineClf_GB(learning_rate, max_features, max_depth, n_jobs, j)
		
		# Run ML algorithm on balanced datasets.
		if ho !='':
			result,current_scores,result_ho = ML.fun.BuildModel_Apply_Performance(df1, clf, cv_num, df_notSel, apply_unk, df_unknowns, ho_df, classes, POS, NEG, j, ALG,THRSHD_test)
			results_ho.append(result_ho)	
		else:
			result,current_scores = ML.fun.BuildModel_Apply_Performance(df1, clf, cv_num, df_notSel, apply_unk, df_unknowns, ho_df, classes, POS, NEG, j, ALG,THRSHD_test)

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
			if r['importances'] != 'na':
				if ALG.lower() == 'rf' or ALG.lower() == 'gb':
					imp[count] = r['importances']
				else:
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
			f1_temp_array = np.insert(arr = r['f1_MC'], obj = 0, values = r['macro_f1'])
			f1_array = np.append(f1_array, [f1_temp_array], axis=0)

	# Output for both binary and multiclass predictions
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	
	# Plot confusion matrix (% class predicted as each class) based on balanced dataframes
	cm_mean = conf_matrices.groupby('Class').mean()
	if CM.lower() == 'true' or CM.lower() == 't':
		cm_mean.to_csv(SAVE + "_cm.csv",sep="\t")
		done = ML.fun.Plot_ConMatrix(cm_mean, SAVE)

	# Unpack results from hold out
	if ho !='':
		AucRoc_ho_array = []
		AucPRc_ho_array = []
		accuracies_ho = []
		f1_array_ho = np.array([np.insert(arr = classes.astype(np.str), obj = 0, values = 'M')])
		
		for r_ho in results_ho:			
			# For binary predictions
			if 'AucRoc' in r_ho:
				AucRoc_ho_array.append(r_ho['AucRoc'])
			if 'AucPRc' in r_ho:
				AucPRc_ho_array.append(r_ho['AucPRc'])

			# For Multi-class predictions
			if 'accuracy' in r_ho:
				accuracies_ho.append(r['accuracy'])
			if 'macro_f1' in r_ho:
				f1_temp_ho_array = np.insert(arr = r_ho['f1_MC'], obj = 0, values = r_ho['macro_f1'])
				f1_array_ho = np.append(f1_array_ho, [f1_temp_ho_array], axis=0)
		

###### Multiclass Specific Output ######
	if len(classes) > 2:
		
		# For each class, get the median and std score
		summary_cols = []
		mc_score_columns = []
		keep_for_summary = ['Class', 'Prediction']
		for class_nm in reversed(classes): # get std
			class_proba_cols = [c for c in df_proba.columns if c.startswith(class_nm+'_score_')]
			df_proba.insert(loc=1, column = class_nm+'_score_stdev', value = df_proba[class_proba_cols].std(axis=1))
			summary_cols.insert(0,class_nm+'_score_stdev')

		for class_nm in reversed(classes): # get median
			summary_cols.insert(0,class_nm+'_score_Median')
			mc_score_columns.append(class_nm+'_score_Median')
			keep_for_summary.append(class_nm + '_score_Median')
			class_proba_cols = [c for c in df_proba.columns if c.startswith(class_nm+'_score_')]
			df_proba.insert(loc=1, column = class_nm+'_score_Median', value = df_proba[class_proba_cols].median(axis=1))
			
		
		# Find the max mc_score and set to Prediction column (remove the _score_Median string)
		df_proba.insert(loc=1, column = 'Prediction', value = df_proba[mc_score_columns].idxmax(axis=1))
		df_proba['Prediction'] = df_proba.Prediction.str.replace('_score_Median','')

		
		# Count the number of times an instance of class x is predicted as class y 		
		summary_df_proba = df_proba[['Class', 'Prediction',class_nm + '_score_Median']].groupby(['Class', 'Prediction']).agg('count').unstack(level=1)
		summary_df_proba.columns = summary_df_proba.columns.droplevel()

		# Check to make sure each class was predicted at least once
		for cl in classes:
			if cl not in list(summary_df_proba):
				print('No instances were predicted as class: %s' % cl)
				summary_df_proba[cl] = 0
		summary_df_proba['n_total'] = summary_df_proba[classes].sum(axis=1)


		for class_nm in classes:
			summary_df_proba[str(class_nm) + '_perc'] = summary_df_proba[class_nm]/summary_df_proba['n_total']


		scores_file = SAVE + "_scores.txt"
		out_scores = open(scores_file,"w")
		if short_scores == True:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba[["Class"]+summary_cols],sep="\t").strip()+"\n")
		else:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba,sep="\t").strip()+"\n")
		out_scores.close()
		
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

		print("\nML Results: \nAccuracy: %03f (+/- stdev %03f)\nF1 (macro): %03f (+/- stdev %03f)\n" % (
		AC, AC_std, MacF1, MacF1_std))

		# Unpack results from hold out
		if ho !='':
			f1_ho = pd.DataFrame(f1_array_ho)
			f1_ho.columns = f1_ho.iloc[0]
			f1_ho = f1_ho[1:]
			f1_ho.columns = [str(col) + '_F1' for col in f1_ho.columns]
			f1_ho = f1_ho.astype(float)	
			AC_ho = np.mean(accuracies_ho)
			AC_std_ho = np.std(accuracies_ho)
			MacF1_ho = f1_ho['M_F1'].mean()
			MacF1_std_ho = f1_ho['M_F1'].std()
			print("\nML Results from Hold Out Validation: \nAccuracy: %03f (+/- stdev %03f)\nF1 (macro): %03f (+/- stdev %03f)\n" % (
				AC_ho, AC_std_ho, MacF1_ho, MacF1_std_ho))

		# Save detailed results file 
		with open(SAVE + "_results.txt", 'w') as out:
			out.write('%s\nID: %s\nTag: %s\nAlgorithm: %s\nTrained on classes: %s\nApplied to: %s\nNumber of features: %i\n' % (
				timestamp, SAVE, TAG, ALG, classes, apply_model, n_features))
			out.write('Min class size: %i\nCV folds: %i\nNumber of balanced datasets: %i\nGrid Search Used: %s\nParameters used:%s\n' % (
				min_size, cv_num, n, GS, parameters_used))

			out.write('\nMetric\tMean\tSD\nAccuracy\t%05f\t%05f\nF1_macro\t%05f\t%05f\n' % (AC, AC_std, MacF1, MacF1_std))
			for cla in f1.columns:
				if 'M_F1' not in cla:
					out.write('%s\t%05f\t%05f\n' % (cla, np.mean(f1[cla]), np.std(f1[cla])))
		
			out.write('\nMean Balanced Confusion Matrix:\n')
			cm_mean.to_csv(out, mode='a', sep='\t')
			out.write('\n\nCount and percent of instances of each class (row) predicted as a class (col):\n')
			summary_df_proba.to_csv(out, mode='a', header=True, sep='\t')
			
			# Add results from hold out
			if ho !='':
				out.write('\n\nResults from the hold out validation set\n')
				out.write('HO Accuracy\t%05f +/-%05f\nHO F1_macro\t%05f +/-%05f' % (AC_ho, AC_std_ho, MacF1_ho, MacF1_std_ho))

###### Binary Prediction Output ######
	else: 
		# Get AUC for ROC and PR curve (mean, sd, se)
		ROC = [np.mean(AucRoc_array), np.std(AucRoc_array), np.std(AucRoc_array)/np.sqrt(len(AucRoc_array))]
		PRc = [np.mean(AucPRc_array), np.std(AucPRc_array), np.std(AucPRc_array)/np.sqrt(len(AucPRc_array))]
		if ho !='':
			ROC_ho = [np.mean(AucRoc_ho_array), np.std(AucRoc_ho_array), np.std(AucRoc_ho_array)/np.sqrt(len(AucRoc_ho_array))]
			PRc_ho = [np.mean(AucPRc_ho_array), np.std(AucPRc_ho_array), np.std(AucPRc_ho_array)/np.sqrt(len(AucPRc_ho_array))]
		else:
			ROC_ho = ['na', 'na', 'na']
			PRc_ho = ['na', 'na', 'na']

		# Find mean threshold
		final_threshold = round(np.mean(threshold_array),2)

		# Determine final prediction call - using the final_threshold on the mean predicted probability.
		proba_columns = [c for c in df_proba.columns if c.startswith('score_')]
		
		df_proba.insert(loc=1, column = 'Median', value = df_proba[proba_columns].median(axis=1))
		df_proba.insert(loc=1, column = 'Mean', value = df_proba[proba_columns].mean(axis=1))
		df_proba.insert(loc=2, column = 'stdev', value = df_proba[proba_columns].std(axis=1))
		Pred_name =  'Predicted_' + str(final_threshold)
		df_proba.insert(loc=3, column = Pred_name, value = df_proba['Class'])
		df_proba[Pred_name] = np.where(df_proba['Mean'] >= final_threshold, POS,NEG)
		

		# Summarize % of each class predicted as POS and NEG		
		summary_df_proba = df_proba[['Class', Pred_name, 'Mean']].groupby(['Class', Pred_name]).agg('count').unstack(level=1)
		summary_df_proba.columns = summary_df_proba.columns.droplevel()
		try:
			summary_df_proba['n_total'] = summary_df_proba[POS] + summary_df_proba[NEG]
			summary_df_proba[str(NEG) + '_perc'] = summary_df_proba[NEG]/summary_df_proba['n_total']
		except:
			summary_df_proba['n_total'] = summary_df_proba[POS]
			summary_df_proba[str(NEG) + '_perc'] = 0
			print('Warning: No instances were classified as negative!')	
		summary_df_proba[str(POS) + '_perc'] = summary_df_proba[POS]/summary_df_proba['n_total']
		
		

		scores_file = SAVE + "_scores.txt"
		out_scores = open(scores_file,"w")
		if short_scores == True:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba[["Class","Mean","Median","stdev",Pred_name]],sep="\t").strip()+"\n")
		else:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba,sep="\t").strip()+"\n")
		out_scores.close()
		# df_proba.to_csv(SAVE + "_scores.txt", sep="\t")
		
		
		# Get model preformance scores using final_threshold
		if ho !='':
			TP,TN,FP,FN,TPR,FPR,FNR,Pr,Ac,F1,Pr_ho,Ac_ho,F1_ho = ML.fun.Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, POS, NEG, ho_instances)
		else:
			TP,TN,FP,FN,TPR,FPR,FNR,Pr,Ac,F1 = ML.fun.Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, POS, NEG, ho_instances)
			Pr_ho, Ac_ho, F1_ho = 0, 0, 0

		# Plot ROC & PR curves
		if plots.lower() == 'true' or plots.lower() == 't':
			print("\nGenerating ROC & PR curves")
			pr = ML.fun.Plots(df_proba, balanced_ids, ROC, PRc, POS, NEG, n, SAVE)
		
		# Export importance scores
		try:
			imp['mean_imp'] = imp.mean(axis=1)
			imp = imp.sort_values('mean_imp', 0, ascending = False)
			imp_out = SAVE + "_imp"
			imp['mean_imp'].to_csv(imp_out, sep = "\t", index=True)
			# imp['mean_imp'].to_csv(imp_out, sep = ",", index=True)
		except:
			pass

		run_time = time.time() - start_total_time
		
		# Save to summary RESULTS file with all models run from the same directory
		
		if not os.path.isfile('RESULTS.txt'):
			out2 = open('RESULTS.txt', 'a')
			out2.write('DateTime\tRunTime\tID\tTag\tAlg\tClasses\tFeatureNum\tBalancedSize\tCVfold\tBalancedRuns\tAUCROC\tAUCROC_sd\tAUCROC_se\t')
			out2.write('AUCPRc\tAUCPRc_sd\tAUCPRc_se\tAc\tAc_sd\tAc_se\tF1\tF1_sd\tF1_se\tPr\tPr_sd\tPr_se\tTPR\tTPR_sd\tTPR_se\t')
			out2.write('FPR\tFPR_sd\tFPR_se\tFNR\tFNR_sd\tFNR_se\tTP\tTP_sd\tTP_se\tTN\tTN_sd\tTN_se\tFP\tFP_sd\tFP_se\t')
			out2.write('FN\tFN_sd\tFN_se\tPr_ho\tAc_ho\tF1_ho\tAUCROC_ho\tAUCROC_ho_sd\tAUCROC_ho_se\tAUCPRc_ho\tAUCPRc_ho_sd\tAUCPRc_ho_se')
			out2.close()
		out2 = open('RESULTS.txt', 'a')
		out2.write('\n%s\t%s\t%s\t%s\t%s\t%s\t%i\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%05f\t%05f\t%05f\t%s\t%s' % (
			str(timestamp), run_time, SAVE, TAG, ALG, [POS,NEG], n_features, min_size, cv_num , n, 
			'\t'.join(str(x) for x in ROC), '\t'.join(str(x) for x in PRc), '\t'.join(str(x) for x in Ac), '\t'.join(str(x) for x in F1),
			'\t'.join(str(x) for x in Pr), '\t'.join(str(x) for x in TPR), '\t'.join(str(x) for x in FPR),
			'\t'.join(str(x) for x in FNR), '\t'.join(str(x) for x in TP), '\t'.join(str(x) for x in TN),
			'\t'.join(str(x) for x in FP), '\t'.join(str(x) for x in FN), Pr_ho, Ac_ho, F1_ho,
			'\t'.join(str(x) for x in ROC_ho),'\t'.join(str(x) for x in PRc_ho)))
			 
			

		# Save detailed results file 
		with open(SAVE + "_results.txt", 'w') as out:
			out.write('%s\nID: %s\nTag: %s\nAlgorithm: %s\nTrained on classes: %s\nApplied to: %s\nNumber of features: %i\n' % (
				timestamp, SAVE, TAG, ALG, classes, apply_model, n_features))
			out.write('Min class size: %i\nCV folds: %i\nNumber of balanced datasets: %i\nGrid Search Used: %s\nParameters used:%s\n' % (
				min_size, cv_num, n, GS, parameters_used))
			
			out.write('\nPrediction threshold: %s\n'%final_threshold)
			out.write('\nMetric\tMean\tSD\tSE\n')
			out.write('AucROC\t%s\nAucPRc\t%s\nAccuracy\t%s\nF1\t%s\nPrecision\t%s\nTPR\t%s\nFPR\t%s\nFNR\t%s\n' % (
				'\t'.join(str(x) for x in ROC),'\t'.join(str(x) for x in PRc), '\t'.join(str(x) for x in Ac), '\t'.join(str(x) for x in F1),
				'\t'.join(str(x) for x in Pr), '\t'.join(str(x) for x in TPR), '\t'.join(str(x) for x in FPR), '\t'.join(str(x) for x in FNR)))
			out.write('TP\t%s\nTN\t%s\nFP\t%s\nFN\t%s\n' % (
				'\t'.join(str(x) for x in TP), '\t'.join(str(x) for x in TN), '\t'.join(str(x) for x in FP), '\t'.join(str(x) for x in FN)))
			out.write('\n\nMean Balanced Confusion Matrix:\n')
			cm_mean.to_csv(out, mode='a', sep='\t')
			out.write('\n\nCount and percent of instances of each class (row) predicted as a class (col):\n')
			summary_df_proba.to_csv(out, mode='a', header=True, sep='\t')
			if ho !='':
				out.write('\n\nResults from the hold out validation set\n')
				out.write('HO Precision\t%05f\nHO Accuracy\t%05f\nHO F1\t%05f\n' % (Pr_ho, Ac_ho, F1_ho))
				out.write('HO AucROC\t%s\nHO AucPRc\t%s\n' % ('\t'.join(str(x) for x in ROC_ho), '\t'.join(str(x) for x in PRc_ho)))



		print("\n\n===>  ML Results  <===")
		print("Testing Set Scores\nAccuracy: %03f (+/- stdev %03f)\nF1: %03f (+/- stdev %03f)\nAUC-ROC: %03f (+/- stdev %03f)\nAUC-PRC: %03f (+/- stdev %03f)" % (
			Ac[0], Ac[1], F1[0], F1[1], ROC[0], ROC[1], PRc[0], PRc[1]))
		if ho !='':
			print('\n\nHold Out Set Scores:\nPrecision: %03f\nAccuracy: %03f\nF1: %03f\nAUC-ROC: %03f (+/- stdev %03f)\nAUC-PRC: %03f (+/- stdev %03f)' % (
				Pr_ho, Ac_ho, F1_ho, ROC_ho[0], ROC_ho[1], PRc_ho[0], PRc_ho[1]))
		print('finished!')	
	

if __name__ == '__main__':
	main()
