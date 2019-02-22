import sys, os, time, argparse
import pandas as pd
import numpy as np
from datetime import datetime
import ML_functions as ML
start_total_time = time.time()

def main():
	
	########################
	### Parse Input Args ###
	########################
	parser = argparse.ArgumentParser(
		description='Machine learning classification pipeline using tools from Scikit-Learn. \
			See README.md for more information about the pipeline and preprocessing/post-analysis tools \
			available. All required packages \
			are available on MSUs HPCC: "export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH"',
		epilog='https://github.com/ShiuLab')
	
	### Input arguments ###
	# Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-df', help='Feature & class dataframe for ML, (example: example_binary.txt) ', required=True)
	req_group.add_argument('-alg', help='ML Algorithm to run (RF, SVM, SVMpoly, SVMrbf, GB, LogReg))', required=True)
	# Optional
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-df2', help='Class data (if not in -df). Need to provide -y_name', default='')
	inp_group.add_argument('-sep', help='Deliminator', default='\t')
	inp_group.add_argument('-y_name', help='Name of column in Y_file to predict', default='Class')
	inp_group.add_argument('-ho', help='File with testing (i.e. holdout) lines', default='')
	inp_group.add_argument('-feat', help='File with list of features (from x) to include', default='all')
	
	# Model behavior 
	pipln_group = parser.add_argument_group(title='CONTROL PIPELINE BEHAVIOR')
	pipln_group.add_argument('-cl_train', help='Classes to include in training. If binary, first listed = pos.', default='all')
	pipln_group.add_argument('-pos', help='Name of positive class for binary classifier (or from -cl_train)', default=1)
	pipln_group.add_argument('-apply', help='all or list of non-training class labels that the models should be applied to', default='')
	pipln_group.add_argument('-n_jobs', '-p', help='Number of processors for parallel computing (max for HPCC = 14)', type=int, default=1)
	pipln_group.add_argument('-n', '-b', help='Number of replicates (unique balanced datasets).', type=int, default=100)
	pipln_group.add_argument('-threshold_test', help='Metric used to define prediction score threshold for classification (F1 or accuracy)).', default='F1')
	pipln_group.add_argument('-x_norm', help='t/f to normalize features (default to T for SVM based algs unless "force_false")', default='f')
	pipln_group.add_argument('-drop_na', help='t/f to drop rows with NAs', default='f')
	pipln_group.add_argument('-cv_num', '-cv', help='Cross validation fold #', type=int, default=10)
	pipln_group.add_argument('-min_size', help='Number instances to downsample to (default = # instances from smallest class', default='')

	# Grid Search Method
	gs_group = parser.add_argument_group(title='CONTROL GRID SEARCH BEHAVIOR')
	gs_group.add_argument('-gs', help='t/f if grid search over parameter space is desired.', type=str, default='t')
	gs_group.add_argument('-gs_reps', '-gs_n', help='Number of Grid Search Reps (will append results if SAVE_GridSearch.csv exists)', type=int, default=10)
	gs_group.add_argument('-gs_score', help='Metric used to select best parameters', type=str, default='roc_auc')
	gs_group.add_argument('-gs_type', help='Full grid search or randomized search (full/random)', type=str, default='full')
	gs_group.add_argument('-gs_full', help='t/f Output full results from the grid search', type=str, default='f')

	# Output arguments
	out_group = parser.add_argument_group(title='OUTPUT OPTIONS')
	out_group.add_argument('-save', help='prefix for output files. CAUTION: will overwrite!', default='')
	out_group.add_argument('-tag', help='Identifier string to add to RESULTS output line', default='')
	out_group.add_argument('-cm', help='t/f Output the confusion matrix & confusion matrix figure', default='f')
	out_group.add_argument('-plots', help='t/f Output ROC and PR curve plots for each model (see ML_plots.py to post-plot', default='f')
	out_group.add_argument('-short', help='Set to T to output only summary prediction scores', default='f')

	# Default Hyperparameters
	params_group = parser.add_argument_group(title='DEFINE HYPERPARAMETERS')
	params_group.add_argument('-n_estimators', help='RF/GB parameter. Grid Search [100, 500, 1000]', type=int, default=500)
	params_group.add_argument('-max_depth', help='RF/GB parameter. Grid Search [3, 5, 10]', type=int, default=5)
	params_group.add_argument('-max_features', help='RF/GB parameter. Grid Search [0.1, 0.5, sqrt, log2, None]', default='sqrt')
	params_group.add_argument('-lr','-learning_rate', help='GB parameter. Grid Search [0.001, 0.01, 0.1, 0.5, 1]', type=float, default=0.1)
	params_group.add_argument('-kernel', help='SVM parameter - not in grid search use -alg SVM, SVMrbf, or SVMpoly')
	params_group.add_argument('-C', help='SVM/LogReg parameter. Grid Search [0.001, 0.01, 0.1, 0.5, 1, 10, 50]', type=float, default=1.0)
	params_group.add_argument('-gamma', help='SVMrbf/SVMpoly parameter. Grid Search [np.logspace(-5,1,7)]', type=float, default=1)
	params_group.add_argument('-degree', help='SVMpoly parameter. Grid Search [2,3,4]', type=int, default=2)
	params_group.add_argument('-penalty', help='LogReg parameter. Grid Search [2,3,4]', default='l2')
	params_group.add_argument('-intercept_scaling', help='LogReg parameter. Grid Search [0.1, 0.5, 1, 2, 5, 10]', type=float, default=1.0)

	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()
	
	# Complex transformations of input parameters
	if ',' in args.cl_train:
		args.cl_train = args.cl_train.strip().split(',')
		args.pos = args.cl_train[0]
	if args.apply != 'all':
		if ',' in args.apply:
			args.apply = args.apply.split(',')
		else:
			args.apply = [args.apply]
	try:
		args.max_features = float(args.max_features)
	except:
		args.max_features =args.max_features
	
	###################################
	### Load and Process Input Data ###
	###################################
	
	df = pd.read_csv(args.df, sep=args.sep, index_col = 0)
	
	# If features  and class info are in separate files, merge them: 
	if args.df2 != '':
		start_dim = df.shape
		df_class = pd.read_csv(args.df2, sep=args.sep, index_col = 0)
		df = pd.concat([df_class[args.y_name], df], axis=1, join='inner')
		print('Merging the feature & class dataframes changed the dimensions from %s to %s (instance, features).' 
			% (str(start_dim), str(df.shape)))

	# Specify class column - default = Class
	if args.y_name != 'Class':
		df = df.rename(columns = {args.y_name:'Class'})
	
	# Filter out features not in feat file given - default: keep all
	if args.feat != 'all':
		print('Using subset of features from: %s' % args.feat)
		with open(args.feat) as f:
			features = f.read().strip().splitlines()
			features = ['Class'] + features
		df = df.loc[:,features]

	# Check for Nas
	if df.isnull().values.any() == True:
		if args.drop_na.lower() in ['t', 'true']:
			start_dim = df.shape
			df = df.dropna(axis=0)
			print('Dropping rows with NA values changed the dimensions from %s to %s.' 
				% (str(start_dim), str(df.shape)))
		else:
			print(df.columns[df.isna().any()].tolist())
			print('There are Na values in your dataframe.\n Impute them or add -drop_na True to remove rows with nas' )
			quit()
	
		# Normalize data frame for SVM algorithms
	if args.alg.lower() in ["svm", "svmpoly", "svmrbf"] or args.x_norm.lower() in ['t','true']:
		if args.x_norm.lower != 'force_false':
			from sklearn import preprocessing
			y = df['Class']
			X = df.drop(['Class'], axis=1)
			min_max_scaler = preprocessing.MinMaxScaler()
			X_scaled = min_max_scaler.fit_transform(X)
			df = pd.DataFrame(X_scaled, columns = X.columns, index = X.index)
			df.insert(loc=0, column = 'Class', value = y)


	# Set up dataframe of unknown instances that the final models will be applied to
	if args.cl_train != 'all' and '' not in args.apply:
		apply_unk = True
		# if apply to all, select all instances with a class not in args.cl_train
		if args.apply == 'all':
			df_unknowns = df[(~df['Class'].isin(args.cl_train))]
		else: # apply to specified classes
			df_unknowns = df[(df['Class'].isin(args.apply))]
	else:
		apply_unk = False
		df_unknowns = ''
	
	# Remove classes that won't be included in the training (e.g. unknowns)
	if args.cl_train != 'all':
		df = df[(df['Class'].isin(args.cl_train))]

	# Set up dataframe of holdout instances that the final models will be applied to
	if args.ho !='':
		df_all = df.copy()
		print('Removing holdout instances to apply model on later...')
		with open(args.ho) as ho_file:
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
	if args.cl_train == 'all':
		classes = df['Class'].unique()
		if len(classes) == 2:
			args.gs_score = 'roc_auc'
			for clss in classes:
				if clss != args.pos:
					NEG = clss
					try:
						NEG = int(NEG)
					except:
						pass
					break
		else:
			NEG = 'multiclass_no_NEG'
			args.gs_score = 'f1_macro'
	else:
		if len(args.cl_train) == 2:
			NEG = args.cl_train[1]
			args.gs_score = 'roc_auc'
		else:
			NEG = 'multiclass_no_NEG'
			args.gs_score = 'f1_macro'
		classes = np.array(args.cl_train)

	classes.sort()
	
	
	# Determine minimum class size (for making balanced datasets)
	if args.min_size == '':
		min_size = (df.groupby('Class').size()).min() - 1
	else:
		min_size = int(args.min_size)
	
	# Define save name if not specified using -save
	if args.save == "":
		if args.tag == "":
			args.save = args.df + "_" + args.alg
		else:
			args.save = args.df + "_" + args.alg + "_" + args.tag
	
	print("Snapshot of data being used:")
	print(df.iloc[:5, :5])
	print("CLASSES:",classes)
	print("POS:",args.pos,type(args.pos))
	print("NEG:",NEG,type(NEG))
	print('Balanced dataset will include %i instances of each class' % min_size)
	n_features = len(list(df)) - 1
	
	###################################
	### Parameter Sweep/Grid Search ###
	###################################
	
	if args.gs.lower() in ['t','true']:
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use, balanced_ids, param_names = ML.fun.GridSearch(df, args.save, args.alg, classes, min_size, args.gs_score, args.n, args.cv_num, args.n_jobs, args.gs_reps, args.gs_type, args.pos, NEG, args.gs_full)
		
		# Print results from grid search
		if args.alg.lower() == 'rf':
			args.max_depth, args.max_features, args.n_estimators = params2use
			print("Parameters selected: max_depth=%s, max_features=%s, n_estimators=%s" % (str(args.max_depth), str(args.max_features), str(args.n_estimators)))
	
		elif args.alg.lower() == 'svm':
			args.C = params2use
			print("Parameters selected: Kernel=Linear, C=%s" % (str(args.C)))
		
		elif args.alg.lower() == "svmpoly":
			args.C, args.degree, args.gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(args.C), str(args.degree), str(args.gamma)))
		
		elif args.alg.lower() == "svmrbf":
			args.C, args.gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, gamma=%s" % (str(kernel), str(args.C), str(args.gamma)))
		
		elif args.alg.lower() == "logreg":
			args.C, args.intercept_scaling, args.penalty = params2use
			print("Parameters selected: penalty=%s, C=%s, intercept_scaling=%s" % (str(args.penalty), str(args.C), str(args.intercept_scaling)))

		elif args.alg.lower() == "gb":
			args.lr, args.max_depth, args.max_features, args.n_estimators = params2use
			print("Parameters selected: learning rate=%s, max_features=%s, max_depth=%s, n_estimators=%s" % (str(args.lr), str(args.max_features), str(args.max_depth), str(args.n_estimators)))
	
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	
	else:
		print('Not running grid search. Using default or given parameters instead')
		balanced_ids = ML.fun.EstablishBalanced(df,classes,int(min_size),args.n)
	
	bal_id = pd.DataFrame(balanced_ids)
	bal_id.to_csv(args.save + '_BalancedIDs', index=False, header=False,sep="\t")

	 
	###############################
	### Train & Apply ML Models ###
	###############################

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
		if args.cl_train != 'all':
			df_notSel = df_notSel[(df_notSel['Class'].isin(args.cl_train))]
		
		# Prime classifier object based on chosen algorithm
		if args.alg.lower() == "rf":
			parameters_used = [args.n_estimators, args.max_depth, args.max_features]
			clf = ML.fun.DefineClf_RandomForest(args.n_estimators,args.max_depth,args.max_features,j,args.n_jobs)
		elif args.alg.lower() == "svm":
			parameters_used = [args.C]
			clf = ML.fun.DefineClf_LinearSVM(args.C,j)
		elif args.alg.lower() == 'svmrbf' or args.alg.lower() == 'svmpoly':
			parameters_used = [args.C, args.degree, args.gamma, kernel]
			clf = ML.fun.DefineClf_SVM(kernel,args.C,args.degree,args.gamma,j)
		elif args.alg.lower() == "logreg":
			parameters_used = [args.C, args.intercept_scaling, args.penalty]
			clf = ML.fun.DefineClf_LogReg(args.penalty, args.C, args.intercept_scaling)
		elif args.alg.lower() == "gb":
			parameters_used = [args.lr, args.max_features, args.max_depth]
			clf = ML.fun.DefineClf_GB(args.n_estimators, args.lr, args.max_features, args.max_depth, args.n_jobs, j)
		
		# Run ML algorithm on balanced datasets.
		if args.ho!='':
			result,current_scores,result_ho = ML.fun.BuildModel_Apply_Performance(df1, clf, args.cv_num, df_notSel, apply_unk, df_unknowns, ho_df, classes, args.pos, NEG, j, args.alg,args.threshold_test)
			results_ho.append(result_ho)	
		else:
			result,current_scores = ML.fun.BuildModel_Apply_Performance(df1, clf, args.cv_num, df_notSel, apply_unk, df_unknowns, ho_df, classes, args.pos, NEG, j, args.alg,args.threshold_test)

		results.append(result)
		try:
			df_proba = pd.concat([df_proba,current_scores], axis = 1)
		except:
			print('\n\nOpps. Something went wrong merging the probability scores...')
			print('Check if you have duplicate instance names in your dataframe!')
			quit()
		#df_proba = df_proba.merge(current_scores, how='outer', left_index=True, right_index=True)

	print("ML Pipeline time: %f seconds" % (time.time() - start_time))


	
	################################
	### Unpack & Save ML Results ###
	################################
	
	## Make empty dataframes
	conf_matrices = pd.DataFrame(columns = np.insert(arr = classes.astype(np.str), obj = 0, values = 'Class'), dtype=float)
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
				if args.alg.lower() == 'rf' or args.alg.lower() == 'gb':
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
	if args.cm.lower() in ['true','t']:
		cm_mean.to_csv(args.save + "_cm.csv",sep="\t")
		done = ML.fun.Plot_ConMatrix(cm_mean, args.save)

	# Unpack results from hold out
	if args.ho!='':
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


		scores_file = args.save + "_scores.txt"
		out_scores = open(scores_file,"w")
		if args.short.lower() in ['t', 'true']:
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
		if args.ho!='':
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
		with open(args.save + "_results.txt", 'w') as out:
			out.write('%s\nID: %s\nTag: %s\nAlgorithm: %s\nTrained on classes: %s\nApplied to: %s\nNumber of features: %i\n' % (
				timestamp, args.save, args.tag, args.alg, classes, args.apply, n_features))
			out.write('Min class size: %i\nCV folds: %i\nNumber of balanced datasets: %i\nGrid Search Used: %s\nParameters used:%s\n' % (
				min_size, args.cv_num, args.n, args.gs , parameters_used))

			out.write('\nMetric\tMean\tSD\nAccuracy\t%05f\t%05f\nF1_macro\t%05f\t%05f\n' % (AC, AC_std, MacF1, MacF1_std))
			for cla in f1.columns:
				if 'M_F1' not in cla:
					out.write('%s\t%05f\t%05f\n' % (cla, np.mean(f1[cla]), np.std(f1[cla])))
		
			out.write('\nMean Balanced Confusion Matrix:\n')
			cm_mean.to_csv(out, mode='a', sep='\t')
			out.write('\n\nCount and percent of instances of each class (row) predicted as a class (col):\n')
			summary_df_proba.to_csv(out, mode='a', header=True, sep='\t')
			
			# Add results from hold out
			if args.ho!='':
				out.write('\n\nResults from the hold out validation set\n')
				out.write('HO Accuracy\t%05f +/-%05f\nHO F1_macro\t%05f +/-%05f' % (AC_ho, AC_std_ho, MacF1_ho, MacF1_std_ho))

###### Binary Prediction Output ######
	
	else: 
		# Get AUC for ROC and PR curve (mean, sd, se)
		ROC = [np.mean(AucRoc_array), np.std(AucRoc_array), np.std(AucRoc_array)/np.sqrt(len(AucRoc_array))]
		PRc = [np.mean(AucPRc_array), np.std(AucPRc_array), np.std(AucPRc_array)/np.sqrt(len(AucPRc_array))]
		if args.ho!='':
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
		df_proba[Pred_name] = np.where(df_proba['Mean'] >= final_threshold, args.pos,NEG)

		# Summarize % of each class predicted as POS and NEG		
		summary_df_proba = df_proba[['Class', Pred_name, 'Mean']].groupby(['Class', Pred_name]).agg('count').unstack(level=1)
		summary_df_proba.columns = summary_df_proba.columns.droplevel()
		try:
			summary_df_proba['n_total'] = summary_df_proba[args.pos] + summary_df_proba[NEG]
			summary_df_proba[str(NEG) + '_perc'] = summary_df_proba[NEG]/summary_df_proba['n_total']
		except:
			summary_df_proba['n_total'] = summary_df_proba[args.pos]
			summary_df_proba[str(NEG) + '_perc'] = 0
			print('Warning: No instances were classified as negative!')	
		summary_df_proba[str(args.pos) + '_perc'] = summary_df_proba[args.pos]/summary_df_proba['n_total']
		

		scores_file = args.save + "_scores.txt"
		out_scores = open(scores_file,"w")
		if args.short.lower() in ['t','true']:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba[["Class","Mean","Median","stdev",Pred_name]],sep="\t").strip()+"\n")
		else:
			out_scores.write("#ID\t"+pd.DataFrame.to_csv(df_proba,sep="\t").strip()+"\n")
		out_scores.close()
		# df_proba.to_csv(args.save + "_scores.txt", sep="\t")
		
		
		# Get model preformance scores using final_threshold
		if args.ho!='':
			TP,TN,FP,FN,TPR,FPR,FNR,Pr,Ac,F1,Pr_ho,Ac_ho,F1_ho = ML.fun.Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, args.pos, NEG, ho_instances)
		else:
			TP,TN,FP,FN,TPR,FPR,FNR,Pr,Ac,F1 = ML.fun.Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, args.pos, NEG, ho_instances)
			Pr_ho, Ac_ho, F1_ho = 0, 0, 0

		# Plot ROC & PR curves
		if args.plots.lower() in['true','t']:
			print("\nGenerating ROC & PR curves")
			pr = ML.fun.Plots(df_proba, balanced_ids, ROC, PRc, args.pos, NEG, args.n, args.save)

		# Export importance scores
		try:
			imp['mean_imp'] = imp.mean(axis=1)
			imp = imp.sort_values('mean_imp', 0, ascending = False)
			imp_out = args.save + "_imp"
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
			str(timestamp), run_time, args.save, args.tag, args.alg, [args.pos,NEG], n_features, min_size, args.cv_num , args.n, 
			'\t'.join(str(x) for x in ROC), '\t'.join(str(x) for x in PRc), '\t'.join(str(x) for x in Ac), '\t'.join(str(x) for x in F1),
			'\t'.join(str(x) for x in Pr), '\t'.join(str(x) for x in TPR), '\t'.join(str(x) for x in FPR),
			'\t'.join(str(x) for x in FNR), '\t'.join(str(x) for x in TP), '\t'.join(str(x) for x in TN),
			'\t'.join(str(x) for x in FP), '\t'.join(str(x) for x in FN), Pr_ho, Ac_ho, F1_ho,
			'\t'.join(str(x) for x in ROC_ho),'\t'.join(str(x) for x in PRc_ho)))
			 
			

		# Save detailed results file 
		with open(args.save + "_results.txt", 'w') as out:
			out.write('%s\nID: %s\nTag: %s\nAlgorithm: %s\nTrained on classes: %s\nApplied to: %s\nNumber of features: %i\n' % (
				timestamp, args.save, args.tag, args.alg, classes, args.apply, n_features))
			out.write('Min class size: %i\nCV folds: %i\nNumber of balanced datasets: %i\nGrid Search Used: %s\nParameters used:%s\n' % (
				min_size, args.cv_num, args.n, args.gs, parameters_used))
			
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
			if args.ho!='':
				out.write('\n\nResults from the hold out validation set\n')
				out.write('HO Precision\t%05f\nHO Accuracy\t%05f\nHO F1\t%05f\n' % (Pr_ho, Ac_ho, F1_ho))
				out.write('HO AucROC\t%s\nHO AucPRc\t%s\n' % ('\t'.join(str(x) for x in ROC_ho), '\t'.join(str(x) for x in PRc_ho)))



		print("\n\n===>  ML Results  <===")
		print("Testing Set Scores\nAccuracy: %03f (+/- stdev %03f)\nF1: %03f (+/- stdev %03f)\nAUC-ROC: %03f (+/- stdev %03f)\nAUC-PRC: %03f (+/- stdev %03f)" % (
			Ac[0], Ac[1], F1[0], F1[1], ROC[0], ROC[1], PRc[0], PRc[1]))
		if args.ho!='':
			print('\n\nHold Out Set Scores:\nPrecision: %03f\nAccuracy: %03f\nF1: %03f\nAUC-ROC: %03f (+/- stdev %03f)\nAUC-PRC: %03f (+/- stdev %03f)' % (
				Pr_ho, Ac_ho, F1_ho, ROC_ho[0], ROC_ho[1], PRc_ho[0], PRc_ho[1]))
		print('finished!')	
	

if __name__ == '__main__':
	main()
