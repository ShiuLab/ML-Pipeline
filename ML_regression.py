"""
PURPOSE:
Machine learning classifications implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
	
	REQUIRED ALL:
	-df       Feature & class dataframe for ML. See "" for an example dataframe
	-alg      Available: RF, SVM (linear), SVMpoly, SVMrbf, GB, and Linear Regression (LR)
	
	OPTIONAL:
	-unknown  String in Y that indicates unknown values you want to predict. Leave as none if you don't have unknowns in your data Default = none
	-gs       Set to True if grid search over parameter space is desired. Default = False
	-normX		T/F to normalize the features (Default = F (except T for SVM))
	-normY	  T/F to normalize the predicted value (Default = F)
	-sep      Set sep for input data (Default = '\t')
	-cv       # of cross-validation folds. Default = 10
	-cv_reps  # of times CV predictions are run
	-cv_set  	File with cv folds defined
	-p        # of processors. Default = 1
	-tag      String for SAVE name and TAG column in RESULTS.txt output.
	-feat     Import file with subset of features to use. If invoked,-tag arg is recommended. Default: keep all features.
	-Y        String for column with what you are trying to predict. Default = Y
	-save     Adjust save name prefix. Default = [df]_[alg]_[tag (if used)], CAUTION: will overwrite!
	-short    Set to True to output only the median and std dev of prediction scores, default = full prediction scores
	-df_Y     File with class information. Use only if df contains the features but not the Y values 
								If more than one column in the class file, specify which column contains Y: -df_class class_file.csv,ColumnName
	
	PLOT OPTIONS:
	-plots    T/F - Do you want to output plots?

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
	n, FEAT, apply, n_jobs, Y_col, plots, cv_num, TAG, SAVE, short_scores, OUTPUT_LOC = 100, 'all','F', 1, 'Y', 'False', 10, '', '', '', ''
	y_name, SEP, THRSHD_test, DF_Y, df_unknowns,UNKNOWN, normX, normY, cv_reps, cv_sets = 'Y', '\t','F1', 'ignore', 'none','unk', 'F', 'F', 10, 'none'

	# Default parameters for Grid search
	GS, gs_score = 'F', 'neg_mean_squared_error'
	
	# Default Random Forest and GB parameters
	n_estimators, max_depth, max_features, learning_rate = 500, 10, "sqrt", 1.0
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'rbf', 0.01, 2, 0.00001, 'hinge', "500"
	
	# Default Logistic Regression paramemter
	penalty, C, intercept_scaling = 'l2', 1.0, 1.0
	
	for i in range (1,len(sys.argv),2):
		if sys.argv[i] == "-df":
			DF = sys.argv[i+1]
		elif sys.argv[i] == "-df_Y":
			DF_Y = sys.argv[i+1]
		elif sys.argv[i] == "-y_name":
			y_name = sys.argv[i+1]
		elif sys.argv[i] == "-sep":
			SEP = sys.argv[i+1]
		elif sys.argv[i] == '-save':
			SAVE = sys.argv[i+1]
		elif sys.argv[i] == '-feat':
			FEAT = sys.argv[i+1]
		elif sys.argv[i] == "-gs":
			GS = sys.argv[i+1]
		elif sys.argv[i] == '-normX':
			normX = sys.argv[i+1].lower()
		elif sys.argv[i] == "-normY":
			normY = sys.argv[i+1].lower()
		elif sys.argv[i] == "-gs_score":
			gs_score = sys.argv[i+1]
		elif sys.argv[i] == "-Y":
			Y = sys.argv[i+1]
		elif sys.argv[i] == "-UNKNOWN":
			UNKNOWN = sys.argv[i+1]
		elif sys.argv[i] == "-n":
			n = int(sys.argv[i+1])
		elif sys.argv[i] == "-b":
			n = int(sys.argv[i+1])
		elif sys.argv[i] == "-alg":
			ALG = sys.argv[i+1]
		elif sys.argv[i] == "-cv":
			cv_num = int(sys.argv[i+1])
		elif sys.argv[i] == "-cv_reps":
			cv_reps = int(sys.argv[i+1])
		elif sys.argv[i] == "-cv_set":
			cv_sets = pd.read_csv(sys.argv[i+1], index_col = 0)
			cv_reps = len(cv_sets.columns)
			cv_num = len(cv_sets.iloc[:,0].unique())
		elif sys.argv[i] == "-plots":
			plots = sys.argv[i+1]
		elif sys.argv[i] == "-tag":
			TAG = sys.argv[i+1]
		elif sys.argv[i] == "-out":
			OUTPUT_LOC = sys.argv[i+1]
		elif sys.argv[i] == "-threshold_test":
			THRSHD_test = sys.argv[i+1]
		elif sys.argv[i] == "-n_jobs" or sys.argv[i] == "-p":
			n_jobs = int(sys.argv[i+1])
		elif sys.argv[i] == "-short":
			scores_len = sys.argv[i+1]
			if scores_len.lower() == "true" or scores_len.lower() == "t":
				short_scores = True

	if len(sys.argv) <= 1:
		print(__doc__)
		exit()
	
	####### Load Dataframe & Pre-process #######
	
	df = pd.read_csv(DF, sep=SEP, index_col = 0)
	# If feature info and class info are in separate files
	if DF_Y != 'ignore':
		df_Y_file, df_Y_col = DF_Y.strip().split(',')
		df_Y = pd.read_csv(df_Y_file, sep=SEP, index_col = 0)
		df[y_name] = df_Y[df_Y_col]
		y_name = df_Y_col

	# Specify Y column - default = Class
	if y_name != 'Y':
		df = df.rename(columns = {y_name:'Y'})
		y_name = y_name
	
	# Filter out features not in feat file given - default: keep all
	if FEAT != 'all':
		with open(FEAT) as f:
			features = f.read().strip().splitlines()
			features = ['Y'] + features
		df = df.loc[:,features]
	
	# Remove instances with NaN or NA values
	df = df.replace("?",np.nan)
	df = df.dropna(axis=0)
	
	# Set up dataframe of unknown instances that the final models will be applied to and drop unknowns from df for model building
	if UNKNOWN in df['Y'].unique():
		df_unknowns = df[(df['Y']==UNKNOWN)]
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		df = df.drop(df_unknowns.index.values)
		print("Model built using %i instances and applied to %i unknown instances (see _scores file for results)" % (len(df.index), len(df_unknowns.index)))
	else:
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		print("Model built using %i instances" % len(df.index))

	
	if SAVE == "":
		if TAG == "":
			if OUTPUT_LOC == "":
				SAVE = DF + "_" + ALG
			else:
				SAVE = OUTPUT_LOC + '/' + DF + "_" + ALG
		else:
			if OUTPUT_LOC == "":
				SAVE = DF + "_" + ALG + "_" + TAG
			else:
				SAVE = OUTPUT_LOC + '/' + DF + "_" + ALG + "_" + TAG
	
	# Normalize feature data (normX)
	if ALG == "SVM" or normX == 't' or normX == 'true':
		from sklearn import preprocessing
		y = df['Y']
		X = df.drop(['Y'], axis=1)
		min_max_scaler = preprocessing.MinMaxScaler()
		X_scaled = min_max_scaler.fit_transform(X)
		df = pd.DataFrame(X_scaled, columns = X.columns, index = X.index)
		df.insert(loc=0, column = 'Y', value = y)

	# Normalize y variable (normY)
	if normY == 't' or normY == 'true':
		print('normY not implemented yet!!!')
	
	
	print("Snapshot of data being used:")
	print(df.head())

	n_features = len(list(df)) - 1
	
	####### Run parameter sweep using a grid search #######
	if GS.lower() == 'true' or GS.lower() == 't':
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use, param_names = ML.fun.RegGridSearch(df, SAVE, ALG, gs_score, n, cv_num, n_jobs)
		
		# Print results from grid search
		if ALG == 'RF':
			max_depth, max_features = params2use
			print("Parameters selected: max_depth=%s, max_features=%s" % (str(max_depth), str(max_features)))
	
		elif ALG == 'SVM':
			C, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s" % (str(kernel), str(C)))
		
		elif ALG == "SVMpoly":
			C, degree, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(C), str(degree), str(gamma)))
		
		elif ALG == "SVMrbf":
			C, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, gamma=%s" % (str(kernel), str(C), str(gamma)))
		
		elif ALG == "LogReg":
			C, intercept_scaling, penalty = params2use
			print("Parameters selected: penalty=%s, C=%s, intercept_scaling=%s" % (str(penalty), str(C), str(intercept_scaling)))

		elif ALG == "GB":
			learning_rate, max_depth, max_features = params2use
			print("Parameters selected: learning rate=%s, max_features=%s, max_depth=%s" % (str(learning_rate), str(max_features), str(max_depth)))
	
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	
	else:
		params2use = "Default parameters used"
	 


	####### Run ML models #######
	start_time = time.time()
	print("\n\n===>  ML Pipeline started  <===")
	
	results = []
	imp = pd.DataFrame(index = list(df.drop(['Y'], axis=1)))

	for j in range(0,cv_reps): 
		print("Running %i of %i" % (j+1, cv_reps))
		rep_name = "rep_" + str(j+1)
		# Prime classifier object based on chosen algorithm
		if ALG == "RF":
			reg = ML.fun.DefineReg_RandomForest(n_estimators,max_depth,max_features,n_jobs,j)
		elif ALG == "SVM" or ALG == 'SVMrbf' or ALG == 'SVMpoly':
			reg = ML.fun.DefineReg_SVM(kernel,C,degree,gamma,j)
		elif ALG == "GB":
			reg = ML.fun.DefineReg_GB(learning_rate,max_features,max_depth,n_jobs,j)
		elif ALG == "LR":
			reg = ML.fun.DefineReg_LinReg()
		
		# Run ML algorithm on balanced datasets.
		result,cv_pred,importance = ML.fun.Run_Regression_Model(df, reg, cv_num, ALG, df_unknowns, cv_sets, j)
		results.append(result)
		predictions[rep_name] = cv_pred
		
		try:
			imp[rep_name] = importance
		except:
			try:
				imp[rep_name] = importance[0]
			except:
				print("Cannot parse importance scores!")

	print("ML Pipeline time: %f seconds" % (time.time() - start_time))


	
	####### Unpack ML results #######
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	mses, evss, r2s, cors = [], [], [], []
	for r in results:
		mses.append(r[0])
		evss.append(r[1])
		r2s.append(r[2])
		cors.append(r[3])

	MSE_stats = [np.mean(mses), np.std(mses), np.std(mses)/np.sqrt(len(mses))]
	EVS_stats = [np.mean(evss), np.std(evss), np.std(evss)/np.sqrt(len(evss))]
	r2_stats = [np.mean(r2s), np.std(r2s), np.std(r2s)/np.sqrt(len(r2s))]
	PCC_stats = [np.mean(cors), np.std(cors), np.std(cors)/np.sqrt(len(cors))]

	# Get average predicted value
	pred_columns = [c for c in predictions.columns if c.startswith('rep_')]
	predictions.insert(loc=1, column = 'Mean', value = predictions[pred_columns].mean(axis=1))
	predictions.insert(loc=2, column = 'stdev', value = predictions[pred_columns].std(axis=1))

	scores_file = SAVE + "_scores.txt"
	if short_scores == True:
			predictions.to_csv(scores_file, sep='\t', columns=['Y','Mean','stdev'])
	else:
		predictions.to_csv(scores_file, sep='\t')

	# Plot results
	if plots.lower() == 'true' or plots.lower() == 't':
		print("\nGenerating prediction plot")
		pr = ML.fun.PlotsReg(predictions, SAVE)
		
	# Export importance scores
	try:
		imp['mean_imp'] = imp.mean(axis=1)
		imp = imp.sort_values('mean_imp', 0, ascending = False)
		imp_out = SAVE + "_imp"
		imp['mean_imp'].to_csv(imp_out, sep = "\t", index=True)
	except:
		pass

	run_time = time.time() - start_total_time

	# Save to summary RESULTS file with all models run from the same directory
	if not os.path.isfile('RESULTS_reg.txt'):
		out2 = open('RESULTS_reg.txt', 'a')
		out2.write('DateTime\tRunTime\tID\tTag\tY\tAlg\tNumInstances\tFeatureNum\tCVfold\tCV_rep\t')
		out2.write('MSE\tMSE_sd\tMSE_se\tEVS\tEVS_sd\tEVS_se\tr2\tr2_sd\tr2_se\tPCC\tPCC_sd\tPCC_se\n')
		out2.close()

	out2 = open('RESULTS_reg.txt', 'a')
	out2.write('%s\t%s\t%s\t%s\t%s\t%s\t%i\t%i\t%i\t%i\t%s\t%s\t%s\t%s\n' % (
		timestamp, run_time, SAVE, TAG, y_name, ALG, len(df.index), n_features, cv_num , cv_reps, 
		'\t'.join(str(x) for x in MSE_stats), '\t'.join(str(x) for x in EVS_stats), 
		'\t'.join(str(x) for x in r2_stats), '\t'.join(str(x) for x in PCC_stats)))


	# Save detailed results file 
	with open(SAVE + "_results.txt", 'w') as out:
		out.write('%s\nID: %s\nTag: %s\nPredicting: %s\nAlgorithm: %s\nNumber of Instances: %s\nNumber of features: %i\n' % (
			timestamp, SAVE, TAG, y_name, ALG, len(df.index), n_features))
		out.write('CV folds: %i\nCV_reps: %i\nParameters used:%s\n' % (cv_num, cv_reps, params2use))
		out.write('Metric\tMean\tstd\tSE\n')
		out.write('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_stats), '\t'.join(str(x) for x in EVS_stats), 
			'\t'.join(str(x) for x in r2_stats), '\t'.join(str(x) for x in PCC_stats)))


	print("\n\n===>  ML Results  <===")
	print('Metric\tMe