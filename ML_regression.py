"""
PURPOSE:
Machine learning regression implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on MSU HPCC first run:
export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH

INPUTS:
	
	REQUIRED:
	-df       Feature & class dataframe for ML
	-alg      Available: RF, SVM (linear), SVMpoly, SVMrbf, GB, and Logistic Regression (LogReg)
	
	OPTIONAL:
	-y_name 	Name of the column to predict (Default = Y)
	-apply	  String in Y column that indicates unknown values you want to predict. (Default = None)
	-sep      Set seperator for input data (Default = '\t')
	-ho       File with list of intances to holdout from feature selection
	-gs       T/F if grid search over parameter space is desired. (Default = True)
	-normX		T/F to normalize the features (Default = F (except T for SVM))
	-drop_na  T/F to drop rows with NAs
	-cv       # of cross-validation folds. (Default = 10)
	-cv_set  	File with cv folds defined
	-n/-b			# of times CV predictions are run (Default = 100)
	-p        # of processors. (Default = 1, max for HPCC = 14)
	-tag      String for SAVE name and TAG column in RESULTS.txt output.
	-feat     Import file with subset of features to use. If invoked,-tag arg is recommended. Default: keep all features.
	-Y        String for column with what you are trying to predict. Default = Y
	-save     Adjust save name prefix. Default = [df]_[alg]_[tag (if used)].
							CAUTION: will overwrite!
	-short    Set to True to output only the median and std dev of prediction scores, default = full prediction scores
	-gs_full 	T/F Output full results from the grid search. (Default = F)
	-gs_reps  Number of replicates of the grid search (Default = 10)
	-gs_type  Full grid search or randomized search (Default = full, alt = random)
	-df2      File with class information. Use only if df contains the features but not the Y values 
							* Need to specifiy what column in df2 is y using -y_name 
	
	PLOT OPTIONS:
	-plots    T/F Output a regression plot showing true vs. predicted Y


OUTPUT:
	-SAVE_imp           Importance scores for each feature
	-SAVE_GridSearch    Results from parameter sweep sorted by F1
	-SAVE_scores		    Mean predicted Y value & individual predictions from each -n
	-SAVE_results		    Detailed results from each model 
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
	n, FEAT, n_jobs, Y_col, plots, cv_num, TAG, SAVE, short_scores, OUTPUT_LOC = 100, 'all', 1, 'Y', 'False', 10, '', '', '', ''
	y_name, SEP, DF2, df_unknowns, apply_model , normX, normY, cv_reps, cv_sets = 'Y', '\t', 'None', 'none','None', 'F', 'F', 10, 'none'
	drop_na, ho = 'f', ''
	
	# Default parameters for Grid search
	GS, gs_score, GS_REPS, GS_TYPE, gs_full = 'f', 'neg_mean_squared_error', 10, 'full', 'f'
	
	# Default Random Forest and GB parameters
	n_estimators, max_depth, max_features, learning_rate = 500, 10, "sqrt", 1.0
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'rbf', 0.01, 2, 0.00001, 'hinge', "500"
	
	# Default Logistic Regression paramemter
	penalty, C, intercept_scaling = 'l2', 1.0, 1.0
	
	for i in range (1,len(sys.argv),2):
		if sys.argv[i].lower() == "-df":
			DF = sys.argv[i+1]
		elif sys.argv[i].lower() == "-df2":
			DF2 = sys.argv[i+1]
		elif sys.argv[i].lower() == "-y_name":
			y_name = sys.argv[i+1]
		elif sys.argv[i].lower() == "-sep":
			SEP = sys.argv[i+1]
		elif sys.argv[i].lower() == '-save':
			SAVE = sys.argv[i+1]
		elif sys.argv[i].lower() == '-feat':
			FEAT = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs":
			GS = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_reps":
			GS_REPS = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-gs_type":
			GS_TYPE = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_full":
			gs_full = sys.argv[i+1]
		elif sys.argv[i].lower() == '-normx':
			normX = sys.argv[i+1]
		elif sys.argv[i].lower() == "-gs_score":
			gs_score = sys.argv[i+1]
		elif sys.argv[i].lower() == "-Y":
			Y = sys.argv[i+1]
		elif sys.argv[i].lower() == "-apply":
			apply_model = sys.argv[i+1]
		elif sys.argv[i].lower() == "-b" or sys.argv[i].lower() == "-n":
			cv_reps = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-drop_na":
			drop_na = sys.argv[i+1]
		if sys.argv[i].lower() == '-ho':
			ho = sys.argv[i+1]
		elif sys.argv[i].lower() == "-alg":
			ALG = sys.argv[i+1]
		elif sys.argv[i].lower() == "-cv":
			cv_num = int(sys.argv[i+1])
		elif sys.argv[i].lower() == "-cv_set":
			cv_sets = pd.read_csv(sys.argv[i+1], index_col = 0)
			cv_reps = len(cv_sets.columns)
			cv_num = len(cv_sets.iloc[:,0].unique())
		elif sys.argv[i].lower() == "-plots":
			plots = sys.argv[i+1]
		elif sys.argv[i].lower() == "-tag":
			TAG = sys.argv[i+1]
		elif sys.argv[i].lower() == "-out":
			OUTPUT_LOC = sys.argv[i+1]
		elif sys.argv[i].lower() == "-n_jobs" or sys.argv[i] == "-p":
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

	# Specify Y column - default = Class
	if y_name != 'Y':
		df = df.rename(columns = {y_name:'Y'})

	# Filter out features not in feat file given - default: keep all
	if FEAT != 'all':
		print('Using subset of features from: %s' % FEAT)
		with open(FEAT) as f:
			features = f.read().strip().splitlines()
			features = ['Y'] + features
		df = df.loc[:,features]
	
	# Check for Nas
	if df.isnull().values.any() == True:
		if drop_na.lower() == 't' or drop_na.lower() == 'true':
			start_dim = df.shape
			df = df.dropna(axis=0)
			print('Dropping rows with NA values changed the dimensions from %s to %s.' 
				% (str(start_dim), str(df.shape)))
		else:
			print(df.columns[df.isnull().any()].tolist())
			print('There are Na values in your dataframe.\n Impute them or add -drop_na True to remove rows with nas')
			quit()

	# Set up dataframe of unknown instances that the final models will be applied to and drop unknowns from df for model building
	if apply_model != 'None':
		df_unknowns = df[(df['Y']==apply_model)]
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		df = df.drop(df_unknowns.index.values)
		print("Model built using %i instances and applied to %i unknown instances (see _scores file for results)" % (len(df.index), len(df_unknowns.index)))
	else:
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		print("Model built using %i instances" % len(df.index))
	
	# Make sure Y is datatype numeric
	df['Y'] = pd.to_numeric(df['Y'], errors = 'raise')

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

		
	print("Snapshot of data being used:")
	print(df.head())

	n_features = len(list(df)) - 1
	
	####### Run parameter sweep using a grid search #######
	
	if GS.lower() == 'true' or GS.lower() == 't':
		start_time = time.time()
		print("\n\n===>  Grid search started  <===") 
		
		params2use, param_names = ML.fun.RegGridSearch(df, SAVE, ALG, gs_score, n, cv_num, n_jobs, GS_REPS, GS_TYPE, gs_full)
		
		# Print results from grid search
		if ALG.lower() == 'rf':
			max_depth, max_features = params2use
			print("Parameters selected: max_depth=%s, max_features=%s" % (str(max_depth), str(max_features)))
	
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
			learning_rate, max_depth, max_features = params2use
			print("Parameters selected: learning rate=%s, max_features=%s, max_depth=%s" % (str(learning_rate), str(max_features), str(max_depth)))
	
		print("Grid search complete. Time: %f seconds" % (time.time() - start_time))
	
	else:
		params2use = "Default parameters used"
	 


	####### Run ML models #######
	start_time = time.time()
	print("\n\n===>  ML Pipeline started  <===")
	
	results = []
	results_ho = []
	imp = pd.DataFrame(index = list(df.drop(['Y'], axis=1)))



		
	for j in range(0,cv_reps): 
		print("Running %i of %i" % (j+1, cv_reps))
		rep_name = "rep_" + str(j+1)
		
		# Prime regressor object based on chosen algorithm
		if ALG.lower() == "rf":
			reg = ML.fun.DefineReg_RandomForest(n_estimators,max_depth,max_features,n_jobs,j)
		elif ALG.lower() == "svm" or ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
			reg = ML.fun.DefineReg_SVM(kernel,C,degree,gamma,j)
		elif ALG.lower() == "gb":
			reg = ML.fun.DefineReg_GB(learning_rate,max_features,max_depth,n_jobs,j)
		elif ALG.lower() == "logreg":
			reg = ML.fun.DefineReg_LinReg()
		else:
			print('Algorithm not available...')
			quit()

		# Run ML algorithm.
		if ho != '':
			result,cv_pred,importance,result_ho = ML.fun.Run_Regression_Model(df, reg, cv_num, ALG, df_unknowns, ho_df, cv_sets, j)
			results_ho.append(result_ho)
		else:
			result,cv_pred,importance = ML.fun.Run_Regression_Model(df, reg, cv_num, ALG, df_unknowns, ho_df, cv_sets, j)
		
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

	# Get scores from hold out validation set:
	if ho != '':
		mses_ho, evss_ho, r2s_ho, cors_ho = [], [], [], []
		for r in results_ho:
			mses_ho.append(r[0])
			evss_ho.append(r[1])
			r2s_ho.append(r[2])
			cors_ho.append(r[3])

		MSE_ho_stats = [np.mean(mses_ho), np.std(mses_ho), np.std(mses_ho)/np.sqrt(len(mses_ho))]
		EVS_ho_stats = [np.mean(evss_ho), np.std(evss_ho), np.std(evss_ho)/np.sqrt(len(evss_ho))]
		r2_ho_stats = [np.mean(r2s_ho), np.std(r2s_ho), np.std(r2s_ho)/np.sqrt(len(r2s_ho))]
		PCC_ho_stats = [np.mean(cors_ho), np.std(cors_ho), np.std(cors_ho)/np.sqrt(len(cors_ho))]
	
	else:
		MSE_ho_stats, EVS_ho_stats, r2_ho_stats, PCC_ho_stats = ['na', 'na', 'na'],['na', 'na', 'na'],['na', 'na', 'na'],['na', 'na', 'na']


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
		out2.write('MSE\tMSE_sd\tMSE_se\tEVS\tEVS_sd\tEVS_se\tr2\tr2_sd\tr2_se\tPCC\tPCC_sd\tPCC_se\t')
		out2.write('MSE_ho\tMSE_ho_sd\tMSE_ho_se\tEVS_ho\tEVS_ho_sd\tEVS_ho_se\tr2_ho\tr2_ho_sd\tr2_ho_se\tPCC_ho\tPCC_ho_sd\tPCC_ho_se\n')
		out2.close()

	out2 = open('RESULTS_reg.txt', 'a')
	out2.write('%s\t%s\t%s\t%s\t%s\t%s\t%i\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
		timestamp, run_time, SAVE, TAG, y_name, ALG, len(df.index), n_features, cv_num , cv_reps, 
		'\t'.join(str(x) for x in MSE_stats), '\t'.join(str(x) for x in EVS_stats), 
		'\t'.join(str(x) for x in r2_stats), '\t'.join(str(x) for x in PCC_stats),
		'\t'.join(str(x) for x in MSE_ho_stats), '\t'.join(str(x) for x in EVS_ho_stats), 
		'\t'.join(str(x) for x in r2_ho_stats), '\t'.join(str(x) for x in PCC_ho_stats),  ))


	# Save detailed results file 
	with open(SAVE + "_results.txt", 'w') as out:
		out.write('%s\nID: %s\nTag: %s\nPredicting: %s\nAlgorithm: %s\nNumber of Instances: %s\nNumber of features: %i\n' % (
			timestamp, SAVE, TAG, y_name, ALG, len(df.index), n_features))
		out.write('CV folds: %i\nCV_reps: %i\nParameters used:%s\n' % (cv_num, cv_reps, params2use))
		out.write('Metric\tMean\tstd\tSE\n')
		out.write('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_stats), '\t'.join(str(x) for x in EVS_stats), 
			'\t'.join(str(x) for x in r2_stats), '\t'.join(str(x) for x in PCC_stats)))

		if ho != '':
			out.write('\n\nResults from the hold out validation set\n')
			out.write('Metric\tMean\tstd\tSE\n')
			out.write('HO MSE\t%s\nHO EVS\t%s\nHO R2\t%s\nHO PCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_ho_stats), '\t'.join(str(x) for x in EVS_ho_stats), 
			'\t'.join(str(x) for x in r2_ho_stats), '\t'.join(str(x) for x in PCC_ho_stats)))


	print("\n\n===>  ML Results  <===")
	print('Metric\tMean\tstd\tSE')
	print('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
		'\t'.join(str(x) for x in MSE_stats), '\t'.join(str(x) for x in EVS_stats), 
		'\t'.join(str(x) for x in r2_stats), '\t'.join(str(x) for x in PCC_stats)))

	if ho !='':
		print('\n\nHold Out Set Scores:\nMetric\tMean\tstd\tSE\n')
		print('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_ho_stats), '\t'.join(str(x) for x in EVS_ho_stats), 
			'\t'.join(str(x) for x in r2_ho_stats), '\t'.join(str(x) for x in PCC_ho_stats)))

	print('\nfinished!')

if __name__ == '__main__':
	main()