import sys, os, argparse, time
import pandas as pd
import numpy as np
from datetime import datetime
import ML_functions as ML
start_total_time = time.time()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():

	########################
	### Parse Input Args ###
	########################
	parser = argparse.ArgumentParser(
		description='Machine learning regression pipeline using tools from '
			'Scikit-Learn. See README.md for more information about the '
			'software needed, the pipeline and preprocessing/post-analysis '
			'tools available!',
		epilog='https://github.com/ShiuLab/ML-Pipeline')

	### Input arguments ###
	# Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-df', help='Feature & class dataframe for ML. '
		'(See example in Workshop directory)', required=True)
	req_group.add_argument('-alg', help='ML Algorithm to run (RF, SVM, '
		'SVMpoly, SVMrbf, GB, LR))', required=True)

	# Optional
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-df2', help='Class data (if not in -df). Need to '
		'provide -y_name', default='')
	inp_group.add_argument('-sep', help='Deliminator', default='\t')
	inp_group.add_argument('-y_name', help='Name of column in Y_file to '
		'predict', default='Y')
	inp_group.add_argument('-test', help='File with testing lines', default='')
	inp_group.add_argument('-feat', help='File with list of features (from x) '
		'to include', default='all')

	# Model behavior 
	pipln_group = parser.add_argument_group(title='CONTROL PIPELINE BEHAVIOR')
	pipln_group.add_argument('-apply', help='Non-training Y labels that the '
		'models should be applied to (e.g. unknown)', default='')
	pipln_group.add_argument('-n_jobs', '-p', help='Number of processors for '
		'parallel computing (max for HPCC = 14)', type=int, default=1)
	pipln_group.add_argument('-n', '-b', help='Number of replicates (unique '
		'balanced datasets).', type=int, default=100)
	pipln_group.add_argument('-threshold_test', help='Metric used to define '
		'prediction score threshold for classification (F1 or accuracy)).',
		default='F1')
	pipln_group.add_argument('-norm', help='t/f to normalize Y values',
		default='f')
	pipln_group.add_argument('-x_norm', help='t/f to normalize features '
		'(default to T for SVM based algs unless "force_false")', default='f')
	pipln_group.add_argument('-y_norm', help='t/f to normalize Y)', default='f')
	pipln_group.add_argument('-drop_na', help='t/f to drop rows with NAs',
		default='f')
	pipln_group.add_argument('-cv_num', '-cv', help='Cross validation fold #',
		type=int, default=10)
	pipln_group.add_argument('-cv_sets', help='File with defined cross '
		'validation folds', default='none')

	# Grid Search Method
	gs_group = parser.add_argument_group(title='CONTROL GRID SEARCH BEHAVIOR')
	gs_group.add_argument('-gs', help='t/f if grid search over parameter space '
		'is desired.', type=str, default='t')
	gs_group.add_argument('-gs_reps', '-gs_n', help='Number of Grid Search '
		'Reps (will append results if SAVE_GridSearch.csv exists)',
		type=int, default=10)
	gs_group.add_argument('-gs_score', help='Metric used to select best '
		'parameters', type=str, default='neg_mean_squared_error')
	gs_group.add_argument('-gs_type', help='Full grid search or randomized '
		'search (full/random)', type=str, default='full')
	gs_group.add_argument('-gs_full', help='t/f Output full results from the '
		'grid search', type=str, default='f')

	# Output arguments
	out_group = parser.add_argument_group(title='OUTPUT OPTIONS')
	out_group.add_argument('-save', help='prefix for output files. CAUTION: '
		'will overwrite!', default='')
	out_group.add_argument('-tag', help='Identifier string to add to RESULTS '
		'output line', default='')
	out_group.add_argument('-out_loc', help='Path to where output files are '
		'saved. Default to cwd.', default='')
	out_group.add_argument('-plots', help='t/f Output ROC and PR curve plots '
		'for each model (see ML_plots.py to post-plot', default='f')
	out_group.add_argument('-short', help='Set to T to output only summary '
		'prediction scores', default='f')

	# Default Hyperparameters
	params_group = parser.add_argument_group(title='DEFINE HYPERPARAMETERS')
	params_group.add_argument('-n_estimators', help='RF/GB parameter.',
		type=int, default=500)
	params_group.add_argument('-max_depth', help='RF/GB parameter. Grid Search '
		'[3, 5, 10]', type=int, default=5)
	params_group.add_argument('-max_features', help='RF/GB parameter. Grid '
		'Search [0.1, 0.25, 0.5, 0.75, sqrt, log2, None]', default='sqrt')
	params_group.add_argument('-lr', '-learning_rate', help='GB parameter. '
		'Grid Search [0.001, 0.01, 0.1, 0.5, 1]', type=float, default=0.1)
	params_group.add_argument('-kernel', help='SVM parameter - not in grid '
		'search use -alg SVM, SVMrbf, or SVMpoly', default='poly')
	params_group.add_argument('-C', help='SVM/LogReg parameter. Grid Search '
		'[0.001, 0.01, 0.1, 0.5, 1, 10, 50]', type=float, default=0.1)
	params_group.add_argument('-gamma', help='SVMrbf/SVMpoly parameter. Grid '
		'Search [np.logspace(-5,1,7)]', type=float, default=0.1)
	params_group.add_argument('-degree', help='SVMpoly parameter. Grid Search '
		'[2,3,4]', type=int, default=2)
	params_group.add_argument('-penalty', help='LogReg parameter. Grid Search '
		'[2,3,4]', default='l2')
	params_group.add_argument('-intercept_scaling', help='LogReg parameter. '
		'Grid Search [0.1, 0.5, 1, 2, 5, 10]', type=float, default=1.0)

	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()

	# Complex transformations of input parameters
	df_unknowns = 'none'

	if args.cv_sets != 'none':
			args.cv_sets = pd.read_csv(args.cv_sets, index_col=0)
			args.cv_reps = len(args.cv_sets.columns)
			args.cv_num = len(args.cv_sets.iloc[:, 0].unique())

	try:
		args.max_features = float(args.max_features)
	except:
		args.max_features =args.max_features

	####### Load Dataframe & Pre-process #######

	df = pd.read_csv(args.df, sep=args.sep, index_col=0)

	# If features  and class info are in separate files, merge them: 
	if args.df2 != '':
		start_dim = df.shape
		df_class = pd.read_csv(args.df2, sep=args.sep, index_col=0)
		df = pd.concat([df_class[args.y_name], df], axis=1, join='inner')
		print('Merging the X and Y dfs. Dim change: %s to %s (instance, feat).'
			% (str(start_dim), str(df.shape)))

	# Specify Y column - default = Class
	if args.y_name != 'Y':
		df = df.rename(columns={args.y_name: 'Y'})


	# Filter out features not in feat file given - default: keep all
	if args.feat != 'all':
		print('Using subset of features from: %s' % args.feat)
		with open(args.feat) as f:
			features = f.read().strip().splitlines()
			features = ['Y'] + features
		df = df.loc[:,features]

	# Check for Nas
	if df.isnull().values.any() == True:
		if args.drop_na.lower() in ['t', 'true']:
			start_dim = df.shape
			df = df.dropna(axis=0)
			print('Dropping rows with NAs... dim change: %s to %s.'
				% (str(start_dim), str(df.shape)))
		else:
			print(df.columns[df.isnull().any()].tolist())
			print('There are Na values in your dataframe.\n '
				'Impute them or add -drop_na True to remove rows with nas')
			quit()

	# Normalize feature data (x_norm)
	if (args.alg.lower() in ["svm", "svmpoly", "svmrbf"] or 
		args.norm.lower() in ['t', 'true']):
		if args.norm.lower != 'force_false':
			from sklearn import preprocessing
			try:
				y = df['Y']
			except:
				print("\n'Y' not found...\nCheck: (1) -y_name is defined or is"
				" 'Y', (2) your data is tab delimited or -sep is specified")
				quit()
			X = df.drop(['Y'], axis=1)
			min_max_scaler = preprocessing.MinMaxScaler()
			X_scaled = min_max_scaler.fit_transform(X)
			df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
			df.insert(loc=0, column='Y', value=y)

	# Set up dataframe of unknown instances that the final models will be 
	# applied to and drop unknowns from df for model building
	if args.apply != '':
		df_unknowns = df[df['Y'].str.match(args.apply)]
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		df = df.drop(df_unknowns.index.values)
		print("Model trained on %i instances & applied to %i unknown instances "
			"(see _scores for unkns)" % (len(df.index), len(df_unknowns.index)))
	else:
		predictions = pd.DataFrame(data=df['Y'], index=df.index, columns=['Y'])
		print("Model built using %i instances" % len(df.index))

	# Make sure Y is datatype numeric
	df['Y'] = pd.to_numeric(df['Y'], errors='raise')
	if args.y_norm in ['t', 'true']:
		print('Normalizing Y...')
		mean = df['Y'].mean(axis=0)
		std = df['Y'].std(axis=0)
		df['Y'] = (df['Y'] - mean) / std

	# Separte test intances from training/validation 
	if args.test != '':
		print('Removing test instances to apply model on later...')
		with open(args.test) as test_file:
			test_instances = test_file.read().splitlines()
		try:
			test_df = df.loc[test_instances, :]
			df = df.drop(test_instances)
		except:
			test_instances = [int(x) for x in test_instances]
			test_df = df.loc[test_instances, :]
			df = df.drop(test_instances)
	else:
		test_df = 'None'
		test_instances = 'None'

	if args.save == "":
		if args.tag == "":
			if args.out_loc == "":
				args.save = args.df + "_" + args.alg
			else:
				args.save = args.out_loc + '/' + args.df + "_" + args.alg
		else:
			if args.out_loc == "":
				args.save = args.df + "_" + args.alg + "_" + args.tag
			else:
				args.save = (args.out_loc + '/' + args.df + "_" + args.alg +
					"_" + args.tag)

	print("Snapshot of data being used:")
	print(df.ix[:5, :5])

	n_features = len(list(df)) - 1

	####### Run parameter sweep using a grid search #######

	if args.gs.lower() in ['true', 't']:
		start_time = time.time()

		if args.alg.lower() == 'lr':
			print("\nNo grid search for linear regression...\n")
			params2use, param_names = 'NA'

		else:
			print("\n\n===>  Grid search started  <===")
			params2use, param_names = ML.fun.RegGridSearch(df, args.save,
				args.alg, args.gs_score, args.n, args.cv_num, args.n_jobs,
				args.gs_reps, args.gs_type, args.gs_full)

			# Print results from grid search
			if args.alg.lower() == 'rf':
				args.max_depth, args.max_features, args.n_estimators = params2use
				print("Parameters selected: max_depth=%s, max_features=%s, "
					"n_estimators=%s" % (str(args.max_depth),
					str(args.max_features), str(args.n_estimators)))

			elif args.alg.lower() == 'svm':
				args.C = params2use
				print("Parameters selected: Kernel=Lin, C=%s" % (str(args.C)))

			elif args.alg.lower() == "svmpoly":
				args.C, args.degree, args.gamma, kernel = params2use
				print("Parameters selected: Kernel=%s, C=%s, degree=%s,"
					" gamma=%s" % (str(kernel), str(args.C), str(args.degree),
					str(args.gamma)))

			elif args.alg.lower() == "svmrbf":
				args.C, args.gamma, kernel = params2use
				print("Parameters selected: Kernel=%s, C=%s, gamma=%s" % (
					str(kernel), str(args.C), str(args.gamma)))

			elif args.alg.lower() == "gb":
				args.lr, args.max_depth, args.max_features, args.n_estimators = params2use
				print("Parameters selected: learning rate=%s, max_features=%s, "
					"max_depth=%s, n_estimators=%s" % (str(args.lr),
						str(args.max_features), str(args.max_depth),
						str(args.n_estimators)))

			else:
				params2use = "No Grid Search... Default parameters used..."
			print("Grid search done. Time: %f s" % (time.time() - start_time))
	else:
		params2use = "No Grid Search... Default parameters used..."

	####### Run ML models #######
	start_time = time.time()
	print("\n\n===>  ML Pipeline started  <===")

	results = []
	results_test = []
	imp = pd.DataFrame(index=list(df.drop(['Y'], axis=1)))

	for j in range(0, args.n):
		print("Running %i of %i" % (j + 1, args.n))
		rep_name = "rep_" + str(j + 1)

		# Prime regressor object based on chosen algorithm
		if args.alg.lower() == "rf":
			reg = ML.fun.DefineReg_RandomForest(args.n_estimators,
				args.max_depth, args.max_features, args.n_jobs, j)
		elif args.alg.lower() == "svm":
			reg = ML.fun.DefineReg_LinearSVM(args.C, j)
		elif args.alg.lower() in ['svmrbf', 'svmpoly']:
			reg = ML.fun.DefineReg_SVM(kernel, args.C, args.degree,
				args.gamma, j)
		elif args.alg.lower() == "gb":
			reg = ML.fun.DefineReg_GB(args.n_estimators, args.lr,
				args.max_features, args.max_depth, args.n_jobs, j)
		elif args.alg.lower() == "lr":
			reg = ML.fun.DefineReg_LinReg()
		else:
			print('Algorithm not available...')
			quit()

		# Run ML algorithm.
		if args.test != '':
			result, cv_pred, importance, result_test = ML.fun.Run_Regression_Model(
				df, reg, args.cv_num, args.alg, df_unknowns, test_df,
				args.cv_sets, j)
			results_test.append(result_test)
		else:
			result, cv_pred, importance = ML.fun.Run_Regression_Model(
				df, reg, args.cv_num, args.alg, df_unknowns,
				test_df, args.cv_sets, j)

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

	MSE_stats = [np.mean(mses), np.std(mses), np.std(mses) / np.sqrt(len(mses))]
	EVS_stats = [np.mean(evss), np.std(evss), np.std(evss) / np.sqrt(len(evss))]
	r2_stats = [np.mean(r2s), np.std(r2s), np.std(r2s) / np.sqrt(len(r2s))]
	PCC_stats = [np.mean(cors), np.std(cors), np.std(cors) / np.sqrt(len(cors))]

	# Get scores from test set:
	if args.test != '':
		mses_test, evss_test, r2s_test, cors_test = [], [], [], []
		for r in results_test:
			mses_test.append(r[0])
			evss_test.append(r[1])
			r2s_test.append(r[2])
			cors_test.append(r[3])

		MSE_test_stats = [np.mean(mses_test), np.std(mses_test),
			(np.std(mses_test) / np.sqrt(len(mses_test)))]
		EVS_test_stats = [np.mean(evss_test), np.std(evss_test),
			np.std(evss_test) / np.sqrt(len(evss_test))]
		r2_test_stats = [np.mean(r2s_test), np.std(r2s_test),
			np.std(r2s_test) / np.sqrt(len(r2s_test))]
		PCC_test_stats = [np.mean(cors_test), np.std(cors_test),
			np.std(cors_test) / np.sqrt(len(cors_test))]

	else:
		MSE_test_stats, EVS_test_stats = ['na', 'na', 'na'], ['na', 'na', 'na']
		r2_test_stats, PCC_test_stats = ['na', 'na', 'na'], ['na', 'na', 'na']


	# Get average predicted value
	pred_columns = [c for c in predictions.columns if c.startswith('rep_')]
	predictions.insert(loc=1, column='Mean',
		value=predictions[pred_columns].mean(axis=1))
	predictions.insert(loc=2, column='stdev',
		value=predictions[pred_columns].std(axis=1))

	scores_file = args.save + "_scores.txt"
	if args.short in ['t', 'true']:
			predictions.to_csv(scores_file, sep='\t',
				columns=['Y', 'Mean', 'stdev'])
	else:
		predictions.to_csv(scores_file, sep='\t')

	# Plot results
	if args.plots.lower() in ['true', 't']:
		print("\nGenerating prediction plot")
		ML.fun.PlotsReg(predictions, args.save)

	# Export importance scores
	try:
		imp['mean_imp'] = imp.mean(axis=1)
		imp = imp.sort_values('mean_imp', 0, ascending=False)
		imp_out = args.save + "_imp"
		imp['mean_imp'].to_csv(imp_out, sep="\t", index=True)
	except:
		pass

	run_time = time.time() - start_total_time

	# Save to summary RESULTS file with all models run from the same directory
	if not os.path.isfile('RESULTS_reg.txt'):
		out2 = open('RESULTS_reg.txt', 'a')
		out2.write('DateTime\tRunTime\tID\tTag\tY\tAlg\tNumInstances\t'
			'FeatureNum\tCVfold\tCV_rep\tMSE_val\tMSE_val_sd\tMSE_val_se\t'
			'EVS_val\tEVS_val_sd\tEVS_val_se\tr2_val\tr2_val_sd\tr2_val_se\t'
			'PCC_val\tPCC_val_sd\tPCC_val_se\tMSE_test\tMSE_test_sd\t'
			'MSE_test_se\tEVS_test\tEVS_test_sd\tEVS_test_se\tr2_test\t'
			'r2_test_sd\tr2_test_se\tPCC_test\tPCC_test_sd\tPCC_test_se\n')
		out2.close()

	out2 = open('RESULTS_reg.txt', 'a')
	out2.write('%s\t%s\t%s\t%s\t%s\t%s\t%i\t%i\t%i\t%i\t%s\t%s\t%s\t%s\t%s\t%s'
		'\t%s\t%s\n' % (
		timestamp, run_time, args.save, args.tag, args.y_name, args.alg,
		len(df.index), n_features, args.cv_num, args.n,
		'\t'.join(str(x) for x in MSE_stats),
		'\t'.join(str(x) for x in EVS_stats),
		'\t'.join(str(x) for x in r2_stats),
		'\t'.join(str(x) for x in PCC_stats),
		'\t'.join(str(x) for x in MSE_test_stats),
		'\t'.join(str(x) for x in EVS_test_stats),
		'\t'.join(str(x) for x in r2_test_stats),
		'\t'.join(str(x) for x in PCC_test_stats)))


	# Save detailed results file 
	with open(args.save + "_results.txt", 'w') as out:
		out.write('%s\nID: %s\nTag: %s\nPredicting: %s\nAlgorithm: %s\nNumber'
		' of Instances: %s\nNumber of features: %i\n' % (
			timestamp, args.save, args.tag, args.y_name, args.alg,
			len(df.index), n_features))
		out.write('CV folds: %i\nCV_reps: %i\nParameters used:%s\n' % (
			args.cv_num, args.n, params2use))
		out.write('\n\nResults from the validation set\nMetric\tMean\tstd'
			'\tSE\nMSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_stats),
			'\t'.join(str(x) for x in EVS_stats),
			'\t'.join(str(x) for x in r2_stats),
			'\t'.join(str(x) for x in PCC_stats)))

		if args.test != '':
			out.write('\n\nResults from the test set\nMetric\tMean\tstd\tSE\n'
				'test MSE\t%s\ntest EVS\t%s\ntest R2\t%s\ntest PCC\t%s\n' % (
					'\t'.join(str(x) for x in MSE_test_stats),
					'\t'.join(str(x) for x in EVS_test_stats),
					'\t'.join(str(x) for x in r2_test_stats),
					'\t'.join(str(x) for x in PCC_test_stats)))

	print("\n\n===>  ML Results  <===")

	print('\nValidation Set Scores:\nMetric\tMean\tstd\tSE\n')
	print('Metric\tMean\tstd\tSE')
	print('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
		'\t'.join(str(x) for x in MSE_stats),
		'\t'.join(str(x) for x in EVS_stats),
		'\t'.join(str(x) for x in r2_stats),
		'\t'.join(str(x) for x in PCC_stats)))

	if args.test !='':
		print('\n\nTest Set Scores:\nMetric\tMean\tstd\tSE\n')
		print('MSE\t%s\nEVS\t%s\nR2\t%s\nPCC\t%s\n' % (
			'\t'.join(str(x) for x in MSE_test_stats),
			'\t'.join(str(x) for x in EVS_test_stats),
			'\t'.join(str(x) for x in r2_test_stats),
			'\t'.join(str(x) for x in PCC_test_stats)))

	print('\nfinished!')

if __name__ == '__main__':
	main()
