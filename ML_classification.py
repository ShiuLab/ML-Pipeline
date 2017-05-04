"""
PURPOSE:
Machine learning classifications implemented in sci-kit learn. 

To access pandas, numpy, and sklearn packages on HPS first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH


INPUTS:
	
	REQUIRED:
	-df       Feature & class dataframe for ML. See "" for an example dataframe
	-alg 	  Available: RF, SVC (LinearSVC)
	
	OPTIONAL:
	-save     Save name. Default = df_alg (caution - will overwrite!)
	-feat     Import file with list of features to keep if desired. Default: keep all.
	-gs       Set to True if parameter sweep is desired. Default = False
	-gs_n     Number of balanced datasets to run for grid search. Default = 25
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
import time

import ML_functions as ML

def main():
	
	# Default code parameters
	n, FEAT, CL_TRAIN, CL_APPLY, n_jobs, class_col, CM, POS = 50, 'all', 'all','none', 28, 'Class', 'False', 1

	# Default parameters for Grid search
	GS, gs_score, gs_n = 'F', None, 25

	# Default Random Forest parameters
	n_estimators, max_depth, max_features = 100, 10, "sqrt"

	# Default Linear SVC parameters
	C, loss, max_iter = 1, 'hinge', "500"

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

	# Set up dataframe of unknown instances that the final model will be applied to
	if CL_APPLY != 'none':
		df_unknowns = df[(df['Class'].isin(CL_APPLY))]

	# Remove classes that won't be included in the training (i.e. to ignore or unknowns)
	if CL_TRAIN != 'all':
		df = df[(df['Class'].isin(CL_TRAIN))]

	# Determine minimum class size (for making balanced datasets)
	min_size = (df.groupby('Class').size()).min() - 1
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
			C, degree, gamma, kernel = params2use
			print("Parameters selected: Kernel=%s, C=%s, degree=%s, gamma=%s" % (str(kernel), str(C), str(degree), str(gamma)))


	

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
			parameters_used = [n_estimators, max_depth, max_features]
			result = ML.fun.RandomForest(df1, classes, POS, n_estimators, max_depth, max_features, n_jobs, j)
		elif ALG == "SVC":
			parameters_used = [C, degree, gamma, kernel]
			result = ML.fun.LinearSVC(df1, classes, POS, C, degree, gamma, kernel, n_jobs, j)
		
		results.append(result)

	print("ML Pipeline time: %f seconds" % (time.time() - start_time))



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

	
	
if __name__ == '__main__':
	main()


	