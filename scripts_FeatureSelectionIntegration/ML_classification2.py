'''
ML_classification2.py

This script is an extension of ML_classification.py. The key difference in
goal is that this script will manually cross-validate the training instances
so that feature selection and parameter grid searches can be performed under
cross-validation. In addition, performing multiple interations of cross-
validation splits is supported (default = 30 iterations).

Compared to ML_classification.py, this script does not currently support 
multi-class classification and has stricter requirements on the input data
frame.

REQUIRED ARGS:
  -df   input data frame
          col1:  instance IDs
          col2:  Class
          col3+: Features
            Classes may also be provided via a
            separate data frame (see -df2)
  -alg  Algorithm, available: RF, LogReg, SVM

OPTIONAL ARGS:
  -df2       data frame with classes (if not included in -df)
  -df2_col   Column to use in df2 (default = column 2)
  -pos       Name of positive class (default = 1)
  -neg       Name of negative class (default = 0)
  -b         # of random balanced datasets to run (default = 50)
               I.e., model replicates
  -prop      Proportion by which to downsample balanced 
               datasets (default = 1, no downsampling)
               EX: 0.75 = select 75% of possible instances
  -save      prefix for file outputs (default = df name)
  -cv        # of cross-validation folds (default = 10)
  -cv_itr    # of CV iterations (default = 30)
  -fs        Perform feature selection with LASSO? y/n (default = y)
  -l1        lambda penalty for LASSO (default = 0.1)
  -feat      file with features to consider during feature selection
               1 column file. (default = all features)
  -gs        Perform a parameter grid search? y/n (default = n)
  -n_jobs    # of processors to use (default = 1)
  -detail    Detail in final reports (default = 1)
              Prediction scores:
               0: Mean/SD of all scores only
               1: Mean/SD of each CV iteration
               2: All scores (NOTE: Can be large: 1,500 scores by default, 
                                    15,000 for non-POS/NEG instances) 
              Feature importances:
               0 or 1: Average feat imp across balanced datasets within a CV fold
		       (n=300 data frames by default)
               2: Feat imp for each balanced dataset (n=15,000 by default)

Author: John Lloyd
Creation Date: 28 June 2018
'''

# import modules
import os, sys
import ML_utils as ML
#import numpy as np
import pandas as pd

# TO reload modules:
# import importlib
# importlib.reload(MODULE)

def main():
	
	if len(sys.argv) == 1:
		print(__doc__)
		sys.exit()
	
	print("\nReading arguments")
	
	# set defaults
	POS = 1
	NEG = 0
	n_folds = 10
	n_CVitr = 30
	n_bal = 50
	save_prefix = None
	
	DF2 = None
	df2_col = 2
	
	FEAT = None
	featSel = True
	gridSearch = False
	full_scores = True
	
	detail = 1
	l1 = 0.1
	prop = 1.0
	NA_axis = 0
	n_jobs = 1	
	
	# read arguments
	for i in range (1,len(sys.argv),2):
		if sys.argv[i] == "-df":
			DF = sys.argv[i+1]
		elif sys.argv[i] == "-alg":
			ALG = sys.argv[i+1]
		elif sys.argv[i] == "-cv":
			n_folds = int(sys.argv[i+1])
		elif sys.argv[i] == "-cv_itr":
			n_CVitr = int(sys.argv[i+1])
		elif sys.argv[i] == "-df2":
			DF2 = sys.argv[i+1]
		elif sys.argv[i] == "-df2_col":
			df2_col = int(sys.argv[i+1])
		elif sys.argv[i] == "-pos":
			POS = sys.argv[i+1]
		elif sys.argv[i] == "-neg":
			NEG = sys.argv[i+1]
		elif sys.argv[i] == "-b":
			n_bal = int(sys.argv[i+1])
		elif sys.argv[i] == "-fs":
			fs = sys.argv[i+1]
			featSel = ML.boolean_argument(fs, "Feature selection", True)
		elif sys.argv[i] == "-gs":
			gs = sys.argv[i+1]
			gridSearch = ML.boolean_argument(gs, "Grid search", False)
		elif sys.argv[i] == "-l1":
			l1 = float(sys.argv[i+1])
		elif sys.argv[i] == "-prop":
			prop = float(sys.argv[i+1])
		elif sys.argv[i] == "-save":
			save_prefix = sys.argv[i+1]
		elif sys.argv[i] == "-feat":
			FEAT = sys.argv[i+1]
		elif sys.argv[i] == "-full":
			full = sys.argv[i+1]
			full_scores = ML.boolean_argument(full, "Full scores", True)
		elif sys.argv[i] == "-detail":
			detail = int(sys.argv[i+1])
		elif sys.argv[i] == "-dropNA":
			na = sys.argv[i+1]
			NA_axis = ML.check_numeric_arg( na, 0 )
		elif sys.argv[i] == "-n_jobs":
			n_jobs = int(sys.argv[i+1])
		elif sys.argv[i].startswith("-"):
			print("WARNING! Flag not recognized and ignored:", sys.argv[i])
	
	if save_prefix == None:
		save_prefix = DF
	
	print("Processing dataframe(s)")
	df, classes, Y_name, df_nonTrain = ML.process_dataframe (DF, DF2, FEAT, POS, NEG, ALG, df2_col, NA_axis)
	print(classes)
	# df_notTrain = None, if only POS and NEG classes in data frame
	
	
	# dicts to store inputs, predictions, and feature importances
	# KEY1 = CV_iteration
	#   KEY2 = string to indicate data type (e.g. "feat_d", "prediction_scores")
	#     KEY3 = CV_fold
	inputs_d = {}
	predictions_d = {}
	featImp_d = {}
	
	for i in range(1, n_CVitr+1):
		
		print("\n\nCROSS-VALIDATION ITERATION %s\n"%i)
		
		inputs_d[i] = {}
		predictions_d[i] = {}
		featImp_d[i] = {}
		
		instanceIDs, foldIDs = ML.CV_folds_by_class (df, Y_name, [1, 0], n_folds)
		feat_d = ML.featSel_wrapper( featSel, df, Y_name, instanceIDs, foldIDs, l1 )
		param_d, parameter_names = ML.runGridSearch_wrapper( gridSearch, df, Y_name, instanceIDs, foldIDs, feat_d, ALG, n_jobs )
		balIDs_d = ML.balancedIDs_CV(df, Y_name, instanceIDs, foldIDs, n_bal, prop)
		
		inputs_d[i]["instanceIDs"] = instanceIDs
		inputs_d[i]["foldIDs"] = foldIDs
		inputs_d[i]["feat_d"] = feat_d
		inputs_d[i]["param_d"] = param_d
		inputs_d[i]["balIDs_d"] = balIDs_d
		
		prediction_scores, nonTrain_scores, featImpCV_d = ML.train_test_CV (df, Y_name, instanceIDs, foldIDs, feat_d, param_d, balIDs_d, ALG, df_nonTrain, i, n_jobs)
		
		predictions_d[i]["prediction_scores"] = prediction_scores
		predictions_d[i]["nonTrain_scores"] = nonTrain_scores
		
		featImp_d[i] = featImpCV_d
	
	ML.write_CV_folds_CVitr(save_prefix, inputs_d, n_CVitr, n_folds)
	ML.write_features_CVitr(save_prefix, featSel, l1, inputs_d)
	ML.write_parameters_CVitr(save_prefix, inputs_d, parameter_names)	
	ML.write_balIDs_CVitr(save_prefix, inputs_d)
	
	#ML.write_featImp_CVitr(save_prefix, featImp_d)
		
	#scores, nonTrain_scores = ML.concat_full_scores(predictions_d, df, df_nonTrain)
	#scores_wMeanSD = ML.make_df_mean_std(scores)
		
	ML.write_score_overview(save_prefix, predictions_d, df, df_nonTrain, Y_name, POS, NEG)
	ML.write_scores(save_prefix, detail, predictions_d, df, df_nonTrain, Y_name, POS, NEG)
	
	ML.write_featImp(save_prefix, detail, featImp_d)
	
	print("Calculating performance metrics")
	scores_name = "%s.d1.scores" % (save_prefix)
	featImp_name = "%s.d1.featImp" % save_prefix
	os.system( "Rscript /home/lloydjp/bin/ML-Pipeline/ML_postprocessing.R -scores %s -featImp %s -pos %s -neg %s -save %s" % (scores_name, featImp_name, POS, NEG, save_prefix) )
	
	print("\nDone!")
	

if __name__ == "__main__":
	main()

