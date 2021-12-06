"""
PURPOSE: Functions for Shiu Lab ML-Pipeline

"""
import sys
import pandas as pd
import numpy as np
import time
import random as rn
import math

class fun(object):
	def __init__(self, filename):
		self.tokenList = open(filename, 'r')

	def EstablishBalanced(df, classes, min_size, gs_n):
		""" Defines which instances will be used for each balanced dataset """
		class_ids_dict = {}
		for cl in classes:
			cl_ids = list(df[df["Class"] == cl].index)
			class_ids_dict[cl] = cl_ids

		# Build a list of lists containing the IDs for balanced datasets
		bal_list = []
		for j in range(gs_n):
			tmp_l = []
			for cl in class_ids_dict:
				bal_samp = rn.sample(class_ids_dict[cl], min_size)
				tmp_l = tmp_l + bal_samp
			bal_list.append(tmp_l)
		return bal_list

	def param_space(ALG, GS_TYPE, n):
		"Define the parameter space for the grid search."
		from numpy import random as npr

		dist_max_features = npr.choice(list(np.arange(0.01, 1., 0.01)) +
			['sqrt', 'log2', None], n)
		dist_max_depth = npr.randint(1, 50, n)
		dist_C = npr.uniform(1e-10, 10, n)
		dist_degree = npr.randint(2, 4, n)
		dist_gamma = npr.uniform(1e-7, 1, n)
		dist_learnrate = npr.uniform(1e-5, 1, n)
		dist_intscaling = npr.uniform(0, 10, n)
		dist_penalty = npr.choice(['l1', 'l2'], n)
		dist_nestimators = npr.choice(range(100, 1000, 100), n)

		if GS_TYPE.lower() == 'rand' or GS_TYPE.lower() == 'random':
			if ALG.lower() == 'rf':
				parameters = {'max_depth': dist_max_depth,
				'max_features': dist_max_features,
				'n_estimators': dist_nestimators}
			elif ALG.lower() == "svm":
				parameters = {'C': dist_C}
			elif ALG.lower() == 'svmpoly':
				parameters = {'kernel': ['poly'] * n, 'C': dist_C,
				'degree': dist_degree, 'gamma': dist_gamma}
			elif ALG.lower() == 'svmrbf':
				parameters = {'kernel': ['rbf'] * n, 'C': dist_C,
				'gamma': dist_gamma}
			elif ALG.lower() == 'logreg':
				parameters = {'C': dist_C, 'intercept_scaling': dist_intscaling,
				'penalty': dist_penalty}
			elif ALG.lower() == 'gb':
				parameters = {'learning_rate': dist_learnrate,
				'max_depth': dist_max_depth, 'max_features': dist_max_features,
				'n_estimators': dist_nestimators}
			else:
				print('Grid search is not available for the algorithm selected')
				exit()

		else:
			if ALG.lower() == 'rf':
				parameters = {'max_depth': [3, 5, 10],
				'max_features': [0.1, 0.5, 'sqrt', 'log2', None],
				'n_estimators': [100, 500, 1000]}
			elif ALG.lower() == "svm":
				parameters = {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100]}
			elif ALG.lower() == 'svmpoly':
				parameters = {'kernel': ['poly'],
				'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100], 'degree': [2, 3],
				'gamma': np.logspace(-5, 1, 7)}
			elif ALG.lower() == 'svmrbf':
				parameters = {'kernel': ['rbf'],
				'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100],
				'gamma': np.logspace(-5, 1, 7)}
			elif ALG.lower() == 'logreg':
				parameters = {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 50],
				'intercept_scaling': [0.1, 0.5, 1, 2, 5, 10],
				'penalty': ['l1','l2']}
			elif ALG.lower() == 'gb':
				parameters = {'learning_rate': [0.1, 0.001, 0.01, 0.1, 1],
				'max_depth': [3, 5, 10],
				'max_features': [0.1, 0.5, 'sqrt', 'log2', None],
				'n_estimators': [100, 500, 1000]}
			else:
				print('Grid search is not available for the algorithm selected')
				exit()

		return parameters

	def GridSearch(df, SAVE, ALG, classes, min_size, gs_score, n, cv_num,
		n_jobs, GS_REPS, GS_TYPE, POS, NEG, gs_full):
		""" Perform a parameter sweep using GridSearchCV from SK-learn.
		Need to edit the hard code to modify what parameters are searched
		"""
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import RandomizedSearchCV
		from sklearn.preprocessing import StandardScaler

		start_time = time.time()
		n_iter = 10
		parameters = fun.param_space(ALG, GS_TYPE, n_iter)

		gs_results = pd.DataFrame(columns=['mean_test_score', 'params'])
		
		bal_ids_list = []
		for j in range(n):
			# Build balanced dataframe and define x & y
			df1 = pd.DataFrame(columns=list(df))
			for cl in classes:
				temp = df[df['Class'] == cl].sample(min_size, random_state=j)
				df1 = pd.concat([df1, temp])
			
			bal_ids_list.append(list(df1.index))

			if j < GS_REPS:
				print("Round %s of %s"%(j+1,GS_REPS))
				y = df1['Class']
				x = df1.drop(['Class'], axis=1)
				
				# Build model, run grid search 10-fold CV, and fit
				if ALG.lower() == 'rf':
					from sklearn.ensemble import RandomForestClassifier
					model = RandomForestClassifier()
				elif ALG.lower() == "svm":
					from sklearn.svm import LinearSVC
					model = LinearSVC()
				elif ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
					from sklearn.svm import SVC
					# x = StandardScaler().fit_transform(x)
					model = SVC(probability=True)
				elif ALG.lower() == "logreg":
					from sklearn.linear_model import LogisticRegression
					model = LogisticRegression()
				elif ALG.lower() == "gb":
					from sklearn.ensemble import GradientBoostingClassifier
					model = GradientBoostingClassifier()
				
				if gs_score.lower() == 'auprc':
					gs_score = 'average_precision'

				if GS_TYPE.lower() == 'rand' or GS_TYPE.lower() == 'random':
					grid_search = RandomizedSearchCV(model,
						param_distributions=parameters, scoring=gs_score,
						n_iter=n_iter, cv=cv_num, n_jobs=n_jobs,
						pre_dispatch=2 * n_jobs, return_train_score=True)
				else:
					grid_search = GridSearchCV(model, param_grid=parameters,
						scoring=gs_score, cv=cv_num, n_jobs=n_jobs,
						pre_dispatch=2 * n_jobs, return_train_score=True)
				
				if len(classes) == 2:
					y = y.replace(to_replace=[POS, NEG], value=[1, 0])
				
				grid_search.fit(x, y)
				
				# Add results to dataframe
				j_results = pd.DataFrame(grid_search.cv_results_)
				gs_results = pd.concat([gs_results, j_results[['params',
					'mean_test_score']]])
			
		# Break params into seperate columns
		gs_results2 = pd.concat([gs_results.drop(['params'], axis=1),
			gs_results['params'].apply(pd.Series)], axis=1)
		param_names = list(gs_results2)[1:]

		if gs_full.lower() == 't' or gs_full.lower() == 'true':
			gs_results2.to_csv(SAVE + "_GridSearchFULL.txt")
		
		# Find the mean score for each set of parameters & select the top set
		gs_results_mean = gs_results2.groupby(param_names).mean()
		gs_results_mean = gs_results_mean.sort_values('mean_test_score', 0,
			ascending=False)
		top_params = gs_results_mean.index[0]

		print("Parameter sweep time: %f seconds" % (time.time() - start_time))

		# Save grid search results
		outName = open(SAVE + "_GridSearch.txt", 'w')
		outName.write('# %f sec\n' % (time.time() - start_time))
		gs_results_mean.to_csv(outName)
		outName.close()
		return top_params,bal_ids_list, param_names
	
	def RegGridSearch(df, SAVE, ALG, gs_score, n, cv_num, n_jobs, GS_REPS,
		GS_TYPE, gs_full):
		""" Perform a parameter sweep using GridSearchCV from SK-learn. 
		Need to edit the hard code to modify what parameters are searched"""
		from sklearn.metrics import mean_squared_error, r2_score
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import RandomizedSearchCV
		from sklearn.preprocessing import StandardScaler

		start_time = time.time()
		n_iter = 10
		parameters = fun.param_space(ALG, GS_TYPE, n_iter)

		y = df['Y']
		x = df.drop(['Y'], axis=1)

		gs_results = pd.DataFrame(columns=['mean_test_score', 'params'])

		#if j < GS_REPS:
		for j in range(GS_REPS):
			print("Round %s of %s" % (j + 1, GS_REPS))

			# Build model
			if ALG.lower() == 'rf':
				from sklearn.ensemble import RandomForestRegressor
				model = RandomForestRegressor()
			elif ALG.lower() == "svm" :
				from sklearn.svm import LinearSVR
				model = LinearSVR()
			elif ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
				from sklearn.svm import SVR
				model = SVR()
			elif ALG.lower() == "gb":
				from sklearn.ensemble import GradientBoostingRegressor
				model = GradientBoostingRegressor()

			# Run grid search with 10-fold cross validation and fit
			if GS_TYPE.lower() == 'rand' or GS_TYPE.lower() == 'random':
				grid_search = RandomizedSearchCV(model, parameters,
					scoring=gs_score, n_iter=n_iter, cv=cv_num, n_jobs=n_jobs,
					pre_dispatch=2 * n_jobs, return_train_score=True)
			else:
				grid_search = GridSearchCV(model, parameters, scoring=gs_score,
					cv=cv_num, n_jobs=n_jobs, pre_dispatch=2 * n_jobs,
					return_train_score=True)
			grid_search.fit(x, y)

			# Add results to dataframe
			j_results = pd.DataFrame(grid_search.cv_results_)
			gs_results = pd.concat([gs_results, j_results[['params',
				'mean_test_score']]])
		
		# Break params into seperate columns
		gs_results2 = pd.concat([gs_results.drop(['params'], axis=1),
			gs_results['params'].apply(pd.Series)], axis=1)
		
		if gs_full.lower() == 't' or gs_full.lower() == 'true':
			gs_results.to_csv(SAVE + "_GridSearchFULL.txt")
		param_names = list(gs_results2)[1:]
		#print('Parameters tested: %s' % param_names)
		
		# Find the mean score for each set of parameters & select the top set
		gs_results_mean = gs_results2.groupby(param_names).mean()
		gs_results_mean = gs_results_mean.sort_values('mean_test_score',
			0, ascending=False)
		top_params = gs_results_mean.index[0]
		print(gs_results_mean.head())
		
		# Save grid search results
		print("Parameter sweep time: %f seconds" % (time.time() - start_time))
		outName = open(SAVE + "_GridSearch.txt", 'w')
		outName.write('# %f sec\n' % (time.time() - start_time))
		gs_results_mean.to_csv(outName)
		outName.close()
		return top_params, param_names

	

	def DefineClf_RandomForest(n_estimators, max_depth, max_features, j, n_jobs):
		from sklearn.ensemble import RandomForestClassifier
		clf = RandomForestClassifier(n_estimators=int(n_estimators),
			max_depth=max_depth,
			max_features=max_features,
			criterion='gini',
			random_state=j,
			n_jobs=n_jobs)
		return clf
	
	def DefineReg_RandomForest(n_estimators, max_depth, max_features, n_jobs, j):
		from sklearn.ensemble import RandomForestRegressor
		reg = RandomForestRegressor(n_estimators=int(n_estimators),
			max_depth=max_depth,
			max_features=max_features,
			criterion='mse',
			random_state=j,
			n_jobs=n_jobs)
		return reg

	def DefineReg_GB(n_estimators, learning_rate, max_features, max_depth, 
		n_jobs, j):
		from sklearn.ensemble import GradientBoostingRegressor
		reg = GradientBoostingRegressor(loss='ls',
			learning_rate=learning_rate,
			max_features=max_features,
			max_depth=max_depth,
			n_estimators=int(n_estimators),
			random_state=j)
		return reg

	def DefineClf_GB(n_estimators, learning_rate, max_features, max_depth,
		n_jobs, j):
		from sklearn.ensemble import GradientBoostingClassifier
		reg = GradientBoostingClassifier(loss='deviance',
			learning_rate=learning_rate,
			max_features=max_features,
			max_depth=max_depth,
			n_estimators=int(n_estimators),
			random_state=j)
		return reg

	def DefineClf_SVM(kernel, C, degree, gamma, j):
		from sklearn.svm import SVC
		clf = SVC(kernel=kernel,
			C=float(C),
			degree=degree,
			gamma=gamma,
			random_state=j,
			probability=True)
		return clf

	def DefineReg_SVM(kernel, C, degree, gamma, j):
		from sklearn.svm import SVR
		reg = SVR(kernel=kernel,
			C=float(C),
			degree=degree,
			gamma=gamma)
		return reg
	
	def DefineClf_LinearSVM(C, j):
		from sklearn.svm import LinearSVC
		clf = LinearSVC(C=float(C), random_state=j)
		return clf

	def DefineReg_LinearSVM(C, j):
		from sklearn.svm import LinearSVR
		reg = LinearSVR(C=float(C))
		return reg

	def DefineClf_LogReg(penalty, C, intercept_scaling):
		from sklearn.linear_model import LogisticRegression
		clf = LogisticRegression(penalty=penalty,
			C=float(C),
			intercept_scaling=intercept_scaling)
		return clf

	def DefineReg_LinReg():
		from sklearn import linear_model
		reg = linear_model.LinearRegression()
		return reg

	def BuildModel_Apply_Performance(df, clf, cv_num, df_notSel, apply_unk,
		df_unknowns, test_df, classes, POS, NEG, j, ALG, THRSHD_test):
		from sklearn.model_selection import cross_val_predict

		# Data from balanced dataframe
		y = df['Class']
		X = df.drop(['Class'], axis=1)

		# For LinearSVM need to have calibrated classifier to get probability
		# scores, but not for importance scores
		if ALG.lower() == 'svm':
			from sklearn.calibration import CalibratedClassifierCV
			clf2 = clf
			clf2.fit(X,y)
			# adds the probability output to linearSVC
			clf = CalibratedClassifierCV(clf, cv=3)
		else:
			clf2 = 'pass'

		# Obtain the predictions using 10 fold cross validation
		# (uses KFold cv by default):
		cv_proba = cross_val_predict(estimator=clf, X=X, y=y, cv=int(cv_num),
			method='predict_proba')
		cv_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=cv_num)

		# Fit a model using all data and apply to 
		# (1) instances that were not selected using cl_train
		# (2) instances with unknown class
		# (3) test instances
		clf.fit(X,y)

		notSel_proba = clf.predict_proba(df_notSel.drop(['Class'], axis=1))
		if apply_unk == True:
			unk_proba = clf.predict_proba(df_unknowns.drop(['Class'], axis=1))
		if not isinstance(test_df, str):
			test_proba = clf.predict_proba(test_df.drop(['Class'], axis=1))
			test_pred = clf.predict(test_df.drop(['Class'], axis=1))

		# Evaluate performance
		if len(classes) == 2:
			i = 0
			for clss in classes:
				if clss == POS:
					POS_IND = i
					break
				i += 1
			scores = cv_proba[:, POS_IND]

			# Generate run statistics from balanced dataset scores
			result = fun.Performance(y, cv_pred, scores, clf, clf2, classes,
				POS, POS_IND, NEG, ALG, THRSHD_test)

			#Generate data frame with all scores
			score_columns=["score_%s"%(j)]
			df_sel_scores = pd.DataFrame(data=cv_proba[:, POS_IND], 
				index=df.index, columns=score_columns)
			df_notSel_scores = pd.DataFrame(data=notSel_proba[:,POS_IND],
				index=df_notSel.index, columns=score_columns)
			current_scores = pd.concat([df_sel_scores, df_notSel_scores],
				axis=0)
			if apply_unk == True:
				df_unk_scores = pd.DataFrame(data=unk_proba[:, POS_IND],
					index=df_unknowns.index, columns=score_columns)
				current_scores =  pd.concat([current_scores,df_unk_scores],
					axis=0)
			if not isinstance(test_df, str):
				df_test_scores = pd.DataFrame(data=test_proba[:,POS_IND],
					index=test_df.index, columns=score_columns)
				current_scores =  pd.concat([current_scores, df_test_scores],
					axis=0)
				scores_test = test_proba[:,POS_IND]
				result_test = fun.Performance(test_df['Class'], test_pred,
					scores_test, clf, clf2, classes, POS, POS_IND, NEG, ALG,
					THRSHD_test)

		else:
			# Generate run statistics from balanced dataset scores
			result = fun.Performance_MC(y, cv_pred, classes)

			#Generate data frame with all scores
			score_columns = []
			for clss in classes:
				score_columns.append("%s_score_%s"%(clss, j))

			df_sel_scores = pd.DataFrame(data=cv_proba, index=df.index,
				columns=score_columns)
			df_notSel_scores = pd.DataFrame(data=notSel_proba,
				index=df_notSel.index, columns=score_columns)
			current_scores = pd.concat([df_sel_scores, df_notSel_scores],
				axis=0)
			if apply_unk:
				df_unk_scores = pd.DataFrame(data=unk_proba,
					index=df_unknowns.index, columns=score_columns)
				current_scores =  pd.concat([current_scores, df_unk_scores],
					axis=0)
			if not isinstance(test_df, str):
				df_test_scores = pd.DataFrame(data=test_proba,
					index=test_df.index, columns=score_columns)
				current_scores = pd.concat([current_scores, df_test_scores],
					axis=0)
				result_test = fun.Performance_MC(test_df['Class'], test_pred,
					classes)

		# also return the fitted clf to be saved, Peipei Wang, 12/06/2021
		if not isinstance(test_df, str):
			return clf, result,current_scores,result_test
		else:
			return clf, result,current_scores

	def Run_Regression_Model(df, reg, cv_num, ALG, df_unknowns, test_df,
		cv_sets, j):
		from sklearn.model_selection import cross_val_predict
		from sklearn.metrics.scorer import make_scorer
		from sklearn.metrics import mean_squared_error, r2_score
		from sklearn.metrics import explained_variance_score
		# Data from balanced dataframe
		y = df['Y']
		X = df.drop(['Y'], axis=1)

		# Obtain the predictions using 10 fold cross validation
		# (uses KFold cv by default):
		if isinstance(cv_sets, pd.DataFrame):
			from sklearn.model_selection import LeaveOneGroupOut
			cv_split = LeaveOneGroupOut()
			cv_folds = cv_split.split(X, y, cv_sets.iloc[:, j])
			cv_pred = cross_val_predict(estimator=reg, X=X, y=y, cv=cv_folds)
		else:
			cv_pred = cross_val_predict(estimator=reg, X=X, y=y, cv=cv_num)

		cv_pred_df = pd.DataFrame(data=cv_pred, index=df.index,
			columns=['pred'])

		# Get performance statistics from cross-validation
		y = y.astype(float)
		mse = mean_squared_error(y, cv_pred)
		evs = explained_variance_score(y, cv_pred)
		r2 = r2_score(y, cv_pred)
		cor = np.corrcoef(np.array(y), cv_pred)
		result = [mse, evs, r2, cor[0, 1]]

		reg.fit(X, y)

		# Apply fit model to unknowns
		if isinstance(df_unknowns, pd.DataFrame):
			unk_pred = reg.predict(df_unknowns.drop(['Y'], axis=1))
			unk_pred_df = pd.DataFrame(data=unk_pred, index=df_unknowns.index,
				columns=['pred'])
			cv_pred_df = cv_pred_df.append(unk_pred_df)

		if not isinstance(test_df, str):
			test_y = test_df['Y']
			test_pred = reg.predict(test_df.drop(['Y'], axis=1))
			test_pred_df = pd.DataFrame(data=test_pred, index=test_df.index,
				columns=['pred'])
			cv_pred_df = cv_pred_df.append(test_pred_df)

			# Get performance stats
			mse_test = mean_squared_error(test_y, test_pred)
			evs_test = explained_variance_score(test_y, test_pred)
			r2_test = r2_score(test_y, test_pred)
			cor_test = np.corrcoef(np.array(test_y), test_pred)
			result_test = [mse_test, evs_test, r2_test, cor_test[0, 1]]

		# Try to extract importance scores 
		try:
			importances = reg.feature_importances_
		except:
			try:
				importances = reg.coef_
			except:
				importances = "na"
				print("Cannot get importance scores")

		# also return the fitted reg to be saved, Peipei Wang, 12/06/2021
		if not isinstance(test_df, str):
			return reg, result, cv_pred_df, importances, result_test
		else:
			return reg, result, cv_pred_df, importances

	def Performance(y, cv_pred, scores, clf, clf2, classes, POS, POS_IND,
		NEG, ALG, THRSHD_test):
		""" For binary predictions: This function calculates the best threshold
		for defining POS/NEG from the prediction probabilities by maximizing
		the f1_score. Then calcuates the area under the ROC and PRc 
		"""
		from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
		from sklearn.metrics import average_precision_score, confusion_matrix

		# Gather balanced model scoring metrics
		cm = confusion_matrix(y, cv_pred, labels=classes)

		# Determine the best threshold cutoff for the balanced run
		y1 = y.replace(to_replace=[POS, NEG], value=[1,0])
		max_f1 = -1
		max_f1_thresh = ''
		for thr in np.arange(0.01, 1, 0.01):
			thr_pred = scores.copy()
			thr_pred[thr_pred >= thr] = 1
			thr_pred[thr_pred < thr] = 0
			# Eliminates cases where all predictions are negative and
			# the f1 and auROC are undefined
			if sum(thr_pred) > 1:
				if (THRSHD_test.lower() == 'f1' or
					THRSHD_test.lower() == 'fmeasure'):
					f1 = f1_score(y1, thr_pred, pos_label=1)
				elif (THRSHD_test.lower() == 'acc' or
					THRSHD_test.lower() == 'a' or
					THRSHD_test.lower() == 'accuracy'):
					f1 = accuracy_score(y1, thr_pred)
				elif THRSHD_test.lower() == 'auprc':
					f1 = average_precision_score(y1, thr_pred)
				else:
					print('%s is not a scoring option for model thresholding' %
						THRSHD_test)
					exit()
				if f1 > max_f1:
					max_f1 = f1
					max_f1_thresh = thr

		# Calculate AUC_ROC and AUC_PRC
		AucRoc = roc_auc_score(y1, scores)
		AucPRc = average_precision_score(y1, scores)

		# Try to extract importance scores 
		if clf2 != 'pass':
			clf = clf2
		try:
			importances = clf.feature_importances_
		except:
			try:
				importances = clf.coef_
			except:
				importances = "na"
				print("Cannot get importance scores")

		return {'cm': cm, 'threshold': max_f1_thresh, 'AucPRc': AucPRc, 
			'AucRoc': AucRoc, 'MaxF1': max_f1, 'importances': importances}

	def Performance_MC(y, cv_pred, classes):
		from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

		cm = confusion_matrix(y, cv_pred, labels=classes)
		accuracy = accuracy_score(y, cv_pred)
		macro_f1 = f1_score(y, cv_pred, average='macro')
		f1 = f1_score(y, cv_pred, average=None)# Returns F1 for each class

		return {'cm': cm, 'accuracy': accuracy, 'macro_f1': macro_f1,
			'f1_MC': f1}


	def Model_Performance_Thresh(df_proba, final_threshold, balanced_ids,
		POS, NEG, test_instances):
		from sklearn.metrics import f1_score, confusion_matrix

		TP, TN, FP, FN, TPR, FPR, FNR, Precision, Accuracy, F1 = \
			[], [], [], [], [], [], [], [], [], []

		df_proba_thresh = df_proba.copy()
		proba_columns = [c for c in df_proba_thresh.columns if c.startswith('score_')]

		for proba_column in proba_columns:
			df_proba_thresh[proba_column] = np.where(
				df_proba_thresh[proba_column] > final_threshold, POS, NEG)
		balanced_count = 0

		if test_instances != 'None':
			df_proba_thresh_test = df_proba_thresh.loc[test_instances, :]
			df_proba_thresh = df_proba_thresh.drop(test_instances)


		# Get predictions scores from the balanced runs with the final threshold
		for i in proba_columns:

			# Get y and yhat for instances that were in the balanced dataset
			y = df_proba_thresh.loc[balanced_ids[balanced_count], 'Class']
			yhat = df_proba_thresh.loc[balanced_ids[balanced_count], i]
			balanced_count += 1

			matrix = confusion_matrix(y, yhat, labels=[POS, NEG])
			TP1, FP1 = matrix[0, 0], matrix[1, 0]
			TN1, FN1 = matrix[1, 1], matrix[0, 1]

			TP.append(TP1)
			FP.append(FP1)
			TN.append(TN1)
			FN.append(FN1)
			TPR.append(TP1 / (TP1 + FN1))  # synonyms: recall, sensitivity
			FPR.append(FP1 / (FP1 + TN1))  # synonyms: fall-out
			FNR.append(FN1 / (FN1 + TP1))  # synonyms: miss rate
			Precision.append(TP1 / (TP1 + FP1))  # synonyms: + predictive value
			Accuracy.append((TP1 + TN1) / (TP1 + TN1 + FP1 + FN1))
			F1.append((2 * TP1) / ((2 * TP1) + FP1 + FN1))

		denominator = np.sqrt(len(TP))
		TP = [np.mean(TP), np.std(TP), np.std(TP) / denominator]
		FP = [np.mean(FP), np.std(FP), np.std(FP) / denominator]
		TN = [np.mean(TN), np.std(TN), np.std(TN) / denominator]
		FN = [np.mean(FN), np.std(FN), np.std(FN) / denominator]
		TPR = [np.mean(TPR), np.std(TPR), np.std(TPR) / denominator]
		FPR = [np.mean(FPR), np.std(FPR), np.std(FPR) / denominator]
		FNR = [np.mean(FNR), np.std(FNR), np.std(FNR) / denominator]
		Precision = [np.mean(Precision), np.std(Precision),
			np.std(Precision) / denominator]
		Accuracy = [np.mean(Accuracy), np.std(Accuracy),
			np.std(Accuracy) / denominator]
		F1 = [np.mean(F1), np.std(F1), np.std(F1) / denominator]

		if test_instances != 'None':
			y_test = df_proba_thresh_test['Class']
			yhat_test = df_proba_thresh_test.filter(regex='Predicted_')
			matrix_test = confusion_matrix(y_test, yhat_test, labels=[POS, NEG])
			TP_test, FP_test = matrix[0, 0], matrix[1, 0],
			TN_test, FN_test = matrix[1, 1], matrix[0, 1]
			Precision_test = TP_test / (TP_test + FP_test)
			Accuracy_test = (TP_test + TN_test) / (TP_test + TN_test +
				FP_test + FN_test)
			F1_test = (2 * TP_test) / ((2 * TP_test) + FP_test + FN_test)
			return (TP, TN, FP, FN, TPR, FPR, FNR, Precision, Accuracy, F1,
				Precision_test, Accuracy_test, F1_test)
		else:
			return TP, TN, FP, FN, TPR, FPR, FNR, Precision, Accuracy, F1

	def Plots(df_proba, balanced_ids, ROC, PRc, POS, NEG, n, SAVE):
		import matplotlib.pyplot as plt
		from sklearn.metrics import roc_curve, auc, confusion_matrix
		plt.switch_backend('agg')

		FPRs = {}
		TPRs = {}
		precisions = {}

		# For each balanced dataset
		for i in range(0, n): 
			FPR = []
			TPR = []
			precis = []
			name = 'score_' + str(i)
			y = df_proba.loc[balanced_ids[i], 'Class']

			# Get decision matrix & scores at each threshold between 0 & 1
			for j in np.arange(0, 1, 0.01):
				yhat = df_proba.loc[balanced_ids[i], name].copy()
				yhat[df_proba[name] >= j] = POS
				yhat[df_proba[name] < j] = NEG
				matrix = confusion_matrix(y, yhat, labels=[POS, NEG])
				TP, FP = matrix[0, 0], matrix[1, 0]
				TN, FN = matrix[1, 1], matrix[0, 1]
				FPR.append(FP / (FP + TN))
				TPR.append(TP / (TP + FN))
				precis.append(TP / (TP + FP))

			FPRs[name] = FPR
			TPRs[name] = TPR
			precisions[name] = precis

		# Convert metric dictionaries into dataframes
		FPRs_df = pd.DataFrame.from_dict(FPRs, orient='columns')
		TPRs_df = pd.DataFrame.from_dict(TPRs, orient='columns')
		precisions_df = pd.DataFrame.from_dict(precisions, orient='columns')

		# Get summary stats 
		FPR_mean = FPRs_df.mean(axis=1)
		FPR_sd = FPRs_df.std(axis=1)
		TPR_mean = TPRs_df.mean(axis=1)
		TPR_sd = TPRs_df.std(axis=1)
		precis_mean = precisions_df.mean(axis=1)
		precis_sd = precisions_df.std(axis=1)

		# Plot the ROC Curve
		plt.title('ROC Curve: ' + SAVE)
		plt.plot(FPR_mean, TPR_mean, lw=3, color='black',
			label='AUC-ROC: ' + str(round(ROC[0], 3)))
		plt.fill_between(FPR_mean, TPR_mean - TPR_sd, TPR_mean + TPR_sd,
			facecolor='black', alpha=0.4, linewidth=0, label='SD_TPR')
		plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Expectation')
		plt.legend(loc='lower right')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		filename = SAVE + "_ROCcurve.pdf"
		plt.savefig(filename, format='pdf')
		plt.clf()

		# Plot the Precision-Recall Curve
		plt.title('PR Curve: ' + SAVE)
		plt.plot(TPR_mean, precis_mean, lw=3, color='black',
			label='AUC-PRc: ' + str(round(PRc[0], 3)))
		plt.fill_between(TPR_mean, precis_mean - precis_sd, precis_mean +
			precis_sd, facecolor='black', alpha=0.4, linewidth=0,
			label='SD_Precision')
		plt.plot([0, 1], [0.5, 0.5], 'r--', lw=2, label='Random Expectation')
		plt.legend(loc='upper right')
		plt.xlim([0, 1])
		plt.ylim([0.45, 1])
		plt.ylabel('Precision')
		plt.xlabel('Recall')
		plt.show()
		filename = SAVE + "_PRcurve.pdf"
		plt.savefig(filename, format='pdf')
		plt.close()

	def PlotsReg(predictions, SAVE):
		import matplotlib.pyplot as plt
		plt.switch_backend('agg')

		y = predictions['Y']
		yhat = predictions['Mean']
		# Plot the ROC Curve
		plt.scatter(y, yhat, edgecolors=(0, 0, 0))
		plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
		plt.ylabel('Predicted')
		plt.xlabel('Measured')
		plt.show()
		filename = SAVE + ".pdf"
		plt.savefig(filename, format='pdf')
		plt.clf()

	def Plot_ConMatrix(cm, SAVE):
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
		plt.switch_backend('agg')

		row_sums = cm.sum(axis=1)
		norm_cm = cm / row_sums

		fig, ax = plt.subplots()
		heatmap = ax.pcolor(norm_cm, vmin=0, vmax=1, cmap=plt.cm.Blues)
		fig = plt.gcf()
		plt.colorbar(heatmap)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		ax.set_frame_on(False)

		# put the major ticks at the middle of each cell
		ax.set_yticks(np.arange(norm_cm.shape[0]) + 0.5, minor=False)
		ax.set_xticks(np.arange(norm_cm.shape[1]) + 0.5, minor=False)
		ax.set_xticklabels(list(norm_cm), minor=False)
		ax.set_yticklabels(norm_cm.index, minor=False)

		filename = SAVE + "_CM.pdf"
		plt.savefig(filename, format='pdf')

		return 'Confusion matrix plotted.'
