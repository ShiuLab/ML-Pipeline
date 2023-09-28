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

	def param_space(GS_TYPE, n):
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
			parameters = {'max_depth': dist_max_depth,
			'max_features': dist_max_features,
			'n_estimators': dist_nestimators}
		else:
			parameters = {'max_depth': [3, 5, 10],
			'max_features': [0.1, 0.5, 'sqrt', 'log2', None],
			'n_estimators': [100, 500, 1000]}
		
		return parameters

	def GridSearch(df, SAVE, classes, min_size, gs_score, n, cv_num,
		n_jobs, GS_REPS, GS_TYPE, POS, NEG, gs_full):
		""" Perform a parameter sweep using GridSearchCV from SK-learn.
		Need to edit the hard code to modify what parameters are searched
		"""
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import RandomizedSearchCV
		from sklearn.preprocessing import StandardScaler

		start_time = time.time()
		n_iter = 10
		parameters = fun.param_space(GS_TYPE, n_iter)

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
				from sklearn.ensemble import RandomForestClassifier
				model = RandomForestClassifier()
				
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
	
	def RegGridSearch(df, SAVE, gs_score, cv_num, n_jobs, GS_REPS, GS_TYPE, gs_full):
		""" Perform a parameter sweep using GridSearchCV from SK-learn. 
		Need to edit the hard code to modify what parameters are searched"""
		from sklearn.metrics import mean_squared_error, r2_score
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import RandomizedSearchCV
		from sklearn.preprocessing import StandardScaler

		start_time = time.time()
		n_iter = 10
		parameters = fun.param_space(GS_TYPE, n_iter)

		y = df['Y']
		x = df.drop(['Y'], axis=1)

		gs_results = pd.DataFrame(columns=['mean_test_score', 'params'])

		for j in range(GS_REPS):
			print("Round %s of %s" % (j + 1, GS_REPS))

			# Build model
			from sklearn.ensemble import RandomForestRegressor
			model = RandomForestRegressor()

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
		gs_results_mean = gs_results_mean.sort_values(by='mean_test_score', \
			axis=0, ascending=False)
		
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
			max_depth=int(max_depth),
			max_features=max_features,
			criterion='friedman_mse',
			random_state=j,
			n_jobs=n_jobs)
		return reg
	
	def Run_Regression_Model(df, reg, cv_num, df_unknowns, test_df,
		cv_sets, j):
		from sklearn.model_selection import cross_val_predict
		from sklearn.metrics import make_scorer
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
			cv_pred_df = pd.concat([cv_pred_df, unk_pred_df])

		if not isinstance(test_df, str):
			test_y = test_df['Y']
			test_pred = reg.predict(test_df.drop(['Y'], axis=1))
			test_pred_df = pd.DataFrame(data=test_pred, index=test_df.index,
				columns=['pred'])
			cv_pred_df = pd.concat([cv_pred_df, test_pred_df])

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