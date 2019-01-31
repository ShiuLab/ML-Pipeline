
import sklearn
import numpy as np
import pandas as pd

def boolean_argument(y_n, name, default):
	BOOL = default
	if y_n.lower() == "n" or y_n.lower() == "no":
		BOOL = False
	elif y_n.lower() == "y" or y_n.lower() == "yes":
		BOOL = True
	else:
		print( "WARNING! %s boolean not recognized: %s" % (name, y_n) )
		print( "         Revert to default: %s" % default )
	return BOOL

def check_numeric_arg(IN, default):
	OUT = default
	if IN.lower() == 'n' or IN.lower() == "no":
		OUT = None
	else:
		OUT = int(IN)
	return OUT
	

def balancedIDs(df, Y_name, n_bal):
	import random
	
	pos_ids = list( df [ df[Y_name] == 1 ].index)
	neg_ids = list( df [ df[Y_name] == 0 ].index)
	
	min_bal = min( len(pos_ids), len(neg_ids))-1
	
	bal_ids = []
	for j in range(n_bal):
		#sub_l = []
		pos_samp = random.sample( pos_ids, min_bal)
		neg_samp = random.sample( neg_ids, min_bal)
		bal_ids.append( pos_samp+neg_samp )
	
	return pos_ids, neg_ids, min_bal, bal_ids

def remove_nonTrain (df, Y_name):
	instancesIDs_full = df.index
	
	P_ind = set( np.where(df[Y_name] == 1)[0] )
	N_ind = set( np.where(df[Y_name] == 0)[0] )
	train_ind = list(P_ind.union(N_ind))
	
	notP_ind = set( np.where(df[Y_name] != 1)[0] )
	notN_ind = set( np.where(df[Y_name] != 0)[0] )
	nonTrain_ind  = list(notP_ind.intersection(notN_ind))
	
	if len(nonTrain_ind) == 0:
		df_train = df
		df_nonTrain = None
	else:
		train_IDs = list(df.index[train_ind])
		nonTrain_IDs = list(df.index[nonTrain_ind])
		
		df_train = df[df.index.isin(train_IDs)]
		df_nonTrain = df[df.index.isin(nonTrain_IDs)]

	return df_train, df_nonTrain

def feat_subset_df (df, Y_name, FEAT):
	if FEAT != None:
		feats = [Y_name]
		
		inp = open(FEAT)
		for line in inp:
			if not line.startswith("#"):
				ft = line.strip()
				if ft != "":
					if ft not in feats:
						feats.append(ft)
		inp.close()
	else:
		feats = df.columns.tolist()
	
	df = df.loc[:,feats]
	
	return df, feats
	

def process_dataframe(DF, DF2, FEAT, POS, NEG, ALG, df2_col, NA_axis):
	
	df = pd.read_csv(DF, sep="\t", index_col = 0)
	
	if DF2 == None:
		Y_name = df.columns.tolist()[0]
	else:
		df2 = pd.read_csv(DF2, sep="\t", index_col = 0)
		Y_name = df2.columns.tolist()[df2_col-1]
		df = pd.concat([df2[Y_name], df], axis=1, join='inner')
	
	df, feats = feat_subset_df (df, Y_name, FEAT)
	#if FEAT != None:
	#	feats = [Y_name]
	#	inp = open(FEAT)
	#	for line in inp:
	#		if not line.startswith("#"):
	#			ft = line.strip()
	#			if ft != "":
	#				if ft not in feats:
	#					feats.append(ft)
	#	inp.close()
	#	
	#	df = df.loc[:,feats]
	
	if NA_axis != None:
		# 0 = drop by row
		# 1 = drop by column
		df = df.dropna( axis = NA_axis )
	
	classes = list(set(df[Y_name]))
	# Ensure integers are read as numeric (doesn't happen with mix of integers and strings)
	for i in range(len(classes)):
		class_nm = classes[i]
		try:
			check = int(class_nm)
			digit = True
		except:
			digit = False
		if digit == True:
			df.loc[df[Y_name] == class_nm, Y_name] = int(class_nm)
			classes[i] = int(class_nm)
	
	if POS != 1:
		#df[Y_name] [df[Y_name] == POS] = 1
		df.loc[df[Y_name] == POS, Y_name] = 1
	if NEG != 0:
		#df[Y_name] [df[Y_name] == NEG] = 0
		df.loc[df[Y_name] == NEG, Y_name] = 0
	
	if ALG.lower() == "svm" or ALG.lower() == "svmpoly" or ALG.lower() == "svmrbf":
		from sklearn import preprocessing
		
		y = df[Y_name]
		X = df.drop(['Class'], axis=1)
		
		min_max_scaler = preprocessing.MinMaxScaler()
		X_scaled = min_max_scaler.fit_transform(X)
		
		df = pd.DataFrame(X_scaled, columns = X.columns, index = X.index)
		df.insert(loc=0, column = Y_name, value = y)
	
	df, df_nonTrain = remove_nonTrain (df, Y_name)
		
	return df, classes, Y_name, df_nonTrain

def make_CV_folds (instanceIDs, n_folds):
	import math
	import random
	
	n_inst = len(instanceIDs)
	max_inst_per_fold = math.ceil( n_inst / n_folds )
	fold_range = list(range(1, n_folds+1))
	
	foldIDs = (fold_range*max_inst_per_fold)[:n_inst]
	random.shuffle(foldIDs)
	random.shuffle(foldIDs) # shuffle twice for good measure
	
	return foldIDs

def CV_folds_by_class (df, Y_name, classes, n_folds):
	instanceIDs = []
	foldIDs = []
	for class_nm in classes:
		classIDs = list( df.index[ df[Y_name] == class_nm ] )
		foldIDs_class = make_CV_folds (classIDs, n_folds)
		
		instanceIDs = instanceIDs + classIDs
		foldIDs = foldIDs + foldIDs_class
	
	return instanceIDs, foldIDs

#def remove_nonTrain (df, Y_name, instanceIDs):
#	notP_ind = set( np.where(df[Y_name] != 1)[0] )
#	notN_ind = set( np.where(df[Y_name] != 0)[0] )
#	nonTrain_ind  = list(notP_ind.intersection(notN_ind))
#	
#	if len(nonTrain_ind) == 0:
#		df_train = df
#		df_nonTrain = None
#	else:
#		nonTrain_IDs = list(df.index[nonTrain_ind])
#		df_train = df[df.index.isin(instanceIDs)]
#		df_nonTrain = df[df.index.isin(nonTrain_IDs)]
#	
#	return df_train, df_nonTrain
	

def pull_train_test_IDs (instanceIDs, foldIDs, CV_fold):
	indices_train = [i for i, x in enumerate(foldIDs) if x != CV_fold]
	indices_test = [i for i, x in enumerate(foldIDs) if x == CV_fold]
	
	trainIDs = [ instanceIDs[i] for i in indices_train ]
	testIDs = [ instanceIDs[i] for i in indices_test ]
	
	return trainIDs, testIDs
	
def LASSO_featSel(df_train, Y_name, l1):
	from sklearn.feature_selection import SelectFromModel
	from sklearn.linear_model import Lasso
	
	X_all = df_train.drop(Y_name, axis=1).values	
	Y_all = df_train.loc[:, Y_name].values
	
	estimator = Lasso(alpha = l1).fit(X_all, Y_all)
	model = SelectFromModel(estimator, prefit=True)
	keep = model.get_support([])
	
	feat_names = np.array(list(df_train)[1:])
	selectedFeats = list(feat_names[keep])
	
	return (selectedFeats)

def featSel_CV_wrapper(df, Y_name, instanceIDs, foldIDs, l1):
	feat_d = {}	
	for CV_fold in set(foldIDs):
		trainIDs, testIDs = pull_train_test_IDs (instanceIDs, foldIDs, CV_fold)
		df_train = df[df.index.isin(trainIDs)]
		
		selectedFeats = LASSO_featSel(df_train, Y_name, l1)
		feat_d [CV_fold] = selectedFeats
		
		n_sel = len(selectedFeats)
		print("  CV fold", CV_fold, "complete. # features selected:", n_sel)
	
	return feat_d

def featSel_wrapper( featSel, df, Y_name, instanceIDs, foldIDs, l1 ):
	if featSel == True:
		print("Performing feature selection")
		feat_d = featSel_CV_wrapper(df, Y_name, instanceIDs, foldIDs, l1)
	else:
		print("Feature selection turned off")
		feat_d = {}
		for CV_fold in set(foldIDs):
			feat_d [CV_fold] = "ALL_FEATURES"
	return feat_d

def write_CV_folds(save_prefix, CV, instanceIDs, foldIDs):
	out = open("%s.CV_folds"%save_prefix, "w")
	out.write("#CV = %s\n"%CV)
	out.write("#instance_ID\tCV_fold\n")
	for i in range(len(instanceIDs)):
		instID = instanceIDs[i]
		foldID = foldIDs[i]
		out.write("%s\t%s\n"%(instID, foldID))
	out.close()

def write_features(save_prefix, featSel, l1, feat_d):
	out = open("%s.featSel"%save_prefix, "w")
	if featSel == True:
		out.write("#l1 = %s\n"%l1)
	elif featSel == False:
		out.write("#l1 = NA\n")
	out.write("#CV_fold\tn\tSelected_features\n")
	for CV_fold in feat_d:
		selectedFeats = feat_d[CV_fold]
		
		if selectedFeats == "ALL_FEATURES":
			n_sel = "NA"
		else:
			n_sel = len(selectedFeats)
		if featSel == True:
			out.write("%s\t%s\t%s\n"%(CV_fold, n_sel, ",".join(selectedFeats)))
		elif featSel == False:
			out.write("%s\t%s\tALL_FEATURES\n"%(CV_fold, n_sel))
	out.close()

def process_CV_file(CVF):
	instanceIDs = []
	foldIDs = []
	
	inp = open(CVF)
	for line in inp:
		if not line.startswith("#"):
			if line.strip() != "":
				instID, foldID = line.strip().split("\t")
				instanceIDs.append(instID)
				foldIDs.append(int(foldID))
	inp.close()
	
	return instanceIDs, foldIDs

def process_featSel_file(FSF):
	feat_d = {}
	
	inp = open(FSF)
	for line in inp:
		if not line.startswith("#"):
			if line.strip() != "":
				CV_fold, n_feat, selectedFeats_str = line.strip().split()
				if selectedFeats != "ALL_FEATURES":
					selectedFeats = selectedFeats_str.split(",")
				feat_d[int(CV_fold)] = selectedFeats
	inp.close()
	
	return feat_d

def pull_featSel_subset(feat_d, CV_fold, df, Y_name):
	selectedFeats = feat_d [CV_fold]
	#print(selectedFeats)
	if selectedFeats == "ALL_FEATURES":
		df_sel = df
	else:
		df_sel = df.loc[:, [Y_name]+selectedFeats ]
	return df_sel

def runGridSearch(df, Y_name, ALG, GS_REPS, n_jobs):
	
	### NOTE: The returned top_params will be in alphabetical order - to be consistent add any additional 
	###       parameters to test in alphabetical order
	if ALG.lower() == 'rf':
		parameters = {'max_depth':[3, 5, 10], 'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None], 'n_estimators': [100,500,1000]}
		
		from sklearn.ensemble import RandomForestClassifier
		model = RandomForestClassifier()
		
	elif ALG.lower() == "svm":
		parameters = {'kernel': ['linear'], 'C':[0.01, 0.1, 0.5, 1, 10, 50, 100]}
		
	elif ALG.lower() == 'svmpoly':
		parameters = {'kernel': ['poly'], 'C':[0.01, 0.1, 0.5, 1, 10, 50, 100],'degree': [2,3,4], 'gamma': np.logspace(-5,1,7)}
		
	elif ALG.lower() == 'svmrbf':
		parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'gamma': np.logspace(-5,1,7)}
		
	elif ALG.lower() == 'logreg':
		parameters = {'C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'intercept_scaling': [0.1, 0.5, 1, 2, 5, 10],'penalty': ['l1','l2']}
		
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression()
		
	elif ALG.lower() == 'gb':
		parameters = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],'max_depth': [3, 5, 10], 'max_features': [0.1, 0.25, 0.5, 0.75, 'sqrt', 'log2', None],'n_estimators': [100,500,1000]}
		
		from sklearn.ensemble import GradientBoostingClassifier
		model = GradientBoostingClassifier()
	else:
		print('Grid search is not available for the algorithm selected')
		exit()
	
	parameter_names = parameters.keys()
		
	if ALG.lower() == "svm" or ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
		from sklearn.svm import SVC
		model = SVC(probability=True)
	
	from sklearn.model_selection import GridSearchCV
	
	y = df[Y_name]
	y = y.astype('int')
	
	x = df.drop([Y_name], axis=1) 
	
	# Build model, run grid search with 10-fold cross validation and fit
	gs_results = pd.DataFrame(columns = ['mean_test_score','params'])
	for i in range(GS_REPS):
		print("    Grid search repetition: %s"%i+1)
		grid_search = GridSearchCV(model, parameters, scoring = 'roc_auc', cv = 2, n_jobs = n_jobs, pre_dispatch=2*n_jobs)
		grid_search.fit(x, y)
	
		gs_results_i = pd.DataFrame(grid_search.cv_results_)
		gs_results = pd.concat([gs_results, gs_results_i[['params','mean_test_score']]])
	
	gs_results2 = pd.concat([gs_results.drop(['params'], axis=1), gs_results['params'].apply(pd.Series)], axis=1)
	param_names = list(gs_results2)[1:]
	
	gs_results_mean = gs_results2.groupby(param_names).mean()
	gs_results_mean = gs_results_mean.sort_values('mean_test_score', 0, ascending = False)
	top_params = gs_results_mean.index[0]
	
	return top_params, parameter_names

def runGridSearch_CV_wrapper (df, Y_name, foldIDs, instanceIDs, feat_d, ALG, n_jobs):
	GS_REPS = 10
	
	print("Running grid search")
	param_d = {}
	for CV_fold in set(foldIDs):
		print("  Working on CV fold: %s"%CV_fold)
		trainIDs, testIDs = pull_train_test_IDs (instanceIDs, foldIDs, CV_fold)
		df_train = df[df.index.isin(trainIDs)]
		df_train_sel = pull_featSel_subset(feat_d, CV_fold, df_train, Y_name)
		top_params, parameter_names = runGridSearch(df_train_sel, Y_name, ALG, GS_REPS, n_jobs)
		param_d [CV_fold] = top_params
	
	return param_d, parameter_names

def pull_default_parameters(ALG):
	
	# Default Random Forest and Gradient Boosting parameters
	n_estimators, max_depth, max_features, learning_rate = 500, 10, "sqrt", 0.1
	
	# Default Linear SVC parameters
	kernel, C, degree, gamma, loss, max_iter = 'linear', 1, 2, 1, 'hinge', "500"
	
	# Default Logistic Regression paramemter
	penalty, C, intercept_scaling = 'l2', 1.0, 1.0
	
	if ALG.lower() == "rf":
		parameters = [ max_depth, max_features, n_estimators ]
		parameter_names = [ "max_depth", "max_features", "n_estimators" ]
	elif ALG.lower() == "svm" or ALG.lower() == 'svmrbf' or ALG.lower() == 'svmpoly':
		parameters = [ C, degree, gamma, kernel]
		parameter_names = [ "C", "degree", "gamma", "kernel" ]
	elif ALG.lower() == "logreg":
		parameters = [ C, intercept_scaling, penalty]
		parameter_names = [ "C", "intercept_scaling", "penalty" ]
	elif ALG.lower() == "gb":
		parameters = [ learning_rate, max_depth, max_features ]
		parameter_names = [ "learning_rate", "max_depth", "max_features" ]
	
	return parameters, parameter_names
	

def runGridSearch_wrapper (gs, df, Y_name, instanceIDs, foldIDs, feat_d, ALG, n_jobs):
	
	if gs == True:
		print("Running grid search")
		param_d, parameter_names = runGridSearch_CV_wrapper (df, Y_name, foldIDs, instanceIDs, feat_d, ALG, n_jobs)
	else:
		print("Grid search turned off")
		param_d = {}
		default_parameters, parameter_names = pull_default_parameters (ALG)
		for CV_fold in set(foldIDs):
			param_d [CV_fold] = default_parameters
	
	return param_d, parameter_names

def balancedIDs_CV(df, Y_name, instanceIDs, foldIDs, n_bal, prop):
	import random
	import math
	
	balIDs_d = {}
		
	for CV_fold in set(foldIDs):
		balIDs_d[CV_fold] = []
		
		trainIDs, testIDs = pull_train_test_IDs (instanceIDs, foldIDs, CV_fold)
		df_train = df[df.index.isin(trainIDs)]
		
		pos_ids = list( df_train [ df_train[Y_name] == 1 ].index)
		neg_ids = list( df_train [ df_train[Y_name] == 0 ].index)
		
		n_pos = df_train [ df_train[ Y_name] == 1 ].shape[0]
		n_neg = df_train [ df_train[ Y_name] == 0 ].shape[0]
		min_bal = min( n_pos, n_neg )-1
		min_bal = int( ( math.ceil( min_bal * prop ) ) )
		
		for i in range(n_bal):
			pos_samp = random.sample( pos_ids, min_bal)
			neg_samp = random.sample( neg_ids, min_bal)
			
			balIDs_d[CV_fold].append(pos_samp+neg_samp)
	
	return(balIDs_d)

def write_balIDs_d(save_prefix, balIDs_d):
	out = open("%s.balIDs"%save_prefix, "w")
	
	out.write("#CV_fold\tbal_iter\tn_class_inst\tselected_IDs\n")
	for CV_fold in balIDs_d:
		IDs_list = balIDs_d[CV_fold]
		
		itr = 1
		for IDs in IDs_list:
			min_bal = int( len(IDs) / 2 )
			out.write("%s\t%s\t%s\t%s\n" % (CV_fold, itr, min_bal, ",".join(IDs)) )
			itr += 1
	out.close()

def DefineClf_RandomForest(parameters, random_seed, n_jobs):
	from sklearn.ensemble import RandomForestClassifier
	
	max_depth, max_features, n_estimators = parameters
	
	clf = RandomForestClassifier(n_estimators=int(n_estimators), 
		max_depth=max_depth, 
		max_features=max_features,
		criterion='gini', 
		random_state=random_seed, 
		n_jobs=n_jobs)
	return clf

def DefineClf_SVM(parameters, random_seed, n_jobs):
	from sklearn.svm import SVC

	C, degree, gamma, kernel = parameters
	
	clf = SVC(kernel = kernel,
		C=float(C), 
		degree = degree,
		gamma = gamma, 
		random_state = random_seed,
		probability = True,
		n_jobs=n_jobs)
	return clf

def DefineClf_LogReg(parameters, n_jobs):
	from sklearn.linear_model import LogisticRegression
	
	C, intercept_scaling, penalty = parameters
	
	clf = LogisticRegression(penalty=penalty,
		C=float(C),
		intercept_scaling=intercept_scaling,
		n_jobs=n_jobs)
	return clf

def init_clf(ALG, parameters, random_seed, n_jobs):
	
	if ALG.lower() == "rf":
		clf = DefineClf_RandomForest(parameters, random_seed, n_jobs)
	elif ALG.lower() == "svm":
		clf = DefineClf_SVM(parameters, random_seed, n_jobs)
	elif ALG.lower() == "logreg":
		clf = DefineClf_LogReg(parameters, n_jobs)
	
	return clf

def train_model (df_train, Y_name, clf):
	y = df_train[Y_name]
	y = y.astype('int')
	X = df_train.drop([Y_name], axis=1)
	
	clf.fit(X,y)

def apply_model (clf, Y_name, df_apply, col_hdr):
	pred_proba = clf.predict_proba( df_apply.drop( [Y_name], axis=1) )
	
	clf_classes = list( clf.classes_ )
	POS_IND = [i for i, v in enumerate(clf_classes) if v == 1 ][0]
	
	pred_scores = pred_proba[:,POS_IND]
	
	df_scores = pd.DataFrame( data = pred_scores, index = df_apply.index, columns = col_hdr )
	
	return(df_scores)

def pull_importances(clf, features, ascending = False, ROUND = False):
	df_imp = None
	warning = "    Feature importances could not be extracted"
	#if ALG.lower() == "rf":
	#	imp = list(clf.feature_importances_)
	#	std = list(np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0))
	#	
	#	df_imp = pd.DataFrame( data = {"imp": imp, "std": std}, index = features )
	#	df_imp = df_imp.sort_values(by = ["imp"], axis = 0, ascending = False)
	#else:
	try:
		imp = list(clf.feature_importances_)
		df_imp = pd.DataFrame( data = {"imp": imp}, index = features )
	except:
		try:
			imp = list(clf.coef_)
			df_imp = pd.DataFrame( data = {"imp": imp}, index = features )
		except:			
			print(warning)
	
	if df_imp is not None:
		df_imp.sort_values(by = ["imp"], axis = 0, ascending = ascending, inplace = True)
		if ROUND != False:
			df_imp = df_imp.round(decimals = ROUND)
	
	return df_imp

def round_and_multiply_df(df):
	# typical score from RF: 0.266 | 5 characters
	# rounded: 0.27 | 4 characters
	# Multiply by 100, encode as integer: 27 | 2 characters
	#  typical compression: 2.5X reduction in size (5/2)
	df_out = df.round(decimals=2)
	df_out = df_out.multiply(100)
	df_out = df_out.astype('int')
	return df_out

def train_test_CV (df, Y_name, instanceIDs, foldIDs, feat_d, param_d, balIDs_d, ALG, df_nonTrain, CV_itr, n_jobs):
	
	print("Training and applying models")
		
	prediction_scores = pd.DataFrame()
	if df_nonTrain is not None:
		nonTrain_scores = pd.DataFrame()
	else:
		nonTrain_scores = None
	
	featImpCV_d = {}
	for CV_fold in set(foldIDs):
		print("  Working on CV fold %s" % CV_fold)
		parameters = param_d [CV_fold]
		df_sel = pull_featSel_subset(feat_d, CV_fold, df, Y_name)
				
		trainIDs, testIDs = pull_train_test_IDs (instanceIDs, foldIDs, CV_fold)
		df_train = df_sel[df_sel.index.isin(trainIDs)]
		df_test = df_sel[df_sel.index.isin(testIDs)]
		
		test_scores = pd.DataFrame(index = df_test.index)
		if df_nonTrain is not None:
			df_nonTrain_sel = pull_featSel_subset(feat_d, CV_fold, df_nonTrain, Y_name)
			nonTrain_scores_CV = pd.DataFrame(index = df_nonTrain.index)
		
		featImpCV_d [CV_fold] = {}
		
		balIDs_list = balIDs_d [ CV_fold ]
		for i in range( len(balIDs_list) ):
			itr = i+1
			print("    Balanced iteration %s"%itr)
			
			balIDs = balIDs_list[i]
			df_train_bal = df_train [ df_train.index.isin(balIDs) ]
			
			# intialize classifier and train model
			random_seed = CV_fold*itr+1000
			clf = init_clf (ALG, parameters, random_seed, n_jobs)
			train_model (df_train_bal, Y_name, clf)
			
			# apply model to test and non-train/test instances
			col_hdr = [ "score_CV%s_B%s" % (CV_itr, itr) ]
			
			test_scores_bal = apply_model (clf, Y_name, df_test, col_hdr)
			test_scores = pd.concat( [ test_scores, test_scores_bal], axis = 1)
			
			features = list(df_train_bal.drop(Y_name, axis = 1).columns)
			df_imp = pull_importances(clf, features, ROUND = 3)
			featImpCV_d [CV_fold] [itr] = df_imp
			
			if df_nonTrain is not None:
				col_hdr2 = [ "score_CV%s_F%s_B%s" % (CV_itr, CV_fold, itr) ]
				nonTrain_scores_bal = apply_model (clf, Y_name, df_nonTrain_sel, col_hdr2)
				nonTrain_scores_CV = pd.concat( [ nonTrain_scores_CV, nonTrain_scores_bal ], axis = 1)
		
		prediction_scores = pd.concat( [ prediction_scores, test_scores], axis = 0 )
		if df_nonTrain is not None:
			nonTrain_scores = pd.concat( [ nonTrain_scores, nonTrain_scores_CV], axis = 1 )
	
	prediction_scores = round_and_multiply_df(prediction_scores)
	if df_nonTrain is not None:
		nonTrain_scores = round_and_multiply_df(nonTrain_scores)
	
	return prediction_scores, nonTrain_scores, featImpCV_d

def write_CV_folds_CVitr(save_prefix, inputs_d, n_CVitr, n_folds):
	
	hdr = []
	out_d = {}
	
	for CV_iter in inputs_d:
		hdr.append( "CV%s" % CV_iter )
		
		instanceIDs = inputs_d[CV_iter]["instanceIDs"]
		foldIDs = inputs_d[CV_iter]["foldIDs"]
		
		for i in range(len(instanceIDs)):

			instID = instanceIDs[i]
			foldID = str(foldIDs[i])
			
			if instID not in out_d:
				out_d[instID] = [foldID]
			else:
				out_d[instID].append(foldID)
	
	out = open("%s.CV_folds"%save_prefix, "w")
	
	out.write("#CViter = %s\n" % n_CVitr)
	out.write("#CV_folds = %s\n" % n_folds)
	
	hdr_str = "\t".join(hdr)
	out.write("#ID\t%s\n" % hdr_str)
	for instID in instanceIDs:
		foldID_l = out_d[instID]
		out.write( "%s\t%s\n" % (instID, "\t".join(foldID_l)) )
	out.close()

def write_features_CVitr(save_prefix, featSel, l1, inputs_d):
	out = open("%s.featSel"%save_prefix, "w")
	if featSel == True:
		out.write("#l1 = %s\n"%l1)
	elif featSel == False:
		out.write("#l1 = NA\n")
	out.write("#CV_iter\tCV_fold\tn\tSelected_features\n")
	
	for CVitr in inputs_d:
		feat_d = inputs_d[CVitr]["feat_d"]
		for CV_fold in feat_d:
			selectedFeats = feat_d[CV_fold]
			
			if selectedFeats == "ALL_FEATURES":
				n_sel = "NA"
			else:
				n_sel = len(selectedFeats)
			if featSel == True:
				out.write("%s\t%s\t%s\t%s\n"%(CVitr, CV_fold, n_sel, ",".join(selectedFeats)))
			elif featSel == False:
				out.write("%s\t%s\t%s\tALL_FEATURES\n"%(CVitr, CV_fold, n_sel))

def write_parameters_CVitr(save_prefix, inputs_d, parameter_names):
	out = open("%s.params"%save_prefix, "w")
	
	out.write("#CV_iter\tCV_fold\t%s\n" % ("\t".join(parameter_names)) )
	for CVitr in inputs_d:
		param_d = inputs_d[CVitr]["param_d"]
		for CV_fold in param_d:
			params = [ str(i) for i in  param_d[CV_fold] ]
			out.write("%s\t%s\t%s\n" % (CVitr, CV_fold, "\t".join(params)) )

def write_balIDs_CVitr(save_prefix, inputs_d):
	out = open("%s.balIDs"%save_prefix, "w")
	
	out.write("#CV_iter\tCV_fold\tbal_iter\tn_class_inst\tselected_IDs\n")
	for CVitr in inputs_d:
		balIDs_d = inputs_d[CVitr]["balIDs_d"]
		for CV_fold in balIDs_d:
			IDs_list = balIDs_d[CV_fold]
			
			itr = 1
			for IDs in IDs_list:
				min_bal = int( len(IDs) / 2 )
				out.write("%s\t%s\t%s\t%s\t%s\n" % (CVitr, CV_fold, itr, min_bal, ",".join(IDs)) )
				itr += 1
	
	out.close()

def concat_full_scores(predictions_d, df, df_nonTrain):
	
	concat_scores = pd.DataFrame()
	if df_nonTrain is not None:
		concat_nonTrain = pd.DataFrame()
	else:
		concat_nonTrain = None
		
	for CV_fold in predictions_d:
		prediction_scores = predictions_d [CV_fold] ["prediction_scores"]
		nonTrain_scores = predictions_d [CV_fold] ["nonTrain_scores"]
		
		concat_scores = pd.concat( [ concat_scores, prediction_scores ], axis = 1 )
		
		if df_nonTrain is not None:
			concat_nonTrain = pd.concat( [ concat_nonTrain, nonTrain_scores ], axis = 1 )
	
	concat_scores = concat_scores.reindex( df.index )
	if df_nonTrain is not None:
		concat_nonTrain = concat_nonTrain.reindex( df_nonTrain.index )
	
	return concat_scores, concat_nonTrain

def make_df_mean(df, ROUND = 0):
	df_mean = df.mean(axis = 1)
	
	if ROUND != False:
		df_mean = df_mean.round(decimals=ROUND)
	if ROUND == 0:
		df_mean = df_mean.astype('int')
	df_mean.columns = ["Mean"]
	return df_mean

def make_df_std(df, ROUND = 1):
	df_std = df.std(axis = 1)
	if ROUND != False:
		df_std = df_std.round(decimals = ROUND)
	df_std.columns = ["SD"]
	return df_std

def make_df_mean_std(df, prefix = "", round_mean = 0, round_std = 1):
	df_mean = make_df_mean(df, ROUND = round_mean)
	df_std = make_df_std(df, ROUND = round_std)
	df_mean_std = pd.concat( [ df_mean, df_std ], axis = 1 )
	df_mean_std.columns = [ "%sMean"%prefix, "%sSD"%prefix ]
	return df_mean_std

def add_classes(df_add, df_classes, Y_name, POS, NEG):
	wClass = pd.concat( [ df_classes[Y_name], df_add ], axis = 1 )
	
	if POS != 1:
		wClass [Y_name] [ wClass [Y_name] == 1] = POS
	if NEG != 0:
		wClass [Y_name] [ wClass [Y_name] == 0] = NEG
	
	return wClass
	

def write_score_overview(save_prefix, predictions_d, df, df_nonTrain, Y_name, POS, NEG):
	
	scores, nonTrain_scores = concat_full_scores(predictions_d, df, df_nonTrain)	
	
	scores_mean_std = make_df_mean_std(scores)
	scores_class_mean_std = add_classes(scores_mean_std, df, Y_name, POS, NEG)
	#scores_class_mean_std = pd.concat( [ df[Y_name], scores_mean_std ], axis = 1 )
	#
	#if POS != 1:
	#	scores_class_mean_std [Y_name] [ scores_class_mean_std [Y_name] == 1] = POS
	#if NEG != 0:
	#	scores_class_mean_std [Y_name] [ scores_class_mean_std [Y_name] == 0] = NEG
	
	if df_nonTrain is not None:
		nonTrain_mean_std = make_df_mean_std(nonTrain_scores)
		#nonTrain_class_mean_std = pd.concat( [ df_nonTrain[Y_name], nonTrain_mean_std ], axis = 1 )
		nonTrain_class_mean_std = add_classes(nonTrain_mean_std, df_nonTrain, Y_name, POS, NEG)
		
		scores_class_mean_std = pd.concat( [ scores_class_mean_std, nonTrain_class_mean_std ], axis = 0 )
	
	out = open("%s.score_mean"%save_prefix, "w")
	out.write(pd.DataFrame.to_csv(scores_class_mean_std,sep="\t").strip()+"\n")
	out.close()

def write_full_scores_DEP(save_prefix, full_scores, scores, nonTrain_scores):
	if full_scores == True:
		out1 = open("%s.full_scores"%save_prefix, "w")
		out1.write(pd.DataFrame.to_csv(scores,sep="\t").strip()+"\n")
		out1.close()
		
		if nonTrain_scores is not None:
			out2 = open("%s.full_scores.nonTrain"%save_prefix, "w")
			out2.write(pd.DataFrame.to_csv(nonTrain_scores,sep="\t").strip()+"\n")
			out2.close()
	else:
		print("Writing of full score set turned off")

def df_mean_sd_CVitr(predictions_d, probe, df):
	df_mean_std_CVitr = pd.DataFrame()
	for CVitr in predictions_d:
		df_scores = predictions_d[CVitr][probe]
		
		prefix = "CV%s_" % CVitr
		df_mean_std = make_df_mean_std(df_scores, prefix = prefix)
		
		df_mean_std_CVitr = pd.concat( [df_mean_std_CVitr, df_mean_std], axis = 1 )
	
	df_mean_std_CVitr = df_mean_std_CVitr.reindex( df.index )
	
	return df_mean_std_CVitr

def add_grandmean(df):
	mean_cols = list(filter( lambda x: 'Mean' in x, list(df.columns) ))
	mean_df = df[mean_cols]
	
	grandmean = mean_df.mean(axis = 1).round(decimals=0).astype('int')
	grandmean_SD = mean_df.std(axis = 1).round(decimals=1)
	
	SD_cols = list(filter( lambda x: 'SD' in x, list(df.columns) ))
	SD_df = df[SD_cols]
	
	SD_mean = SD_df.mean(axis = 1).round(decimals=1)
	
	df_w_grandmean = pd.concat( [grandmean, grandmean_SD, SD_mean, df], axis = 1)
	df_w_grandmean.rename( columns = {0: "grand_mean", 1: "grand_mean_SD", 2: "SD_mean"}, inplace = True )
	
	return df_w_grandmean

def write_abridged_scores(save_prefix, predictions_d, df, df_nonTrain, Y_name, POS, NEG):
	
	scores_mean_std_CVitr = df_mean_sd_CVitr (predictions_d, "prediction_scores", df)
	scores_mean_std_CVitr_GM = add_grandmean (scores_mean_std_CVitr)
	abr_scr = add_classes(scores_mean_std_CVitr_GM, df, Y_name, POS, NEG)
	
	if df_nonTrain is not None:
		nonTrain_mean_str_CVitr = df_mean_sd_CVitr (predictions_d, "nonTrain_scores", df_nonTrain)
		nonTrain_mean_str_CVitr_GM = add_grandmean(nonTrain_mean_str_CVitr)
		nonTrain_abr = add_classes(nonTrain_mean_str_CVitr_GM, df_nonTrain, Y_name, POS, NEG)
		
		abr_scr = pd.concat( [ abr_scr, nonTrain_abr ], axis = 0)
	
	#scores_mean_std_CVitr_GM = add_grandmean(scores_mean_std_CVitr)
		
	out = open("%s.d1.scores" % (save_prefix), "w")
	out.write(pd.DataFrame.to_csv(abr_scr, sep="\t").strip()+"\n")
	out.close()

def write_full_scores(save_prefix, predictions_d, df, df_nonTrain):
	
	scores, nonTrain_scores = concat_full_scores(predictions_d, df, df_nonTrain)
	
	out1 = open("%s.d2.scores" % (save_prefix), "w")
	out1.write(pd.DataFrame.to_csv(scores,sep="\t").strip()+"\n")
	out1.close()
	
	if nonTrain_scores is not None:
		out2 = open("%s.d2.scores.nonTrain" % (save_prefix), "w")
		out2.write(pd.DataFrame.to_csv(nonTrain_scores,sep="\t").strip()+"\n")
		out2.close()

def write_scores(save_prefix, detail, predictions_d, df, df_nonTrain, Y_name, POS, NEG):
	if detail == 0:
		print("Writing of abridged or full score set turned off")
	elif detail == 1:
		print("Writing abridged scores")
		write_abridged_scores(save_prefix, predictions_d, df, df_nonTrain, Y_name, POS, NEG)
	elif detail == 2:
		print("Writing full scores")
		write_full_scores(save_prefix, predictions_d, df, df_nonTrain)
	else:
		print("WARNING: Parameter for score detail writing not recognized: %s" % detail)
		print("         Default: Writing abridged scores")
		write_abridged_scores(save_prefix, predictions_d, df, df_nonTrain)

def write_abrigded_featImp_CVitr (save_prefix, featImp_d):
	
	abr_featImp = pd.DataFrame()
	for CVitr in featImp_d:
		featImpCV_d = featImp_d [CVitr]
		for CV_fold in featImpCV_d:
			featImpCVbal_d = featImpCV_d[CV_fold]
			
			df_CVfold = pd.DataFrame()
			for bal_itr in featImpCVbal_d:
				df_imp = featImpCVbal_d[bal_itr]
				df_CVfold = pd.concat( [df_CVfold, df_imp], axis = 1 )
			
			df_imp_meanSD = make_df_mean_std(df_CVfold, prefix = "", round_mean = 3, round_std = 3)
			df_imp_meanSD.sort_values(by = ["Mean"], axis = 0, ascending = False, inplace = True)
			
			n_feat = df_imp_meanSD.shape[0]
			CVitr_l = [ CVitr for i in range(n_feat) ]
			CV_fold_l = [ CV_fold for i in range(n_feat) ]
			
			CVitr_df = pd.DataFrame(data = CVitr_l, index = df_imp_meanSD.index, columns = ["CV_iter"] )
			CV_fold_df = pd.DataFrame(data = CV_fold_l, index = df_imp_meanSD.index, columns = ["CV_fold"] )
			
			desig_df = pd.concat( [ CVitr_df, CV_fold_df ], axis = 1 )
						
			featImp_out_df = pd.concat( [ df_imp_meanSD, desig_df ], axis = 1)
			
			abr_featImp = pd.concat( [abr_featImp, featImp_out_df], axis = 0)
	
	out = open("%s.d1.featImp" % save_prefix, "w")
	out.write("#feat\t"+pd.DataFrame.to_csv(abr_featImp,sep="\t").strip()+"\n")
	out.close()


def write_full_featImp_CVitr(save_prefix, featImp_d):
	
	full_featImp = pd.DataFrame()
	for CVitr in featImp_d:
		featImpCV_d = featImp_d [CVitr]
		for CV_fold in featImpCV_d:
			featImpCVbal_d = featImpCV_d[CV_fold]
			for bal_itr in featImpCVbal_d:
				df_imp = featImpCVbal_d[bal_itr]
				
				n_feat = df_imp.shape[0]
				CVitr_l = [ CVitr for i in range(n_feat) ]
				CV_fold_l = [ CV_fold for i in range(n_feat) ]
				bal_itr_l = [ bal_itr for i in range(n_feat) ]
				
				CVitr_df = pd.DataFrame(data = CVitr_l, index = df_imp.index, columns = ["CV_iter"] )
				CV_fold_df = pd.DataFrame(data = CV_fold_l, index = df_imp.index, columns = ["CV_fold"] )
				bal_itr_df = pd.DataFrame(data = bal_itr_l, index = df_imp.index, columns = ["bal_iter"] )
				
				#desig_df = pd.DataFrame( data = { "CV_iter": CVitr_l, "CV_fold": CV_fold_l,"bal_iter": bal_itr_l }, index = df_imp.index )
				desig_df = pd.concat( [ CVitr_df, CV_fold_df, bal_itr_df ], axis = 1 )
					
				featImp_out_df = pd.concat( [ df_imp, desig_df ], axis = 1)
				
				full_featImp = pd.concat([full_featImp, featImp_out_df], axis = 0)
	
	out = open("%s.d2.featImp"%save_prefix, "w")
	out.write("#feat\t"+pd.DataFrame.to_csv(full_featImp,sep="\t").strip()+"\n")
	out.close()

def write_featImp(save_prefix, detail, featImp_d):
	if detail == 0 or detail == 1:
		write_abrigded_featImp_CVitr (save_prefix, featImp_d)
	elif detail == 2:
		write_full_featImp_CVitr(save_prefix, featImp_d)
	else:
		print("WARNING! Detail parameter not recognized: %s" % detail)
		print("         Default: Write abridged feature importance")
		write_abrigded_featImp_CVitr (save_prefix, featImp_d)

