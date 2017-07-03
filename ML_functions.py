"""
PURPOSE:

Functions for SKlearn machine learning pipeline

GridSearch
RandomForest
LinearSVC
Performance
Performance_MC
Plots (PR & ROC)
Plot_ConMatrix

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
		class_ids_dict = {}
		for cl in classes:
			cl_ids = list(df[df["Class"]==cl].index)
			class_ids_dict[cl] = cl_ids
		
		# Build a list of lists containing the IDs for balanced datasets
		bal_list = []
		for j in range(gs_n):
			tmp_l = []
			for cl in class_ids_dict:
				bal_samp = rn.sample(class_ids_dict[cl],min_size)
				tmp_l = tmp_l+bal_samp
			bal_list.append(tmp_l)
		return bal_list
	
	# def GridSearch(df, balanced_list, SAVE, ALG, classes, min_size, gs_score, gs_n, n_jobs):
	def GridSearch(df, SAVE, ALG, classes, min_size, gs_score, n, cv_num, n_jobs, POS, NEG):
		""" Perform a parameter sweep using grid search CV implemented in SK-learn
		"""
		from sklearn.model_selection import GridSearchCV
		from sklearn.preprocessing import StandardScaler
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.svm import SVC
		
		start_time = time.time()
		
		### NOTE: The returned top_params will be in alphabetical order - to be consistent add any additional 
		###       parameters to test in alphabetical order
		if ALG == 'RF':
			parameters = {'max_depth':[3, 5, 10, 50], 'max_features': [0.1, 0.25, 0.5, 0.75, 0.9999, 'sqrt', 'log2']}
			
		elif ALG == "SVM":
			parameters = {'kernel': ['linear'], 'C':[0.01, 0.1, 0.5, 1, 10, 50, 100]}

		elif ALG == 'SVMpoly':
			parameters = {'kernel': ['poly'], 'C':[0.01, 0.1, 0.5, 1, 10, 50, 100],'degree': [2,3,4], 'gamma': np.logspace(-5,1,7)}

		elif ALG == 'SVMrbf':
			parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 0.5, 1, 10, 50, 100], 'gamma': np.logspace(-5,1,7)}
				
		
		else:
			print('Grid search is not available for the algorithm selected')
			exit()
		
		gs_results = pd.DataFrame(columns = ['mean_test_score','params'])
		
		bal_ids_list = []
		for j in range(n):
			print("  Round %s of %s"%(j+1,n))
			
			# Build balanced dataframe and define x & y
			df1 = pd.DataFrame(columns=list(df))
			for cl in classes:
				temp = df[df['Class'] == cl].sample(min_size, random_state=j)
				df1 = pd.concat([df1, temp])
			
			bal_ids_list.append(list(df1.index))
			
			y = df1['Class']
			x = df1.drop(['Class'], axis=1) 
			
			# Build model, run grid search with 10-fold cross validation and fit
			if ALG == 'RF':
				model = RandomForestClassifier()
			elif ALG == "SVM" or ALG == 'SVMrbf' or ALG == 'SVMpoly':
				x = StandardScaler().fit_transform(x)
				model = SVC(probability=True)
			
			grid_search = GridSearchCV(model, parameters, scoring = gs_score, cv = cv_num, n_jobs = n_jobs, pre_dispatch=2*n_jobs)
			
			if len(classes) == 2:
				y = y.replace(to_replace = [POS, NEG], value = [1,0])
			
			grid_search.fit(x, y)
			
			# Add results to dataframe
			j_results = pd.DataFrame(grid_search.cv_results_)
			gs_results = pd.concat([gs_results, j_results[['params','mean_test_score']]])
		
		# Break params into seperate columns
		gs_results2 = pd.concat([gs_results.drop(['params'], axis=1), gs_results['params'].apply(pd.Series)], axis=1)
		param_names = list(gs_results2)[1:]
		#print('Parameters tested: %s' % param_names)
		
		# Find the mean score for each set of parameters & select the top set
		gs_results_mean = gs_results2.groupby(param_names).mean()
		gs_results_mean = gs_results_mean.sort_values('mean_test_score', 0, ascending = False)
		top_params = gs_results_mean.index[0]
		
		print("Parameter sweep time: %f seconds" % (time.time() - start_time))
		outName = SAVE + "_GridSearch.txt"
		gs_results_mean.to_csv(outName)
		return top_params,bal_ids_list, param_names
	
	def DefineClf_RandomForest(n_estimators,max_depth,max_features,j,n_jobs):
		from sklearn.ensemble import RandomForestClassifier
		clf = RandomForestClassifier(n_estimators=int(n_estimators), 
			max_depth=max_depth, 
			max_features=max_features,
			criterion='gini', 
			random_state=j, 
			n_jobs=n_jobs)
		return clf
	
	def DefineClf_SVM(kernel,C,degree,gamma,j):
		from sklearn.svm import SVC
		clf = SVC(kernel = kernel,
			C=float(C), 
			degree = degree,
			gamma = gamma, 
			random_state=j,
			probability=True)
		return clf
		
	# def MakeScoreFrame(cv_proba,POS_IND,sel_labels,score_columns,notSel_proba,notSel_labels,apply_unk,unk_proba,unk_labels):
		# df_sel_scores = pd.DataFrame(data=cv_proba[POS_IND],index=sel_labels,columns=score_columns)
		# df_notSel_scores = pd.DataFrame(data=notSel_proba[POS_IND],index=df_notSel.index,columns=score_columns)
		# current_scores = pd.concat([df_sel_scores,df_notSel_scores], axis = 0)
		# if apply_unk == True:
			# df_unk_scores = pd.DataFrame(data=unk_proba[POS_IND],index=df_unknowns.index,columns=score_columns)
			# current_scores = pd.concat([current_scores,df_unk_scores], axis = 0)
		# return current_scores
	
	def BuildModel_Apply_Performance(df, clf, cv_num, df_notSel, apply_unk, df_unknowns, classes, POS, NEG, j):
		from sklearn.model_selection import cross_val_predict
		
		y = df['Class']
		X = df.drop(['Class'], axis=1) 

		# Obtain the predictions using 10 fold cross validation (uses KFold cv by default):
		cv_proba = cross_val_predict(estimator=clf, X=X, y=y, cv=cv_num, method='predict_proba')
		cv_pred = cross_val_predict(estimator=clf, X=X, y=y, cv=cv_num)
		
		# Fit a model using all data and apply to instances that were not selected and of unknown class
		clf.fit(X,y)
		notSel_proba = clf.predict_proba(df_notSel.drop(['Class'], axis=1))
		if apply_unk == True:
			unk_proba = clf.predict_proba(df_unknowns.drop(['Class'], axis=1))
		
		# Evaluate performance
		if len(classes) == 2:
			i = 0
			for clss in classes:
				if clss == POS:
					POS_IND = i
					break
				i += 1
			scores = cv_proba[:,POS_IND]
			#Generate data frame with all scores
			score_columns=["score_%s"%(j)]
			df_sel_scores = pd.DataFrame(data=cv_proba[:,POS_IND],index=df.index,columns=score_columns)
			df_notSel_scores = pd.DataFrame(data=notSel_proba[:,POS_IND],index=df_notSel.index,columns=score_columns)
			current_scores =  pd.concat([df_sel_scores,df_notSel_scores], axis = 0)
			if apply_unk == True:
				df_unk_scores = pd.DataFrame(data=unk_proba[:,POS_IND],index=df_unknowns.index,columns=score_columns)
				current_scores =  pd.concat([current_scores,df_unk_scores], axis = 0)
			result = fun.Performance(y, cv_pred, scores, clf, classes, POS, POS_IND, NEG)
		else:
			
			#Generate data frame with all scores
			score_columns = []
			for clss in classes:
				score_columns.append("%s_score_%s"%(clss,j))
			
			df_sel_scores = pd.DataFrame(data=cv_proba,index=df.index,columns=score_columns)
			df_notSel_scores = pd.DataFrame(data=notSel_proba,index=df_notSel.index,columns=score_columns)
			current_scores =  pd.concat([df_sel_scores,df_notSel_scores], axis = 0)
			if apply_unk == True:
				df_unk_scores = pd.DataFrame(data=unk_proba,index=df_unknowns.index,columns=score_columns)
				current_scores =  pd.concat([current_scores,df_unk_scores], axis = 0)
			
			
			result = fun.Performance_MC(y, cv_pred, classes)
		return result,current_scores

	def Performance(y, cv_pred, scores, clf, classes, POS, POS_IND, NEG):
		from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

		
		# Gather balanced model scoring metrics
		cm = confusion_matrix(y, cv_pred, labels=classes)
		
		# Determine the best threshold cutoff for the balanced run
		y1 = y.replace(to_replace = [POS, NEG], value = [1,0])
		max_f1 = [-1,-1]
		max_f1_thresh = ''
		for thr in np.arange(0.01, 1, 0.01):
			thr_pred = scores.copy()
			thr_pred[thr_pred>=thr] = 1
			thr_pred[thr_pred<thr] = 0
			f1 = f1_score(y1, thr_pred, average=None)	# Returns F1 for each class
			# try:
				# f1 = f1_score(y1, thr_pred, average=None)	# Returns F1 for each class
			# except UndefinedMetricWarning: # THIS DOESN"T GET FLAGGED
				# f1 = 0
			if list(f1)[POS_IND] > list(max_f1)[POS_IND]:
				max_f1 = f1
				max_f1_thresh = thr
		
		# Calculate AUC_ROC and AUC_PRC (based on scores, so threshold doesn't matter)
		AucRoc = roc_auc_score(y1, scores) 
		AucPRc = average_precision_score(y1, scores)

		# Try to extract importance scores 
		try:
			importances = clf.feature_importances_  # Works for RF
		except:
			try:
				importances = clf.coef_	# Works for SVM
			except:
				print("Cannot get importance scores")
				return {'cm':cm, 'threshold':max_f1_thresh,'AucPRc':AucPRc, 'AucRoc':AucRoc, 'MaxF1': max_f1}
		
		
		return {'cm':cm, 'threshold':max_f1_thresh,'AucPRc':AucPRc, 'AucRoc':AucRoc, 'MaxF1': max_f1, 'importances':importances}
	


	def Performance_MC(y, cv_pred, classes):
		from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
		
		cm = confusion_matrix(y, cv_pred, labels=classes)
		accuracy = accuracy_score(y, cv_pred)
		macro_f1 = f1_score(y, cv_pred, average='macro')	# 
		f1 = f1_score(y, cv_pred, average=None)	# Returns F1 for each class
		
		return {'cm':cm, 'accuracy':accuracy,'macro_f1':macro_f1,'f1':f1}


	def Model_Performance_Thresh(df_proba, final_threshold, balanced_ids, POS, NEG):
		
		from sklearn.metrics import f1_score, confusion_matrix
		
		TP,TN,FP,FN,TPR,FPR,FNR,Precision,Accuracy,F1  = [],[],[],[],[],[],[],[],[],[]
		
		
		df_proba_thresh = df_proba.copy()
		proba_columns = [c for c in df_proba_thresh.columns if c.startswith('score_')]
		for proba_column in proba_columns:
			df_proba_thresh[df_proba[proba_column] >= final_threshold] = POS
			df_proba_thresh[df_proba[proba_column] < final_threshold] = NEG
		balanced_count = 0
		
		# Get predictions scores from the balanced runs using the final threshold
		for i in proba_columns:

			# Get y and yhat for instances that were in the balanced dataset
			y = df_proba_thresh.ix[balanced_ids[balanced_count], 'Class'] #,'Class']
			yhat = df_proba_thresh.ix[balanced_ids[balanced_count], i]
			balanced_count += 1

			matrix = confusion_matrix(y, yhat, labels = [POS,NEG])
			TP1, FP1, TN1, FN1 = matrix[0,0], matrix[1,0], matrix[1,1], matrix[0,1]

			TP.append(TP1)
			FP.append(FP1)
			TN.append(TN1)
			FN.append(FN1)
			TPR.append(TP1/(TP1 + FN1))  # synonyms: recall, sensitivity, hit rate
			FPR.append(FP1/(FP1 + TN1))  # synonyms: fall-out
			FNR.append(FN1/(FN1 + TP1))  # synonyms: miss rate
			Precision.append(TP1/(TP1+FP1)) # synonyms: positive predictive value
			Accuracy.append((TP1 + TN1)/ (TP1 + TN1 + FP1 + FN1))
			F1.append((2*TP1)/((2*TP1) + FP1 + FN1))

		denominator = np.sqrt(len(TP))
		TP = [np.mean(TP), np.std(TP), np.std(TP)/denominator]
		FP = [np.mean(FP), np.std(FP), np.std(FP)/denominator]
		TN = [np.mean(TN), np.std(TN), np.std(TN)/denominator]
		FN = [np.mean(FN), np.std(FN), np.std(FN)/denominator]
		TPR = [np.mean(TPR), np.std(TPR), np.std(TPR)/denominator]
		FPR = [np.mean(FPR), np.std(FPR), np.std(FPR)/denominator]
		FNR = [np.mean(FNR), np.std(FNR), np.std(FNR)/denominator]
		Precision = [np.mean(Precision), np.std(Precision), np.std(Precision)/denominator]
		Accuracy = [np.mean(Accuracy), np.std(Accuracy), np.std(Accuracy)/denominator]
		F1 = [np.mean(F1), np.std(F1), np.std(F1)/denominator]
		return TP,TN,FP,FN,TPR,FPR,FNR,Precision,Accuracy,F1


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
			y = df_proba.ix[balanced_ids[i], 'Class'] #,'Class']
			
			# Get decision matrix & scores at each threshold between 0 & 1
			for j in np.arange(0, 1, 0.01):
				yhat = df_proba.ix[balanced_ids[i], name].copy()
				yhat[df_proba[name] >= j] = POS
				yhat[df_proba[name] < j] = NEG
				matrix = confusion_matrix(y, yhat, labels = [POS,NEG])
				TP, FP, TN, FN = matrix[0,0], matrix[1,0], matrix[1,1], matrix[0,1]
				FPR.append(FP/(FP + TN))
				TPR.append(TP/(TP + FN))
				precis.append(TP/(TP+FP))
			
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
		plt.plot(FPR_mean, TPR_mean, lw=3, color= 'black', label='AUC-ROC: ' + str(round(ROC[0], 3)))
		plt.fill_between(FPR_mean, TPR_mean-TPR_sd, TPR_mean+TPR_sd, facecolor='black', alpha=0.4, linewidth = 0, label='SD_TPR')
		#plt.plot(FPRs_df.median(axis=1), TPRs_df.mean(axis=1), lw=2, color= 'blue', label='AUC-ROC: ' + str(round(ROC[0], 3)))
		#plt.fill_between(FPRs_df.median(axis=1), TPRs_df.min(axis=1), TPRs_df.max(axis=1), facecolor='blue', alpha=0.5, label='SD_TPR')
		plt.plot([0,1],[0,1],'r--', lw = 2, label = 'Random Expectation')
		plt.legend(loc='lower right')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		filename = SAVE + "_ROCcurve.png"
		plt.savefig(filename)
		plt.clf()
		
		# Plot the Precision-Recall Curve
		plt.title('PR Curve: ' + SAVE)
		plt.plot(TPR_mean, precis_mean, lw=3, color= 'black', label='AUC-PRc: ' + str(round(PRc[0], 3)))
		plt.fill_between(TPR_mean, precis_mean-precis_sd, precis_mean+precis_sd, facecolor='black', alpha=0.4, linewidth = 0, label='SD_Precision')
		#plt.plot(TPRs_df.median(axis=1), precisions_df.median(axis=1), lw=2, color= 'blue', label='AUC-PRc: ' + str(round(PRc[0], 3)))
		#plt.fill_between(TPRs_df.median(axis=1), precisions_df.min(axis=1), precisions_df.max(axis=1), facecolor='blue', alpha=0.5, label='SD_Precision')
		plt.plot([0,1],[0.5,0.5],'r--', lw=2, label = 'Random Expectation')
		plt.legend(loc='upper right')
		plt.xlim([0,1])
		plt.ylim([0.45,1])
		plt.ylabel('Precision')
		plt.xlabel('Recall')
		plt.show()
		filename = SAVE + "_PRcurve.png"
		plt.savefig(filename)
		plt.close()



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

		
		
		filename = SAVE + "_CM.png"
		plt.savefig(filename) 
		
		return 'Confusion matrix plotted.'

