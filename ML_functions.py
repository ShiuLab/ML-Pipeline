"""
PURPOSE:
Bianary classifications using RandomForestClassifier and LinearSVC implemented in sci-kit learn. 

Optional grid seach function (-gs True) to run a parameter sweep on a subset of the balanced datasets

To access pandas, numpy, and sklearn packages on HPS first run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH


INPUTS:
  
  REQUIRED:
  -df       Feature & class dataframe for ML. See "" for an example dataframe
  -save     Unique save name (caution - will overwrite!)
  
  OPTIONAL:
  -feat     Import file with list of features to keep if desired. Default: keep all.
  -gs       Set to True if parameter sweep is desired. Default = False
  -alg      Algorithm to use. Currently available: RandomForest (RF)(Default), LinearSVC (SVC)
  -n        Number of random balanced datasets to run. Default = 50
  -pos      String for what codes for the positive example (i.e. UUN) Default = 1
  -neg      String for what codes for the negative example (i.e. NNN) Default = 0
  -class    String for what column has the class. Default = Class


OUTPUT:
  -SAVE_imp           Importance scores for each feature
  -SAVE_GridSearch    Results from parameter sweep sorted by F1
  -RESULTS_XX.txt     Accumulates results from all ML runs done in a specific folder - use unique save names! XX = RF or SVC

"""
import sys
import pandas as pd
import numpy as np
import time

class fun(object):
  def __init__(self, filename):
    self.tokenList = open(filename, 'r')

  def GridSearch(df, SAVE, ALG, classes, min_size, gs_score, gs_n, n_jobs):
    """ Perform a parameter sweep using grid search CV implemented in SK-learn
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC

    start_time = time.time()

    ### NOTE: The returned top_params will be in alphabetical order - to be consistent add any additional 
    ###       parameters to test in alphabetical order
    if ALG == 'RF':
      #parameters = {'max_depth':(3, 5, 10, 50, 100), 'max_features': (0.1, 0.25, 0.5, 0.75, 0.9999, 'sqrt', 'log2'),'n_estimators':(50, 100, 500)}
      parameters = {'max_depth':(2, 5), 'max_features': (0.1, 0.5), 'n_estimators':(10, 50)}
    
    elif ALG == "SVC":
      #parameters = {'C':(0.1, 1), 'loss':('hinge', 'squared_hinge'), 'max_iter':(10,100)}
      parameters = {'C':(0.01, 0.1, 0.5, 1, 10, 50, 100), 'loss':('hinge', 'squared_hinge'), 'max_iter':(10,100,1000)}

    else:
      print('Grid search is not available for the algorithm selected')
      exit()
    
    gs_results = pd.DataFrame(columns = ['mean_test_score','params'])
    
    for j in range(gs_n):
      # Build random balanced dataset and define x & y
      df1 = pd.DataFrame(columns=list(df))
      for cl in classes:
        temp = df[df['Class'] == cl].sample(min_size, random_state=j)
        df1 = pd.concat([df1, temp])

      y = df1['Class']
      x = df1.drop(['Class'], axis=1) 

      # Build model, run grid search with 10-fold cross validation and fit
      if ALG == 'RF':
        model = RandomForestClassifier()
      elif ALG == "SVC":
        x = StandardScaler().fit_transform(x)
        model = LinearSVC()
      
      grid_search = GridSearchCV(model, parameters, scoring = gs_score, cv = 10, n_jobs = n_jobs, pre_dispatch=2*n_jobs)
      grid_search.fit(x, y)

      # Add results to dataframe
      j_results = pd.DataFrame(grid_search.cv_results_)
      gs_results = pd.concat([gs_results, j_results[['params','mean_test_score']]])
    
    # Break params into seperate columns
    gs_results2 = pd.concat([gs_results.drop(['params'], axis=1), gs_results['params'].apply(pd.Series)], axis=1)
    param_names = list(gs_results2)[1:]
    print('Parameters tested: %s' % param_names)

    # Find the mean score for each set of parameters & select the top set
    gs_results_mean = gs_results2.groupby(param_names).mean()
    gs_results_mean = gs_results_mean.sort_values('mean_test_score', 0, ascending = False)
    top_params = gs_results_mean.index[0]

    print("Parameter sweep time: %f seconds" % (time.time() - start_time))
    outName = SAVE + "_GridSearch"
    gs_results_mean.to_csv(outName)

    return top_params



  def RandomForest(df, classes, POS, n_estimators, max_depth, max_features, n_jobs, j):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    y = df['Class']
    x = df.drop(['Class'], axis=1) 
    
    # Create the classifier
    clf = RandomForestClassifier(n_estimators=int(n_estimators), 
      max_depth=max_depth, 
      max_features=max_features,
      criterion='gini', 
      random_state=j, 
      n_jobs=n_jobs)
    
    # Obtain the predictions using 10 fold cross validation (uses KFold cv by default):
    cv_pred = cross_val_predict(estimator=clf, X=x, y=y, cv=10)
    
    # Gather scoring metrics
    cm = confusion_matrix(y, cv_pred, labels=classes)
    accuracy = accuracy_score(y, cv_pred)
    macro_f1 = f1_score(y, cv_pred, average='macro')  # 
    f1 = f1_score(y, cv_pred, average=None)  # Returns F1 for each class
    
    # Calculate additional scores for binary predictions 
    if len(df['Class'].unique()) == 2:
      clf.fit(x,y)
      importances = clf.feature_importances_
      print(POS)
      AucRoc = roc_auc_score(y, cv_pred,  pos_label = POS)
      return [cm, accuracy, macro_f1, f1, AucRoc, importances]
    
    else:
      return [cm, accuracy, macro_f1, f1]


  def LinearSVC(df, classes, POS, C, loss, max_iter, n_jobs, j):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    
    y = df['Class']
    x = StandardScaler().fit_transform(df.drop(['Class'], axis=1))

    # Create the classifier
    clf = LinearSVC(C=float(C), 
      loss=loss, 
      penalty='l2', 
      max_iter=int(max_iter), 
      random_state=j)

    #Obtain the predictions using 10 fold cross validation (uses KFold cv by default):
    cv_pred = cross_val_predict(estimator=clf, X=x, y=y, cv=10, n_jobs = n_jobs)

    #return(y, cv_pred)
    # Gather scoring metrics
    cm = confusion_matrix(y, cv_pred, labels=classes)
    accuracy = accuracy_score(y, cv_pred)
    macro_f1 = f1_score(y, cv_pred, average='macro')  # use macro instead of weighted because models are balanced
    f1 = f1_score(y, cv_pred, average=None)  # Returns F1 for each class
    
    # Calculate additional scores for binary predictions 
    if len(df['Class'].unique()) == 2:
      clf.fit(x,y)
      importances = clf.feature_importances_
      AucRoc = roc_auc_score(y, cv_pred,  pos_label = POS)
      return [cm, accuracy, macro_f1, f1, AucRoc, importances]
    
    else:
      return [cm, accuracy, macro_f1, f1]

   

  def PR_Curve(y_pred, SAVE):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    
    f1_summary = f1_score(y_true, y_pred_round)
    plt.plot(recall, precision, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PR Curve: %s\nSummary F1 = %0.2f' % (SAVE, f1_summary))
    plt.show()
    # save a PDF file named for the CSV file (but in the current directory)
    filename = SAVE + "_PRcurve.png"
    plt.savefig(filename)

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

