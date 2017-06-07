"""
PURPOSE:
Run feature selection method available from sci-kit learn on a given dataframe

Must set path to Miniconda in HPC:  export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH


INPUT:
  -df       Feature dataframe for ML. Format -> Col 1 = example.name, Col 2 = Class, Col 3-... = Features.
  -f        Feature selection method to use (Available = Chi2, DecisionTree, L1 (need -p, -type))


OPTIONAL INPUT:
  -feat     Default: all (i.e. everything in the dataframe given). Can import txt file with list of features to keep.
  -pos      String for what codes for the positive example (i.e. UUN) Default = 1
  -neg      String for what codes for the negative example (i.e. NNN) Default = 0
  -type     r = regression, c = classification
  -p        Parameter value for L1.
            If type = r: need alpha value, try 0.01, 0.001. (larger = fewer features selected)
            If type = c: need C which controls the sparcity, try 0.01, 0.1 (smaller = fewer features selected)
  -n        Number of features you would like to keep

OUTPUT:
  -df_f.txt    New dataframe with columns only from feature selection


AUTHOR: Christina Azodi

REVISIONS:   Submitted 8/16/2016

"""
import pandas as pd
import numpy as np
import sys, os

def DecisionTree(df, n):
  """Feature selection using DecisionTree on the whole dataframe
  Feature importance from the Random Forest Classifier is the Gini importance
  (i.e. the normalized total reduction of the criterion for the decendent nodes
    compared to the parent node brought by that feature across all trees.)
  """
  from sklearn.ensemble import RandomForestClassifier
  from math import sqrt

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values

  fean_num_feat_sel = len(list(df.columns.values)[1:])
  feat_sel_forest = RandomForestClassifier(criterion='entropy', max_features= round(sqrt(fean_num_feat_sel)), n_estimators=500, n_jobs=8)
  
  #Train the model & derive importance scores
  feat_sel_forest = feat_sel_forest.fit(X_all, Y_all)
  importances = feat_sel_forest.feature_importances_

  # Sort importance scores and keep top n
  feat_names = list(df.columns.values)[1:]
  temp_imp = pd.DataFrame(importances, columns = ["imp"], index=feat_names) 
  indices = np.argsort(importances)[::-1]
  indices_keep = indices[0:n]
  fixed_index = []

  # Translate keep indices into the indices in the df
  for i in indices_keep:
    new_i = i + 1
    fixed_index.append(new_i)
  fixed_index = [0] + fixed_index

  good = [df.columns[i] for i in fixed_index]

  df = df.loc[:,good]
  print("Features selected using DecisionTree feature selection: %s" % str(good))
  return(df)
  
def Chi2(df, n):
  """Feature selection using Chi2 on the whole dataframe. 
  Chi2 measures the dependence between stochastic variables, this method 
  weeds out features that are most likely to be independent of class"""
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values

  # Set selection to chi2 with n to keep
  ch2 = SelectKBest(chi2, k=n)
  X_new = ch2.fit_transform(X_all, Y_all)
  index = ch2.get_support(indices=True)

  # Translate keep indices into the indices in the df
  fixed_index = []
  for i in index:
    new_i = i + 1
    fixed_index.append(new_i)
  fixed_index = [0] + fixed_index

  good = [df.columns[i] for i in fixed_index]
  
  print("Features selected using Chi2 feature selection: %s" % str(good))
  df = df.loc[:,good]
  return(df)

def L1(df, PARAMETER, TYPE):
  """Apply a linear model with a L1 penalty and select features who's coefficients aren't 
  shrunk to zero. Unlike Chi2, this method accounts for the effect of all of the
  other features when determining if a feature is a good predictor.
  For a regression problem, it uses linear_model.Lasso
  For a classification problem, it uses svm.LinearSVC """

  from sklearn.feature_selection import SelectFromModel
  from sklearn.svm import LinearSVC
  from sklearn.linear_model import Lasso

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values
  print(Y_all)

  if TYPE == 'c' or TYPE == 'classification':
    estimator = LinearSVC(C = PARAMETER, penalty='l1', dual=False).fit(X_all, Y_all)
  elif TYPE == 'r' or TYPE == 'regression':
    estimator = Lasso(alpha = PARAMETER).fit(X_all, Y_all)
  
  model = SelectFromModel(estimator, prefit=True)
  keep = model.get_support([])

  X_new = model.transform(X_all)
  feat_names = np.array(list(df)[1:])
  good = feat_names[keep]
  
  print('Number of features selected using l2 (parameter = %s): %i' % (str(PARAMETER), X_new.shape[1]))
  print("Features selected using l2: %s" % str(good))
  df2 = pd.DataFrame(data = X_new, columns=good, index=df.index)
  df2.insert(0, 'Class', Y_all, )
  return(df2)


if __name__ == "__main__":
  
  #Default parameters
  FEAT = 'all'    #Features to include from dataframe. Default = all (i.e. don't remove any from the given dataframe)
  neg = int(0)    #Default value for negative class = 0
  pos = int(1)    #Default value for positive class = 1

  for i in range (1,len(sys.argv),2):

        if sys.argv[i] == "-df":
          DF = sys.argv[i+1]
        if sys.argv[i] == '-save':
          SAVE = sys.argv[i+1]
        if sys.argv[i] == '-f':
          F = sys.argv[i+1]
        if sys.argv[i] == '-n':
          N = int(sys.argv[i+1])
        if sys.argv[i] == '-feat':
          FEAT = sys.argv[i+1]
        if sys.argv[i] == '-p':
          PARAMETER = float(sys.argv[i+1])        
        if sys.argv[i] == '-type':
          TYPE = sys.argv[i+1]

  if len(sys.argv) <= 1:
    print(__doc__)
    exit()

  #Load feature matrix and save feature names 
  if isinstance(DF, str):
    df = pd.read_csv(DF, sep='\t', index_col = 0)
  else:
    df = DF

  print('Original dataframe contained %i features' % df.shape[1])

  #Recode class as 1 for positive and 0 for negative, then divide into two dataframes.
  df["Class"] = df["Class"].replace(pos, 1)
  df["Class"] = df["Class"].replace(neg, 0)

  #If 'features to keep' list given, remove columns not in that list
  if FEAT != 'all':
    with open(FEAT) as f:
      features = f.read().splitlines()
      features = ['Class'] + features
    df = df.loc[:,features]

  # Run feature selection

  if F == "decisiontree" or F == "DecisionTree":
    df_feat = DecisionTree(df, N)
    save_name = DF.split("/")[-1] + "_" + F + "_" + str(N)
  elif F == "chi2" or F == "Chi2":
    df_feat = Chi2(df, N)
    save_name = DF.split("/")[-1] + "_" + F + "_" + str(N)
  elif F == "L1" or F == "l1":
    df_feat = L1(df, PARAMETER, TYPE)
    save_name = DF.split("/")[-1] + "_" + F + "_" + TYPE + '_' + str(PARAMETER)
  else:
    print("Feature selection method not available in this script")


  
  df_feat.to_csv(save_name, sep='\t', quoting=None)

