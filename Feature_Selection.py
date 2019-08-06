"""
PURPOSE:
Run feature selection mettestd available from sci-kit learn on a given dataframe

Must set path to Miniconda in HPC:  export PATH=/mnt/testme/azodichr/miniconda3/bin:$PATH


INPUT:
  -df       Feature file for ML. If class/Y values are in a separate file use -df for features and -df2 for class/Y
  -alg        Feature selection mettestd to use 
                - Chi2 
                    need: -n
                - RandomForest 
                    need: -n, -type
                - Enrichment using Fisher's Exact (for classification with binary feats only)
                    need: -p (default=0.05)
                - LASSO 
                    need: -p, -type, n
                - Bayesian LASSO (bl) 
                    need: -n
                - Elastic Net (EN)
                    need: -n -p (default=0.5, proportion of L1 to L2 penalty)
                - Relief (https://github.com/EpistasisLab/scikit-rebate) (currently for regression only)
                    need: -n, 
                - BayesA regression (for regression only)
                    need: -n
                - rrBLUP (i.e. mixed.solve) (for regression only)
                    need: -n
                - Random
                    need: -n

OPTIONAL INPUT:

  -n        Number(s) of features you would like to keep (required for chi2, RF, relief, bayesA)
              Example: -n 10 or -n 10,50,100
  -test       File with list of intances to testldout from feature selection
  -save     Save name for list of features selected. Will automatically append _n to the name
              Default: df_F_n or df_f_cvJobNum_n
  -cl_train   Since only RF works for multi-class problems, use cl_train to give a list of what classes you want to include (Default = 'all')   
              If binary, first label = positive class.
  -sep      Set seperator for input data (Default = '\t')
  -df2      File with class information. Use only if df contains the features but not the classes 
              * Need to specifiy what column in df2 is y using -y_name 
  -y_name   Name of the column to predict (Default = Class)
  -drop_na  T/F to drop rows with NAs
  -feat     File containing the features you want to use from the df (one feature per line)
              Default: all (i.e. everything in the dataframe given).
  -type     r = regression, c = classification (required for LASSO and RF)
  -p        Parameter value for LASSO, EN, or Fisher's Exact Test.
            Fishers: pvalue cut off (Default = 0.05)
            EN: Ratio of L1 to L2 regularization (larger = fewer features selected)
            LASSO: If type = r: need alpha value, try 0.01, 0.001. (larger = fewer features selected)
            LASSO: If type = c: need C which controls the sparcity, try 0.01, 0.1 (smaller = fewer features selected)
  -pos      String for what codes for the positive example (i.e. UUN) Default = 1
  -neg      String for what codes for the negative example (i.e. NNN) Default = 0
  -cvs      To run feat. sel. withing cross-validation scheme provide a CVs matrix and -JobNum
              CVs maxtrix: rows = instances, columns = CV replicates, value are the CV-fold each instance belongs to.
  -scores   T/F to output scores/coefficients for each feature (Not available for: FET, LASSO, or Chi2)

OUTPUT:
  -df_alg.txt    New dataframe with columns only from feature selection


AUTtestR: Christina Azodi

REVISIONS:   Written 8/16/2016
             Added relief algorithm 10/22/2017
             Added BayesA algorithm 3/23/2018

"""
import pandas as pd
import numpy as np
import subprocess as sp
import sys, os, time

start_time = time.time()

def SaveTopFeats(top, save_name):

    try:
      top.remove('Class')
    except:
      pass

    out = open(save_name, 'w')
    for f in top:
      out.write(f + '\n')




def DecisionTree(df, n, TYPE, save_name, SCORES):
  """Feature selection using DecisionTree on the wtestle dataframe
  Feature importance from the Random Forest Classifier is the Gini importance
  (i.e. the normalized total reduction of the criterion for the decendent nodes
    compared to the parent node brought by that feature across all trees.)
  """
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import RandomForestRegressor
  from math import sqrt

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values

  fean_num_feat_sel = len(list(df.columns.values)[1:])
  if TYPE.lower() == 'c':
    feat_sel_forest = RandomForestClassifier(criterion='entropy', max_features= round(sqrt(fean_num_feat_sel)), n_estimators=500, n_jobs=1)
  elif TYPE.lower() == 'r':
    Y_all = Y_all.astype('float')
    feat_sel_forest = RandomForestRegressor(max_features= round(sqrt(fean_num_feat_sel)), n_estimators=500, n_jobs=1)
  else:
    print('Need to specify -type r/c (regression/classification)')
    exit()

  print("=====* Running decision tree based feature selection *=====")

  #Train the model & derive importance scores
  feat_sel_forest = feat_sel_forest.fit(X_all, Y_all)
  importances = feat_sel_forest.feature_importances_

  # Sort importance scores and keep top n
  feat_names = list(df.columns.values)[1:]
  temp_imp = pd.DataFrame(importances, columns = ["imp"], index=feat_names) 
  indices = np.argsort(importances)[::-1]

  if SCORES.lower() != 'f':
    save_scores = save_name + '_RFScores.txt'
    temp_imp.to_csv(save_scores)

  for n_size in n:
    indices_keep = indices[0:int(n_size)]
    fixed_index = []
    # Translate keep indices into the indices in the df
    for i in indices_keep:
      new_i = i + 1
      fixed_index.append(new_i)
    fixed_index = [0] + fixed_index
    good = [df.columns[i] for i in fixed_index]
    print("Features selected using DecisionTree feature selection: %s" % str(good.remove('Class')))

    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(good, save_name2)

  
  
def Chi2(df, n, save_name):
  """Feature selection using Chi2 on the wtestle dataframe. 
  Chi2 measures the dependence between stochastic variables, this mettestd 
  weeds out features that are most likely to be independent of class"""
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2
  from sklearn.feature_selection import mutual_info_classif 

  print('This function might not be working right now.... Bug Christina if you need it!')
  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values
  #Y_all = Y_all.astype('int')
  print(Y_all)
  # Set selection to chi2 with n to keep
  for n_size in n:
    # Set selection to chi2 with n to keep
    ch2 = SelectKBest(chi2, k=n)
    ch2.fit_transform(X_all, Y_all)
    index = ch2.get_support(indices=True)

    # Translate keep indices into the indices in the df
    fixed_index = []
    for i in index:
      new_i = i + 1
      fixed_index.append(new_i)
    fixed_index = [0] + fixed_index

    good = [df.columns[i] for i in fixed_index]
    print("Features selected using DecisionTree feature selection: %s" % str(good))

    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(good, save_name2)


def Relief(df, n, n_jobs, save_name, SCORES):
  """Feature selection using Relief on the wtestle dataframe."""
  from skrebate import ReliefF

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values
  Y_all = Y_all.astype('int')

  feature_names = list(df)
  feature_names.remove('Class')
  print("=====* Running relief/rebase based feature selection *=====")

  # Set selection to relief
  fs = ReliefF(n_jobs = int(n_jobs))
  fs.fit(X_all, Y_all)
  imp = pd.DataFrame(fs.feature_importances_, index = feature_names, columns = ['relief_imp'])
  imp_top = imp.sort_values(by='relief_imp', ascending=False)

  if SCORES.lower() != 'f':
    save_scores = save_name + '_ReliefScores.txt'
    imp_top.to_csv(save_scores)

  for n_size in n:
    keep = imp_top.index.values[0:int(n_size)]
    print("Features selected using Relief from rebase: %s" % str(keep))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(keep, save_name2)



def L1(df, PARAMETER, TYPE, save_name):
  """Apply a linear model with a L1 penalty and select features wtest's coefficients aren't 
  shrunk to zero. Unlike Chi2, this mettestd accounts for the effect of all of the
  other features when determining if a feature is a good predictor.
  For a regression problem, it uses linear_model.Lasso
  For a classification problem, it uses svm.LinearSVC """

  from sklearn.feature_selection import SelectFromModel
  from sklearn.svm import LinearSVC
  from sklearn.linear_model import Lasso

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values
  Y_all = Y_all.astype('int')

  if TYPE == 'c' or TYPE == 'classification':
    estimator = LinearSVC(C = PARAMETER, penalty='l1', dual=False).fit(X_all, Y_all)
  elif TYPE == 'r' or TYPE == 'regression':
    estimator = Lasso(alpha = PARAMETER).fit(X_all, Y_all)

  print("=====* Running L1/LASSO based feature selection *=====")
  
  model = SelectFromModel(estimator, prefit=True)
  keep = model.get_support([])

  X_new = model.transform(X_all)
  feat_names = np.array(list(df)[1:])
  good = feat_names[keep]
  
  print("Features selected using LASSO: %s" % str(good))
  print('\nNumber of features selected using LASSO (sparcity parameter = %s): %i' % (str(PARAMETER), X_new.shape[1]))
  
  save_name2 = save_name 
  SaveTopFeats(good, save_name2)

def EN(df, PARAMETER, n, save_name, SCORES):
  """Apply Elastic-Net based feature selection. Can tune l1:l2 (penalty:zero) ratio (Default = 0.5)"""

  from sklearn.linear_model import ElasticNet

  X_all = df.drop('Class', axis=1).values  
  Y_all = df.loc[:, 'Class'].values
  feature_names = list(df)
  feature_names.remove('Class')
  
  enet = ElasticNet(alpha=0.5, l1_ratio=PARAMETER, fit_intercept=True, positive=False).fit(X_all, Y_all)

  imp = pd.DataFrame(enet.coef_, index = feature_names, columns = ['EN_coef'])
  imp = imp.abs()
  imp_top = imp.sort_values(by='EN_coef', ascending=False)
  
  # Count the number of coefficients that were not zero
  non_zero = imp_top[imp_top > 0 ].count() 
  if SCORES.lower() != 'f':
    save_scores = save_name + '_ENScores.txt'
    imp_top.to_csv(save_scores)


  for n_size in n:
    keep = imp_top.index.values[0:int(n_size)]
    if int(n_size) > int(non_zero):
      print("!!!!!!!!!!!WARNING!!!!!!!!!!!! ONLY %i FEATURES HAD A NON-ZERO COEFFICIENT." % non_zero)
      print('!!!!!!!!!!!WARNING!!!!!!!!!!!! THIS LIST CONTAINS SOME FEATURES THAT HAD A COEF = 0')
    print("Features selected using Elastic-Net with l1 ratio = %0.6f: %s..." % (PARAMETER, str(keep[:5])))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(keep, save_name2)

  print("Note that using a l1 ratio = %.6f, there were %i non-zero features" % (PARAMETER, non_zero))



def BayesA(df_use, n, save_name, SCORES):
  """ Use BayesA from BGLR package to select features with largest
  abs(coefficients) """
  
  cwd = os.getcwd()
  temp_name = 'temp_' + save_name
  df_use.to_csv(temp_name)

  # Write temp Rscript  
  tmpR=open("%s_BayA.R" % temp_name,"w")
  tmpR.write('library(BGLR)\n')
  tmpR.write("setwd('%s')\n" % cwd)
  tmpR.write("df <- read.csv('%s', sep=',', header=TRUE, row.names=1)\n" % temp_name)
  tmpR.write("Y <- df[, 'Class']\n")
  tmpR.write("X <- df[, !colnames(df) %in% c('Class')]\n")
  tmpR.write("X=scale(X)\n")
  tmpR.write("ETA=list(list(X=X,model='BayesA'))\n")
  tmpR.write("fm=BGLR(y=Y,ETA=ETA,verbose=FALSE, nIter=12000,burnIn=2000)\n")
  tmpR.write("coef <- abs(fm$ETA[[1]]$b)\n")
  tmpR.write("coef_df <- as.data.frame(coef)\n")
  tmpR.write("write.table(coef_df, file='%s', sep=',', row.names=TRUE, quote=FALSE)\n" % (temp_name + '_BayAScores.txt'))
  tmpR.close()
  
  print('Running bayesA model from BGLR inplemented in R.')

  process = sp.Popen('module load R && export R_LIBS_USER=~/R/library && R CMD BATCH %s_BayA.R' % temp_name, shell=True)
  process.wait()

  coefs = pd.read_csv(temp_name + '_BayAScores.txt', sep = ',')
  coefs['coef_abs'] = coefs.coef.abs()
  coefs_top = coefs.sort_values(by='coef_abs', ascending=False)

  if SCORES.lower() == 'f':
  	os.system("rm %s_BayAScores.txt" % temp_name)
  else:
    os.system("mv %s_BayAScores.txt %s_BayAScores.txt" % (temp_name, save_name))
  os.system("rm %s %s_BayA.R varE.dat mu.dat ETA_1_ScaleBayesA.dat ETA_1_lambda.dat" % (temp_name, temp_name))
  
  for n_size in n:
    keep = coefs_top.index.values[0:int(n_size)]
    print("Top %s features selected using BayesA from BGLR: %s" % (str(n_size), str(keep)))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(keep, save_name2)

def BLASSO(df_use, n, save_name, SCORES):
  """ Use BayesA from BGLR package to select features with largest
  abs(coefficients) """
  
  cwd = os.getcwd()
  temp_name = 'temp_' + save_name
  df_use.to_csv(temp_name)

  # Write temp Rscript  
  tmpR=open("%s_BL.R" % temp_name,"w")
  tmpR.write('library(BGLR)\n')
  tmpR.write("setwd('%s')\n" % cwd)
  tmpR.write("df <- read.csv('%s', sep=',', header=TRUE, row.names=1)\n" % temp_name)
  tmpR.write("Y <- df[, 'Class']\n")
  tmpR.write("X <- df[, !colnames(df) %in% c('Class')]\n")
  tmpR.write("ETA=list(list(X=X,model='BL'))\n")
  tmpR.write("fm=BGLR(y=Y,ETA=ETA,verbose=FALSE, nIter=12000,burnIn=2000)\n")
  tmpR.write("coef <- abs(fm$ETA[[1]]$b)\n")
  tmpR.write("coef_df <- as.data.frame(coef)\n")
  tmpR.write("write.table(coef_df, file='%s', sep=',', row.names=TRUE, quote=FALSE)\n" % (temp_name + '_BLScores.txt'))
  tmpR.close()
  
  print('Running bayesA model from BGLR inplemented in R.')
  process = sp.Popen('module load R && export R_LIBS_USER=~/R/library && R CMD BATCH %s_BL.R' % temp_name, shell=True)
  process.wait()

  coefs = pd.read_csv(temp_name + '_BLScores.txt', sep = ',')
  coefs['coef_abs'] = coefs.coef.abs()
  coefs_top = coefs.sort_values(by='coef_abs', ascending=False)

  if SCORES.lower() == 'f':
    os.system("rm %s_BLScores.txt" % temp_name)
  else:
    os.system("mv %s_BLScores.txt %s_BLScores.txt" % (temp_name, save_name))
  os.system("rm %s %s_rrB.R varE.dat mu.dat ETA_1_ScaleBL.dat ETA_1_lambda.dat" % (temp_name, temp_name, temp_name))

  for n_size in n:
    keep = coefs_top.index.values[0:int(n_size)]
    print("Top %s features selected using BL from BGLR: %s" % (str(n_size), str(keep)))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(keep, save_name2)
  
def rrBLUP(df_use, n, save_name, SCORES):
  """ Use BayesA from BGLR package to select features with largest
  abs(coefficients) """
  
  cwd = os.getcwd()
  temp_name = 'temp_' + save_name
  df_use.to_csv(temp_name)

  # Write temp Rscript  
  tmpR=open("%s_rrB.R" % temp_name,"w")
  tmpR.write("setwd('%s')\n" % cwd)
  tmpR.write('library(rrBLUP)\n')
  tmpR.write("df <- read.csv('%s', sep=',', header=TRUE, row.names=1)\n" % temp_name)
  tmpR.write("Y <- df[, 'Class']\n")
  tmpR.write("X <- df[, !colnames(df) %in% c('Class')]\n")
  tmpR.write("mod <- mixed.solve(Y, Z=X, K=NULL, SE=FALSE, return.Hinv=FALSE)\n")
  tmpR.write("coef <- mod$u\n")
  tmpR.write("coef_df <- as.data.frame(coef)\n")
  tmpR.write("write.table(coef_df, file='%s', sep=',', row.names=TRUE, quote=FALSE)\n" % (temp_name + '_rrBScores.txt'))
  tmpR.close()
  
  print('Running rrBLUP using mixed.solve in R.')
  process = sp.Popen('module load R && export R_LIBS_USER=~/R/library && R CMD BATCH %s_rrB.R' % temp_name, shell=True)
  process.wait()

  coefs = pd.read_csv(temp_name + '_rrBScores.txt', sep = ',')
  coefs['coef_abs'] = coefs.coef.abs()
  coefs_top = coefs.sort_values(by='coef_abs', ascending=False)
  
  if SCORES.lower() == 'f':
	  os.system("rm %s_rrBScores.txt" % temp_name)
  else:
    os.system("mv %s_rrBScores.txt %s_rrBScores.txt" % (temp_name, save_name))
  os.system("rm %s %s_rrB.R varE.dat mu.dat ETA_1_ScalerrB.dat" % (temp_name, temp_name))

  for n_size in n:
    keep = coefs_top.index.values[0:int(n_size)]
    print("Top %s features selected using mixed.solve in R (similar to rrBLUP): %s" % (str(n_size), str(keep)))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(keep, save_name2)

def FET(df, PARAMETER, pos, neg, save_name):
  """Use Fisher's Exact Test to look for enriched features"""
  from scipy.stats import fisher_exact

  kmers = list(df)
  kmers.remove(CL)

  enriched = [CL]
  
  print("=====* Running enrichement based feature selection *=====")
  
  for k in kmers:
    temp = df.groupby([CL, k]).size().reset_index(name="Count")
    try:
      TP = temp.loc[(temp[CL] == pos) & (temp[k] == 1), 'Count'].iloc[0]
    except:
      TP = 0
    try:
      TN = temp.loc[(temp[CL] == neg) & (temp[k] == 0), 'Count'].iloc[0]
    except:
      TN = 0
    try:
      FP = temp.loc[(temp[CL] == neg) & (temp[k] == 1), 'Count'].iloc[0]
    except:
      FP = 0
    try:
      FN = temp.loc[(temp[CL] == pos) & (temp[k] == 0), 'Count'].iloc[0]
    except:
      FN = 0

    oddsratio,pvalue = fisher_exact([[TP,FN],[FP,TN]],alternative='greater')
      
    if pvalue <= PARAMETER:
      enriched.append(k)

  save_name2 = save_name 
  SaveTopFeats(enriched, save_name2)


def Random(df, n, save_name):
  """Randomly select n features"""
  from random import sample
  feat_names = list(df)
  feat_names.remove('Class')

  for n_size in n:
    rand_feats = sample(feat_names, int(n_size))
    save_name2 = save_name + "_" + str(n_size)
    SaveTopFeats(rand_feats, save_name2)

if __name__ == "__main__":
  
  #Default parameters
  FEAT = 'all'    #Features to include from dataframe. Default = all (i.e. don't remove any from the given dataframe)
  neg = 0    #Default value for negative class = 0
  pos = 1    #Default value for positive class = 1
  save_list = 'false'
  p = 0.05
  CL = 'Class'
  TYPE = 'c'
  n_jobs = 1
  CVs, REPS = 'pass', 1
  SEP = '\t'
  SAVE, DF2 = 'default', 'None'
  UNKNOWN = 'unk'
  y_name = 'Class'
  test = ''
  cl_train = ''
  drop_na = 'f'
  NA = 'na'
  SCORES = 'f'
  N = 10

  for i in range (1,len(sys.argv),2):

    if sys.argv[i].lower() == "-df":
      DF = sys.argv[i+1]
    if sys.argv[i].lower() == "-df2":
      DF2 = sys.argv[i+1]
    if sys.argv[i].lower() == "-sep":
      SEP = sys.argv[i+1]
    if sys.argv[i].lower() == '-save':
      SAVE = sys.argv[i+1]
    if sys.argv[i].lower() == '-alg':
      alg = sys.argv[i+1]
      if alg.lower() == 'en':
        PARAMETER = 0.5
    if sys.argv[i].lower() == '-n':
      N = sys.argv[i+1]
    if sys.argv[i].lower() == '-n_jobs':
      n_jobs = int(sys.argv[i+1])
    if sys.argv[i].lower() == '-feat':
      FEAT = sys.argv[i+1]
    if sys.argv[i].lower() == '-cl_train':
      cl_train = sys.argv[i+1]
    if sys.argv[i].lower() == '-p':
      PARAMETER = float(sys.argv[i+1])        
    if sys.argv[i].lower() == '-type':
      TYPE = sys.argv[i+1]
    if sys.argv[i].lower() == '-y_name':
      y_name = sys.argv[i+1]
    if sys.argv[i].lower() == '-pos':
      pos = sys.argv[i+1]
    if sys.argv[i].lower() == '-neg':
      neg = sys.argv[i+1]
    if sys.argv[i].lower() == '-cvs':
      CVs = sys.argv[i+1]
    if sys.argv[i].lower() == '-jobnum':
      jobNum = sys.argv[i+1]
    if sys.argv[i].lower() == '-test':
      test = sys.argv[i+1]
    if sys.argv[i].lower() == '-drop_na':
      drop_na = sys.argv[i+1]
    if sys.argv[i].lower() == '-scores':
      SCORES = sys.argv[i+1]


  if len(sys.argv) <= 1:
    print(__doc__)
    exit()
  #Load feature matrix and save feature names 
  df = pd.read_csv(DF, sep=SEP, index_col = 0)

  # If features  and class info are in separate files, merge them: 
  if DF2 != 'None':
    start_dim = df.shape
    df_class = pd.read_csv(DF2, sep=SEP, index_col = 0)
    df = pd.concat([df_class[y_name], df], axis=1, join='inner')
    print('Merging the feature & class dataframes changed the dimensions from %s to %s (instance, features).' 
      % (str(start_dim), str(df.shape)))

  # Specify class column - default = Class
  if y_name != 'Class':
    df = df.rename(columns = {y_name:'Class'})


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
  
  # Drop instances in testld out set if provided
  if test !='':
    print('Removing testldout instances...')
    with open(test) as test_file:
      test_instances = test_file.read().splitlines()
    try:
      df = df.drop(test_instances)
    except:
      print('Trying converting instance names to int')
      test_instances = [int(i) for i in test_instances]
      df = df.drop(test_instances)
  
    # Drop instances that aren't in the listed classes (i.e. make binary matrix)
  if cl_train !='':
    start_dim = df.shape
    use_classes = cl_train.strip().split(',')
    df = df[df['Class'].isin(use_classes)]
    print('Dropping instances that are not in %s, changed dimensions from %s to %s (instance, features).' 
      % (str(use_classes), str(start_dim), str(df.shape)))
  
  df = df[df['Class'] != UNKNOWN]
  #Recode class as 1 for positive and 0 for negative
  if TYPE.lower() == 'c':
    if cl_train != '':
      use_classes = cl_train.strip().split(',')
      pos = use_classes[0]
      neg = use_classes[1]
    df["Class"] = df["Class"].replace(pos, 1)
    df["Class"] = df["Class"].replace(neg, 0) 

  # If requesting multiple n, convert to list
  try:
    N = N.strip().split(',')
  except:
    N = [N]

  #If 'features to keep' list given, remove columns not in that list
  if FEAT != 'all':
    with open(FEAT) as f:
      features = f.read().splitlines()
      features = ['Class'] + features
    df = df.loc[:,features]
  
  print('\nSnapshot of data:')
  print(df.iloc[:6, :5])

  # Run feature selection
  df_use = df.copy()
  
  # Run FS within a cross-validation scheme
  if CVs != 'pass':
    print("Working on cv_%s" % str(jobNum))
    cv_folds = pd.read_csv(CVs, sep=',', index_col=0)
    cv = cv_folds['cv_' + str(jobNum)]
    df_use['Class'][cv==5] = 'unk'


  if SAVE != 'default':
    save_name = SAVE
  else:
    try:
      save_name = DF.split("/")[-1] + "_" + y_name + '_'+ alg + '_cv' + str(jobNum)
    except:
      save_name = DF.split("/")[-1] + "_" + alg


  if alg.lower() == "randomforest" or alg.lower() == "rf":
    DecisionTree(df_use, N, TYPE, save_name, SCORES)
    
  elif alg.lower() == "chi2" or alg.lower() == "c2":
    Chi2(df_use, N, save_name)
    
  elif alg.lower() == "l1" or alg.lower() == "lasso":
    if SAVE == 'default':
      save_name = save_name + '_' + str(PARAMETER)
    L1(df_use, PARAMETER, TYPE, save_name)

  elif alg.lower() == "en" or alg.lower() == "elasticnet":
    if SAVE == 'default':
      save_name = save_name + '_' + str(PARAMETER)
    EN(df_use, PARAMETER, N, save_name, SCORES)
    
  elif alg.lower() == "relief" or alg.lower() == "rebate" or alg.lower() == "rl":
    Relief(df_use, N, n_jobs, save_name, SCORES)

  elif alg.lower() == "bayesa" or alg.lower() == "ba":
    BayesA(df_use, N, save_name, SCORES)
    
  elif alg.lower() == "blasso" or alg.lower() == "bl":
    BLASSO(df_use, N, save_name, SCORES)

  elif alg.lower() == "rrblup" or alg.lower() == "rrb":
    rrBLUP(df_use, N, save_name, SCORES)

  elif alg.lower() == "fisher" or alg.lower() == "fet" or alg.lower() == 'enrich':
    if SAVE == 'default':
      save_name = save_name + '_' + str(PARAMETER)
    FET(df_use, PARAMETER, pos, neg, save_name)
  
  elif alg.lower() == "random" or alg.lower() == "rand" or alg.lower() == 'ran':
    Random(df_use, N, save_name)


run_time = time.time() - start_time
print('Run time (sec):' + str(round(run_time,2)))
print('Done!')
