"""
Generate AUC-PR and AUC-ROC curves for multiple ML runs in the same figure. 
Plots mean score over the balanced runs with stdev error bars.

*** Need _BalancedIDs.csv file in the same directory as the _scores.txt files.

To run:
$ export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
$ python ML_plots.py [SAVE_NAME] name1 [Path_to_1st_scores_file] name3 [Path_to_2nd_scores_file] etc.


"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 
SAVE = sys.argv[1]
POS = 1
NEG = 0
items = {}

# Organize all _scores and _BalancedIDs files into dictionary
for i in range (2,len(sys.argv),2):
  items[sys.argv[i]] = [sys.argv[i+1], sys.argv[i+1].replace('_scores.txt', '_BalancedIDs.csv')]

n_lines = len(items)

# To collect summary stats for each item to plot
FPR_all = {}
TPR_all = {}
Prec_all = {}

for i in items:
  # Read in scores and which genes were part of the balanced run for each run number
  df_proba = pd.read_csv(items[i][0], sep='\t', index_col = 0)
  n = len([c for c in df_proba.columns if c.lower().startswith('score_')])
  
  balanced_ids = []
  with open(items[i][1], 'r') as ids:
    balanced_ids = ids.readlines()
  balanced_ids = [x.strip().split(',') for x in balanced_ids]
  

  FPRs = {}
  TPRs = {}
  precisions = {}
  
  # For each balanced run
  for k in range(0, n): 
    FPR = []
    TPR = []
    precis = []
    name = 'score_' + str(k)
    y = df_proba.ix[balanced_ids[k], 'Class'] #,'Class']
    
    # Get decision matrix & scores at each threshold between 0 & 1
    for j in np.arange(0, 1, 0.01):
      yhat = df_proba.ix[balanced_ids[k], name].copy()
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

  # Add mean and stdev to summary stats 
  FPR_all[i] = [FPRs_df.mean(axis=1), FPRs_df.std(axis=1)]
  TPR_all[i] = [TPRs_df.mean(axis=1), TPRs_df.std(axis=1)]
  Prec_all[i] = [precisions_df.mean(axis=1), precisions_df.std(axis=1)]



colors = plt.cm.get_cmap('Set1')
# Plot the ROC Curve
plt.title('ROC Curve: ' + SAVE, fontsize=18)
count = 0
for i in items:
  c = colors(count/float(n_lines))
  plt.plot(FPR_all[i][0], TPR_all[i][0], lw=3, color= c, alpha=0.9, label = i)
  plt.fill_between(FPR_all[i][0], TPR_all[i][0]-TPR_all[i][1], TPR_all[i][0]+TPR_all[i][1], facecolor=c, alpha=0.2, linewidth = 0)
  count += 1
plt.plot([0,1],[0,1],'r--', lw = 2)
plt.legend(loc='lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=16)
plt.show()
filename = SAVE + "_ROCcurve.png"
plt.savefig(filename)
plt.clf()


# Plot the Precision-Recall Curve
plt.title('PR Curve: ' + SAVE, fontsize=18)
count2 = 0
for i in items:
  c = colors(count2/float(n_lines))
  plt.plot(TPR_all[i][0], Prec_all[i][0], lw=3, color= c, alpha=0.9, label = i)
  plt.fill_between(TPR_all[i][0], Prec_all[i][0]-Prec_all[i][1], Prec_all[i][0]+Prec_all[i][1], facecolor=c, alpha=0.2, linewidth = 0)
  count2 += 1
plt.plot([0,1],[0.5,0.5],'r--', lw=2)
plt.legend(loc='upper right')
plt.xlim([0,1])
plt.ylim([0.45,1])
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.show()
filename = SAVE + "_PRcurve.png"
plt.savefig(filename)
plt.close()


