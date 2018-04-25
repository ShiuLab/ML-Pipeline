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
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from cycler import cycler
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 
if len(sys.argv) <= 1:
  print(__doc__)
  exit()

SAVE = sys.argv[1]
POS = sys.argv[2]
NEG = sys.argv[3]
items = OrderedDict()

# Organize all _scores and _BalancedIDs files into dictionary
for i in range (4,len(sys.argv),2):
  print(sys.argv[i])
  items[sys.argv[i]] = [sys.argv[i+1], sys.argv[i+1].replace('_scores.txt', '_BalancedIDs')]


n_lines = len(items)

# To collect summary stats for each item to plot
FPR_all = {}
TPR_all = {}
Prec_all = {}

for i in items:
  # Read in scores and which genes were part of the balanced run for each run number
  df_proba = pd.read_csv(items[i][0], sep='\t', index_col = 0)
  n = len([c for c in df_proba.columns if c.lower().startswith('score_')])
  print(df_proba.head())
  balanced_ids = []
  with open(items[i][1], 'r') as ids:
    balanced_ids = ids.readlines()
  balanced_ids = [x.strip().split('\t') for x in balanced_ids]

  FPRs = {}
  TPRs = {}
  precisions = {}
  
  # For each balanced run
  for k in range(0, n): 
    print("Working on",i,k)
    FPR = []
    TPR = []
    precis = []
    name = 'score_' + str(k)
    y = df_proba.ix[balanced_ids[k], 'Class'] 

    # Get decision matrix & scores at each threshold between 0 & 1
    for j in np.arange(0, 1, 0.01):
      yhat = df_proba.ix[balanced_ids[k], name].copy()
      
      yhat[df_proba[name] >= float(j)] = POS
      yhat[df_proba[name] < float(j)] = NEG
      print(y.head())
      print(yhat.head())
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


matplotlib.rc('pdf', fonttype=42)
colors = ['#a1d581','#fea13c','#8abedc','#8761ad','#f27171','#9a9a9a']
#colors = plt.cm.get_cmap('Set1')

# Plot the ROC Curve
f = plt.figure()
plt.title('ROC Curve: ' + SAVE, fontsize=18)
count = 0
for i in items:
  #c = colors(count/float(n_lines))
  #c = colors[0:n_lines]
  plt.plot(FPR_all[i][0], TPR_all[i][0], lw=3, color= colors[count], alpha=1, label = i)
  plt.fill_between(FPR_all[i][0], TPR_all[i][0]-TPR_all[i][1], TPR_all[i][0]+TPR_all[i][1], facecolor=colors[count], alpha=0.3, linewidth = 0)
  count += 1
plt.plot([-0.03,1.03],[-0.03,1.03],'r--', lw = 2)
plt.legend(loc='lower right')
plt.xlim([-0.03,1.03])
plt.ylim([-0.03,1.03])
plt.axes().set_aspect('equal')
plt.ylabel('True Positive Rate', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=16)
plt.show()

filename = SAVE + "_ROCcurve.png"
plt.savefig(filename)
filename_pdf = SAVE + "_ROCcurve.pdf"
f.savefig(filename_pdf, bbox_inches='tight')
plt.clf()

# Plot the Precision-Recall Curve
f = plt.figure()
plt.title('PR Curve: ' + SAVE, fontsize=18)
count2 = 0
for i in items:
  #c = colors(count2/float(n_lines))
  #c = colors[0:n_lines]
  plt.plot(TPR_all[i][0], Prec_all[i][0], lw=3, color= colors[count2], alpha=1, label = i)
  plt.fill_between(TPR_all[i][0], Prec_all[i][0]-Prec_all[i][1], Prec_all[i][0]+Prec_all[i][1], facecolor=colors[count2], alpha=0.3, linewidth = 0)
  count2 += 1
plt.plot([-0.03,1.03],[0.5,0.5],'r--', lw=2)
plt.legend(loc='upper right')
plt.xlim([-0.03,1.03])
plt.ylim([0.485,1.015])
plt.axes().set_aspect(2)
plt.ylabel('Precision', fontsize=16)
plt.xlabel('Recall', fontsize=16)
plt.show()

filename = SAVE + "_PRcurve.png"
plt.savefig(filename)
filename_pdf = SAVE + "_PRcurve.pdf"
f.savefig(filename_pdf, bbox_inches='tight')
plt.close()

print("Done!")
