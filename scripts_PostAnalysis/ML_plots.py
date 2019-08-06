import sys, os, argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from cycler import cycler
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

###### Parse input parameters #######

parser = argparse.ArgumentParser(
	description='Code to make ROC and PR plots from multiple ML runs (need _scores.txt files). \
	Plots mean score over the balanced runs with stdev error bars.',
	epilog='https://github.com/ShiuLab/ML_Pipeline/')

# Info about input data
parser.add_argument('-save', help='Output save name', default='plots')
parser.add_argument('-cl_train', help='Positive and negative class label (pos,neg), where pos is first in -cl_train', default=['1','0'], nargs='+', type=str)
parser.add_argument('-scores', help='List of scores files (e.g. -scores test_SVM_scores.txt, test_RF_scores.txt)', required=True, nargs='+', type=str)
parser.add_argument('-names', help='List of names to assign to each score file (e.g. -names SVM, RF)', required=True, nargs='+', type=str)

if len(sys.argv)==1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

POS, NEG = args.cl_train[0], args.cl_train[1]
items = OrderedDict()

# Organize all _scores and _BalancedIDs files into dictionary
for i in range(len(args.names)):
  items[args.names[i]] = [args.scores[i], args.scores[i].replace('_scores.txt', '_BalancedIDs')]


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
  balanced_ids = [x.strip().split('\t') for x in balanced_ids]

  FPRs = {}
  TPRs = {}
  precisions = {}
  
  # For each balanced run
  for k in range(0, n): 
    print("Processing model %s, replicate %i" % (i,k))
    FPR = []
    TPR = []
    precis = []
    name = 'score_' + str(k)
    y = df_proba.ix[balanced_ids[k], 'Class'] 
    y = y.astype(str)

    # Get decision matrix & scores at each threshold between 0 & 1
    for j in np.arange(0, 1, 0.01):
      yhat = df_proba.ix[balanced_ids[k], name].copy()
      yhat[df_proba[name] >= float(j)] = POS
      yhat[df_proba[name] < float(j)] = NEG
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
plt.title('ROC Curve: ' + args.save, fontsize=18)
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

filename = args.save + "_ROCcurve.png"
plt.savefig(filename)
filename_pdf = args.save + "_ROCcurve.pdf"
f.savefig(filename_pdf, bbox_inches='tight')
plt.clf()

# Plot the Precision-Recall Curve
f = plt.figure()
plt.title('PR Curve: ' + args.save, fontsize=18)
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

filename = args.save + "_PRcurve.png"
plt.savefig(filename)
filename_pdf = args.save + "_PRcurve.pdf"
f.savefig(filename_pdf, bbox_inches='tight')
plt.close()

print("Done!")
