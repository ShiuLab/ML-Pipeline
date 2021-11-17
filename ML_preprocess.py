import sys, os, argparse
import pandas as pd
import numpy as np
from scipy import stats


###### Parse input parameters #######

parser = argparse.ArgumentParser(
	description='Code to:\n'+\
				' [1] remove/impute NAs,\n'+\
				' [2] t/f one-hot-encode categorical features,\n'+\
				' [3] t/f remove duplicate rows,\n'+\
				' [4] keep/drop columns.',
	epilog='https://github.com/ShiuLab/ML_Pipeline/')

# Info about input data
parser.add_argument(
	'-df', 
	help='Feature & class dataframe. Must be specified',
	required=True)
parser.add_argument(
	'-y_name', 
	help='Name of lable column in dataframe, default=Class',
	default='Class')
parser.add_argument(
	'-sep',
	help='Deliminator, default="\t"',
	default='\t')

# Imputation parameters
parser.add_argument(
	'-na_method', 
	help='Mode for inputation (options: drop, mean, median, mode). Will '+\
		'default to mode if feature is categorical (i.e. a string), '+\
		'otherwise default=median',
	default='median')
parser.add_argument( #### Shiu: this is ambiguous, should be -drop_proportion
	'-drop_percent', 
	help='If > drop_percent of data is missing, feature will be dropped '+\
		'instead of imputed, default=0.5', 
	default=0.5)

# One-Hot-Encoding Parameters
parser.add_argument(
	'-onehot', 
	help='t/f. If onehot encoding should be done if a column contains '+\
		'strings, default = t', 
	default='t')
parser.add_argument(
	'-onehot_list', 
	help='list of columns to be one-hot-encoded (will default to default to '+\
		'any column of type object - i.e. strings)',
	default='default')

# Other parameters
parser.add_argument(
	'-remove_dups', 
	help='t/f. Removes rows with duplicate row names (1st column value),' +\
		'default=t',
	default='t')
parser.add_argument(
	'-keep', 
	help='List of column names to keep, drop the rest (except index and '+\
		'y_name) - note this can be done in ML_classification/ML_regression, '+\
		'default="na"', 
	default='na')
parser.add_argument(
	'-drop', 
	help='List of column names to drop, default="na"', 
	default='na')

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit(0)
args = parser.parse_args()

###### Read in data #######

df = pd.read_csv(args.df, sep=args.sep, index_col=0)

df = df.replace(['?', 'NA', 'na', 'n/a', '', '.'], np.nan)

print('Snapshot of input data...')
print(df.iloc[:5, :5])

# Shiu: Catch situaton when label column name is incorrectly specified
try:
	df_classes = df[args.y_name]
except KeyError:
	print("\nERR: y_name is specified as %s: does not exist\n" % args.y_name)
	sys.exit(0)
	
df = df.drop(args.y_name, 1)

###### Remove NAs with too much data missing or if na_method = 0 #######

print('\n\n### Dropping/imputing NAs... ###')
cols_with_na = df.columns[df.isna().any()].tolist()
print('\nNumber of columns with NAs: %i' % len(cols_with_na))

# Shiu: Fix two issues,
#  1) drop_percent is misleading and people can be giving percent number
#  2) If user does provide a drop_percent, it is not properly converted to
#     a floating point number and a TypeError will be thrown.
args.drop_percent = float(args.drop_percent)
if args.drop_percent > 1 or args.drop_percent < 0:
	print('\nERR: drop_percent is between 0 and 1, but %f is specified\n' %\
		args.drop_percent)
	sys.exit(0)

dropped = []
if len(cols_with_na) > 0:
	if args.na_method == 'drop':
		df = df.drop(cols_with_na, 1)
	else:
		for col in cols_with_na:
			missing = df.loc[:, col].isnull().sum()
			miss_pct = missing / len(df)
			if miss_pct > args.drop_percent:
				dropped.append(col)

if len(dropped) > 0:
	print('\nFeatures dropped because missing > %.2f%% of data: %s' % \
		(args.drop_percent * 100, dropped))
	df.drop(dropped, 1, inplace=True)

cols_to_impute = [x for x in cols_with_na if x not in dropped]
print('Number of columns to impute: %i' % len(cols_to_impute))

###### Impute remaining NAs ####### 

if len(cols_to_impute) > 0 and args.na_method != 'drop':
	for col in cols_to_impute:
		col_type = df[col].dtypes

		if col_type == 'object':
			df[col].fillna(df[col].mode()[0], inplace=True)

		elif args.na_method == 'mean':
			df[col].fillna(df[col].mean(), inplace=True)

		elif args.na_method == 'median':
			df[col].fillna(df[col].median(), inplace=True)

		else:
			print('Need to specify method for imputation')
			quit()

###### One-Hot-Encode any categorical features ####### 

if args.onehot.lower() == 't':
	print('\n\n### One Hot Encoding... ###')
	if args.onehot_list == 'default':
		cols_cat = list(df.select_dtypes(include=['object']).columns)
	else:
		with open(args.onehot_list) as f:
			cols_cat = f.read().splitlines()

	print('\nFeatures to one-hot-encode: %s' % cols_cat)
	start_shape = df.shape

	for col in cols_cat:
		df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
		df.drop([col], axis=1, inplace=True)

	end_shape = df.shape
	print('Dataframe shape (rows, cols) before and after one-hot-encoding:\n'+\
		'Before: %s\nAfter: %s' % (start_shape, end_shape))

###### Remove duplicate rows #######

if args.remove_dups.lower() in ['t', 'true']:
	dups_count = df.index.size - df.index.nunique()
	print('\nNumber of duplicate row names to delete: %i' % dups_count)

	df = df[~df.index.duplicated(keep='first')]
	df_classes = df_classes[~df_classes.index.duplicated(keep='first')]
###### Keep/Drop given columns #######

if args.keep.lower() != 'na':
	print('Using subset of features from: %s' % args.keep)
	with open(args.keep) as f:
		f_keep = f.read().strip().splitlines()
		f_keep = [args.y_name] + f_keep
	df = df.loc[:, f_keep]

if args.drop.lower() != 'na':
	print('Dropping features from: %s' % args.drop)
	with open(args.drop) as f:
		f_drop = f.read().strip().splitlines()
	df = df.drop(f_drop, axis=1)

###### Add class column back in and save ######

df = pd.concat([df_classes, df], axis=1)
print('\nSnapshot of imputed data...')
print(df.iloc[:5, :5])

save_name = args.df.replace('.txt','') + '_mod.txt'
df.to_csv(save_name, sep=args.sep, header=True)

print('\nOutput file saved as: %s' % save_name)
print('\nDone!')
