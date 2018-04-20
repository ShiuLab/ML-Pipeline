"""
PURPOSE:

Define a validation set to hold out during feature selection/model training.
For regression models will hold out a ranomd X percent.
For classification models, will hold out X percent of each class


INPUTS:
	
	REQUIRED:
	-df       Feature & class dataframe for ML
	-type     c/r (classification vs. regression)
	-p 		  Percent of values to hold out (0.1 = 10%)
	
	OPTIONAL:
	-y_name   Name of the column to predict (Default = Class)
	-apply	  A list of what classes you want to include in the hold out (Default = all)
	-skip     A list of what classes you don't want in the hold out (i.e. unknown classes) (Default = none)
	-sep      Set seperator for input data (Default = '\t')
	-drop_na  T/F to drop rows with NAs
	-save     Adjust save name prefix. Default = [df]_holdout.
	-df2      File with class information. Use only if df contains the features but not the Y values 
							* Need to specifiy what column in df2 is y using -y_name 

OUTPUT:
	-df_holdout  List of instances to hold out. 
				 Use for input into feature selection and ML_classification/ML_regression (-ho)


"""
import sys, os
import pandas as pd

DF2, SAVE, SEP, y_name = 'None','', '\t', 'Class'
apply_ho, drop_na, SKIP =  'all', 'f', 'None'

for i in range (1,len(sys.argv),2):
	if sys.argv[i].lower() == "-df":
		DF = sys.argv[i+1]
	elif sys.argv[i].lower() == "-df2":
		DF2 = sys.argv[i+1]
	elif sys.argv[i].lower() == "-y_name":
		y_name = sys.argv[i+1]
	elif sys.argv[i].lower() == "-sep":
		SEP = sys.argv[i+1]
	elif sys.argv[i].lower() == '-save':
		SAVE = sys.argv[i+1]
	elif sys.argv[i].lower() == '-skip':
		SKIP = sys.argv[i+1]
		if SKIP != "None":
			SKIP = sys.argv[i+1].split(',')
	elif sys.argv[i].lower() == "-apply":
		apply_ho = sys.argv[i+1]
	elif sys.argv[i].lower() == "-drop_na":
		drop_na = sys.argv[i+1]
	elif sys.argv[i].lower() == "-type":
		model_type = sys.argv[i+1]
	elif sys.argv[i].lower() == "-p":
		p = sys.argv[i+1]

if len(sys.argv) <= 1:
	print(__doc__)
	exit()


df = pd.read_csv(DF, sep=SEP, index_col = 0)

# If features  and class info are in separate files, merge them: 
if DF2 != 'None':
	start_dim = df.shape
	df_class = pd.read_csv(DF2, sep=SEP, index_col = 0)
	df = pd.concat([df_class[y_name], df], axis=1, join='inner')
	print('Merging the feature & class dataframes changed the dimensions from %s to %s (instance, features).' 
		% (str(start_dim), str(df.shape)))

# Specify Y column - default = Class
if model_type.lower() == 'c' or model_type.lower() == 'classificaton':
	if y_name != 'Class':
		df = df.rename(columns = {y_name:'Class'})
elif model_type.lower() == 'r' or model_type.lower() == 'regression':
	if y_name != 'Y':
		df = df.rename(columns = {y_name:'Y'})
else:
	print('Model type not recognized, define as c or r')
	exit()

if SKIP != 'None':
	try:
		df = df[~(df['Class'].isin(SKIP))]
	except:
		df = df[~(df['Y'].isin(SKIP))]

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

# Set Save Name
if SAVE == "":
	save_name = DF + "_holdout.txt"
else:
	save_name = SAVE



per_ho = float(p)
print('Holding out %.1f percent' % (per_ho*100))

# Define hold out for regression
holdout = []
if model_type.lower() == 'c' or model_type.lower() == 'classificaton':
	if apply_ho == 'all':
		classes = df.Class.unique()
		print('Pulling holdout set from classes: %s' % str(classes))
		for cl in classes:
			temp = df[(df['Class']==cl)] 
			temp_sample = temp.sample(frac = per_ho)
			keep_ho = list(temp_sample.index)
			holdout.extend(keep_ho)
			print(holdout)
	else:
		apply_to = apply_ho.strip().split(',')
		for cl in apply_to:
			temp = df[(df['Class']==cl)] 
			temp_sample = temp.sample(frac = per_ho)
			keep_ho = list(temp_sample.index)
			holdout.extend(keep_ho)
			print(holdout)


elif model_type.lower() == 'r' or model_type.lower() == 'regression':
			temp_sample = df.sample(frac = per_ho)
			keep_ho = list(temp_sample.index)
			holdout.extend(keep_ho)

else:
	print('Model type not recognized, define as c or r')
	exit()	

print('%i instances in holdout' % len(holdout))

out = open(save_name, 'w')
for ho in holdout:
	out.write('%s\n' % ho)

print('finished!')