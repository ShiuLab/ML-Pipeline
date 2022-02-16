import sys, os, argparse
import pandas as pd


########################
### Parse Input Args ###
########################

parser = argparse.ArgumentParser(
	description='Define a test set that will be held out during feature selection \
	and model training. For regression models (-type r), test will be a random X percent or n.\
	For classification models (-type c), test will be X percent or number from each class',
	epilog='https://github.com/ShiuLab')

### Input arguments ###
# Required
req_group = parser.add_argument_group(title='REQUIRED INPUT')
req_group.add_argument('-df', help='Feature & class dataframe for ML, (example: example_binary.txt) ', required=True)
req_group.add_argument('-type', help='c/r (classification vs. regression)', required=True)
req_group.add_argument('-p', '-percent', help='Percent of instances to hold out (0.1 = 10%), can also use -n', required=False, type=float, default=0)
req_group.add_argument('-n', '-num', help='Number of instances to hold out, can also use -p', required=False, type=int, default=0)

# Optional
inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
inp_group.add_argument('-y_name', help='Name of column to predict', default='Class')
inp_group.add_argument('-df2', help='Class data (if not in -df). Need to provide -a.y_name', default='')
inp_group.add_argument('-sep', help='Deliminator', default='\t')
inp_group.add_argument('-use', help='List of classes to include in test set', default='all')
inp_group.add_argument('-skip', help='List of classes to not include in test set (i.e. unknown)', default='')
inp_group.add_argument('-drop_na', help='T/F to drop rows with NAs', default='f')
inp_group.add_argument('-save', help='Adjust save name prefix. Default = [df]_test.', default='default')

a = parser.parse_args()
if a.skip != "":
	skip = a.skip.split(',')
if a.save == 'default':
	a.save = a.df + "_test.txt"


#########################
### Read in dataframe ###
#########################

df = pd.read_csv(a.df, sep=a.sep, index_col = 0)

# If features  and class info are in separate files, merge them: 
if a.df2 != '':
	start_dim = a.df.shape
	df_class = pd.read_csv(a.df2, sep=a.sep, index_col = 0)
	df = pd.concat([df_class[a.y_name], df], axis=1, join='inner')
	print('Merging the feature & class dataframes changed the dimensions from %s to %s (instance, features).' 
		% (str(start_dim), str(df.shape)))

# Specify Y column - default = Class
if a.type.lower() == 'c' or a.type.lower() == 'classificaton':
	if a.y_name != 'Class':
		df = df.rename(columns = {a.y_name:'Class'})
elif a.type.lower() == 'r' or a.type.lower() == 'regression':
	if a.y_name != 'Y':
		df = df.rename(columns = {a.y_name:'Y'})
else:
	print('Model type not recognized, define as classification (c) or rregression (r)')
	exit()

if a.skip != '':
	try:
		df = df[~(df['Class'].isin(a.skip))]
	except:
		df = df[~(df['Y'].isin(a.skip))]

# Check for Nas
if df.isnull().values.any() == True:
	if a.drop_na.lower() == 't' or a.drop_na.lower() == 'true':
		start_dim = df.shape
		df = df.dropna(axis=0)
		print('Dropping rows with NA values changed the dimensions from %s to %s.' 
			% (str(start_dim), str(df.shape)))
	else:
		print(df.columns[df.isnull().any()].tolist())
		print('There are Na values in your dataframe.\n Impute them or add -drop_na True to remove rows with nas')
		quit()

if a.p != 0.0:
	print('Holding out %.1f percent' % (a.p*100))
elif a.n != 0:
	print('Holding out %i instances per class' % (a.n))
else:
	print('Either -p or -n is required!')
	quit()



#######################
### Define test set ###
#######################

def pull_sample(temp, p, n):
	if p != 0.0:
		temp_sample = temp.sample(frac = p)
	elif n != 0:
		temp_sample = temp.sample(n = n)
	return temp_sample

test = []

if a.type.lower() == 'c' or a.type.lower() == 'classificaton':
	if a.use == 'all':
		use_list = df.Class.unique()
	else:
		use_list = a.use.strip().split(',')

	print('Pulling test set from classes: %s' % str(use_list))
	min_size = (df.groupby('Class').size()).min() - 1
	for cl in use_list:
		temp = df[df['Class']==cl].sample(min_size, random_state=42) 
		temp_sample = pull_sample(temp, a.p, a.n)
		keep_test = list(temp_sample.index)
		test.extend(keep_test)

elif a.type.lower() == 'r' or a.type.lower() == 'regression':
	if a.p != 0:
		temp_sample = df.sample(frac = a.p)
	elif a.n != 0:
		temp_sample = df.sample(n = a.n)
	
	keep_test = list(temp_sample.index)
	test.extend(keep_test)

else:
	print('Model type not recognized, define as c or r')
	exit()	

print('%i instances in test set' % len(test))

out = open(a.save, 'w')
for ho in test:
	out.write('%s\n' % ho)
out.close()
print('finished!')

