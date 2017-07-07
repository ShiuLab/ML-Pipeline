
#import modules
import sys
import os

def print_help():
	print'''
inp1  = input data frame
inp2  = algorithm: RF, SVM, SVMpoly, or SVMrbf
inp3  = working directory: use $PWD for current directory
inp4  = processors: recommended, 14
inp5  = shell script name
inp6+ = any additional arguments for running ML_classification.py


ADDITIONAL arguments for ML_classification:
	-p        # of processors. Default = 1
	-cl_train List of classes to include in the training set. Default = all classes. If binary, first label = positive class.
	-pos      name of positive class (default = 1 or first class provided with -cl_train)
	-gs       Set to True if parameter sweep is desired. Default = False
	-cv       # of cross-validation folds. Default = 10
	-b        # of random balanced datasets to run. Default = 100
	-apply    To which non-training class labels should the models be applied? Enter 'all' or a list (comma-delimit if >1)
	-save     Save name. Default = [df]_[alg] (caution - will overwrite!)
	-class    String for column with class. Default = Class
	-feat     Import file with list of features to keep if desired. Default: keep all.
	-tag      String for the TAG column in the RESULTS.txt output.
	
	PLOT OPTIONS:
	-cm       T/F - Do you want to output the confusion matrix & confusion matrix figure? (Default = False)
	-plots    T/F - Do you want to output ROC and PR curve plots for each model? (Default = False)

'''

def base_shell(p,wd,inp_fl,alg):
	base_shell_cmd = '''#!/bin/bash -login
#PBS -q main
#PBS -l nodes=1:ppn=%s,walltime=3:59:00,mem=20gb
#PBS -d %s
export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python /mnt/home/lloydjo1/GitHub/ML-Pipeline/ML_classification.py -p %s -df %s -alg %s'''%(p,wd,p,inp_fl,alg)
	return base_shell_cmd

def add_additional(base_shell_cmd,addtnl):
	if addtnl != "":
		addtnl_str = " "+" ".join(addtnl)
		shell_cmd = base_shell_cmd+addtnl_str
		return shell_cmd
	else:
		return base_shell_cmd

def write_and_submit(shell_nm,shell_cmd):
	out= open(shell_nm,"w")
	out.write(shell_cmd)
	out.close()
	
	os.system("qsub %s"%(shell_nm))

def main():
	if len(sys.argv) == 1 or "-h" in sys.argv:
		print_help()
		sys.exit()
	
	try:
		inp_fl = sys.argv[1]
		alg = sys.argv[2]
		wd = sys.argv[3]
		p = sys.argv[4]
		shell_nm = sys.argv[5]
		if shell_nm.endswith(".sh"):
			pass
		else:
			shell_nm = shell_nm+".sh"
		if len(sys.argv) > 6:
			addtnl = sys.argv[6:]
		else:
			addtnl = ""
	except:
		print_help()
		print "Error reading arguments, quitting!"
		sys.exit()
	
	base_shell_cmd = base_shell(p,wd,inp_fl,alg)
	shell_cmd = add_additional(base_shell_cmd,addtnl)
	print
	print "GENERATED SHELL SCRIPT:"
	print
	print shell_cmd
	print
	write_and_submit(shell_nm,shell_cmd)

if __name__ == "__main__":
	main()
