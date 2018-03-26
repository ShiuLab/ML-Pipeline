# ML-Pipeline
Scripts for Shiu Lab Machine Learning

## Environment Requirements
* biopython                 1.68
* matplotlib                1.5.1
* numpy                     1.11.3
* pandas                    0.18.1
* python                    3.4.4
* scikit-learn              0.18.1
* scipy                     0.18.1

Example: 

    wget http://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda install biopython
    conda install matplotlib
    conda install pandas
    conda install scikit-learn
    
MSU HPCC: export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
Calculon2: 

## Data Preprocessing

### Feature Selection (Azodi)
Available feature selection tools: RandomForest, Chi2, LASSO (L1 penalty), enrichement (Fisher's Exact test - binary features only), and BayesA (regression only). For parameters needed for each feature selection algorithm run Feature_Selection.py with no parameters.

Example:

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python Feature_Selection.py -df [path/to/dataframe] -f [rf/chi2/lasso/fet/bayesA] -n [needed for chi2/rf] -p [needed for LASSO/FET] -type [needed for LASSO] -list T 
    * For more info/additional options run Feature_Selection.py with no parameters

### Impute Data (Moore)

This script imputes msising data in a given matrix. Matrices must be separate if data types are different (ie. numeric separate from binary). This script takes at least one matrix, you only need one if you have only one type of data, but if you have multiple types you need to input them separate.

Inputs:
Data matrix- columns should be gene|Class|feature1|feature2| (-df*)

Data types (-dtype*):
1. numeric: n
2. categorical: c
3. binary: b

Available imputation methods (-mv):
1. impute data from a random selection from your data distribution (-mv 1)
2. impute data by using the median for numeric data and the mode for categorical/binary data (-mv 2)
3. drop all rows with NAs (-mv 0)

Example:
    python impute_data.py -df1 [path/to/dataframe] -dtype1 [n,c,b] -mv [0,1,2] -df2 [path/to/dataframe2] -dtype2 [n,c,b] -df3 [path/to/dataframe3] -dtype3 [n,c,b]

### Convert categorical data to binary (Moore)

This script converts a categorical matrix to a binary matrix to run on machine-learning algorithms

    python get_cat_as_binmatrix.py [categorical_matrix]

output: [categorical_matrix]_binary.matrix.txt

## Building Models

### Classification
See ML_classification.py docstrings for additional options (and ML_clf_functions.py)

Available model algorithms: RF, Gradent Boosting (GB), SVM, SVMpoly, SVMrbf, Logistic Regression (LogReg)

Example binary classification:

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python ML_classification.py -df example_Bin.txt -alg [ALG]

Example multiclass classification:

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python ML_classification.py -df example_MC.txt -alg [ALG] -class Biotech_cluster -cl_train a,b,c -cm T

* For more info/additional options run ML_classification.py with no parameters

### Regression
See ML_regression.py docstrings for additional options (and ML_clf_functions.py)

Available model algorithms: RF, SVM, SVMpoly, SVMrbf, Gradient Boosting (GB), Logistic Regression (LogReg - no grid search needed)

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python ML_regression.py -df data.txt -alg [ALG]

* For more info/additional options run ML_regression.py with no parameters

## Post-Processing

### AUC-ROC & AUC-PR Plots
Use this code to build plots with multiple classification _scores files.

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python ML_plots.py [SAVE_NAME] name1 [Path_to_1st_scores_file] name3 [Path_to_2nd_scores_file] etc.

### Compare classifiers (Venn-Diagrams)
Given a set of *_scores.txt results files, output a list of which instances were classified correctly and which incorrectly and summarize with a table of overlaps.
Example:

    export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
    python compare_classifiers.py -scores [comma sep list of scores files] -ids [comma sep list of classifier names] -save [out_name] 



## TO DO LIST
- Merge pre-processing scripts so you can deal with NAs, imputations, and one-hot encoding categorical variables in one script.
- Add additional classification models: Naive Bayes, basic neural network (1-2 layers)
- Add validation set hold out option

