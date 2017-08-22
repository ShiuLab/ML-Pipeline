# ML-Pipeline
Scripts for Shiu Lab Machine Learning

## Environment Requirements
biopython                 1.68
matplotlib                1.5.1
numpy                     1.11.3
pandas                    0.18.1
python                    3.4.4
scikit-learn              0.18.1
scipy                     0.18.1

## Data Preprocessing

### Feature Selection (Azodi)
Available feature selection tools: RandomForest, Chi2, LASSO (L1 penalty), enrichement (Fisher's Exact test).

Example:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python Feature_Selection_sklearn.py -df [path/to/dataframe] -f [rf/chi2/lasso/fet] -n [needed for chi2/rf] -p [needed for LASSO/FET] -type [needed for LASSO] -list T </code></pre>
Use -list T/F to either just save a list of selected features (can use as -feat input during model building) or a filtered data frame

### Impute Data (Moore)
Available imputation methods:

Example:
<pre><code>export python impute_data.py -df [path/to/dataframe] -dtype [] -mv [] </code></pre>


## Building Models

### Classification
See ML_classification.py docstrings for additional options (and ML_clf_functions.py)

Available model algorithms: RF, SVM, SVMpoly, SVMrbf

Example binary classification:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_classification.py -df example_Bin.txt -alg [ALG] </code></pre>

Example multiclass classification:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_classification.py -df example_MC.txt -alg [ALG] -class Biotech_cluster -cl_train a,b,c -cm T</code></pre>

*Note: To run tests use -gs T -gs_n 3 -n 5 

## Post-Processing

### AUC-ROC & AUC-PR Plots
Use this code to build plots with multiple results files.

<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_plots.py [SAVE_NAME] name1 [Path_to_1st_scores_file] name3 [Path_to_2nd_scores_file] etc.</code></pre>

### Compare classifiers (Venn-Diagrams)
Given a set of *_scores.txt results files, output a list of which instances were classified correctly and which incorrectly and summarize with a table of overlaps.
Example:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python compare_classifiers.py -scores [comma sep list of scores files] -ids [comma sep list of classifier names] -save [out_name]</code></pre>



## TO DO LIST

- Add script for regression models
- Merge pre-processing scripts so you can deal with NAs, imputations, and one-hot encoding categorical variables in one script.
- Add additional classification models: Naive Bayes, basic neural network (1-2 layers)
- Add validation set hold out option

