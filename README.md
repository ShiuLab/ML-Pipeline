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

### Feature Selection
Available feature selection tools: DecisionTree (uses RandomForest), Chi2.
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python Feature_Selection_sklearn.py -df [path/to/dataframe] -f [Chi2/DecisionTree] -n [Int or Fraction]</code></pre>

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

## TO DO LIST

- Make pipeline for regression
- Add to preprocessing section to allow for imputing NAs, dropping NAs, normalizing/scaling features, etc
- Add additional classification models: Naive Bayes, basic neural network (1-2 layers)
- Look into perfect TPR/FPR/FNR/FP/FN scores in RESULTS output file
- Categorical data is throwing errors, as scikit is trying treat it as numeric data
- Random forest imporance scores are all identical
- Add validation set hold out option
