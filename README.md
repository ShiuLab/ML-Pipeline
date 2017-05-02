# ML-Pipeline
Scripts for Shiu Lab Machine Learning


## Data Preprocessing

### Feature Selection
Available feature selection tools: DecisionTree (uses RandomForest), Chi2.
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python Feature_Selection_sklearn.py -df [path/to/dataframe] -f [Chi2/DecisionTree] -n [Int or Fraction]</code></pre>

### Classification
See ML_classification.py (and ML_clf_functions.py)

Example binary classification:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_classification.py -df example_Bin.txt -alg [RF/SVC] -gs T </code></pre>

Example multiclass classification:
<pre><code>export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
python ML_classification.py -df example_MC.txt -alg [RF/SVC] -class Biotech_cluster -cl_train a,b,c -gs T -cm T</code></pre>

*Note: To run tests use -gs T -gs_n 3 -n 5 



## TO DO LIST

- Make pipeline for regression
- Add to preprocessing section to allow for imputing NAs, dropping NAs, normalizing/scaling features, etc
- Apply model to "unknowns" to make predictions
- Add additional classification models: SVC with kernel, Naive Bayes, basic neural network (1-2 layers)
- Output for AUC-ROC and Precision-Recall figures... (Either figures directly or data)
