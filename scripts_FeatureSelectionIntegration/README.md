This script is an extension of ML_classification.py. The key difference in
goal is that this script will manually cross-validate the training instances
so that feature selection and parameter grid searches can be performed under
cross-validation. In addition, performing multiple interations of cross-
validation splits is supported (default = 30 iterations).
Compared to ML_classification.py, this script does not currently support 
multi-class classification and has stricter requirements on the input data
frame.
