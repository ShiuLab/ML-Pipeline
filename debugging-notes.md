# 09/28/2023: Create conda environment and debug the regression pipeline with dummy dataset
This repository is exactly the same as the one published on GitHub. The version I have in ~/Shiu_Lab/Project/External_software/ML-Pipeline/ has been modified for scikit-learn version 0.24.2.

The goal for today is to ensure that the pipeline is working for users who clone the current repository.

## __Create the conda environment:__
conda create -n ml-pipeline python=3.6.4
conda activate ml-pipeline
conda update python # decided we should have the most recent python version (now v3.11.5)
conda install -c conda-forge matplotlib biopython numpy pandas scikit-learn scipy
conda list -e > requirements.txt

## __Debugging the Random Forest regression pipeline:__
The dummy dataset is in the folder `test_data/`. It comes from the Peter et al. (2018) Nature paper's biallelic SNP dataset that I cleaned for my yeast fitness prediction project.

The commands I ran are in `test_rf_pipeline.sh`, where I run the RF regression algorithm using a full gridsearch and a randomized gridsearch.
```bash
# Run full grid search
python ../ML_regression_09282023.py \
    -df features.csv \
    -df2 labels.csv \
    -test test_instances.txt \
    -sep , -y_name YPACETATE \
    -alg RF -n 2 -cv_num 2 \
    -save test_rf_pipeline \
    -gs_type full -gs_reps 2 \
    -tag YPACETATE -plots t

# Rum randomized grid search
python ../ML_regression_09282023.py \
    -df features.csv \
    -df2 labels.csv \
    -test test_instances.txt \
    -sep , -y_name YPACETATE \
    -alg RF -n 2 -cv_num 2 \
    -save test_rf_pipeline \
    -gs_type random -gs_reps 2 \
    -tag YPACETATE -plots t
```

## __Errors and Solutions__:<br>
### For full grid search:<br>
  __Error__:
  ```bash 
  Traceback (most recent call last):
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 494, in <module>
      main()
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 262, in main
      params2use, param_names = RF.fun.RegGridSearch(df, args.save, \
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/RF_model.py", line 187, in RegGridSearch
      gs_results_mean = gs_results_mean.sort_values('mean_test_score', \
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TypeError: DataFrame.sort_values() takes 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given
  ```
  __Solution__: add `\` to the ends of lines where the arguments do not all fit in one line (ML_regression_09282023.py, line 262; RF_model.py, line 187) and also specify the `by=` and `axis=` arguments to `.sort_values()` on line 187.

  __Error__:
  ```bash
  Traceback (most recent call last):
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 494, in <module>
      main()
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 332, in main
      reg, result, cv_pred, importance, result_test = RF.fun.Run_Regression_Model(
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/RF_model.py", line 240, in Run_Regression_Model
      cv_pred = cross_val_predict(estimator=reg, X=X, y=y, cv=cv_num)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 986, in cross_val_predict
      predictions = parallel(
                    ^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 63, in __call__
      return super().__call__(iterable_with_config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/joblib/parallel.py", line 1863, in __call__
      return output if self.return_generator else list(output)
                                                  ^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/joblib/parallel.py", line 1792, in _get_sequential_output
      res = func(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 123, in __call__
      return self.function(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 1068, in _fit_and_predict
      estimator.fit(X_train, y_train, **fit_params)
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/ensemble/_forest.py", line 340, in fit
      self._validate_params()
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/base.py", line 600, in _validate_params
      validate_parameter_constraints(
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/_param_validation.py", line 97, in validate_parameter_constraints
      raise InvalidParameterError(
  sklearn.utils._param_validation.InvalidParameterError: The 'criterion' parameter of RandomForestRegressor must be a str among {'friedman_mse', 'absolute_error', 'poisson', 'squared_error'}. Got 'mse' instead.
  ```
  __Solution__: Changed line 217 in RF_model.py, from `'mse'` to `'friedman_mse'`.

  __Error__:
  ```bash
  Traceback (most recent call last):
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 494, in <module>
      main()
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 332, in main
      reg, result, cv_pred, importance, result_test = RF.fun.Run_Regression_Model(
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/RF_model.py", line 240, in Run_Regression_Model
      cv_pred = cross_val_predict(estimator=reg, X=X, y=y, cv=cv_num)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 986, in cross_val_predict
      predictions = parallel(
                    ^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 63, in __call__
      return super().__call__(iterable_with_config)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/joblib/parallel.py", line 1863, in __call__
      return output if self.return_generator else list(output)
                                                  ^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/joblib/parallel.py", line 1792, in _get_sequential_output
      res = func(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/parallel.py", line 123, in __call__
      return self.function(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/model_selection/_validation.py", line 1068, in _fit_and_predict
      estimator.fit(X_train, y_train, **fit_params)
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/ensemble/_forest.py", line 340, in fit
      self._validate_params()
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/base.py", line 600, in _validate_params
      validate_parameter_constraints(
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/sklearn/utils/_param_validation.py", line 97, in validate_parameter_constraints
      raise InvalidParameterError(
  sklearn.utils._param_validation.InvalidParameterError: The 'max_depth' parameter of RandomForestRegressor must be an int in the range [1, inf) or None. Got 10.0 instead.
  ```
  __Solution__: Changed line 215 in RF_model.py from `max_depth=max_depth,` to `max_depth=int(max_depth),`.

  __Error__:
  ```bash
  Traceback (most recent call last):
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 494, in <module>
      main()
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/test_data/../ML_regression_09282023.py", line 332, in main
      reg, result, cv_pred, importance, result_test = RF.fun.Run_Regression_Model(
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/mnt/ufs18/home-056/seguraab/ML-Pipeline/RF_model.py", line 267, in Run_Regression_Model
      cv_pred_df = cv_pred_df.append(test_pred_df)
                  ^^^^^^^^^^^^^^^^^
    File "/mnt/home/seguraab/miniconda3/envs/ml-pipeline/lib/python3.11/site-packages/pandas/core/generic.py", line 5989, in __getattr__
      return object.__getattribute__(self, name)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  AttributeError: 'DataFrame' object has no attribute 'append'. Did you mean: '_append'?
  ```
  __Solution__: Changed line 260 and 267 in RF_model.py from `cv_pred_df = cv_pred_df.append(unk_pred_df)` to `cv_pred_df = pd.concat([cv_pred_df, unk_pred_df])` and `cv_pred_df = cv_pred_df.append(test_pred_df)` to `cv_pred_df = pd.concat([cv_pred_df, test_pred_df])`, respectively. `.append()` is a deprecated pandas method since version 1.4.0.

  The pipeline is now fully working on the dummy dataset.

## __Adding the changes to a new branch in the Shiu Lab GitHub__:
```bash
git status
# On branch master
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#
#       ../ML_regression_09282023.py
#       ../RF_model.py
#       ../debugging-notes.md
#       ../requirements.txt
#       ./
# nothing added to commit but untracked files present (use "git add" to track)
git checkout -b kenia-dev-update-rf
# On branch kenia-dev-update-rf
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#
#       ../ML_regression_09282023.py
#       ../RF_model.py
#       ../debugging-notes.md
#       ../requirements.txt
#       ./
# nothing added to commit but untracked files present (use "git add" to track)
git add ../ML_regression_09282023.py
git add ../RF_model.py
git commit -m "Update RF pipeline package versions"
git commit --amend -v # Added more details
git add ../debugging-notes.md
git add ../requirements.txt
git commit -v # Added more descriptions
git push --set-upstream origin kenia-dev-update-rf # publish branch
```
