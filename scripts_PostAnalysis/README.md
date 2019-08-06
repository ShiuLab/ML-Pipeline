# Post processing functions available

- **ML_plots.py:** Generate AUC-PR and AUC-ROC curves for multiple ML runs in the same figure. Plots mean score over the balanced runs with stdev error bars.
- **compare_classifiers.py:** Given a set of _scores.txt results files, output a list of which instances were classified correctly and which incorrectly and summarize with a table of overlaps
- **get_average_imp_rank.py:** Get mean importance rank among many imp files
- **get_scaled_imp_binary.py:** This script parses importance file (-imp) and input matrix (-df) to get scaled directional importance. Since imp files are only obtained from binary models, this script is set up to use a positive and negative class only
- **plot_predprob.R:** This script makes histogram plots of scores for binary models. The input file is the score file from your ML run.
- **venn.py:** Used by the compare_classifiers.py script to make a venn diagram of instances that were correctly classified by different models. 


## Details


### ML_plots.py 

```python ML_plots.py [SAVE_NAME] name1 [Path_to_1st_scores_file] name3 [Path_to_2nd_scores_file] etc.```

### compare_classifiers.py (Venn-Diagrams)

Given a set of *_scores.txt results files, output a list of which instances were classified correctly and which incorrectly and summarize with a table of overlaps.
  
```python compare_classifiers.py -scores [comma sep list of scores files] -ids [comma sep list of classifier names] -save [out_name] ```
