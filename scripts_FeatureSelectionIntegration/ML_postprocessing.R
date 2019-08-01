

##
## RENAME TO ML_performance.detail1.R
##

## Make sure R_utils.R and ML_performance-fn.R are included in distribution
##

POS = 1
NEG = 0

LABEL_IND <- 1
SCORE_IND <- 2

args = commandArgs(TRUE)

for( i in 1:length(args) ){
  if(args[i] == "-scores"){
    FILE.scores <- normalizePath(args[i+1])
  }else if(args[i] == "-featImp"){
    FILE.featImp <- normalizePath(args[i+1])
  }else if(args[i] == "-pos"){
    POS <- args[i+1]
  }else if(args[i] == "-neg"){
    NEG <- args[i+1]
  }else if(args[i] == "-save"){
    save_prefix <- args[i+1]
  }
}

#### Pull full directory of executed script ####
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
fp <- normalizePath(script.name)
slash_ind = which(strsplit(fp, "")[[1]]=="/")
util_dir <- substr(x = fp, start = 1, stop = max(slash_ind))

utils <- paste(util_dir, "R_utils.R", sep = "")
fn <- paste(util_dir, "ML_postprocessing-fn.R", sep = "")
#utils <- "R_utils.R"
#fn <- "ML_postprocessing-fn.R"

source(utils)
source(fn)

#### Import scores ####
df_scores <- import_file(FILE.scores)
df_scores [LABEL_IND] <- as.character( df_scores [ ,LABEL_IND ] )

#### Define classes ####

classes_all <- unique( df_scores [ ,LABEL_IND ] )
nonTrain_classes <- setdiff(classes_all, c(POS, NEG))
classes <- c(POS, NEG, nonTrain_classes )

#### Calculate AUC-ROC ####

aucroc <- aucroc_from_df (df_scores, label_ind = 1, score_ind = 2, POS = POS, NEG = NEG)
print(paste("  AUC-ROC:", aucroc))

#### Generate ROC and PR prediction objects ####

df_scores.POS_NEG <- pos_neg_rename2 (df_scores, LABEL_IND, POS, NEG)

labels <- df_scores.POS_NEG [ , LABEL_IND ]
POS_freq <- length( which( labels == 1) ) / length (labels)

roc_obj <- performance_object (df_scores.POS_NEG, labels, col_start = 5, metric1 = "tpr", metric2 = "fpr")
pr_obj <- performance_object (df_scores.POS_NEG, labels, col_start = 5, metric1 = "prec", metric2 = "rec")

#### Calculate threshold-based performance measures ####

df_scores.sort <- df_scores.POS_NEG [ order( df_scores.POS_NEG [SCORE_IND], decreasing = TRUE ),  ]
scores.sort <- df_scores.sort [ , SCORE_IND ]
labels.sort <- df_scores.sort [ , LABEL_IND ]

thresh_metrics <- metrics_by_threshold (labels.sort, scores.sort, thr_l = "")

thr.FPR <- thresh_by_proportion_cutoff.monotonic (thresh_metrics$thresholds, thresh_metrics$FPR, cutoff = 0.05)
thr.FNR <- thresh_by_proportion_cutoff.monotonic (thresh_metrics$thresholds, thresh_metrics$FNR, cutoff = 0.05)

max_MCC <- max(thresh_metrics$MCC, na.rm = T)
max_Fmeas <- max(thresh_metrics$Fmeas, na.rm = T)
exp_Fmeas <- (2 * POS_freq * 0.5) / (POS_freq + 0.5)
exp_Fmeas2 <- (2 * POS_freq * 1) / (POS_freq + 1)
max_Fmeas_bal <- max(thresh_metrics$Fmeas_bal, na.rm = T)

PPV_at_FPR <- value_at_threshold (thresh_metrics$thresholds, thresh_metrics$PPV_bal, thr.FPR)
NPV_at_FNR <- value_at_threshold (thresh_metrics$thresholds, thresh_metrics$NPV_bal, thr.FNR)

#### Report performance metrics ####

#metrics <- c("AUC-ROC", "MCC", "F-meas (bal)", "F-meas", "F-meas (exp: 0.5)", "F-meas (exp: 1.0)")
#values <- c(aucroc, max_MCC, max_Fmeas_bal, max_Fmeas, exp_Fmeas, exp_Fmeas2)
metrics <- c("AUC-ROC", "MCC", "F-meas (bal)", "F-meas",
             "F-meas (exp: 0.5)", "F-meas (exp: 1.0)",
             "FPR_thr", "FNR_thr",
             "PPV at FPR_thr", "NPV at FNR_thr")
values <- c(aucroc, max_MCC, max_Fmeas_bal, max_Fmeas, exp_Fmeas, exp_Fmeas2, thr.FPR, thr.FNR, PPV_at_FPR, NPV_at_FNR)
values <- round(values, 3)
df_report <- data.frame( metric = metrics, value = values)

report_name <- paste(save_prefix, "performance", sep = ".")
write.table(x = df_report, file = report_name, quote = F, append = F, sep = "\t", row.names = F, col.names = T)

#### ROC, PR, and score plots
pdf_name <- paste(save_prefix, "ROC_PR_scores","pdf", sep = ".")
ROC_text <- paste("AUC-ROC:", round(aucroc, 2) )

pdf(file = pdf_name, width = 8.5, height = 7)
par ( mfrow = c(2, 3), pty = "s")
plot_AUC_objects ( list(roc_obj) )
mtext(ROC_text)
plot_PR_objects ( list(pr_obj), POS_freq = POS_freq )
for(Class in classes){
  class_scores <- sort( df_scores [ which(df_scores[LABEL_IND] == Class), SCORE_IND], decreasing = T )
  plot_scores (scores = class_scores, thr1 = thr.FNR, thr2 = thr.FPR, main = Class)
}
dev.off()

#### Summarize feature importance ####

df_featImp <- read.table(file = FILE.featImp, comment.char = "", header = T, sep = "\t", stringsAsFactors = F)
names(df_featImp)[1] <- "feat"

featImp_dists <- pull_featImp_dists (df_featImp)
feats_summary <- summarize_featImp (df_featImp, featImp_dists)
feats_summary <- feats_summary[ order(feats_summary$impScore, decreasing = T), ]

featSum_name <- paste(save_prefix, "feature_imp_summary", sep = ".")
write_df(df = feats_summary, rowNm_header = "#feat", outfile = featSum_name)


