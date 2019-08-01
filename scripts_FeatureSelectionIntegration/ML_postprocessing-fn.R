
pos_neg_rename <- function(df, pos_nm, neg_nm){
  df.pos <- m.FL[ row.names(df)[ which(df$Class == pos_nm) ], ]
  df.neg <- m.FL[ row.names(df)[ which(df$Class == neg_nm) ], ]
  
  df.pos.rnm <- cbind( rep(1, nrow(df.pos)), df.pos$Mean)
  df.neg.rnm <- cbind( rep(0, nrow(df.neg)), df.neg$Mean)
  
  df.pos_neg <- rbind(df.pos.rnm, df.neg.rnm)
  df.pos_neg.sort <- df.pos_neg[ order(df.pos_neg[,2], decreasing = T), ]
  
  return(df.pos_neg.sort)
}

aucroc_calc <- function(labels, values){
  library(ROCR)
  roc_pred <- prediction( values, labels ) # arg1 = prediction, arg2 = labels
  auc_obj <- performance(roc_pred, "auc")
  auc_roc <- auc_obj@y.values[[1]]
  return(auc_roc)
}

pos_neg_rename2 <- function(x, label_ind, POS, NEG){
  df.pos <- x[ which( x[ label_ind ] == POS), ]
  df.neg <- x[ which( x[ label_ind ] == NEG), ]
  
  df.pos [ label_ind ] <- 1
  df.neg [ label_ind ] <- 0
  
  df.rnm <- rbind(df.pos, df.neg)
  
  return(df.rnm)
}

aucroc_from_df <- function(x, label_ind = 1, score_ind = 2, POS = "1", NEG = "0"){
  
  target_x <- pos_neg_rename2 (x, label_ind, POS, NEG)
  target_x <- target_x[ order(target_x[score_ind], decreasing = T), ]
  
  labels <- target_x[ , label_ind ]
  scores <- target_x[ , score_ind ]
  
  aucroc <- aucroc_calc (labels, scores)
  
  return(aucroc)
}

balance_df <- function(df, n_samp){
  p_i <- which(df[,1] == 1)
  n_i <- which(df[,1] == 0)
  
  p_i.samp <- sample(p_i, n_samp)
  n_i.samp <- sample(n_i, n_samp)
  
  pn_i.samp <- sort( c( p_i.samp, n_i.samp ) )
  
  df.bal <- df[ pn_i.samp, , drop = F]
  
  return(df.bal)
}

balanced_PR_object <- function(df, pos_nm, neg_nm, n_samp, n_iter){
  
  df.bal.group <- data.frame(row.names = 1:n_samp*2)
  
  for(i in 1:n_iter){
    df.pos_neg.bal <- balance_df ( df.pos_neg , n_samp)
    df.bal.group <- cbind( df.bal.group, df.pos_neg.bal)
  }
  
  library(ROCR)
  ncol = dim(df.bal.group)[2]
  pred <- prediction( df.bal.group[,seq(2,ncol,2)], df.bal.group[,seq(1,ncol,2)] )
  perf <- performance(pred, "prec", "rec")
  
  return(perf)
  #plot( 1, type = "n", ylim = c( 0.45 ,1 ), ylab = "Precision",
  #     xlim = c( 0, 1), xlab = "Recall")
  
  #plot(perf, avg='vertical', spread.estimate='stddev', ylim=c(0,1), col = "red",lwd=2,add=TRUE)
  #abline(h = 0.5, col = "gray", lty = "dashed")
}

PR_object <- function(df, pos_nm, neg_nm, n_samp, n_iter){
  
  library(ROCR)
  ncol = dim(df.bal.group)[2]
  pred <- prediction( df.bal.group[,seq(2,ncol,2)], df.bal.group[,seq(1,ncol,2)] )
  perf <- performance(pred, "prec", "rec")
  
  return(perf)
  #plot( 1, type = "n", ylim = c( 0.45 ,1 ), ylab = "Precision",
  #     xlim = c( 0, 1), xlab = "Recall")
  
  #plot(perf, avg='vertical', spread.estimate='stddev', ylim=c(0,1), col = "red",lwd=2,add=TRUE)
  #abline(h = 0.5, col = "gray", lty = "dashed")
}

performance_object <- function(df, labels, col_start, metric1 = "phi", metric2 = ""){
  # PR: metric1 = "prec", metric2 = "rec"
  # ROC: metric1 = "tpr", metric2 = "fpr"
  
  
  library(ROCR)
  ncol = dim(df)[2]
  CV_ind <- seq(col_start, ncol, 2)
  
  m.prediction <- df[ ,CV_ind ]
  m.label <- matrix( rep(labels, ncol(m.prediction)), ncol = ncol(m.prediction) )
  
  pred <- prediction( predictions = m.prediction, labels = m.label )
  
  
  perf <- performance(pred, metric1, metric2)
  #perf <- performance(pred, "tpr", "fpr")
  #perf <- performance(pred, "prec", "rec")
  
  return(perf)
}

aucroc_object <- function(df, labels, col_start){
  
  library(ROCR)
  ncol = dim(df)[2]
  CV_ind <- seq(col_start, ncol, 2)
  
  m.prediction <- df[ ,CV_ind ]
  m.label <- matrix( rep(labels, ncol(m.prediction)), ncol = ncol(m.prediction) )
  
  pred <- prediction( predictions = m.prediction, labels = m.label )
  #perf <- performance(pred, "tpr", "fpr")
  perf <- performance(pred, "prec", "prec")
  
  return(perf)
}

plot_PR_objects <- function(perf_list, POS_freq){
  plot( 1, type = "n", ylim = c( 0, 1 ), ylab = "Precision",
        xlim = c( 0, 1), xlab = "Recall")
  abline(h = POS_freq, col = "gray", lty = "dashed")
  
  lines_colors <- c("red", "blue", "orange")
  i <- 1
  for(perf in perf_list){
    plot(perf, avg='vertical', spread.estimate='stderror', ylim= c(0, 1),
         col = lines_colors[i], lwd = 2, add = TRUE)
    i <- i + 1
  }
}

plot_PR_objects.balanced <- function(perf_list){
  plot( 1, type = "n", ylim = c( 0.45 ,1 ), ylab = "Precision",
        xlim = c( 0, 1), xlab = "Recall")
  abline(h = 0.5, col = "gray", lty = "dashed")
  
  lines_colors <- c("red", "blue", "orange")
  i <- 1
  for(perf in perf_list){
    plot(perf, avg='vertical', spread.estimate='stderror', ylim= c(0, 1),
         col = lines_colors[i], lwd = 2, add = TRUE)
    i <- i + 1
  }
}

plot_AUC_objects <- function(perf_list){
  plot( 1, type = "n", ylim = c( 0, 1 ), ylab = "True Positive Rate",
        xlim = c( 0, 1), xlab = "False Positive Rate")
  abline(0, 1, col = "gray", lty = "dashed")
  
  lines_colors <- c("red", "blue", "orange")
  i <- 1
  for(perf in perf_list){
    plot(perf, avg='vertical', spread.estimate='stderror', ylim= c(0, 1),
         col = lines_colors[i], lwd = 2, add = TRUE)
    i <- i + 1
  }
}

make_confusion_matrix <- function(observed, predicted){
  obs_P = which(observed == 1)
  obs_N = which(observed == 0)
  
  prd_P = which(predicted == 1)
  prd_N = which(predicted == 0)
  
  cnt_TP <- length( intersect( obs_P, prd_P ) )
  cnt_FP <- length( intersect( obs_N, prd_P ) )
  cnt_FN <- length( intersect( obs_P, prd_N ) )
  cnt_TN <- length( intersect( obs_N, prd_N ) )
  
  confusion_matrix <- list( TP = cnt_TP, FP = cnt_FP, FN = cnt_FN, TN = cnt_TN )
  confusion_matrix <- unlist(confusion_matrix)
  
  return( confusion_matrix )
}

calc_MCC <- function( TP, FP, FN, TN ){
  
  NUMER = ( TP * TN ) - ( FP * FN )
  DENOM = sqrt( ( TP + FP ) * ( TP + FN ) * ( TN + FP ) * ( TN + FN ) )
  MCC = NUMER / DENOM
  names(MCC) <- NULL
  
  return (MCC)
  
}

thresh_by_MCC <- function(labels, scores, thr_l = ""){
  
  if (thr_l == ""){
    thr_l = sort(unique(scores))
  }
  
  MCC_l <- c()
  for(thr in thr_l){
    pred_labels <- scores
    pred_labels [ which (scores > thr) ] <- 1
    pred_labels [ which (scores <= thr) ] <- 0
    
    confusion_matrix <- make_confusion_matrix (labels, pred_labels)
    MCC <- calc_MCC ( confusion_matrix[1], confusion_matrix[2], confusion_matrix[3], confusion_matrix[4] )
    MCC_l <- c( MCC_l, MCC )
  }
  
  max_MCC <- max( MCC_l, na.rm = T )
  max_MCC_ind <- which(MCC_l == max_MCC)
  max_MCC_thr <- thr_l [max_MCC_ind]
  
  return(max_MCC_thr)
}

metrics_by_threshold <- function(labels, scores, thr_l = ""){
  
  if (thr_l == ""){
    thr_l = sort(unique(scores))
  }
  
  TPR_l <- c()
  FPR_l <- c()
  FNR_l <- c()
  TNR_l <- c()
  
  PPV_l <- c()
  NPV_l <- c()
  PPV_bal_l <- c()
  NPV_bal_l <- c()
  
  Fmeas_l <- c()
  Fmeas_bal_l <- c()
  
  MCC_l <- c()
  
  for(thr in thr_l){
    
    pred_labels <- scores
    pred_labels [ which (scores > thr) ] <- 1
    pred_labels [ which (scores <= thr) ] <- 0
    
    confusion_matrix <- make_confusion_matrix (labels, pred_labels)
    
    names(confusion_matrix) <- NULL
    TP <- confusion_matrix[1]
    FP <- confusion_matrix[2]
    FN <- confusion_matrix[3]
    TN <- confusion_matrix[4]
    
    TPR <- TP / (TP + FN)
    FPR <- FP / (FP + TN)
    FNR <- FN / (TP + FN)
    TNR <- TN / (FP + TN)
    
    PPV <- TP / (TP + FP)
    NPV <- TN / (TN + FN)
    PPV_bal <- TPR / (TPR + FPR)
    NPV_bal <- TNR / (TNR + FNR)
    
    Fmeas <- (2 * PPV * TPR) / (PPV + TPR)
    Fmeas_bal <- (2 * PPV_bal * TPR) / (PPV_bal + TPR)
    
    MCC <- calc_MCC ( confusion_matrix[1], confusion_matrix[2], confusion_matrix[3], confusion_matrix[4] )
    
    TPR_l <- c(TPR_l, TPR)
    FPR_l <- c(FPR_l, FPR)
    FNR_l <- c(FNR_l, FNR)
    TNR_l <- c(TNR_l, TNR)
    
    PPV_l <- c(PPV_l, PPV)
    NPV_l <- c(NPV_l, NPV)
    PPV_bal_l <- c(PPV_bal_l, PPV_bal)
    NPV_bal_l <- c(NPV_bal_l, NPV_bal)
    
    Fmeas_l <- c(Fmeas_l, Fmeas)
    Fmeas_bal_l <- c(Fmeas_bal_l, Fmeas_bal)
    
    MCC_l <- c(MCC_l, MCC)
  }
  
  thresh_metrics <- list(thresholds = thr_l, TPR = TPR_l, FPR = FPR_l, FNR = FNR_l, TNR = TNR_l,
                         PPV = PPV_l, NPV = NPV_l, PPV_bal = PPV_bal_l, NPV_bal = NPV_bal_l,
                         Fmeas = Fmeas_l, Fmeas_bal = Fmeas_bal_l, MCC = MCC_l)
  
  return(thresh_metrics)
}

thresh_by_max_value <- function(thresholds, values){
  max_val <- max(values, na.rm = T)
  max_ind <- which(values == max_val)
  max_thr <- thresholds[max_ind]
  return(max_thr)
}

thresh_by_proportion_cutoff.monotonic <- function(thresholds, values, cutoff = 0.05){
  diff_from_cutoff <- values - cutoff
  neg_ind <- which(diff_from_cutoff < 0)
  max_neg <- max( diff_from_cutoff[neg_ind] )
  max_ind <- which( diff_from_cutoff == max_neg )
  max_thr <- min(thresholds[max_ind])
  return(max_thr)
}

thresh_by_proportion_cutoff <- function(thresholds, values, cutoff = 0.95, abv_blw = "abv", lo_hi = "lo" ){
  
  if ( abv_blw == "abv" ){
    meet_cutoff <- which(values >= cutoff)
  }else if ( abv_blw == "blw" ){
    meet_cutoff <- which(values <= cutoff)
  }else{
    print(paste("abv_blw parameter not recognized:", abv_blw))
  }
  
  thr_candidates <- thresholds[meet_cutoff]
  
  if(lo_hi == "lo"){
    out_thr = min(thr_candidates)
  }else if(lo_hi == "hi"){
    out_thr = max(thr_candidates)
  }else{
    print(paste("lo_hi parameter not recognized:", lo_hi))
  }
  
  return(out_thr)
}

value_at_threshold <- function(thresholds, values, thr){
  thr_ind <- which(thresholds == thr)
  val_at_ind <- values[thr_ind]
  return(val_at_ind)
}

plot_scores <- function(scores, thr1, thr2, from = 0, to = 100, by = 2, color_lo = "darkred", color_mid = "gray", color_hi = "dodgerblue", main = NA){
  #thr1 <- 0.3
  #thr2 <- 0.6
  breaks <- seq(from = from, to = to, by = by)
  
  # color vector
  cnt_lo <- length( which(breaks <= thr1) )
  cnt_mid <- length( which(breaks > thr1 & breaks <= thr2) )
  cnt_hi <- length( which(breaks > thr2) ) - 1 # there is one fewer histogram bar than break
  
  hist_colors <- c( rep(color_lo, cnt_lo), rep(color_mid, cnt_mid), rep(color_hi, cnt_hi)  )
  
  hist( scores, breaks = breaks, col = hist_colors, border = F, xlab = "Prediction score", main = main )
  abline(v = thr1, lty = "dotted", col = "black")
  abline(v = thr2, lty = "dotted", col = "black")
  
  cnt_lo <- length( which ( scores <= thr1 ))
  cnt_md <- length( which ( scores > thr1 & scores <= thr2 ))
  cnt_hi <- length( which ( scores > thr2))
  
  per_lo <- round( cnt_lo/length(scores)*100, digits = 2)
  per_md <- round( cnt_md/length(scores)*100, digits = 2)
  per_hi <- round( cnt_hi/length(scores)*100, digits = 2)
  per_md_hi <- per_md+per_hi
  
  mtext(per_lo, side = 3, adj = 0)
  mtext(per_md, side = 3, adj = 0.33)
  mtext(per_hi, side = 3, adj = 0.66)
  mtext(per_md_hi, side = 3, adj = 1)
  
}

pull_featImp_dists <- function(df_featImp){
  CV_iters <- unique(df_featImp$CV_iter)
  CV_folds <- unique(df_featImp$CV_fold)
  featImp_dists <- list()
  for(i in CV_iters){
    featImp_dists[[i]] <- list()
    for(j in CV_folds){
      itr_ind <- which(df_featImp$CV_iter == i)
      fold_ind <- which(df_featImp$CV_fold == j)
      itr_fold <- intersect(itr_ind, fold_ind)
      
      imps <- sort( df_featImp[ itr_fold, "Mean" ], decreasing = T )
      
      featImp_dists[[i]][[j]] <- imps
    }
  }
  
  return(featImp_dists)
}

calc_max_sel <- function(df_featImp){
  n_CVitr <- length(unique(df_featImp$CV_iter))
  n_CV_fold <- length(unique(df_featImp$CV_fold))
  max_sel <- n_CVitr * n_CV_fold
  return(max_sel)
}

summarize_featImp <- function(df_featImp, featImp_dists){
  features <- unique(df_featImp$feat)
  
  max_sel <- calc_max_sel (df_featImp)
  feats_summary <- data.frame()
  
  for(ft in features){
    ft_ind <- which(df_featImp$feat == ft)
    sub_df <- df_featImp [ ft_ind, ]
    
    n_selected <- nrow(sub_df)
    n_sel_norm <- n_selected / max_sel
    
    imp_l <- c()
    imp_norm_l <- c()
    
    rank_l <- c()
    rank_norm_l <- c()
    
    for(i in 1:nrow(sub_df)){
      imp <- sub_df$Mean[i]
      
      itr <- sub_df$CV_iter[i]
      fold <- sub_df$CV_fold[i]
      imp_dist <- featImp_dists[[itr]][[fold]]
      
      min_imp <- min(imp_dist)
      max_imp <- max(imp_dist)
      imp_norm <- (imp - min_imp) / (max_imp - min_imp)
      
      rank <- which( imp_dist == imp)
      rank2 <- which( rev(imp_dist) == imp)
      rank_norm <- rank2/ length(imp_dist)
      
      imp_l <- c(imp_l, imp)
      imp_norm_l <- c(imp_norm_l, imp_norm)
      rank_l <- c(rank_l, rank)
      rank_norm_l <- c(rank_norm_l, rank_norm)
    }
    
    ave_imp <- mean(imp_l)
    ave_imp_norm <- mean(imp_norm_l)
    ave_rank <- mean(rank_l)
    ave_rank_norm <- mean(rank_norm_l)
    
    imp_score <- (2 * n_sel_norm * ave_rank_norm) / (n_sel_norm + ave_rank_norm)
    
    ft_summary <- data.frame( n_sel = n_selected, n_selNorm = round(n_sel_norm, 3),
                              imp = round(ave_imp, 3), impNorm = round(ave_imp_norm, 3),
                              rank = round(ave_rank, 1), rankNorm = round(ave_rank_norm, 3),
                              impScore = round(imp_score, 3), row.names = ft)
    feats_summary <- rbind(feats_summary, ft_summary)
  }
  
  return(feats_summary)
}
