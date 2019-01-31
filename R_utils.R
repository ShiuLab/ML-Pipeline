
corner <- function(m, n = 10){
  print(m[1:n, 1:n])
}

import_file <- function(FILE_NM, header = TRUE){
  m <- read.table(FILE_NM, header=header, comment.char = "", row.names=1, sep="\t", quote = "", stringsAsFactors = F)
  m.df <- data.frame(m, stringsAsFactors = FALSE)
  return (m.df)
}

write_df <- function (df, rowNm_header, outfile) {
  
  df.rowNm <- data.frame(val = row.names(df))
  names(df.rowNm) <- rowNm_header
  
  df.out <- cbind ( df.rowNm, df )
  
  write.table( df.out, file = outfile, quote = F, sep = "\t", row.names = F )
}

median_center_cols <- function (df.raw){
  rowmed <- apply(df.raw, 2, median)
  df.center <- data.frame(scale(df.raw, center= rowmed, scale = FALSE))
  return(df.center)
}

remove_nonvarying_columns <- function (df.raw, max_repeat){
  max_repeats <- apply(df.raw, 2, function(y) max(table(y)))
  max_repeats.ind <- which(max_repeats <= max_repeat)
  df <- df.raw[max_repeats.ind]
  return(df)
}

make_CV_folds <- function(names_vec, n_folds = 10){
  # https://stats.stackexchange.com/questions/61090/how-to-split-a-data-set-to-do-10-fold-cross-validation
  folds <- cut( seq(1, length(names_vec)), breaks = n_folds, labels = FALSE)
  folds.shuffle <- sample(folds)
  return(folds.shuffle)
}

calc_MSE <- function(x1, x2){
  diff <- x1 - x2
  sqr <- diff*diff
  sm <- sum(sqr)
  MSE <- sm/length(sqr)
  return ( MSE )
}

calc_median_error <- function (x1, x2){
  diff <- x1 - x2
  abs_diff <- abs(diff)
  median_error <- median(abs_diff)
  return (median_error)
}

plot_YvX <- function (x, y, main_title, xlab = "Observed", ylab = "Predicted", plim = "", fit_line = T, sd = ""){
  if(plim == ""){
    mn <- min(c(x, y))
    mx <- max(c(x, y))
    plim <- c(mn, mx)
  }
  
  #par(pty = "s")
  #par(pty = "s")
  plot(x = 0, type = "n", xlim = plim, ylim = plim, xlab = xlab, ylab = ylab, main = main_title)
  abline(0, 1, col = "lightgray")
  
  if(sd != ""){
    sd_up = y+sd
    sd_down = y-sd
    segments( x0 = x, y0 = sd_up, x1 = x, y1 = sd_down, col = "darkgray"  )
  }
  
  if(fit_line == T){
    abline(lm(y~x), lty = "dashed", col = "black")
    
    r <- cor(x, y)
    r2 <- r**2
    
    residuals <- y-x
    residuals2 <- residuals**2
    rss <- sum(residuals2)
    
    cor_test_obj <- cor.test(x = x, y = y, method = "pearson")
    cor.p <- cor_test_obj$p.value
    
    mtext(paste("r=",round(r, 2), sep = ""), adj=0)
    mtext(paste("r2=",round(r2, 2), sep = ""), adj=0.5)
    if(cor.p < 0.01){
      mtext(text = paste("p=",formatC(cor.p, format = "e", digits = 0), sep = ""), adj = 1)
    }else{
      mtext(text = paste("p=",round(cor.p, 2), sep = ""), adj = 1)
    }
    
  }
  points(x = x, y = y, pch = 16)
}

add_column <- function( df_full, df_col, col_nm){
  df_full <- cbind(df_full, df_col)
  names(df_full)[ncol(df_full)] <- col_nm
  return(df_full)
}

add_meanSD <- function(x){
  mean_pred <- apply(X = x, MARGIN = 1, mean)
  pred_sd <- apply(X = x, MARGIN = 1, sd)
  
  stats_df <- data.frame( mean_pred = mean_pred, pred_sd = pred_sd )
  df_w_stats <- cbind(stats_df, x)
}

outersect <- function(x, y) {
  osect <- sort(c(setdiff(x, y),
         setdiff(y, x)))
  return(osect)
}