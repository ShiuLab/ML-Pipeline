cat("
This script makes histogram plots of scores for binary models.
The input file is the score file from your ML run.

To run:
First do module purge, then load all R modules:
module load GCC/7.3.0-2.30
module load OpenMPI/3.1.1
module load R/3.5.1-X11-20180604

Next run the script via command line:
R --vanilla --slave --args <arg 1> <arg 2> <arg 3> <arg 4> > plot_predprob.R
args:
inp1 = Input file (col1 = id ,col2 = Class, col3 = scores), header required
inp2 = threshold
inp3 = number of rows of plots
inp4 = number of columns of plots

You can also put the arguments in manually and run on your desktop
    ")

args = commandArgs(TRUE)

scores_file <- args[1]
threshold_raw <- args[2]
threshold <- as.numeric(threshold_raw)
nrows <- args[3]
ncols <- args[4]


# round threshold and make color vector
pos_col <- "orangered2"
neg_col <- "cyan2"
if(threshold<0.025){
	round_thresh <- 0.00
	color_vector <- c(pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.025&&threshold<0.075){
	round_thresh <- 0.05
	color_vector <- c(neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.075&&threshold<0.125){
	round_thresh <- 0.10
	color_vector <- c(neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.125&&threshold<0.175){
	round_thresh <- 0.15
	color_vector <- c(neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.175&&threshold<0.225){
	round_thresh <- 0.20
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.225&&threshold<0.275){
	round_thresh <- 0.25
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.275&&threshold<0.325){
	round_thresh <- 0.30
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.325&&threshold<0.375){
	round_thresh <- 0.35
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.375&&threshold<0.425){
	round_thresh <- 0.40
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.425&&threshold<0.475){
	round_thresh <- 0.45
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.475&&threshold<0.525){
	round_thresh <- 0.50
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.525&&threshold<0.575){
	round_thresh <- 0.55
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.575&&threshold<0.625){
	round_thresh <- 0.60
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.625&&threshold<0.675){
	round_thresh <- 0.65
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.675&&threshold<0.725){
	round_thresh <- 0.70
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.725&&threshold<0.775){
	round_thresh <- 0.75
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.775&&threshold<0.825){
	round_thresh <- 0.80
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.825&&threshold<0.875){
	round_thresh <- 0.85
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col,pos_col)
}else if(threshold>=0.875&&threshold<0.925){
	round_thresh <- 0.90
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col,pos_col)
}else if(threshold>=0.925&&threshold<0.975){
	round_thresh <- 0.95
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,pos_col)
}else if(threshold>=0.975){
	round_thresh <- 1.00
	color_vector <- c(neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col,neg_col)
}else{
	print("threshold unround-able")
	print("expected value between 0 and 1")
	print(threshold)
}

# read file and creat unique types vector
m <- read.table(scores_file,header=TRUE,sep="\t",comment.char="")
types <- unique(m[,2])
out_nm = paste(scores_file,".hist.pdf",sep="")
pdf(out_nm,width = 8.5, height = 11)
par(mfrow=c(strtoi(nrows),strtoi(ncols)))
text_x_1 <- round_thresh-0.1
text_x_2 <- round_thresh+0.1
for(typ in types){
	#print(typ)
	#pull values for each type
	indices <- c(which(m$Class==typ))
	#print(indices)
	type_values <- m[indices,][,3]
	
	#plot histogram and add vertical threshold line
	hist_obj <- hist(as.numeric(type_values),breaks=c(seq(from=0,to=1,by=0.05)),main=typ,xlab="confidence_score",ylab="count",col=color_vector,border=FALSE)
	abline(v=round_thresh,lty=2,lwd=1)
	
	# add text
	text_y <- max(hist_obj$counts)
	total_cnt <- length(type_values)
	abv_cnt <- sum(type_values > round_thresh)
	abv_per <- round(abv_cnt/total_cnt*100,digits=2)
	blw_cnt <- sum(type_values <= round_thresh)
	blw_per <- round(blw_cnt/total_cnt*100,digits=2)
	text(x=text_x_1,y=text_y,labels=toString(blw_per))
	text(x=text_x_2,y=text_y,labels=toString(abv_per))
}
dev.off()

