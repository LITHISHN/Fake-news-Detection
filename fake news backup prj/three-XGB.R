#setwd("D:/Main project src code/Datasets")
train <- read.csv(file="L:/Fake news detection project/three_training.csv",header = T,stringsAsFactors=TRUE)
test <- read.csv(file="L:/Fake news detection project/three_testing.csv",header = T,stringsAsFactors=TRUE)

library(xgboost)
library(mlr)

train$Label <- as.numeric(train$Label)-1 #multinomial three (0-2)
test$Label <- as.numeric(test$Label)-1  #multinomial three (0-2)



#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-32,-33)]),label=train$Label)  #32=Search, 33=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-32,-33)]),label=test$Label)   #32=Search, 33=Label


watchlist <- list(train=dtrain,test=dtest)
# Calculate # of folds for cross-validation
# merror is for multiclass error with multi:softprob
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective = "multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection


watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection


#Model 1: train-merror:0.397994	test-merror:0.420109 


#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-33)]),label=train$Label)  #33=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-33)]),label=test$Label)   #33=Label

watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection


#Model 2: train-merror:0.387379	test-merror:0.425565 


