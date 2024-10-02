setwd("D:/Main project src code/Datasets")
train <- read.csv(file="multi_training.csv",header = T,stringsAsFactors=TRUE)
test <- read.csv(file="multi_testing.csv",header = T,stringsAsFactors=TRUE)

library(xgboost)
library(mlr)

train$Label <- as.numeric(train$Label)-1 #multinomial (0-5)
test$Label <- as.numeric(test$Label)-1  #multinomial (0-5)


#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-34,-35)]),label=train$Label)  #34=Search, 35=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-34,-35)]),label=test$Label)   #34=Search, 35=Label


watchlist <- list(train=dtrain,test=dtest)
# Calculate # of folds for cross-validation
# merror is for multiclass error with multi:softprob
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective = "multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection


watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 1: train-merror:0.531405	test-merror:0.586906


#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-35)]),label=train$Label)  #35=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-35)]),label=test$Label)   #35=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-35)]),label=valid$Label)   #35=Label
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection


#Model 2: train-merror:0.518454	test-merror:0.576773

