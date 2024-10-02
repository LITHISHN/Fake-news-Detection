setwd("D:/Main project src code/Datasets")
train <- read.csv(file="binary_training.csv", header=T)
test <- read.csv(file="binary_testing.csv", header = T)
#Extreme Gradient Boosting (xgboost) is similar to gradient boosting framework but more efficient. It has both linear model solver and tree learning algorithms
library(xgboost)
#Extreme Gradient Boosting (xgboost) is similar to gradient boosting framework but more efficient. It has both linear model solver and tree learning algorithms
library(mlr)


#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-31,-32)]),label=train$Label) #31=Search, 32=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-31,-32)]),label=test$Label) #31=Search, 32=Label


watchlist <- list(train=dtrain,test=dtest)
#Estimated Time of Arrival or Expected Time of Arrival
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection
#train-error:0.304022	test-error:0.317757

watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection


#https://www.kaggle.com/general/17120
#Model 1: train-error:0.304022	test-error:0.296181 

#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-32)]),label=train$Label) #32=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-32)]),label=test$Label) #32=Label
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selecting
#https://www.kaggle.com/general/17120

#Model 2: train-error:0.301100	test-error:0.296181


