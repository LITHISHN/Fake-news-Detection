# Random Forest
library("randomForest")
set.seed(1234)
setwd("D:/Main project src code/Datasets")
train <- read.csv(file="binary_training.csv", header=T)
test <- read.csv(file="binary_testing.csv", header = T)


#-------------------- Model 1: FULL MODEL ------------------

#Grid Search for Tuning Hyperparameters using CARET
library(caret)
?trainControl
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")#repeated cross-validation, 3-fold cross-validation, 10 resampling iterations, grid search
set.seed(1234)
tunegrid <- expand.grid(.mtry=c(1:15))
#Tuning hyperparameters using Validation set
rf_gridsearch <- train(factor(Label)~., data=test[,-31], method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #31=Search
print(rf_gridsearch)
plot(rf_gridsearch)
#6    0.6612086  0.3200555
#Creating the model based on the best value for mtry (Number of variables randomly sampled as candidates at each split).
?randomForest
rf <- randomForest(factor(train$Label) ~ ., data=train[,-31], keep.forest=TRUE, ntree=500, mtry=6) #31=Search
#error rate

print(rf) #OOB estimate of  error rate: 32.51%
varImpPlot(rf)
#confusion Matrix
table(train$Label, predict(rf, train[,-32], type="response", norm.votes=TRUE)) #32=Label
RFpred <- predict(rf, test[,-32], type="response", norm.votes=TRUE) #32=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 68.9%")


#Tuning using Algorithm Tools
#Algorithm Tune using TuneRF
x <- train[,-32]
y <- train[,32]
y <- as.factor(y)
bestmtry <- tuneRF(x,y,stepFactor=1.5,improve=1e-5, ntreeTry = 500)
print(bestmtry)
rf1 <- randomForest(factor(train$Label) ~ ., data=train[,-31], keep.forest=TRUE, ntree=500, mtry=3) #31=Search
print(rf1) #OOB estimate of  error rate: 32.86%
varImpPlot(rf1)
#confusion Matrix
table(train$Label, predict(rf1, train[,-32], type="response", norm.votes=TRUE)) #32=Label
RFpred <- predict(rf1, test[,-32], type="response", norm.votes=TRUE) #32=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 66.33%")

#----------------- Model 2: FULL MODEL + SEARCH -------------------
rf1 <- randomForest(factor(train$Label) ~ ., data=train, keep.forest=TRUE, ntree=500, mtry=6)
print(rf1) #OOB estimate of  error rate: 32.37%
varImpPlot(rf1)
table(train$Label, predict(rf1, train[,-32], type="response", norm.votes=TRUE))
RFpred <- predict(rf1, test[,-32], type="response", norm.votes=TRUE)
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 68.75%")