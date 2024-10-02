library(keras)
install_keras()
#setwd("D:/Main project src code/Datasets")
train <- read.csv(file="L:/Fake news detection project/multi_training.csv",header = T,stringsAsFactors=TRUE)
test <- read.csv(file="L:/Fake news detection project/multi_testing.csv",header = T,stringsAsFactors=TRUE)

levels(train$Label)
train$Label = factor(train$Label,levels(train$Label)[c(5,2,1,3,4,6)])
train$Label <- as.numeric(train$Label)
#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6

levels(test$Label)
test$Label = factor(test$Label,levels(test$Label)[c(5,2,1,3,4,6)])
test$Label <- as.numeric(test$Label)
#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6


#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6

#------------ Model 1: FULL MODEL -------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train[,-34]) #Without Search = 34th Column
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test[,-34]) #Without Search = 34th Column
dimnames(newtest) <- NULL


#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:33] <- normalize(newtrain[,1:33]) #Independent = Predictors
newtrain[,34] <- as.numeric(newtrain[,34]) -1  #Label = 0,1,2,3,4,5
summary(newtrain)
training <- newtrain[,1:33]
traintarget <- newtrain[,34]

newtest[,1:33] <- normalize(newtest[,1:33]) #Independent = Predictors
newtest[,34] <- as.numeric(newtest[,34]) -1  #Label = 0,1,2,3,4,5
summary(newtest)
testing <- newtest[,1:33]
testtarget <- newtest[,34]


#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)


#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(33))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#33 independent * 9 layers= 297 + 9 constant values for each node = 306

#Compile the model
#For 6 class problem we will use "categorical_crossentropy"
model %>%
  compile(loss ='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

#Fit the model (Multi-layer Perceptron Neural Network)
#Using Train data on Validation set
history <- model %>%
  fit(training,
      trainlabels,
      epoch=25,
      validation_data = list(testing,testlabels)
  )
plot(history)

#Evaluate the model using Test Data
model1 <- model %>%
  evaluate(testing,testlabels)

#Prediction and Confusion Matrix - test data
prob <- model %>%
  predict(testing)
pred <- model %>%
  predict(testing)%>%k_argmax()


cbind(prob,pred,testtarget)

model1


#------------ Model 2: FULL MODEL + SEARCH-------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train)
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test)
dimnames(newtest) <- NULL


#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:34] <- normalize(newtrain[,1:34])
newtrain[,35] <- as.numeric(newtrain[,35]) -1
summary(newtrain)
training <- newtrain[,1:34]
traintarget <- newtrain[,35]

newtest[,1:34] <- normalize(newtest[,1:34])
newtest[,35] <- as.numeric(newtest[,35]) -1
summary(newtest)
testing <- newtest[,1:34]
testtarget <- newtest[,35]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(34))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

#Fit the model (Multi-layer Perceptron Neural Network)
#Using Train data on Validation set
history <- model %>%
  fit(training,
      trainlabels,
      epoch=25,
      validation_data = list(testing,testlabels))
plot(history)

#Evaluate the model using Test Data
model2 <- model %>%
  evaluate(testing,testlabels)

#Prediction and Confusion Matrix - test data
prob <- model %>%
  predict_proba(testing)
pred <- model %>%
  predict_classes(testing)
table2 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model2
table2
#model1 vs model2, SEARCH improves the accuracy by 5% (40 to 45) and reduces loss by 8.7%
