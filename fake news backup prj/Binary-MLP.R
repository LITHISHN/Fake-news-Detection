library(keras)
install_keras()
setwd("D:/Main project src code/Datasets")
train <- read.csv(file="binary_training.csv",header = T)
test <- read.csv(file="binary_testing.csv",header = T)


#------------ Model 1: FULL MODEL -------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train[,-31]) #Without Search
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test[,-31]) #Without Search
dimnames(newtest) <- NULL


#Normalize the matrix
#Turn off scientific notation for global variable
#The scipen value is an indicator of the integer’s prompt for exponential notation
options(scipen = 999)
#Check the attributes used

#normalize the data to bring all the variables to the same range
newtrain[,1:30] <- normalize(newtrain[,1:30]) #Independent = Predictors
#or manually normalize the matrix
#data=as.data.frame(newtrain[,1:30])
#minMax<- function(x){
# (x-min(x))/(max(x)-min(x))
#}
#newtrain[,1:30]<- as.data.frame(lapply(data,minMax))
#head(newtrain)
newtrain[,31] <- as.numeric(newtrain[,31]) #Label = Target
summary(newtrain)
training <- newtrain[,1:30]
traintarget <- newtrain[,31]

newtest[,1:30] <- normalize(newtest[,1:30]) #Independent = Predictors
newtest[,31] <- as.numeric(newtest[,31]) #Label = Target
summary(newtest)
testing <- newtest[,1:30]
testtarget <- newtest[,31]


#One hot encoding
#One-hot encoding is used to convert categorical variables into a format that can be used by machine learning algorithms
#preprocessing needed for some machine learning algorithms to improve performance.
#'mltools' is package and use 'one_hot' method and convert data to data.table
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)


#Create Sequential Model
# 30 independent variables
# Keras Model composed of a linear stack of layers Usage keras_model_sequential(layers = NULL, name = NULL, ...)
# It makes use of a single set of input as to value and a single set of output as per flow.
model <- keras_model_sequential()
#Pipe symbol %>%
#Add a densely-connected NN layer to an output using layer_dense
#NN layer(Neural Network)Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it(like Travelling Sales person problem in Design and Analysis of Algorithms)
#relu- used in hidden layer to avoid vanishing the problem and better computation performance
#softmax- use in last output layer
model %>% 
          layer_dense(units = 8, activation = 'relu', 
                      input_shape = c(30))%>%
          layer_dense(units = 8, activation = 'relu') %>%
          layer_dense(units = 2, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#30 independent * 8 layers=240 + 8 constant values for each node = 248
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
          compile(loss ='binary_crossentropy',
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
model2<- model%>%evaluate(training,trainlabels)
#Prediction and Confusion Matrix - test data
prob <- model %>%
        predict(testing)
pred <- model %>%
        predict(testing)%>%k_argmax()

table1 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model1
#plotting
p<- plot(model1,col="red",type="p",xlab="Scale",ylab="Accuracy and Loss",main="Scale Vs Accuracy and Loss")
grid()
abline(h=0.5816230,col="blue")
abline(h=0.6968043,col="blue")


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

newtrain[,1:31] <- normalize(newtrain[,1:31])
newtrain[,32] <- as.numeric(newtrain[,32])
summary(newtrain)
training <- newtrain[,1:31]
traintarget <- newtrain[,32]

newtest[,1:31] <- normalize(newtest[,1:31])
newtest[,32] <- as.numeric(newtest[,32])
summary(newtest)
testing <- newtest[,1:31]
testtarget <- newtest[,32]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)


#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 8, activation = 'relu', 
              input_shape = c(31))%>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='binary_crossentropy',
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
  predict(testing)
pred <- model %>%
  predict(testing)%>%k_argmax()
table2 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model2
table2


#model1 vs model2
model1 #69.21% accuracy
model2 #69.60% accuracy
