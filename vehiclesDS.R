#ata set: vehicles.csv
#Your task is to predict the vehicle type (type) knowing the values of the following
#variables: engine, horse, weight, length and fuelcap. The machine learning techniques
#that you must use are:
#  - logistic regression
#- linear discriminant analysis
#- naïve Bayes estimation
#- support vector machine
#- neural networks
#Which method gives the highest prediction accuracy in the test set?

Vehicles <-read.csv('vehicles.csv')
#Check for missing values 
sapply(Vehicles, function(x) sum(is.na(x)))

str(Vehicles)

Reg_model <- glm(type~engine+horse+weight+length+fuelcap, data=Vehicles, family=binomial())
summary(Reg_model)

library(car)
vif(Reg_model)
#Checking for high correlation values >10

##Compute the Antilogs of the coefficients 

expb <-exp(coef(Reg_model))
print(expb)

#Predict probability for type
pred_probs <- predict(Reg_model, type="response")
head(pred_probs)

#Preditct auto  or truck
pred <-ifelse(pred_probs<0.5, "0","1")
head(pred)
tail(pred)

table(Vehicles$type, pred)

Accuracy <-mean(pred == Vehicles$type)
Accuracy

library(ROCR)

#Plot the Roc Curve
pr <-prediction(pred_probs, Vehicles$type)
perf<-performance(pr, x.measure = "fpr", measure = "tpr")
plot(perf)

#compute auc of 96%
auc <-performance(pr, measure="auc")
auc

######Validate 

n<-sample(155, 75)
veh_train <- Vehicles[n, ]
veh_test <-Vehicles[-n, ]

fit <- glm(type~engine+horse+weight+length+fuelcap, data=veh_train, family=binomial())
###Compute the prediction accuracy on the test set 

pred_probs <-predict.glm(fit, newdata = veh_test, type="response")
head(pred_probs)

pred <- ifelse(pred_probs<0.5, "0", "1")
head(pred)
tail(pred)

library(ROCR)
#Build Roc Curve for Test set
pr <-prediction(pred_probs, veh_test$type)
perf<-performance(pr, x.measure = "fpr", measure = "tpr")
plot(perf)

#compute auc of 97% for Test set 
auc <-performance(pr, measure="auc")
auc

######Linear Discriminant Analysis

library(MASS)
fit <- lda(type~engine+horse+weight+length+fuelcap, data=Vehicles)
fit

pred <- predict(fit) #list of predicted 
head(pred)

class<-pred$class #The estimated class
head(class)

table(Vehicles$type, class)

correct <-mean(Vehicles$type==class)
correct
##93% accuracy

#### Validate Linear Discriminant Analysis

n <-sample(10000, 5000)

fit <- lda(type~engine+horse+weight+length+fuelcap, data=veh_train)
#Predict on the test set 
pred_test <- predict(fit, newdata=veh_test)

#list of predicted values
class <-pred_test$class
head(class)

correct <-mean(veh_test$type==class)
correct
#92.5% 

plot(fit)

#####Naive Bayes 

library(e1071)

bayes <- naiveBayes(type~engine+horse+weight+length+fuelcap, data=veh_train)
bayes

pred <-predict(bayes, veh_test)
head(pred)

correct <- mean(veh_test$type==class)
correct

#92.5%

###Suport Vector Machine

fit <- svm(type~., data=veh_train, 
           type= "C-classification", kernel="linear", cost=4)
pred <-predict(fit, veh_test)
mean(pred==veh_test$type)

#92.5%

###To find best cost (Improve accuracy)
###10 fold cv 
t_lin <- tune.svm(type~., data=veh_train, cost = 2^(2:8), kernal="linear")
t_lin$best.parameters

#####Neural Networks 
library(neuralnet)

net <- neuralnet(type~engine+horse+weight+length+fuelcap, 
                 data=veh_train, hidden=1, algorithm ="rprop+", 
                 err.fct = "sse", act.fct = "logistic", 
                 rep = 1, stepmax = 1e06, threshold = 0.01, linear.output = F)

plot(net)

##Plot neural network without weights
plot(net, show.weights = F)

#Generate the main results error 71.1094
net$result.matrix

#Generate weights list 
net$weights

## Predictions in the test set 
pred <-compute(net, veh_test[,-6])
pred2 <-pred$net.result
head(pred2,5)

## Create a categorical predicted value 
predcat <-ifelse(pred2<0.5, 0,1)

###Classification table 70%
table(predcat, veh_test$type)
mean(predcat==veh_test$type)




