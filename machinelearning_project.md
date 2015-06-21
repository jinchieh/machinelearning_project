---
title: "Machine Learning Course Project"
output: html_document
---

# Machine Learning Course Project

## Download Dataset

```r
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainingURL,"pml-training.csv",method="curl")
download.file(testingURL, "pml-testing.csv", method="curl")
```

## Load Libraries

```r
library(gdata)
library(caret)
library(randomForest)
```

## Load Datasets

```r
# Build training dataset
training <- read.csv("C://Users/vchieh/Downloads/pml-training.csv",header=TRUE, na.strings="NA")
training <-training[,c(8:11,37:49,60:68,84:86,102,113:124,140,151:160)]

# Split training set into 2 to cut down processing time
# Can also use model from 1st set to predict 2nd set
set.seed(123)
inTrain <- createDataPartition(y=training$classe,p=0.4,list=FALSE)
training1 <- training[inTrain,]
training2 <- training[-inTrain,]

# Build testing dataset
testing <- read.csv("C://Users/vchieh/Downloads/pml-testing.csv",header=TRUE, na.strings="NA")
testing <-testing[,c(8:11,37:49,60:68,84:86,102,113:124,140,151:160)]
```

Training and testing sets are subsetting in certain columns that have appropriate values to allow us to perform predictions.

## Analysis


```r
# Check for variance (remove if almost 0 since it does not affect the model too much, to reduce processing times)
variance <- data.frame(nrow=1,ncol=52)
for(i in 1:52){
  variance[1,i] <- var(training1[,i])
  names(variance)[i] <- names(training1)[i]
}
```

Most variances are not 0 or close to 0. We will keep all the measurements to save time in transforming the dataset.

## Build Model


```r
# Parallel Core Processing
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
```

Parallel Core Processing done to speed up the process of training the model.

In this case, we use the random forests method to build a prediction model.


```r
# Define training control (Cross Validation with Random Forest method 'oob')
trControlRF <- trainControl(method="oob", number=3, repeats=1)

# Train the Model
modFit <- train(classe~ .,data=training1, ntree=500, method="rf", trControl = trControlRF)

# Make Predictions on 2nd training set
predicted <- predict(modFit, newdata=training2)

# Summarize Results on 2nd training set
confusionMatrix(predicted, training2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3336   32    0    0    0
##          B    9 2211   13    0    2
##          C    2   33 2035   39    8
##          D    0    2    5 1888   12
##          E    1    0    0    2 2142
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9864          
##                  95% CI : (0.9841, 0.9884)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9828          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9706   0.9912   0.9787   0.9898
## Specificity            0.9962   0.9975   0.9916   0.9981   0.9997
## Pos Pred Value         0.9905   0.9893   0.9613   0.9900   0.9986
## Neg Pred Value         0.9986   0.9930   0.9981   0.9958   0.9977
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2834   0.1878   0.1729   0.1604   0.1820
## Detection Prevalence   0.2861   0.1899   0.1798   0.1620   0.1822
## Balanced Accuracy      0.9963   0.9840   0.9914   0.9884   0.9948
```

The model has a very high out-of-sample accuracy of 98.64% with an out-of-sample error of 1.36%, and we will use the model to make predictions on the test set.


```r
# Make Predictions on test set
answers <- predict(modFit, newdata=testing)
```

## End

```r
# Terminate Multiple Processes
stopCluster(cl)
registerDoSEQ()
```
