train = read.csv("C:/Users/daxpa/downloads/train_comp2.csv")
test = read.csv("C:/Users/daxpa/downloads/test_comp2.csv")


#############################################################

# Account for the frequency of the words in order to find
# the crimes of type UNKNOWN

#############################################################
train$NARRATIVE = as.character(train$NARRATIVE)

Description_train = train %>% select(DR, Narrative = NARRATIVE)

# breaking trainnarrative to tokens
Words_train = Description %>% unnest_tokens(word, Narrative, to_lower = T)
head(Words_train)

# remove stopwords from the list
remove_stopwords_train = Words_train %>% anti_join(stop_words)
head(remove_stopwords_train)


crime_count_train = remove_stopwords_train %>% count(DR, word, sort = TRUE) %>% 
  group_by(word) %>% summarise(unique_words = n())


test$NARRATIVE = as.character(test$NARRATIVE)

Description_test = test %>% select(DR, Narrative = NARRATIVE)

# breaking trainnarrative to tokens
Words_test = Description_test %>% unnest_tokens(word, Narrative, to_lower = T)
head(Words)

# remove stopwords from the list
remove_stopwords_test = Words_test %>% anti_join(stop_words)
head(remove_stopwords_test)


crime_count_test = remove_stopwords_test %>% count(DR, word, sort = TRUE) %>% 
  group_by(word) %>% summarise(unique_words = n())

write.csv(crime_count_test, "C:/Users/daxpa/Downloads/CCTest.csv")

write.csv(crime_count_train, "C:/Users/daxpa/Downloads/CCTrain.csv")

TrainSubset = train[, c(1,2)]
CombinedData = rbind(TrainSubset, test)

library(tidytext)
library(tidyverse)

library(tm)

CombinedData$NARRATIVE = as.character(CombinedData$NARRATIVE)

Description = CombinedData %>% select(DR, Narrative = NARRATIVE)

# breaking trainnarrative to tokens
Words = Description %>% unnest_tokens(word, Narrative, to_lower = T)
head(Words)

# remove stopwords from the list
remove_stopwords = Words %>% anti_join(stop_words)
head(remove_stopwords)


crime_count = remove_stopwords %>% count(DR, word, sort = TRUE) %>% 
  group_by(word) %>% summarise(unique_words = n())

data = Corpus(VectorSource(CombinedData$NARRATIVE))

data = tm_map(data, content_transformer(tolower))
data = tm_map(data, removeNumbers)
data = tm_map(data, removePunctuation)
data = tm_map(data, removeWords, c("the", "and", stopwords("english")))
data = tm_map(data, stripWhitespace)

inspect(data[1])

CombinedMatrix = DocumentTermMatrix(data)
CombinedMatrix
inspect(CombinedMatrix[500:505, 500:505])

CombinedMatrix = removeSparseTerms(CombinedMatrix, 0.99)
CombinedMatrix
inspect(CombinedMatrix[1, 1:20])

Matrix_tfidf = DocumentTermMatrix(data, control = list(weighting = weightTfIdf))
Matrix_tfidf = removeSparseTerms(Matrix_tfidf, 0.95)
Matrix_tfidf
inspect(Matrix_tfidf[1,1:20])

sample = as.data.frame(as.matrix(CombinedMatrix))

Train_Sample = sample[1:11500,]
Test_Sample = sample[11501:23675, ]

TrainData = cbind(train, Train_Sample)
TestData = cbind(test, Test_Sample)


TrainData$CRIMETYPE = as.numeric(as.factor(train$CRIMETYPE))

ModelTrain = TrainData[,3:177]

#####################################################################################

# Using SVM model

###################################################################################
model_svm = svm(CRIMETYPE~., data = ModelTrain)

TestData$CRIMETYPE = predict(model_svm, TestData)

TestData$CRIMETYPE = ceiling(TestData$CRIMETYPE)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"
submission_svm = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission_svm, "C:/Users/daxpa/Downloads/SVM.csv")

#####################################################################################

# Using GLM

###################################################################################
library(gbm)

model = glm(CRIMETYPE ~ ., data = ModelTrain, family = "poisson")

TestData$CRIMETYPE = predict(model, TestData,type = "response")

TestData$CRIMETYPE = ceiling(TestData$CRIMETYPE)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"

submission = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission, "C:/Users/daxpa/Downloads/GBM.csv")

#####################################################################################

# Using GBM model

###################################################################################

model_gbm = gbm(CRIMETYPE ~ ., data = ModelTrain, distribution="gaussian", 
                interaction.depth = 3, n.trees=100)

pred_gbm = predict(model_gbm, TestData, n.trees=100, type="response")

TestData$CRIMETYPE = ceiling(pred_gbm)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"

submission2 = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission2, "C:/Users/daxpa/Downloads/GBM.csv")

#####################################################################################

# Using random forest, and rpart

###################################################################################

model_rf = randomForest(CRIMETYPE ~ ., data = ModelTrain)

pred_gbm = predict(model_rf, TestData, type="response")

TestData$CRIMETYPE = ceiling(pred_gbm)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"

submission3 = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission3, "C:/Users/daxpa/Downloads/RandomForest.csv")

#####################################################################################

# Using Naive Bayes

###################################################################################
library(e1071)

model = naiveBayes(CRIMETYPE ~ ., data = ModelTrain)

TestData$CRIMETYPE = predict(model, TestData, type = "raw")

TestData$CRIMETYPE = ceiling(TestData$CRIMETYPE)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"

submission = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission, "C:/Users/daxpa/Downloads/NaiveBayes.csv")

#####################################################################################

# Using random forest, and rpart

###################################################################################

#use random forest
library(caret)

data_control = trainControl(method = "cv", number = 5, verboseIter = F)

model6 = train(CRIMETYPE ~ ., data = ModelTrain, method = "rf", 
               trControl = data_control)

#using rpart
library(rpart)
library(rpart.plot)
rpart_model <- rpart(formula = CRIMETYPE ~ ., data = ModelTrain ,method = "class")

#creating a confusion matrix
# Generate predicted classes using the model object
TestData$CRIMETYPE = predict(object = rpart_model, newdata = TestData, 
                             type = "prob")  


TestData$CRIMETYPE = ceiling(TestData$CRIMETYPE)

TestData$CRIMETYPE[TestData$CRIMETYPE==1] <- "AGG"
TestData$CRIMETYPE[TestData$CRIMETYPE==2] <- "BTFV"
TestData$CRIMETYPE[TestData$CRIMETYPE==3] <- "BURG"
TestData$CRIMETYPE[TestData$CRIMETYPE==4] <- "GTA"
TestData$CRIMETYPE[TestData$CRIMETYPE==5] <- "ROBB"
TestData$CRIMETYPE[TestData$CRIMETYPE==6] <- "THEFT"

submission2 = TestData[,c("DR", "CRIMETYPE")]

write.csv(submission2, "C:/Users/daxpa/Downloads/rpart.csv")



