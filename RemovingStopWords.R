train = read.csv("C:/Users/daxpa/downloads/train_comp2.csv")
test = read.csv("C:/Users/daxpa/downloads/test_comp2.csv")

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

data = Corpus(VectorSource(CombinedData$NARRATIVE))

data = tm_map(data, content_transformer(tolower))
data = tm_map(data, removeNumbers)
data = tm_map(data, removePunctuation)
data = tm_map(data, remove_stopwords)
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
