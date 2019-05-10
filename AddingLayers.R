train = read.csv("../input/train_comp2.csv")
test = read.csv("../input/test_comp2.csv")
library(text2vec)
library(keras)
library(tensorflow)

train$NARRATIVE = as.character(train$NARRATIVE)

txt = as.character(train$NARRATIVE)
it = itoken(txt, tolower, word_tokenizer, n_chunks = 10)
vocab = create_vocabulary(it)
vocab = prune_vocabulary(vocab, term_count_min = 100, doc_proportion_max = 0.25)

word_vectorizer = vocab_vectorizer(vocab)

it_train = itoken(train$NARRATIVE, preprocessor = tolower, tokenizer = word_tokenizer, ids = train$DR, progressbar = FALSE)
dtm_train = create_dtm(it_train, word_vectorizer)

test$NARRATIVE = as.character(test$NARRATIVE)

it_test = itoken(test$NARRATIVE, preprocessor = tolower, tokenizer = word_tokenizer, ids = test$DR, progressbar = FALSE)
dtm_test = create_dtm(it_test, word_vectorizer)

dim(dtm_test)

label_fac = as.numeric(as.factor(train$CRIMETYPE))
label_fac = label_fac - 1
ytrain = label_fac

y_train = to_categorical(ytrain)

model = keras_model_sequential()

model = keras_model_sequential() %>%   
  layer_dense(units = 64, activation = "relu", input_shape = c(dim(dtm_train[2]))) %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = length(unique(ytrain)), activation = 'softmax')

#model %>% layer_dense(units = 20, activation = 'relu', input_shape = c(dim(dtm_train[2]))) %>% 
#layer_dropout(rate = 0.2) %>%
#layer_dense(units = 15, activation = 'relu', input_shape = c(dim(dtm_train[2]))) %>% 
#layer_dropout(rate = 0.2) %>%
#layer_dense(units = 10, activation = 'relu', input_shape = c(dim(dtm_train[2]))) %>% 
#layer_dropout(rate = 0.2) %>%
#layer_dense(units = length(unique(ytrain)), activation = 'softmax')

SGD = optimizer_sgd(lr = 0.01)
ADAM = optimizer_adam(lr = 0.001)
model %>% compile( loss = 'categorical_crossentropy', optimizer = SGD, metrics = c('accuracy'))

train_matrix = as.matrix(dtm_train)
history = model %>% fit (train_matrix, y_train, epochs = 100, batch_size = 100, validation_split = 0.2)

model %>% evaluate(train_matrix, y_train)
preds = predict_classes(object = model, x = train_matrix)  %>% as.vector()

test_matrix = as.matrix(dtm_test)

test$CRIMETYPE = predict_classes(object = model, x = test_matrix)  %>% as.vector()

test$CRIMETYPE[test$CRIMETYPE==0] <- "AGG"
test$CRIMETYPE[test$CRIMETYPE==1] <- "BTFV"
test$CRIMETYPE[test$CRIMETYPE==2] <- "BURG"
test$CRIMETYPE[test$CRIMETYPE==3] <- "GTA"
test$CRIMETYPE[test$CRIMETYPE==4] <- "ROBB"
test$CRIMETYPE[test$CRIMETYPE==5] <- "THEFT"

submission = test[,c("DR", "CRIMETYPE")]

write.csv(submission, "Layers128.csv")