library(text2vec)
library(tensorflow)
library(keras)

train = read.csv("C:/Users/daxpa/downloads/train_comp2.csv")
test = read.csv("C:/Users/daxpa/downloads/test_comp2.csv")

train$NARRATIVE = as.character(train$NARRATIVE)
txt = as.character(train$NARRATIVE)

it = itoken(txt, tolower, word_tokenizer, n_chunks = 10)
vocab = create_vocabulary(it)

vocab = prune_vocabulary(vocab, term_count_min = 100, doc_proportion_max = 0.25)
word_vectorizer = vocab_vectorizer(vocab)

it_train = itoken(train$NARRATIVE, preprocessor = tolower, tokenizer = word_tokenizer
, ids = train$DR, progressbar = FALSE)

dtm_train = create_dtm(it_train, word_vectorizer)
dim(dtm_train)

vocab
label_fac = as.numeric(as.factor(train$CRIMETYPE))
label_fac = label_fac - 1

y_train = label_fac
ytrain = label_fac

y_train = to_categorical(ytrain)

model = keras_model_sequential()
model %>% 
  layer_dense(units =  20, activation = 'relu', input_shape = c(dim(dtm_train)[2])) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = length(unique(ytrain)), activation = 'softmax')

model %>% compile(
  
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
  
)
  