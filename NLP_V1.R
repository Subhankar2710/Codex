library(data.table)
library(dplyr)
library(keras)
library(caret)

DT <- fread('movie_review.csv')
DT[,.N,tag] #overview of the negative and positive comments

# Checking the distribution of the words
DT$text %>% strsplit(" ") %>% unique() %>% sapply(length) %>% summary()

# Defining a text vectorization layer
num_words <- 10000
max_length <- 50
text_vectorization <- layer_text_vectorization(max_tokens = num_words,
                                               output_sequence_length = max_length)

text_vectorization %>% adapt(DT$text)
get_vocabulary(text_vectorization)

# Checking the vectorization for 1st sentence
text_vectorization(matrix(DT$text[1],ncol=1))

# Train and test split
set.seed(1234)
sampleid <- sample.int(nrow(DT),size = 0.8*nrow(DT),replace = F)
train_DT <- DT[sampleid,]
test_DT <- DT[-sampleid,]
train_DT_y <- to_categorical(as.numeric(train_DT[,tag]=='pos'))
test_DT_y <- to_categorical(as.numeric(test_DT[,tag]=='pos'))

# Stacking the layers for the model sequentially
input <- layer_input(shape = c(1), dtype = "string")

output <- input %>%
  text_vectorization() %>%
  layer_embedding(input_dim = num_words+1,output_dim = 32) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 32,activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 16,activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2,activation = 'sigmoid')

model <- keras_model(inputs = input,outputs = output)
model %>% compile(optimizer = optimizer_adam(lr=0.001),
  loss = 'binary_crossentropy',
  metrics = list('accuracy'))
history <- model %>% fit(train_DT[,text],
                          train_DT_y,
                          epoch=10,
                          batch_size = 15,
                          validation_split =0.1)

plot(history)
model %>% evaluate(train_DT[,text],train_DT_y)

prob <- model %>% predict(train_DT[,text])
pred_class <- sapply(1:nrow(prob),function(x) ifelse(prob[x,1] < 0.5, 'pos','neg'))
confusionMatrix(data = factor(pred_class),reference = factor(train_DT[,tag]),
                mode='everything',positive = 'pos')

model %>% evaluate(test_DT[,text],test_DT_y)
prob_test <- model %>% predict(test_DT[,text])
pred_class_test <- sapply(1:nrow(prob_test),function(x) ifelse(prob_test[x,1] < 0.5, 'pos','neg'))
confusionMatrix(data = factor(pred_class_test),reference = factor(test_DT[,tag]),
                mode='everything',positive = 'pos')

saveRDS(model,'Text_Classifier_V1.rds')
