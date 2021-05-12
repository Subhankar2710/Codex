library(EBImage)
library(keras)
library(caret)

# Loading the images
files <- list.files('Dogs and Cats/')
images <- lapply(files,function(x) readImage(files = paste0('Dogs and Cats/',x)))

# Resizing the images
image_resized <- lapply(1:length(images),function(x) resize(images[[x]],225,225))
# Reshaping the images
image_resized <- lapply(1:length(image_resized),function(x) array_reshape(image_resized[[x]],c(225,225,3)))
image_resized <- do.call(rbind,image_resized)

# Creating Test and Train Split
training_x <- image_resized[3:32,]
training_y <- rep(0:1,each=15)

test_x <- image_resized[c(1,2,33,34),]
test_y <- rep(0:1,each=2)

# One Hot Encoding
trainLabels <- to_categorical(training_y)
testLabels <- to_categorical(test_y)

# Model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 512, activation = 'relu', input_shape = c(151875)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)
# Compile
model %>%
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_adam(lr = 0.0001),
          metrics = c('accuracy'))
# Fit Model
model %>%
  fit(training_x,
      trainLabels,
      epochs = 50,
      batch_size = 10,
      validation_split = 0.2)

model %>% evaluate(training_x,trainLabels)
predicted_train <- model %>% predict_classes(training_x)
confusionMatrix(data=factor(predicted_train),reference = factor(training_y),
                mode = 'everything',positive = '1')

predicted_test <- model %>% predict_classes(test_x)
confusionMatrix(data=factor(predicted_test),reference = factor(test_y),
                mode = 'everything',positive = '1')
model %>% evaluate(test_x,testLabels)
saveRDS(model,'ImageRcg1_obj.rds')
