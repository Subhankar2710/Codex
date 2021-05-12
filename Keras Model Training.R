library(data.table)
library(caret)
library(keras)
library(fastDummies)
train <- fread('train_LZdllcl.csv',na.strings = '')
names(train)
train[,.N,is_promoted]


cat_vars <- Filter(f = is.character,x = train)
cat_vars[is.na(cat_vars)] <- 'unknown'

num_vars <- Filter(f = is.numeric,x = train)
num_vars[is.na(num_vars)] <- 0

train <- cbind(num_vars,cat_vars)

factor_vars <- c(names(cat_vars),'KPIs_met >80%','is_promoted','awards_won?')
train[,factor_vars] <- lapply(train[,factor_vars,with=FALSE],as.factor)
numeric_variables <- names(train[,!c(factor_vars,'employee_id'),with=FALSE])
train[,numeric_variables] <- lapply(train[,numeric_variables,with=FALSE],as.numeric)

scaling <- function(train, numeric_variables)
{
  DT <- train[,numeric_variables,with=FALSE] 
  DT <- scale(DT,center = colMeans(DT,na.rm = F),scale = apply(DT,2,sd))
  return(cbind(train[,!numeric_variables,with=FALSE],DT))
}
train <- scaling(train,numeric_variables)
train_dummy <- dummy_cols(train,remove_first_dummy = T)
train_dummy <- train_dummy[,!factor_vars,with=FALSE]
X_train <- as.matrix(train_dummy[,!c('is_promoted_1','employee_id'),with=FALSE])
Y_train <- to_categorical(train_dummy[,is_promoted_1])

f1score <- function(y_true,y_pred)
{
  return(f1score <- MLmetrics::F1_Score(y_true,y_pred))
}

model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(X_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr=0.0005), #adam
  metrics = c('AUC'))

model %>% fit(X_train, Y_train, 
              epochs = 7, 
              batch_size = 10,
              validation_split = 0.2,
              callbacks = callback_early_stopping(monitor = 'val_loss'))
model %>% evaluate(X_train,Y_train)
prob <-  model %>% predict(X_train)
prob <- data.table(prob)
prob[,class:=ifelse(V1>0.73,0,1)]

confusionMatrix(data=factor(prob[,class]),
                reference = factor(train[,is_promoted]),
                positive = '1',mode='everything')

saveRDS(model,'model_keras_20210508.rds')
# Test Data
test <- fread('test_2umaH9m.csv',na.strings = '')
cat_vars <- Filter(f = is.character,x = test)
cat_vars[is.na(cat_vars)] <- 'unknown'

num_vars <- Filter(f = is.numeric,x = test)
num_vars[is.na(num_vars)] <- 0

test <- cbind(num_vars,cat_vars)

factor_vars <- c(names(cat_vars),'KPIs_met >80%','awards_won?')
test[,factor_vars] <- lapply(test[,factor_vars,with=FALSE],as.factor)
numeric_variables <- names(test[,!c(factor_vars,'employee_id'),with=FALSE])
test[,numeric_variables] <- lapply(test[,numeric_variables,with=FALSE],as.numeric)
test <- scaling(test,numeric_variables)
test_dummy <- dummy_cols(test,remove_first_dummy = T)
test_dummy <- test_dummy[,!factor_vars,with=FALSE]
X_test <- as.matrix(test_dummy[,!c('is_promoted_1','employee_id'),with=FALSE])
prob_test <-  model %>% predict(X_test)
prob_test <- data.table(prob_test)
prob_test[,class:=ifelse(V1>0.73,0,1)]

result <- data.table(employee_id =test[,employee_id], is_promoted = prob_test$class)
fwrite(result,'Submission2_20210508.csv')
