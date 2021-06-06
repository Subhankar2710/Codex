library(data.table)
library(fastDummies)
DT <- fread('train_s3TEQDk.csv')
DT[Credit_Product=='',Credit_Product:='Unknown']
DT[,Avg_Account_Balance:=log(Avg_Account_Balance+exp(1))]
DT[,Is_Lead:=factor(Is_Lead)]
DT[,ID:=NULL]
chr_cols <- names(Filter(f = is.character,DT))
DT[,chr_cols] <- lapply(DT[,chr_cols,with=FALSE], as.factor)

num_vars <- names(Filter(f = is.numeric,x = DT))
scaling <- function(train, numeric_variables)
{
  DT <- train[,numeric_variables,with=FALSE] 
  DT <- scale(DT,center = colMeans(DT,na.rm = F),scale = apply(DT,2,sd))
  return(cbind(train[,!numeric_variables,with=FALSE],DT))
}
DT <- scaling(DT,num_vars)
DT <- dummy_cols(DT,remove_first_dummy = T)
DT <- DT[,!chr_cols,with=FALSE]

# Splitting between Train and Validation Set
set.seed(100)
sampleID <- sample(1:nrow(DT),size = 0.8*nrow(DT))
train_DT <- DT[sampleID,]
valid_DT <- DT[-sampleID,]

options(na.action = 'na.pass')

library(keras)
library(kerastuneR)
library(dplyr)
train_X <- as.matrix(train_DT[,!c('Is_Lead','Is_Lead_1'),with=FALSE])
train_Y <- to_categorical(train_DT[,Is_Lead])

build_model <- function(hp)
{
  model <- keras_model_sequential() %>% 
    layer_dense(units = hp$Int('units',min_value=128,max_value=512,step=64),
                activation = hp$Choice('activation',values= c('relu','tanh')),
                input_shape = ncol(train_X)) %>% 
    layer_dropout(rate = hp$Float('rate',min_value=0.2,max_value=0.4,step=0.05)) %>% 
    layer_dense(units = hp$Int('units',min_value=32,max_value=128,step=32),
                activation = hp$Choice('activation',values= c('relu','tanh'))) %>%
    layer_dropout(rate = hp$Float('rate',min_value=0.1,max_value=0.3,step=0.05)) %>%
    layer_dense(units = 2, activation = 'sigmoid')
  
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(lr=hp$Choice('lr',values=c(0.15,0.1,0.075,0.05,0.025,0.01))),
    metrics = 'AUC')
  return(model)
}
tuner = RandomSearch(
  build_model,
  objective = Objective(name='val_auc',direction = 'max'),
  max_trials = 5,
  executions_per_trial = 3)

tuner %>% search_summary()
tuner %>% fit_tuner(train_X,train_Y,
                    epochs=3, 
                    batch_size = 10,
                    validation_split = 0.2)

kerastuneR::plot_tuner(tuner = tuner)
best_model <- tuner %>% get_best_models(1)
best_model <- best_model[[1]]

model <- keras_model_sequential() %>% 
  layer_dense(units = 96,
              activation = 'relu',
              input_shape = ncol(train_X)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 96,
              activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr=0.01),
  metrics = 'AUC')
model %>% fit(train_X,train_Y,
                   epochs=5,
                   batch_size=10,
                   validation_split =0.2)

model %>% evaluate(train_X,train_Y)
pred <- model %>% predict(train_X)
pROC::roc(train_DT[,Is_Lead],pred[,1])

# Test AUC Check
test_X <- as.matrix(valid_DT[,!c('Is_Lead','Is_Lead_1'),with=FALSE])
test_Y <- to_categorical(valid_DT[,Is_Lead])
pROC::roc(valid_DT[,Is_Lead],as.matrix(model %>% predict(test_X))[,2])
