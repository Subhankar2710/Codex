library(data.table)
library(xgboost)
library(fastDummies)
train <- fread('train_NIR5Yl1.csv')
options(scipen = 99)

# Simple Data Exploration
train[,.N,.(Username)]
train[,.(Tot_Upvotes=mean(Upvotes)),Tag][order(Tag)]
# plot(x = log(train[,Views]+1),y = log(train[,Upvotes]+1))
cor(train[Upvotes<10,Views],train[Upvotes<10,Upvotes])
names(train)
EDA1 <- train[,list(Tot_Upvotes=mean(Upvotes),Tot_Rep=mean(Reputation)),Username][order(-Tot_Upvotes)][Tot_Upvotes>1000]
cor(EDA1$Tot_Upvotes,EDA1$Tot_Rep)

Reputation_QT <- as.vector(quantile(train$Reputation))
train[,Reputation_band := as.numeric(cut(Reputation,breaks=as.vector(quantile(Reputation)),
                                     labels = 1:4))][is.na(Reputation_band),Reputation_band:=0]
quantile_views_by_username <- as.vector(train[,.(Views=mean(Views)),.(Username,Tag,Reputation_band)][,quantile(x = Views)])
Username_tags <- train[,.(Views=mean(Views)),.(Username,Tag,Reputation_band)]
Username_tags[,User_Popularity:=as.numeric(cut(Views, breaks=quantile_views_by_username,
                                               labels=c(1:4)))][is.na(User_Popularity),User_Popularity:=1]

# Outlier Capping
train[,c('Reputation','Views','Upvotes'):=list(ifelse(Reputation>as.numeric(quantile(Reputation,0.95)),
                                                      as.numeric(quantile(Reputation,0.95)),Reputation),
                                               ifelse(Views>as.numeric(quantile(Views,0.95)),
                                                      as.numeric(quantile(Views,0.95)),Views),
                                               ifelse(Upvotes>as.numeric(quantile(Upvotes,0.95)),
                                                      as.numeric(quantile(Upvotes,0.95)),Upvotes))]

# Converting Upvotes and Views in Logarithmic Scale
train[,Views:=log(Views+1)]
train[,Upvotes:=log(Upvotes+1)]
train[,Reputation:=log(Reputation+1)]

train <- merge(train,Username_tags[,.(Username,User_Popularity,Tag)],by=c('Username','Tag'),all.x = T)
str(train)
train_DT <- dummy_cols(train,remove_selected_columns = F)
train_DT <- train_DT[,!c('Username','ID','Tag'),with=FALSE]

# Data Splitting into Actual Training and Validation Set
set.seed(1000)
sampleID <- sample(1:nrow(train_DT),size = 0.8*nrow(train_DT),replace = FALSE)
training <- train_DT[sampleID,]
validation <- train_DT[-sampleID,]

# XGBoost Modelling:
options(na.action = 'na.pass')
training_matrix <-model.matrix(Upvotes ~.-1, data = training)
validation_matrix <-model.matrix(Upvotes ~.-1, data = validation)
full_train_matrix <- model.matrix(Upvotes~.-1, data = train_DT)

dtrain <- xgb.DMatrix(data = training_matrix, label = training[,Upvotes]) 
dvalid <- xgb.DMatrix(data = validation_matrix, label = validation[,Upvotes])
dtrain_full <- xgb.DMatrix(data=full_train_matrix, label = train_DT[,Upvotes])

# Take start time to measure time of random search algorithm
start.time <- Sys.time()

# Create empty lists
lowest_error_list = list()
parameters_list = list()

# Create 100 rows with random hyperparameters
set.seed(1234)
for (iter in 1:100){
  param <- list(booster = sample(c('gbtree'),1),
                objective = "reg:squaredlogerror",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}
# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 200 XGBoost-models
for (row in 1:nrow(parameters_df)){
  set.seed(20)
  mdcv <- xgb.train(data=dtrain,
                    booster = parameters_df$booster[row],
                    objective = parameters_df$objective[row],
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= 100,
                    eval_metric = "rmse",
                    early_stopping_rounds= 30,
                    print_every_n = 100,
                    watchlist = list(train= dtrain, val= dvalid)
  )
  lowest_error <- as.data.frame(min(mdcv$evaluation_log$val_rmse))
  lowest_error_list[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = data.table(cbind(lowest_error_df, parameters_df))
setnames(randomsearch,'min(mdcv$evaluation_log$val_rmse)','validation_error')

# Quickly display highest val error
randomsearch[which(abs(randomsearch$validation_error)==min(abs(randomsearch$validation_error))),]
optim <- randomsearch[which(abs(randomsearch$validation_error)==min(abs(randomsearch$validation_error))),][1,]

# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# Model Fitting
xgb_tuned <- xgb.train(data=dtrain_full,
                       booster = optim$booster,
                       objective = optim$objective,
                       max_depth = optim$max_depth,
                       eta = optim$eta,
                       subsample = optim$subsample,
                       colsample_bytree = optim$colsample_bytree,
                       min_child_weight = optim$min_child_weight,
                       nrounds= 300,
                       eval_metric = "rmse",
                       print_every_n = 100)

Upvotes_predicted <- predict(xgb_tuned,newdata = dtrain_full)
train[,Upvotes_predicted:=Upvotes_predicted]
evaluation_matrix <- train[,list(Upvotes=exp(Upvotes)-1,
                                 Upvotes_predicted=exp(Upvotes_predicted)-1)]
# evaluation_matrix <- train[,list(Upvotes=Upvotes,
#                                  Upvotes_predicted=Upvotes_predicted)]
evaluation_matrix[Upvotes_predicted<0,Upvotes_predicted:=0]
evaluation_matrix[,abs_diff:=abs(Upvotes-Upvotes_predicted)]
rmse <- (mean(evaluation_matrix[,abs_diff]^2))^0.5
rmse
saveRDS(xgb_tuned,'XGB_20210509.rds')

# Fitting in Test Data
test <- fread('test_8i3B3FC.csv')
test[,Reputation_band := as.numeric(cut(Reputation,breaks=Reputation_QT,
                                         labels = 1:4))][is.na(Reputation_band),Reputation_band:=0]
Username_tags_test <- test[,.(Views=mean(Views)),.(Username,Tag,Reputation_band)]
Username_tags_test[,User_Popularity:=as.numeric(cut(Views, breaks=quantile_views_by_username,
                                               labels=c(1:4)))][is.na(User_Popularity),User_Popularity:=1]

test <- merge(test,Username_tags_test[,.(Username,User_Popularity,Tag)],by=c('Username','Tag'),all.x = T)
test <- test[!duplicated(ID)]
test[,Views:=log(Views+1)]
# test[,Upvotes:=log(Upvotes+1)]
test[,Reputation:=log(Reputation+1)]
test_DT <- dummy_cols(test,remove_selected_columns = F)
test_DT <- test_DT[,!c('Username','ID','Tag'),with=FALSE]
full_test_matrix <- model.matrix(~.-1,data = test_DT)
dtest_full <- xgb.DMatrix(data=full_test_matrix)
pred_test <- exp(predict(xgb_tuned,dtest_full))-1
# pred_test <- predict(xgb_tuned,dtest_full)

test_result <- data.table(ID=test[,ID],Upvotes=pred_test)
fwrite(test_result,'XGB_Result_20210509.csv')
