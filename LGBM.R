# remotes::install_github("mlr-org/mlr3extralearners") # use if mlr3extralearners are not available
library(data.table)
library(mlr3)
library(mlr3extralearners)
library(mlr3tuning)
library(paradox)

DT <- fread('D:/Practice Codes/Akash Statistics/Credit Card AV/train_s3TEQDk.csv')
options(na.action = 'na.pass')

# Changing the blank Credit Product to 'Unknown' label
DT[Credit_Product=='',Credit_Product:='Unknown']
# DT[,Avg_Account_Balance:=log(Avg_Account_Balance+exp(1))]

# removing ID variable
DT[,ID:=NULL]

# Filtering out the categorical variables and converting to factor datatype
chr_cols <- names(Filter(f = is.character,DT))
DT[,chr_cols] <- lapply(DT[,chr_cols,with=FALSE], as.factor)

# Performing one-hot encoding
DT <- fastDummies::dummy_columns(DT,remove_first_dummy = T,remove_selected_columns = T)

# Converting the target variable into factor datatype 
DT[,Is_Lead:=factor(Is_Lead)]


# Splitting between Train and Validation Set
set.seed(1234)
sampleID <- sample(1:nrow(DT),size = 0.8*nrow(DT))
train_DT <- DT[sampleID,]
valid_DT <- DT[-sampleID,]

task <- TaskClassif$new(id = 'LGBM',backend = train_DT,
                        target = "Is_Lead",positive = '1')

task$col_roles$stratum <- task$col_roles$target
classifier <- lrn("classif.lightgbm",predict_type = 'prob')

search_space <- ps(
  objective = p_fct(c('binary')),
  learning_rate = p_dbl(0.1,0.2),
  num_leaves = p_int(30L,50L),
  max_depth = p_int(3L,10L),
  min_data_in_leaf = p_int(25L,60L),
  # min_sum_hessian_in_leaf = p_dbl(6,16),
  num_iterations = p_int(50L,100L),
  # metric= p_fct(c('auc')),
  feature_fraction = p_dbl(0.4,0.7),
  bagging_fraction = p_dbl(0.75,1.0)
)

stopping_criteria <- trm("evals",n_evals = 10)
tuner <- tnr("grid_search", resolution = 10) # "random_search"
# tuner <- tnr("random_search")

instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = classifier,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc"),
  search_space = search_space,
  terminator = stopping_criteria,
  store_benchmark_result = T
)

resultcv <- tuner$optimize(instance)
resultcv

classifier$param_set$values <- instance$result_learner_param_vals
lightgbm_model <- classifier$train(task = task)

valid_task <- TaskClassif$new(id = 'LGBM',backend = valid_DT,
                              target = "Is_Lead",positive = '1')

model_pred_train <- lightgbm_model$predict(task)
model_pred_test <- lightgbm_model$predict(valid_task)

train_result <- model_pred_train$score(list(msr("classif.recall"), 
                                            msr("classif.precision"), 
                                            msr("classif.acc"),
                                            msr("classif.auc")))

test_result <- model_pred_test$score(list(msr("classif.recall"),
                                          msr("classif.precision"), 
                                          msr("classif.acc"),
                                          msr("classif.auc")))


data.table(metric=c('Recall','Precision','Accuracy','AUC'),cbind(train_result,test_result))

