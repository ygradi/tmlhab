# pipeline.R
# Minimal multiclass pipeline

config <- list(
  data_path    = "input_data.csv",
  id_col       = "id",
  split_col    = "tt",
  label_col    = "HER23",    # three-class label column
  output_dir   = "results",
  preserve_ids = FALSE,
  seed         = 2025,
  run_synthetic = FALSE
)

required_pkgs <- c("dplyr","tidyr","ggplot2","caret","pROC","glmnet",
                   "kernlab","nnet","xgboost","shapviz","kernelshap",
                   "readr","digest","patchwork")

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) stop("Install required packages: ", paste(missing_pkgs, collapse = ", "))

has_fastshap <- requireNamespace("fastshap", quietly = TRUE)
has_themis <- requireNamespace("themis", quietly = TRUE)

library(dplyr); library(tidyr); library(ggplot2)
library(caret); library(pROC); library(glmnet)
library(kernlab); library(nnet); library(xgboost)
library(shapviz); library(kernelshap); library(readr); library(digest); library(patchwork)

dir.create(config$output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(config$seed)

generate_synthetic_data <- function(path, n = 200, seed = 2025) {
  set.seed(seed)
  df <- data.frame(
    id = sprintf("ID%04d", seq_len(n)),
    tt = sample(c(1,0), n, replace = TRUE, prob = c(0.7,0.3)),
    HER23 = sample(0:2, n, replace = TRUE)
  )
  for (i in seq_len(20)) df[[paste0("X", i)]] <- rnorm(n) + as.numeric(df$HER23) * runif(1, -0.5, 0.5)
  write.csv(df, path, row.names = FALSE)
  invisible(df)
}

if (!file.exists(config$data_path)) {
  if (isTRUE(config$run_synthetic)) generate_synthetic_data(config$data_path, n = 300, seed = config$seed)
  else stop("Data file not found: ", config$data_path)
}

data <- readr::read_csv(config$data_path, show_col_types = FALSE)

required_cols <- c(config$id_col, config$split_col, config$label_col)
missing_cols <- setdiff(required_cols, names(data))
if (length(missing_cols) > 0) stop("Missing required cols: ", paste(missing_cols, collapse = ", "))

rownames(data) <- as.character(data[[config$id_col]])

if (!all(data[[config$label_col]] %in% c(0,1,2))) stop("Label must be 0/1/2")
data[[config$label_col]] <- factor(c("zero","low","posi")[data[[config$label_col]] + 1L],
                                   levels = c("zero","low","posi"), ordered = FALSE)

train_idx <- which(data[[config$split_col]] == 1)
test_idx  <- which(data[[config$split_col]] == 0)

X_train <- data[train_idx, setdiff(names(data), c(config$id_col, config$split_col, config$label_col)), drop = FALSE]
X_test  <- data[test_idx,  setdiff(names(data), c(config$id_col, config$split_col, config$label_col)), drop = FALSE]
y_train <- data[[config$label_col]][train_idx]
y_test  <- data[[config$label_col]][test_idx]
label_var <- config$label_col

final_features <- NULL
if (is.null(final_features) || (length(final_features)==1 && final_features=="selected_features")) {
  cand <- names(X_train)[vapply(X_train, is.numeric, logical(1))]
  if (length(cand) == 0) stop("No numeric features detected; set final_features")
  final_features <- cand
}

preproc <- preProcess(X_train[, final_features, drop = FALSE], method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train[, final_features, drop = FALSE])
X_test_scaled  <- predict(preproc, X_test[, final_features, drop = FALSE])

cor_mat <- cor(X_train_scaled, use = "pairwise.complete.obs")
high_idx <- caret::findCorrelation(cor_mat, cutoff = 0.9)
if (length(high_idx) > 0) {
  keep_idx <- setdiff(seq_len(ncol(cor_mat)), high_idx)
  X_train_red <- X_train_scaled[, keep_idx, drop = FALSE]
  X_test_red  <- X_test_scaled[, keep_idx, drop = FALSE]
  final_features <- final_features[keep_idx]
} else {
  X_train_red <- X_train_scaled; X_test_red <- X_test_scaled
}

traindata <- as.data.frame(cbind(y = y_train, X_train_red), stringsAsFactors = FALSE)
testdata  <- as.data.frame(cbind(y = y_test,  X_test_red),  stringsAsFactors = FALSE)
names(traindata)[1] <- label_var; names(testdata)[1] <- label_var

orig_ids_all <- as.character(data[[config$id_col]])
anon_ids_all <- vapply(orig_ids_all, function(x) substr(digest(x, algo = "sha256"), 1, 12), character(1))
id_map <- data.frame(orig_id = orig_ids_all, anon_id = anon_ids_all, stringsAsFactors = FALSE)

if (!isTRUE(config$preserve_ids)) {
  rownames(traindata) <- id_map$anon_id[train_idx]
  rownames(testdata)  <- id_map$anon_id[test_idx]
} else {
  rownames(traindata) <- rownames(data)[train_idx]
  rownames(testdata)  <- rownames(data)[test_idx]
}

cv_control <- trainControl(
  method = "repeatedcv", number = 5, repeats = 3,
  sampling = if (has_themis) "smote" else NULL,
  classProbs = TRUE, summaryFunction = caret::multiClassSummary,
  verboseIter = FALSE, savePredictions = "all", returnResamp = "all", search = "grid"
)

calculate_multiclass_metrics <- function(truth, pred_class, probs) {
  cm <- caret::confusionMatrix(pred_class, truth)
  byClass <- cm$byClass
  mean_metric <- function(metric_name) {
    if (is.null(byClass)) return(NA_real_)
    if (is.matrix(byClass)) {
      if (metric_name %in% colnames(byClass)) return(mean(byClass[, metric_name], na.rm = TRUE))
      else return(NA_real_)
    } else {
      if (metric_name %in% names(byClass)) return(as.numeric(byClass[metric_name]))
      else return(NA_real_)
    }
  }
  auc_val <- tryCatch({
    probs_mat <- as.matrix(probs); probs_mat <- probs_mat[, levels(truth), drop = FALSE]
    as.numeric(pROC::multiclass.roc(truth, probs_mat)$auc)
  }, error = function(e) NA_real_)
  tibble::tibble(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Kappa = as.numeric(cm$overall["Kappa"]),
    Sensitivity = mean_metric("Sensitivity"),
    Specificity = mean_metric("Specificity"),
    Precision = mean_metric("Precision"),
    F1 = mean_metric("F1"),
    AUC = auc_val
  )
}

ncol_data <- ncol(traindata) - 1L
algorithms <- list(
  list(name = "DecisionTree", method = "rpart", tuneGrid = expand.grid(cp = 10^seq(-5, -1, length.out = 5))),
  list(name = "RandomForest", method = "rf", tuneGrid = expand.grid(mtry = unique(pmax(2, floor(seq(2, max(2, ncol_data %/% 2), length.out = 3)))))),
  list(name = "XGBoost", method = "xgbTree", tuneGrid = expand.grid(nrounds = c(100), max_depth = c(3), eta = c(0.1), gamma = 0, colsample_bytree = 0.7, min_child_weight = 1, subsample = 0.7)),
  list(name = "MultinomialLogistic", method = "multinom", tuneGrid = data.frame(decay = 0))
)

all_train_metrics <- list(); all_test_metrics <- list(); all_models <- list(); training_times <- list(); best_tune_params <- list()

dir.create(file.path(config$output_dir, "Feature_importance_ranking"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(config$output_dir, "CV_Detailed_Results"), showWarnings = FALSE, recursive = TRUE)

for (algo in algorithms) {
  algo_name <- algo$name
  set.seed(config$seed)
  train_x <- traindata[, !names(traindata) %in% label_var, drop = FALSE]
  train_y <- traindata[[label_var]]
  train_args <- list(x = train_x, y = train_y, method = algo$method, trControl = cv_control, metric = "Accuracy")
  if (!is.null(algo$tuneGrid)) train_args$tuneGrid <- algo$tuneGrid
  start_time <- Sys.time()
  model <- tryCatch(do.call(caret::train, train_args), error = function(e) { message("Train failed: ", e$message); NULL })
  training_times[[algo_name]] <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  if (is.null(model)) {
    all_train_metrics[[algo_name]] <- tibble::tibble(Accuracy=NA, Kappa=NA, Sensitivity=NA, Specificity=NA, Precision=NA, F1=NA, AUC=NA)
    all_test_metrics[[algo_name]]  <- all_train_metrics[[algo_name]]; best_tune_params[[algo_name]] <- NA; all_models[[algo_name]] <- NULL
    next
  }
  best_tune_params[[algo_name]] <- model$bestTune
  all_models[[algo_name]] <- model
  saveRDS(model, file.path(config$output_dir, paste0(algo_name, "_model.rds")))
  
  predictors <- setdiff(names(traindata), label_var)
  pred_train_prob <- tryCatch(predict(model, newdata = traindata[, predictors, drop = FALSE], type = "prob"), error = function(e) NULL)
  pred_test_prob  <- tryCatch(predict(model, newdata = testdata[, predictors, drop = FALSE], type = "prob"), error = function(e) NULL)
  train_pred_class <- tryCatch(predict(model, newdata = traindata[, predictors, drop = FALSE]), error = function(e) factor(rep(NA, nrow(traindata))))
  test_pred_class  <- tryCatch(predict(model, newdata = testdata[, predictors, drop = FALSE]), error = function(e) factor(rep(NA, nrow(testdata))))
  
  all_train_metrics[[algo_name]] <- calculate_multiclass_metrics(traindata[[label_var]], train_pred_class, pred_train_prob)
  all_test_metrics[[algo_name]]  <- calculate_multiclass_metrics(testdata[[label_var]],  test_pred_class,  pred_test_prob)
  
  vi <- tryCatch(varImp(model), error = function(e) NULL)
  if (!is.null(vi) && !is.null(vi$importance)) {
    imp_df <- data.frame(Variable = rownames(vi$importance), Importance = vi$importance[,1], stringsAsFactors = FALSE)
    p <- ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) + geom_col(fill = "steelblue") + coord_flip() + theme_minimal()
    ggsave(file.path(config$output_dir, "Feature_importance_ranking", paste0(algo_name, "_importance.pdf")), p, width = 6, height = 4)
  }
}

train_summary <- bind_rows(lapply(names(all_train_metrics), function(n) mutate(all_train_metrics[[n]], Model = n))) %>% select(Model, everything())
test_summary  <- bind_rows(lapply(names(all_test_metrics), function(n)  mutate(all_test_metrics[[n]],  Model = n))) %>% select(Model, everything())
write.csv(train_summary, file.path(config$output_dir, "All_Models_Train_Performance.csv"), row.names = FALSE)
write.csv(test_summary,  file.path(config$output_dir, "All_Models_Test_Performance.csv"),  row.names = FALSE)

selected_model <- "MultinomialLogistic"
if (selected_model %in% names(all_models) && !is.null(all_models[[selected_model]])) {
  best_model <- all_models[[selected_model]]
  result_dir <- file.path(config$output_dir, paste0("Best_Model_", selected_model)); dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)
  saveRDS(best_model, file.path(result_dir, paste0(selected_model, "_Final_Model.rds")))
  
  predictors <- setdiff(names(traindata), label_var)
  train_pred_prob  <- tryCatch(predict(best_model, newdata = traindata[, predictors, drop = FALSE], type = "prob"), error = function(e) NULL)
  test_pred_prob   <- tryCatch(predict(best_model, newdata = testdata[, predictors, drop = FALSE],  type = "prob"), error = function(e) NULL)
  train_pred_class <- tryCatch(predict(best_model, newdata = traindata[, predictors, drop = FALSE]), error = function(e) NULL)
  test_pred_class  <- tryCatch(predict(best_model, newdata = testdata[, predictors, drop = FALSE]), error = function(e) NULL)
  
  write.csv(calculate_multiclass_metrics(traindata[[label_var]], train_pred_class, train_pred_prob), file.path(result_dir, "Train_Performance.csv"), row.names = FALSE)
  write.csv(calculate_multiclass_metrics(testdata[[label_var]],  test_pred_class,  test_pred_prob),  file.path(result_dir, "Test_Performance.csv"),  row.names = FALSE)
  
  shap_dir <- file.path(result_dir, "SHAP"); dir.create(shap_dir, showWarnings = FALSE, recursive = TRUE)
  shap_ok <- FALSE
  X_raw <- traindata[, predictors, drop = FALSE]
  X_for_shap <- if (is.null(best_model$preProcess)) X_raw else predict(best_model$preProcess, X_raw)
  p_test <- tryCatch(as.matrix(predict(best_model, newdata = X_for_shap[1:min(10,nrow(X_for_shap)), , drop = FALSE], type = "prob")), error = function(e) NULL)
  if (!is.null(p_test) && all(colnames(p_test) == levels(traindata[[label_var]]))) {
    bg_sample <- X_for_shap[sample(nrow(X_for_shap), min(20, nrow(X_for_shap))), , drop = FALSE]
    shap_res <- tryCatch(kernelshap::kernelshap(best_model, X = X_for_shap[1:min(30,nrow(X_for_shap)), , drop = FALSE], bg_X = bg_sample,
                                                pred_fun = function(model, newdata) as.matrix(predict(model, newdata = as.data.frame(newdata), type = "prob"))),
                         error = function(e) NULL)
    if (!is.null(shap_res)) {
      sv <- shapviz::shapviz(shap_res, X = traindata[, predictors, drop = FALSE])
      png(file.path(shap_dir, "shap_importance.png"), width = 800, height = 600); print(shapviz::sv_importance(sv)); dev.off()
      shap_ok <- TRUE
    }
  }
  if (!shap_ok && has_fastshap) {
    for (cls in levels(traindata[[label_var]])) {
      pred_cls <- function(object, newdata) predict(object, newdata = as.data.frame(newdata), type = "prob")[, cls]
      shap_mat <- tryCatch(fastshap::explain(object = best_model, X = X_for_shap, pred_wrapper = pred_cls, nsim = 100), error = function(e) NULL)
      if (!is.null(shap_mat)) write.csv(shap_mat, file.path(shap_dir, paste0("fastshap_class_", cls, ".csv")), row.names = FALSE)
    }
  }
}

train_with_id <- data[train_idx, , drop = FALSE]; test_with_id <- data[test_idx, , drop = FALSE]
train_with_id$anon_id <- id_map$anon_id[train_idx]; test_with_id$anon_id <- id_map$anon_id[test_idx]
train_with_id <- train_with_id[match(rownames(traindata), if (!isTRUE(config$preserve_ids)) train_with_id$anon_id else train_with_id[[config$id_col]]), , drop = FALSE]
test_with_id  <- test_with_id[match(rownames(testdata),  if (!isTRUE(config$preserve_ids)) test_with_id$anon_id  else test_with_id[[config$id_col]]), , drop = FALSE]

if (exists("train_pred_class") && exists("test_pred_class")) {
  train_pred_num <- if (!is.null(train_pred_class)) as.numeric(train_pred_class) - 1 else rep(NA_integer_, nrow(traindata))
  test_pred_num  <- if (!is.null(test_pred_class))  as.numeric(test_pred_class)  - 1 else rep(NA_integer_, nrow(testdata))
  prob_cols <- if (exists("test_pred_prob") && !is.null(test_pred_prob)) colnames(test_pred_prob) else levels(data[[config$label_col]])
  prob_names <- paste0("Prob_", prob_cols)
  train_out <- data.frame(ID = rownames(traindata), tt = 1, True = if (!is.null(train_with_id[[config$label_col]])) as.numeric(train_with_id[[config$label_col]]) - 1 else NA, Pred = train_pred_num, stringsAsFactors = FALSE)
  if (exists("train_pred_prob") && !is.null(train_pred_prob)) train_out <- cbind(train_out, setNames(as.data.frame(train_pred_prob), prob_names))
  test_out <- data.frame(ID = rownames(testdata), tt = 0, True = if (!is.null(test_with_id[[config$label_col]])) as.numeric(test_with_id[[config$label_col]]) - 1 else NA, Pred = test_pred_num, stringsAsFactors = FALSE)
  if (exists("test_pred_prob") && !is.null(test_pred_prob)) test_out <- cbind(test_out, setNames(as.data.frame(test_pred_prob), prob_names))
  all_pred <- rbind(train_out, test_out); all_pred <- all_pred[order(all_pred$ID), ]
  write.csv(all_pred, file.path(config$output_dir, "Predictions_All_Samples.csv"), row.names = FALSE)
}

message("Done. Outputs (if any) in: ", normalizePath(config$output_dir))