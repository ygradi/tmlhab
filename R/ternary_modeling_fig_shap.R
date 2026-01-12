# Ternary classification Modeling & Visualization & SHAP

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(pROC)
library(rpart.plot)
library(glmnet)
library(kernlab)
library(nnet)
library(xgboost)
library(shapviz)
library(kernelshap)
library(readr)
library(gridExtra)
library(patchwork)

set.seed(123)

# Paths and inputs
data_file <- "TT_features.csv"
out_root  <- "."
dir.create(out_root, showWarnings = FALSE)

if (!file.exists(data_file)) stop("Data file not found: ", data_file)
data <- read_csv(data_file, col_types = cols(.default = "n"))

required_cols <- c("id", "tt", "HER23")
if (any(!required_cols %in% names(data))) stop("Missing required columns: ", paste(setdiff(required_cols, names(data)), collapse = ", "))

rownames(data) <- data$id

# Convert label
data$HER23 <- factor(ifelse(data$HER23 == 0, "zero", ifelse(data$HER23 == 1, "low", "posi")),
                     levels = c("zero", "low", "posi"))

# Split
train_idx <- which(data$tt == 1)
test_idx  <- which(data$tt == 0)
response_var <- "HER23"

# Preselected features
final_features <- c("X1","X2","X3","X4","X5")
missing_feats <- setdiff(final_features, names(data))
if (length(missing_feats) > 0) stop("Missing selected features: ", paste(missing_feats, collapse = ", "))

# Prepare raw feature matrices
X_train_raw <- data[train_idx, final_features, drop = FALSE]
X_test_raw  <- data[test_idx,  final_features, drop = FALSE]
rownames(X_train_raw) <- data$id[train_idx]
rownames(X_test_raw)  <- data$id[test_idx]

X_train_red <- X_train_raw
X_test_red  <- X_test_raw
feature_names_red <- final_features

# Remove high-correlation features (|r| > 0.9)
if (ncol(X_train_red) > 1) {
  cor_mat <- cor(X_train_red, use = "pairwise.complete.obs")
  high_cor_idx <- findCorrelation(cor_mat, cutoff = 0.9)
  if (length(high_cor_idx) > 0) {
    keep_idx <- setdiff(seq_len(ncol(X_train_red)), high_cor_idx)
    X_train_red <- X_train_red[, keep_idx, drop = FALSE]
    X_test_red  <- X_test_red[, keep_idx, drop = FALSE]
    feature_names_red <- feature_names_red[keep_idx]
  }
}

traindata <- data[train_idx, response_var, drop = FALSE]
traindata <- cbind(traindata, X_train_red)
rownames(traindata) <- data$id[train_idx]

testdata <- data[test_idx, response_var, drop = FALSE]
testdata <- cbind(testdata, X_test_red)
rownames(testdata) <- data$id[test_idx]

# Train control
cv_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  verboseIter = TRUE,
  savePredictions = "all",
  returnResamp = "all",
  search = "grid"
)

calculate_multiclass_metrics <- function(truth, pred_class, probs) {
  cm <- confusionMatrix(pred_class, truth)
  byclass <- cm$byClass
  safe_mean <- function(x) if (is.null(x)) NA_real_ else mean(x, na.rm = TRUE)
  metrics <- tibble::tibble(
    Accuracy    = cm$overall["Accuracy"][[1]],
    Kappa       = cm$overall["Kappa"][[1]],
    Sensitivity = safe_mean(byclass[, "Sensitivity"]),
    Specificity = safe_mean(byclass[, "Specificity"]),
    Precision   = safe_mean(byclass[, "Precision"]),
    F1          = safe_mean(byclass[, "F1"])
  )
  auc_val <- tryCatch(as.numeric(multiclass.roc(truth, as.matrix(probs))$auc), error = function(e) NA_real_)
  metrics$AUC <- auc_val
  metrics
}

# Algorithms and tuning grids
ncol_data <- ncol(traindata) - 1L
algorithms <- list(
  list(name = "DecisionTree", method = "rpart", tuneGrid = expand.grid(cp = 10^seq(-5, -1, length.out = 10))),
  list(name = "RandomForest", method = "rf", tuneGrid = expand.grid(mtry = seq(2, max(2, ncol_data %/% 2), length.out = 5))),
  list(name = "XGBoost", method = "xgbTree", preProcess = c("center", "scale"),
       tuneGrid = expand.grid(nrounds = c(100,200), max_depth = c(3,6), eta = c(0.01,0.1), gamma = 0,
                              colsample_bytree = c(0.6,0.8), min_child_weight = 1, subsample = c(0.7,0.8))),
  list(name = "Lasso", method = "glmnet", preProcess = c("center", "scale"),
       tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4,0,length.out = 20))),
  list(name = "Ridge", method = "glmnet", preProcess = c("center", "scale"),
       tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4,0,length.out = 20))),
  list(name = "NeuralNetwork", method = "nnet", preProcess = c("center", "scale"), MaxNWts = 10000,
       tuneGrid = expand.grid(size = c(5,10,20), decay = c(0.001,0.01,0.1))),
  list(name = "LinearRegression", method = "multinom", preProcess = c("center", "scale"), tuneGrid = data.frame(decay = 0)),
  list(name = "KNN", method = "knn", preProcess = c("center", "scale"), tuneGrid = expand.grid(k = seq(3,21,by = 2))),
  list(name = "NaiveBayes", method = "naive_bayes", preProcess = c("center", "scale"),
       tuneGrid = expand.grid(usekernel = c(TRUE, FALSE), adjust = c(0.5,1,1.5), laplace = c(0,0.5,1)))
)

# Storage
all_train_metrics <- list()
all_test_metrics  <- list()
all_models        <- list()
training_times    <- list()
best_tune_params  <- list()
cv_results_list   <- list()

dir.create(file.path(out_root, "Feature_importance_ranking"), showWarnings = FALSE)
dir.create(file.path(out_root, "CV_Detailed_Results"), showWarnings = FALSE)

total_start <- Sys.time()

for (algo in algorithms) {
  algo_name <- algo$name
  start_time <- Sys.time()
  safe_run <- tryCatch({
    args <- list(
      x = traindata[, !names(traindata) %in% response_var],
      y = traindata[[response_var]],
      method = algo$method,
      trControl = cv_control,
      metric = "Accuracy",
      maximize = TRUE
    )
    if (!is.null(algo$tuneGrid))   args$tuneGrid   <- algo$tuneGrid
    if (!is.null(algo$preProcess)) args$preProcess <- algo$preProcess
    if (!is.null(algo$MaxNWts))    args$MaxNWts    <- algo$MaxNWts
    
    model <- do.call(train, args)
    elapsed <- round(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 1)
    training_times[[algo_name]] <- elapsed
    best_tune_params[[algo_name]] <- model$bestTune
    all_models[[algo_name]] <- model
    saveRDS(model, file.path(out_root, paste0(algo_name, "_model.rds")))
    
    if (!is.null(model$resample)) {
      resample_df <- model$resample %>% mutate(Model = algo_name)
      write.csv(resample_df, file.path(out_root, "CV_Detailed_Results", paste0(algo_name, "_CV_Resample.csv")), row.names = FALSE)
      cv_results_list[[algo_name]] <- resample_df
    }
    
    pred_train_prob <- predict(model, traindata[, !names(traindata) %in% response_var], type = "prob")
    pred_test_prob  <- predict(model, testdata[,  !names(testdata) %in% response_var], type = "prob")
    train_pred_class <- predict(model, traindata[, !names(traindata) %in% response_var])
    test_pred_class  <- predict(model, testdata[,  !names(testdata) %in% response_var])
    
    all_train_metrics[[algo_name]] <- calculate_multiclass_metrics(traindata[[response_var]], train_pred_class, pred_train_prob)
    all_test_metrics[[algo_name]]  <- calculate_multiclass_metrics(testdata[[response_var]],  test_pred_class,  pred_test_prob)
    
    vi <- tryCatch(varImp(model), error = function(e) NULL)
    if (!is.null(vi)) {
      imp <- vi$importance
      imp_df <- data.frame(Variable = rownames(imp), Importance = imp[,1], row.names = NULL)
      p <- ggplot(imp_df, aes(reorder(Variable, Importance), Importance)) +
        geom_col(fill = "steelblue") + coord_flip() + theme_minimal()
      ggsave(file.path(out_root, "Feature_importance_ranking", paste0(algo_name, "_importance.pdf")), p, width = 8, height = 6)
    }
    if (algo_name == "DecisionTree") {
      pdf(file.path(out_root, "Feature_importance_ranking", "DecisionTree_plot.pdf"), width = 10, height = 8)
      rpart.plot(model$finalModel)
      dev.off()
    }
    NULL
  }, error = function(e) e)
  if (inherits(safe_run, "error")) {
    training_times[[algo_name]] <- ifelse(is.null(training_times[[algo_name]]), NA, training_times[[algo_name]])
    best_tune_params[[algo_name]] <- NA
    all_models[[algo_name]] <- NULL
    all_train_metrics[[algo_name]] <- tibble::tibble(Accuracy = NA, Kappa = NA, Sensitivity = NA, Specificity = NA, Precision = NA, F1 = NA, AUC = NA)
    all_test_metrics[[algo_name]]  <- all_train_metrics[[algo_name]]
  }
}

# Summary results
train_perf <- bind_rows(lapply(names(all_train_metrics), function(n) mutate(all_train_metrics[[n]], Model = n))) %>% select(Model, everything())
test_perf  <- bind_rows(lapply(names(all_test_metrics),  function(n) mutate(all_test_metrics[[n]],  Model = n))) %>% select(Model, everything())

write.csv(train_perf, file.path(out_root, "1_All_Models_Train_Performance.csv"), row.names = FALSE)
write.csv(test_perf,  file.path(out_root, "1_All_Models_Test_Performance.csv"),  row.names = FALSE)

params_list <- lapply(names(best_tune_params), function(name) {
  params <- best_tune_params[[name]]
  if (is.data.frame(params)) {
    params %>% pivot_longer(everything()) %>% mutate(Model = name) %>% rename(Parameter = name, Value = value)
  } else {
    tibble::tibble(Model = name, Parameter = "Note", Value = "No tuning or failed")
  }
}) %>% bind_rows()
write.csv(params_list, file.path(out_root, "2_All_Models_Best_Parameters.csv"), row.names = FALSE)

time_df <- tibble::tibble(Model = names(training_times), Time_seconds = unlist(training_times))
write.csv(time_df, file.path(out_root, "3_All_Models_Training_Times.csv"), row.names = FALSE)

# Performance heatmap
heatmap_data <- bind_rows(
  test_perf %>% select(Model, Accuracy, AUC, Sensitivity, Specificity, Precision, F1) %>%
    pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>% mutate(Dataset = "Test"),
  train_perf %>% select(Model, AUC) %>% mutate(Metric = "AUC", Dataset = "Train") %>% rename(Value = AUC)
) %>%
  mutate(MetricLabel = ifelse(Dataset == "Test", as.character(Metric), paste0(Metric, "\n(Train)")))

p_heatmap <- ggplot(heatmap_data, aes(x = Model, y = MetricLabel, fill = Value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = ifelse(is.na(Value), "NA", sprintf("%.3f", Value))), size = 3.5) +
  scale_fill_gradient(low = "white", high = "steelblue", na.value = "lightgray") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text(face = "bold"), panel.grid = element_blank()) +
  labs(title = "Model Performance Heatmap")

ggsave(file.path(out_root, "4_Model_Performance_Heatmap.pdf"), p_heatmap, width = 12, height = 8, dpi = 300)

# Select a model
specified_model_name <- "LinearRegression"
best_model_name <- specified_model_name
best_model <- all_models[[best_model_name]]
if (is.null(best_model)) stop("Selected model not available: ", best_model_name)

result_dir <- file.path(out_root, paste0("Best_Model_", best_model_name))
dir.create(result_dir, showWarnings = FALSE)
saveRDS(best_model, file.path(result_dir, paste0(best_model_name, "_Final_Model.rds")))
if (!is.null(best_model$bestTune)) write.csv(as.data.frame(best_model$bestTune), file.path(result_dir, "Best_Parameters.csv"), row.names = FALSE)
write.csv(time_df %>% filter(Model == best_model_name), file.path(result_dir, "Training_Time.csv"), row.names = FALSE)

# Predictions and metrics for selected model
train_pred_class <- predict(best_model, traindata[, !names(traindata) %in% response_var])
test_pred_class  <- predict(best_model, testdata[,  !names(testdata) %in% response_var])
train_pred_prob  <- predict(best_model, traindata[, !names(traindata) %in% response_var], type = "prob")
test_pred_prob   <- predict(best_model, testdata[,  !names(testdata) %in% response_var], type = "prob")

train_metrics <- calculate_multiclass_metrics(traindata[[response_var]], train_pred_class, train_pred_prob)
test_metrics  <- calculate_multiclass_metrics(testdata[[response_var]],  test_pred_class,  test_pred_prob)
write.csv(train_metrics, file.path(result_dir, "Train_Performance.csv"), row.names = FALSE)
write.csv(test_metrics,  file.path(result_dir, "Test_Performance.csv"),  row.names = FALSE)

# ROC OvR plotting function
plot_roc_ovr_with_auc <- function(true_labels, pred_probs, title_suffix = "") {
  roc_list <- list(); auc_annotations <- data.frame(); classes <- levels(true_labels)
  x_pos <- 0.60; y_start <- 0.15; y_step <- 0.08
  for (cls in classes) {
    binary_label <- factor(ifelse(true_labels == cls, cls, "Other"), levels = c("Other", cls))
    roc_obj <- tryCatch(roc(binary_label, pred_probs[, cls], quiet = TRUE), error = function(e) NULL)
    if (is.null(roc_obj)) next
    df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities, Class = cls)
    roc_list[[cls]] <- df
    auc_val <- as.numeric(roc_obj$auc)
    auc_annotations <- rbind(auc_annotations, data.frame(Class = cls, x = x_pos, y = y_start, label = paste(cls, "=", format(round(auc_val, 3), nsmall = 3)), stringsAsFactors = FALSE))
    y_start <- y_start + y_step
  }
  if (length(roc_list) == 0) return(list(plot = ggplot() + ggtitle("No ROC data"), roc_df = NULL, auc_annotations = NULL))
  roc_df <- do.call(rbind, roc_list)
  p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Class)) +
    geom_line(size = 1) + geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
    geom_text(data = auc_annotations, aes(x = x, y = y, label = label), color = "black", size = 4, hjust = 0, fontface = "bold") +
    labs(title = paste("ROC Curves (", title_suffix, ")"), x = "FPR", y = "TPR") +
    theme_minimal() + theme(plot.title = element_text(hjust = 0.5), legend.position = "none") + xlim(0,1) + ylim(0,1)
  list(plot = p, roc_df = roc_df, auc_annotations = auc_annotations)
}

roc_train_res <- plot_roc_ovr_with_auc(traindata[[response_var]], train_pred_prob, "Train")
roc_test_res  <- plot_roc_ovr_with_auc(testdata[[response_var]],  test_pred_prob, "Test")
p_train_roc <- roc_train_res$plot
p_test_roc  <- roc_test_res$plot

# Confusion matrices
cm_train <- confusionMatrix(train_pred_class, traindata[[response_var]])
cm_train_df <- as.data.frame(cm_train$table) %>% setNames(c("Predicted","Actual","Freq"))
cm_test <- confusionMatrix(test_pred_class, testdata[[response_var]])
cm_test_df <- as.data.frame(cm_test$table) %>% setNames(c("Predicted","Actual","Freq"))

levels_order <- levels(traindata[[response_var]])
cm_train_df$Predicted <- factor(cm_train_df$Predicted, levels = levels_order)
cm_train_df$Actual    <- factor(cm_train_df$Actual, levels = levels_order)
cm_test_df$Predicted  <- factor(cm_test_df$Predicted, levels = levels_order)
cm_test_df$Actual     <- factor(cm_test_df$Actual, levels = levels_order)

p_cm_train <- ggplot(cm_train_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") + geom_text(aes(label = Freq), color = "black", fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#4682B4") + theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_cm_test <- ggplot(cm_test_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") + geom_text(aes(label = Freq), color = "black", fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#4682B4") + theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p_roc_col <- (p_train_roc + labs(title = paste(best_model_name, "- Train ROC"))) / (p_test_roc + labs(title = paste(best_model_name, "- Test ROC")))
p_cm_col  <- p_cm_train / p_cm_test
p_all <- p_roc_col | p_cm_col
ggsave(file.path(result_dir, "ROC_ConfusionMatrix_Combined_TrainTest.pdf"), p_all, width = 16, height = 12, dpi = 300)

# SHAP
if (inherits(best_model, "train")) {
  X_raw <- traindata[, !names(traindata) %in% response_var]
  X <- X_raw
  bg_sample <- X[sample(nrow(X), min(20, nrow(X))), , drop = FALSE]
  pred_fun <- function(model, newdata) {
    newdata <- as.data.frame(newdata)
    prob <- predict(model, newdata = newdata, type = "prob")
    as.matrix(prob)
  }
  shap_values <- tryCatch(kernelshap::kernelshap(best_model, X = X, bg_X = bg_sample, pred_fun = pred_fun), error = function(e) NULL)
  if (!is.null(shap_values)) {
    sv <- shapviz(shap_values, X = X)
    p_imp <- sv_importance(sv) + theme_minimal()
    ggsave(file.path(result_dir, "SHAP_Feature_Importance.pdf"), p_imp, width = 8, height = 6)
    p_bee <- sv_importance(sv, kind = "beeswarm") + theme_minimal()
    ggsave(file.path(result_dir, "SHAP_Beeswarm.pdf"), p_bee, width = 8, height = 4)
    p_water <- sv_waterfall(sv, row_id = 1) + theme_minimal()
    ggsave(file.path(result_dir, "SHAP_Waterfall_row1.pdf"), p_water, width = 10, height = 6)
    p_force <- sv_force(sv, row_id = 1) + theme_minimal()
    ggsave(file.path(result_dir, "SHAP_Force_row1.pdf"), p_force, width = 6, height = 6)
    imp_df <- sv_importance(sv)$data
    top_feats <- imp_df$feature[1:min(2, nrow(imp_df))]
    for (feat in top_feats) {
      p_dep <- sv_dependence(sv, v = feat, color_var = "auto") + theme_minimal()
      ggsave(file.path(result_dir, paste0("SHAP_Dependence_", make.names(feat), ".pdf")), p_dep, width = 12, height = 6)
    }
  }
}

# Export SHAP numeric files
if (exists("shap_values") && !is.null(shap_values) && !is.null(shap_values$S)) {
  sv_list <- shap_values$S
  for (i in seq_along(sv_list)) write.csv(sv_list[[i]], file.path(result_dir, sprintf("shap_values_class_%d.csv", i)), row.names = FALSE)
  write.csv(as.data.frame(X_train_red), file.path(result_dir, "X_train.csv"), row.names = FALSE)
  write.csv(data.frame(feature = feature_names_red), file.path(result_dir, "feature_names.csv"), row.names = FALSE)
  write.csv(data.frame(class_names = names(sv_list)), file.path(result_dir, "class_names.csv"), row.names = FALSE)
}

# Export all predictions
data$HER23_num <- as.numeric(data$HER23) - 1
train_with_id <- data[match(rownames(traindata), data$id), ]
test_with_id  <- data[match(rownames(testdata),  data$id), ]

train_pred_num <- as.numeric(train_pred_class) - 1
test_pred_num  <- as.numeric(test_pred_class) - 1

prob_names <- paste0("Prob_", levels(testdata[[response_var]]))
train_out <- data.frame(ID = rownames(traindata), tt = 1, HER23_true = train_with_id$HER23_num, HER23_pred = train_pred_num, stringsAsFactors = FALSE)
test_out  <- data.frame(ID = rownames(testdata),  tt = 0, HER23_true = test_with_id$HER23_num,  HER23_pred = test_pred_num,  stringsAsFactors = FALSE)
train_out <- cbind(train_out, setNames(as.data.frame(train_pred_prob), prob_names))
test_out  <- cbind(test_out,  setNames(as.data.frame(test_pred_prob),  prob_names))
all_pred <- rbind(train_out, test_out)
all_pred <- all_pred[order(all_pred$ID), ]
rownames(all_pred) <- NULL
write.csv(all_pred, file.path(result_dir, "Predictions_All_Samples.csv"), row.names = FALSE)

message("Done. Results in: ", normalizePath(result_dir))