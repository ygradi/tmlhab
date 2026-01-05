# script.R
# config
data_path <- "path/to/data.csv"
output_dir <- "path/to/results"
out_kw <- file.path(output_dir, "kw_features.csv")
out_final <- file.path(output_dir, "final_features.csv")
seed <- 123

# packages
pkgs <- c("readr","dplyr","caret","dunn.test","glmnet")
missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) stop("Install required packages")
invisible(lapply(pkgs, library, character.only = TRUE))

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(seed)

# read data
data <- readr::read_csv(data_path,
                        col_types = readr::cols(
                          id = readr::col_character(),
                          tt = readr::col_integer(),
                          HER23 = readr::col_integer(),
                          .default = readr::col_double()
                        )
)

# basic checks
required_cols <- c("id","tt","HER23")
if (!all(required_cols %in% names(data))) stop("Missing required columns")

# response as ordered factor
data$HER23 <- factor(c("zero","low","posi")[data$HER23 + 1L],
                     levels = c("zero","low","posi"),
                     ordered = TRUE)

# training subset
train_data <- dplyr::filter(data, tt == 1)
if (nrow(train_data) == 0) stop("No training rows")

# feature matrix
feature_names <- setdiff(names(train_data), c("id","tt","HER23"))
if (length(feature_names) == 0) stop("No feature columns")
X_train <- as.data.frame(train_data[, feature_names, drop = FALSE], stringsAsFactors = FALSE)
colnames(X_train)[is.na(colnames(X_train))] <- paste0("feature_", which(is.na(colnames(X_train))))

# Kruskal-Wallis univariate filter (alpha = 0.05)
alpha <- 0.05
kw_significant_features <- character(0)
y_train_ord <- train_data$HER23
for (i in seq_len(ncol(X_train))) {
  xi <- X_train[[i]]
  if (all(is.na(xi)) || length(unique(xi[!is.na(xi)])) <= 1) next
  kt <- try(kruskal.test(xi ~ y_train_ord), silent = TRUE)
  if (inherits(kt, "try-error")) next
  if (!is.na(kt$p.value) && kt$p.value < alpha) kw_significant_features <- c(kw_significant_features, colnames(X_train)[i])
}
write.csv(data.frame(feature = kw_significant_features), file = out_kw, row.names = FALSE)
if (length(kw_significant_features) == 0) stop("No features passed Kruskal-Wallis")

# Multiclass LASSO selection
features_lasso_selected <- character(0)
X_lasso_df <- train_data[, kw_significant_features, drop = FALSE]
valid_cols <- vapply(X_lasso_df, function(col) !all(is.na(col)) && length(unique(na.omit(col))) > 1, logical(1))
if (sum(valid_cols) == 0) {
  features_lasso_selected <- kw_significant_features
} else {
  X_lasso_df <- X_lasso_df[, valid_cols, drop = FALSE]
  y_lasso <- as.factor(as.character(train_data$HER23))
  class_table <- table(y_lasso)
  if (min(class_table) < 8) {
    features_lasso_selected <- colnames(X_lasso_df)
  } else {
    preproc <- caret::preProcess(X_lasso_df, method = c("center", "scale"))
    X_scaled <- predict(preproc, X_lasso_df)
    X_matrix <- as.matrix(X_scaled)
    set.seed(seed)
    cv_fit <- tryCatch(
      cv.glmnet(X_matrix, y_lasso, family = "multinomial", alpha = 1, nfolds = 5, type.multinomial = "grouped"),
      error = function(e) e
    )
    if (inherits(cv_fit, "error")) {
      features_lasso_selected <- colnames(X_lasso_df)
    } else {
      coef_list <- coef(cv_fit, s = "lambda.min")
      nonzero <- character(0)
      for (cls in names(coef_list)) {
        coefs <- as.matrix(coef_list[[cls]])
        vars <- rownames(coefs)
        nz_idx <- which(abs(coefs[,1]) > 0 & vars != "(Intercept)")
        if (length(nz_idx) > 0) nonzero <- c(nonzero, vars[nz_idx])
      }
      features_lasso_selected <- unique(nonzero)
      if (length(features_lasso_selected) == 0) features_lasso_selected <- colnames(X_lasso_df)
    }
  }
}

final_modeling_features <- sort(features_lasso_selected)
if (length(final_modeling_features) == 0) stop("No final features selected")
write.csv(data.frame(feature = final_modeling_features), file = out_final, row.names = FALSE)

cat("KW_count:", length(kw_significant_features), " LASSO_count:", length(features_lasso_selected),
    " Final_count:", length(final_modeling_features), "\n")