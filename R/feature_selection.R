# Feature selection

library(readr)
library(dplyr)
library(caret)
library(glmnet)
library(dunn.test)

# config
seed <- 123
set.seed(seed)
data_file <- file.path("data", "TT_features.csv")
out_dir <- file.path("results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 1. data
data <- readr::read_csv(data_file, col_types = cols(.default = "n"))
stopifnot(all(c("id","tt","HER23") %in% names(data)))

train_data <- filter(data, tt == 1)
if (nrow(train_data) == 0) stop("No training samples with tt==1")

y <- factor(train_data$HER23, levels = c(0,1,2), labels = c("zero","low","posi"))
feature_names <- setdiff(names(train_data), c("id","tt","HER23"))
X <- train_data[, feature_names, drop = FALSE]

# 2. Kruskal-Wallis
kw_select <- function(X, y, alpha = 0.05) {
  sig <- character(0)
  for (i in seq_len(ncol(X))) {
    xi <- X[[i]]
    if (all(is.na(xi)) || length(unique(xi[!is.na(xi)])) <= 1) next
    res <- try(kruskal.test(xi ~ y), silent = TRUE)
    if (inherits(res, "try-error")) next
    if (!is.na(res$p.value) && res$p.value < alpha) sig <- c(sig, colnames(X)[i])
  }
  unique(sig)
}
kw_feats <- kw_select(X, y, alpha = 0.05)
write.csv(data.frame(feature = kw_feats), file = file.path(out_dir, "KW_significant_features.csv"), row.names = FALSE)
if (length(kw_feats) == 0) stop("No KW features")

# 3. LASSO
X_lasso <- train_data[, kw_feats, drop = FALSE]
valid <- sapply(X_lasso, function(col) !all(is.na(col)) && length(unique(na.omit(col))) > 1)
X_lasso <- X_lasso[, valid, drop = FALSE]
y_lasso <- y

orig_feat_names <- colnames(X_lasso)

features_selected <- character(0)
cvfit <- NULL

try({
  if (ncol(X_lasso) == 0) stop("No valid features for LASSO")
  Xm <- as.matrix(X_lasso)
  nfolds <- max(2, min(5, nrow(Xm)))
  set.seed(seed)
  cvfit <- tryCatch(
    cv.glmnet(Xm, y_lasso,
              family = "multinomial",
              alpha = 1,
              nfolds = nfolds,
              type.multinomial = "grouped",
              standardize = TRUE,
              parallel = FALSE),
    error = function(e) e
  )
  
  if (inherits(cvfit, "error")) {
    warning("cv.glmnet failed: ", cvfit$message)
    features_selected <- orig_feat_names
  } else {
    coefs <- coef(cvfit, s = "lambda.min")
    nz <- unique(unlist(lapply(coefs, function(m) {
      mm <- as.matrix(m)
      rownames(mm)[which(mm[,1] != 0)]
    })))
    nz <- setdiff(nz, "(Intercept)")
    nz <- intersect(nz, orig_feat_names)
    if (length(nz) == 0) {
      warning("LASSO NA")
      features_selected <- orig_feat_names
    } else {
      features_selected <- nz
    }
  }
}, silent = FALSE)

if (length(features_selected) == 0) {
  features_selected <- orig_feat_names
}

write.csv(data.frame(feature = sort(features_selected)), file = file.path(out_dir, "Final_Modeling_Features.csv"), row.names = FALSE)
saveRDS(list(features = features_selected, cvfit = if (!is.null(cvfit) && !inherits(cvfit, "error")) cvfit else NULL),
        file = file.path(out_dir, "lasso_result.rds"))