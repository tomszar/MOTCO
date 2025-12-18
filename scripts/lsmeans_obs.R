#!/usr/bin/env Rscript

# Native R implementation to compute lsmeans.obs exactly as in
# tests/data/reference/evo_649_sm_suppmat.r, but parameterized.
#
# Usage:
#   Rscript scripts/lsmeans_obs.R --data PATH --group-col COL --level-col COL [--pcs K]
#
# Output:
#   CSV to stdout with columns: key, PC1..PCk, where key = paste(group, level, sep=":")

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NA) {
  i <- which(args == name)
  if (length(i) == 1 && i < length(args)) {
    return(args[i + 1])
  } else {
    return(default)
  }
}

data_path <- get_arg("--data")
group_col <- get_arg("--group-col")
level_col <- get_arg("--level-col")
pcs_arg   <- get_arg("--pcs", NA)

if (is.na(data_path) || is.na(group_col) || is.na(level_col)) {
  stop("Missing required arguments. Usage: --data PATH --group-col COL --level-col COL [--pcs K]")
}

pcs <- suppressWarnings(as.integer(pcs_arg))
if (is.na(pcs)) pcs <- 2L

# Read data
data <- read.csv(data_path, header = TRUE, stringsAsFactors = FALSE)

if (!(group_col %in% names(data))) stop(paste0("Group column not found: ", group_col))
if (!(level_col %in% names(data))) stop(paste0("Level column not found: ", level_col))

taxa  <- as.factor(data[[group_col]])
level <- as.factor(data[[level_col]])

# Numeric feature columns excluding factors
is_num <- vapply(data, is.numeric, logical(1))
feat_cols <- setdiff(names(data)[is_num], c(group_col, level_col))
if (length(feat_cols) == 0) stop("No numeric feature columns found in the dataset.")

Y <- as.matrix(data[, feat_cols, drop = FALSE])

# PCA: center = TRUE, scale. = FALSE (matches prcomp defaults in reference)
Ypc <- prcomp(Y)$x

# Choose first k PCs (bounded by available PCs)
k_avail <- ncol(Ypc)
k <- max(1L, min(as.integer(pcs), as.integer(k_avail)))
Yk <- Ypc[, seq_len(k), drop = FALSE]

# Full model and predictions
lm.full <- lm(Yk ~ taxa * level, model = TRUE, x = TRUE, y = TRUE, qr = TRUE)
yhat.full <- predict(lm.full)

# Factor for LS-means keys and its levels order
taxaBylevel <- as.factor(paste(taxa, level, sep = ":"))
levs <- levels(taxaBylevel)

# Build lsmeans.obs exactly as reference: mean of fitted values per cell
lsmeans.obs <- NULL
for (i in seq_len(k)) {
  mean.temp <- tapply(yhat.full[, i], taxaBylevel, mean)
  # Ensure consistent row ordering by factor levels
  mean.temp <- mean.temp[levs]
  lsmeans.obs <- cbind(lsmeans.obs, mean.temp)
}

colnames(lsmeans.obs) <- paste0("PC", seq_len(k))
rownames(lsmeans.obs) <- levs

# Write to stdout as CSV with a leading 'key' column for the row names
out <- data.frame(key = rownames(lsmeans.obs), lsmeans.obs, row.names = NULL, check.names = FALSE)
write.csv(out, file = stdout(), row.names = FALSE)
