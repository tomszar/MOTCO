# One-time InterSIM side of the generator-fidelity validation.
#
# Runs InterSIM `n_intersim` times per `(delta, p.DMP)` grid cell and writes the
# per-cell distribution of a statistic battery (`stats.csv`) plus provenance.
# `build_fidelity_fixture_from_export` (Python) packs the export into the
# committed `.npz` fixture; CI then compares the numpy generator against it with
# no R dependency. The statistic battery below mirrors `compute_statistics` in
# `fidelity.py` exactly, so the two sides are comparable by construction.
#
# Usage (R is only needed here, never at runtime):
#   nix develop --command Rscript src/motco/simulations/fidelity_intersim.R \
#       --output-dir <dir> [--seed 20260604] [--n-sample 500] [--n-intersim 30]

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop(paste("Unexpected positional argument:", key))
    if (i == length(args)) stop(paste("Missing value for argument:", key))
    parsed[[substring(key, 3)]] <- args[[i + 1]]
    i <- i + 2
  }
  parsed
}

parsed <- parse_args(args)
output_dir <- parsed[["output-dir"]]
if (is.null(output_dir) || output_dir == "") stop("Missing required argument: --output-dir")

opt_num <- function(key, default) {
  v <- parsed[[key]]
  if (is.null(v) || v == "") default else as.numeric(v)
}
opt_vec <- function(key, default) {
  v <- parsed[[key]]
  if (is.null(v) || v == "") default else as.numeric(strsplit(v, ",", fixed = TRUE)[[1]])
}

# Grid + protocol defaults — keep in sync with `fidelity.default_grid()`.
deltas <- opt_vec("deltas", c(0.0, 1.0, 2.0))
p_dmps <- opt_vec("p-dmps", c(0.2, 0.5))
n_sample <- opt_num("n-sample", 500)
cluster_prop <- opt_vec("cluster-prop", c(0.3, 0.3, 0.4))
n_intersim <- as.integer(opt_num("n-intersim", 30))
base_seed <- as.integer(opt_num("seed", 20260604))
diff_threshold <- opt_num("diff-threshold", 1.0)

if (!requireNamespace("InterSIM", quietly = TRUE)) {
  stop("The R package InterSIM is not installed or cannot be loaded.")
}
suppressMessages(library(InterSIM))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Per-omic statistic battery. MUST match `fidelity._omic_statistics`:
#   moments use the population convention (/n); quantiles are type 7 (R default
#   == numpy default); eta2 = mean_j SSB_j / SST_j; diff_rate = fraction of
#   features whose standardised cluster-mean range exceeds `threshold`;
#   cov_frob = Frobenius norm of the empirical covariance (/n).
omic_statistics <- function(X, cluster_id, threshold) {
  X <- as.matrix(X)
  n <- nrow(X)
  labels <- sort(unique(cluster_id))
  grand <- colMeans(X)
  centered <- sweep(X, 2, grand, "-")
  sst <- colSums(centered^2)

  ssb <- numeric(ncol(X))
  cluster_means <- matrix(0, nrow = length(labels), ncol = ncol(X))
  for (k in seq_along(labels)) {
    rows <- X[cluster_id == labels[k], , drop = FALSE]
    cm <- colMeans(rows)
    cluster_means[k, ] <- cm
    ssb <- ssb + nrow(rows) * (cm - grand)^2
  }
  eta2 <- mean(ifelse(sst > 0, ssb / sst, NA), na.rm = TRUE)

  feat_sd <- sqrt(sst / n)
  rng <- apply(cluster_means, 2, max) - apply(cluster_means, 2, min)
  std_range <- ifelse(feat_sd > 0, rng / feat_sd, 0)
  diff_rate <- mean(std_range > threshold)

  cov <- crossprod(centered) / n
  cov_frob <- sqrt(sum(cov^2))

  flat <- as.numeric(X)
  qs <- quantile(flat, c(0.1, 0.5, 0.9), names = FALSE, type = 7)
  c(
    mean = mean(flat), sd = sqrt(mean((flat - mean(flat))^2)),
    q10 = qs[1], q50 = qs[2], q90 = qs[3],
    eta2 = eta2, diff_rate = diff_rate, cov_frob = cov_frob
  )
}

omics <- c("methylation", "expression", "proteomics")
suffixes <- c("mean", "sd", "q10", "q50", "q90", "eta2", "diff_rate", "cov_frob")
stat_cols <- as.vector(t(outer(omics, suffixes, paste, sep = "_")))

records <- list()
row_i <- 1
for (delta in deltas) {
  for (p_dmp in p_dmps) {
    for (rep in seq_len(n_intersim)) {
      # Deterministic per-(cell, replicate) seed for reproducibility.
      set.seed(base_seed + row_i)
      sim <- InterSIM(
        n.sample = n_sample,
        cluster.sample.prop = cluster_prop,
        delta.methyl = delta, delta.expr = delta, delta.protein = delta,
        p.DMP = p_dmp,
        do.plot = FALSE
      )
      cluster_id <- sim$clustering.assignment$cluster.id
      sm <- omic_statistics(sim$dat.methyl, cluster_id, diff_threshold)
      se <- omic_statistics(sim$dat.expr, cluster_id, diff_threshold)
      sp <- omic_statistics(sim$dat.protein, cluster_id, diff_threshold)
      values <- c(sm, se, sp)
      names(values) <- stat_cols
      records[[row_i]] <- c(delta = delta, p_dmp = p_dmp, replicate = rep, values)
      row_i <- row_i + 1
    }
  }
}

stats <- do.call(rbind, lapply(records, function(r) as.data.frame(as.list(r))))
write.csv(stats, file.path(output_dir, "stats.csv"), row.names = FALSE)

provenance <- c(
  paste0("intersim_version: ", as.character(packageVersion("InterSIM"))),
  paste0("r_version: ", as.character(getRversion())),
  paste0("generation_date: ", format(Sys.Date(), "%Y-%m-%d")),
  paste0("generation_script: fidelity_intersim.R"),
  paste0("base_seed: ", base_seed),
  paste0("numpy_seed: ", base_seed),
  paste0("n_sample: ", n_sample),
  paste0("cluster_prop: ", paste(cluster_prop, collapse = ",")),
  paste0("deltas: ", paste(deltas, collapse = ",")),
  paste0("p_dmps: ", paste(p_dmps, collapse = ",")),
  paste0("n_intersim: ", n_intersim),
  paste0("n_numpy: ", n_intersim),
  paste0("diff_threshold: ", diff_threshold)
)
writeLines(provenance, file.path(output_dir, "provenance.txt"))
cat("Wrote InterSIM fidelity distributions for", length(records), "runs to", output_dir, "\n")