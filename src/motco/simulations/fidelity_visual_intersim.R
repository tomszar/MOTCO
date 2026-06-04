# One-time InterSIM side of the *qualitative* fidelity supplement.
#
# Unlike `fidelity_intersim.R` (which writes summary-statistic distributions),
# this writes the raw data needed for the side-by-side visual comparison:
#   * density:   a few replicates' subsampled per-omic values (marginal shape),
#   * heatmap/PCA/coupling/scatter: ONE full replicate's matrices + cluster ids,
#     generated at a non-trivial cluster count (default 4).
# The numpy side is generated live (no R) by `fidelity_visual.py`, so only the
# InterSIM raw data is committed. `build_visual_fixture_from_export` (Python)
# packs this export into `tests/data/intersim_visual_fixture.npz`.
#
# Usage:
#   nix develop --command Rscript src/motco/simulations/fidelity_visual_intersim.R \
#       --output-dir <dir>

args <- commandArgs(trailingOnly = TRUE)
parse_args <- function(args) {
  parsed <- list(); i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop(paste("Unexpected positional argument:", key))
    if (i == length(args)) stop(paste("Missing value for argument:", key))
    parsed[[substring(key, 3)]] <- args[[i + 1]]; i <- i + 2
  }
  parsed
}
parsed <- parse_args(args)
output_dir <- parsed[["output-dir"]]
if (is.null(output_dir) || output_dir == "") stop("Missing required argument: --output-dir")

opt_num <- function(key, default) {
  v <- parsed[[key]]; if (is.null(v) || v == "") default else as.numeric(v)
}

n_sample <- opt_num("n-sample", 300)
n_cluster <- as.integer(opt_num("n-cluster", 4))
delta <- opt_num("delta", 2.0)
p_dmp <- opt_num("p-dmp", 0.2)
n_rep_density <- as.integer(opt_num("n-rep-density", 3))
subsample <- as.integer(opt_num("subsample", 5000))
base_seed <- as.integer(opt_num("seed", 20260604))

if (!requireNamespace("InterSIM", quietly = TRUE)) {
  stop("The R package InterSIM is not installed or cannot be loaded.")
}
suppressMessages(library(InterSIM))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cluster_prop <- rep(1 / n_cluster, n_cluster)

run_one <- function(seed) {
  set.seed(seed)
  InterSIM(
    n.sample = n_sample, cluster.sample.prop = cluster_prop,
    delta.methyl = delta, delta.expr = delta, delta.protein = delta,
    p.DMP = p_dmp, do.plot = FALSE
  )
}

# --- Density: a few replicates' subsampled per-omic values -------------------
set.seed(base_seed + 10000)  # reproducible subsampling indices
dens <- list(methylation = NULL, expression = NULL, proteomics = NULL)
for (rep in seq_len(n_rep_density)) {
  sim <- run_one(base_seed + rep)
  for (key in names(dens)) {
    flat <- as.numeric(switch(key,
      methylation = sim$dat.methyl, expression = sim$dat.expr,
      proteomics = sim$dat.protein))
    idx <- sample.int(length(flat), size = min(subsample, length(flat)))
    dens[[key]] <- rbind(dens[[key]], flat[idx])
  }
}
for (key in names(dens)) {
  write.table(dens[[key]], file.path(output_dir, paste0("density_", key, ".csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
}

# --- One full replicate for heatmap / PCA / coupling / moment scatter --------
sim <- run_one(base_seed + 1)  # first density replicate, reused
cluster_id <- sim$clustering.assignment$cluster.id
write_matrix <- function(x, file) {
  write.table(as.matrix(x), file.path(output_dir, file), sep = ",",
              row.names = FALSE, col.names = FALSE)
}
write_matrix(sim$dat.methyl, "matrix_methylation.csv")
write_matrix(sim$dat.expr, "matrix_expression.csv")
write_matrix(sim$dat.protein, "matrix_proteomics.csv")
write.table(data.frame(cluster = cluster_id), file.path(output_dir, "cluster_ids.csv"),
            sep = ",", row.names = FALSE, col.names = TRUE)

provenance <- c(
  paste0("intersim_version: ", as.character(packageVersion("InterSIM"))),
  paste0("r_version: ", as.character(getRversion())),
  paste0("generation_date: ", format(Sys.Date(), "%Y-%m-%d")),
  paste0("generation_script: fidelity_visual_intersim.R"),
  paste0("base_seed: ", base_seed),
  paste0("n_sample: ", n_sample),
  paste0("n_cluster: ", n_cluster),
  paste0("delta: ", delta),
  paste0("p_dmp: ", p_dmp),
  paste0("n_rep_density: ", n_rep_density),
  paste0("subsample: ", subsample)
)
writeLines(provenance, file.path(output_dir, "provenance.txt"))
cat("Wrote InterSIM visual fixture export to", output_dir, "\n")