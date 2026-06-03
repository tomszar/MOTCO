# One-time export of InterSIM's reference data for the numpy generator.
#
# InterSIM's generative model is fully determined by a set of package-level
# reference objects (means, covariances, cross-omic maps and correlation
# vectors). This script reads those objects from the InterSIM namespace and
# writes them to a plain-CSV export directory, which `reference.py` then packs
# into the committed `.npz` cache. R is needed only to produce this export;
# runtime generation reads the `.npz` without R.
#
# Usage:
#   Rscript export_reference.R --output-dir <dir>

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(paste("Unexpected positional argument:", key))
    }
    if (i == length(args)) {
      stop(paste("Missing value for argument:", key))
    }
    parsed[[substring(key, 3)]] <- args[[i + 1]]
    i <- i + 2
  }
  parsed
}

parsed <- parse_args(args)
output_dir <- parsed[["output-dir"]]
if (is.null(output_dir) || output_dir == "") {
  stop("Missing required argument: --output-dir")
}

if (!requireNamespace("InterSIM", quietly = TRUE)) {
  stop("The R package InterSIM is not installed or cannot be loaded.")
}
suppressMessages(library(InterSIM))
ns <- asNamespace("InterSIM")
g <- function(name) get(name, envir = ns)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

mean_M <- g("mean.M")
mean_expr <- g("mean.expr")
mean_protein <- g("mean.protein")
cov_M <- g("cov.M")
cov_expr <- g("cov.expr")
cov_protein <- g("cov.protein")
methyl_gene_level_mean <- g("methyl.gene.level.mean")
mean_expr_with_mapped_protein <- g("mean.expr.with.mapped.protein")
rho_methyl_expr <- g("rho.methyl.expr")
rho_expr_protein <- g("rho.expr.protein")
CpG.gene.map.for.DEG <- g("CpG.gene.map.for.DEG")
protein.gene.map.for.DEP <- g("protein.gene.map.for.DEP")

cpg_names <- names(mean_M)
gene_names <- names(mean_expr)
protein_names <- names(mean_protein)

# Build the CpG -> gene incidence matrix in InterSIM's exact indexing:
#   DEG[g, cluster] = 1 iff some active CpG maps (via tmp.gene) to gene g.
gene_of_cpg <- as.character(CpG.gene.map.for.DEG[cpg_names, ]$tmp.gene)
incidence_cpg_gene <- matrix(0L, nrow = length(cpg_names), ncol = length(gene_names))
gene_index <- match(gene_of_cpg, gene_names)
for (i in seq_along(cpg_names)) {
  j <- gene_index[i]
  if (!is.na(j)) incidence_cpg_gene[i, j] <- 1L
}

# Build the gene -> protein incidence matrix:
#   DEP[p, cluster] = 1 iff protein p's mapped gene is active.
gene_of_protein <- as.character(protein.gene.map.for.DEP[protein_names, ]$gene)
incidence_gene_protein <- matrix(0L, nrow = length(gene_names), ncol = length(protein_names))
gene_index_p <- match(gene_of_protein, gene_names)
for (p in seq_along(protein_names)) {
  gi <- gene_index_p[p]
  if (!is.na(gi)) incidence_gene_protein[gi, p] <- 1L
}

write_named <- function(x, file) {
  write.csv(data.frame(name = names(x), value = unname(x)),
            file = file.path(output_dir, file), row.names = FALSE)
}
write_matrix <- function(x, file) {
  write.table(x, file = file.path(output_dir, file), sep = ",",
              row.names = FALSE, col.names = FALSE)
}
write_vector <- function(x, file) {
  write.table(data.frame(value = unname(x)), file = file.path(output_dir, file),
              sep = ",", row.names = FALSE, col.names = TRUE)
}

write_named(mean_M, "mean_M.csv")
write_named(mean_expr, "mean_expr.csv")
write_named(mean_protein, "mean_protein.csv")
write_named(methyl_gene_level_mean, "methyl_gene_level_mean.csv")
write_named(mean_expr_with_mapped_protein, "mean_expr_with_mapped_protein.csv")
write_matrix(cov_M, "cov_M.csv")
write_matrix(cov_expr, "cov_expr.csv")
write_matrix(cov_protein, "cov_protein.csv")
write_vector(rho_methyl_expr, "rho_methyl_expr.csv")
write_vector(rho_expr_protein, "rho_expr_protein.csv")
write_matrix(incidence_cpg_gene, "incidence_cpg_gene.csv")
write_matrix(incidence_gene_protein, "incidence_gene_protein.csv")

provenance <- c(
  paste0("intersim_version: ", as.character(packageVersion("InterSIM"))),
  paste0("r_version: ", as.character(getRversion())),
  paste0("export_date: ", format(Sys.Date(), "%Y-%m-%d")),
  paste0("export_script: export_reference.R"),
  paste0("n_cpg: ", length(cpg_names)),
  paste0("n_gene: ", length(gene_names)),
  paste0("n_protein: ", length(protein_names))
)
writeLines(provenance, file.path(output_dir, "provenance.txt"))

cat("Exported InterSIM reference data to", output_dir, "\n")
