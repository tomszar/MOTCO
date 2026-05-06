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

get_required <- function(parsed, key) {
  value <- parsed[[key]]
  if (is.null(value) || value == "") {
    stop(paste("Missing required argument:", paste0("--", key)))
  }
  value
}

get_numeric <- function(parsed, key) {
  value <- parsed[[key]]
  if (is.null(value) || value == "") {
    return(NULL)
  }
  as.numeric(value)
}

get_character <- function(parsed, key) {
  value <- parsed[[key]]
  if (is.null(value) || value == "") {
    return(NULL)
  }
  value
}

get_numeric_vector <- function(parsed, key) {
  value <- parsed[[key]]
  if (is.null(value) || value == "") {
    return(NULL)
  }
  as.numeric(strsplit(value, ",", fixed = TRUE)[[1]])
}

parsed <- parse_args(args)
output_dir <- get_required(parsed, "output-dir")
seed <- as.integer(get_required(parsed, "seed"))

if (!requireNamespace("InterSIM", quietly = TRUE)) {
  stop("The R package InterSIM is not installed or cannot be loaded.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

call_args <- list(
  n.sample = get_numeric(parsed, "n-sample"),
  cluster.sample.prop = get_numeric_vector(parsed, "cluster-sample-prop"),
  delta.methyl = get_numeric(parsed, "delta-methyl"),
  delta.expr = get_numeric(parsed, "delta-expr"),
  delta.protein = get_numeric(parsed, "delta-protein"),
  p.DMP = get_numeric(parsed, "p-dmp"),
  p.DEG = get_numeric(parsed, "p-deg"),
  p.DEP = get_numeric(parsed, "p-dep"),
  sigma.methyl = get_character(parsed, "sigma-methyl"),
  sigma.expr = get_character(parsed, "sigma-expr"),
  sigma.protein = get_character(parsed, "sigma-protein"),
  cor.methyl.expr = get_numeric(parsed, "cor-methyl-expr"),
  cor.expr.protein = get_numeric(parsed, "cor-expr-protein"),
  do.plot = FALSE
)
call_args <- call_args[!vapply(call_args, is.null, logical(1))]

sim <- do.call(InterSIM::InterSIM, call_args)

clusters <- sim$clustering.assignment
names(clusters) <- c("sample_id", "cluster")

write.csv(sim$dat.methyl, file.path(output_dir, "methylation.csv"))
write.csv(sim$dat.expr, file.path(output_dir, "expression.csv"))
write.csv(sim$dat.protein, file.path(output_dir, "proteomics.csv"))
write.csv(clusters, file.path(output_dir, "clusters.csv"), row.names = FALSE)
