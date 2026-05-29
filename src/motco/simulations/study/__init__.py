"""Declarative, cluster-executable trajectory power study."""

from motco.simulations.study.config import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    SpecificityTarget,
    StudyConfig,
    StudyConfigError,
    TypeIControlTarget,
    dump_study_config,
    load_study_config,
)
from motco.simulations.study.enumerate import enumerate_study
from motco.simulations.study.merge import StudyMergeError, merge_shards
from motco.simulations.study.report import (
    StudyReportError,
    build_power_curves,
    build_specificity_matrix,
    build_type_i_table,
    render_power_curves,
    render_specificity_matrix,
    render_type_i_plot,
    write_report_csvs,
)
from motco.simulations.study.sharding import (
    StudyShardError,
    enumerate_units,
    partition_unit,
    run_shard,
)
from motco.simulations.study.summary import (
    CombinedRuleSummary,
    summarize_combined_rule,
    summarize_study,
)
from motco.simulations.study.targets import (
    TargetEvaluation,
    evaluate_targets,
    write_target_report,
)

__all__ = [
    "AcceptanceTargets",
    "CombinedRuleSummary",
    "PowerMonotonicityTarget",
    "SpecificityTarget",
    "StudyConfig",
    "StudyConfigError",
    "StudyMergeError",
    "StudyReportError",
    "StudyShardError",
    "TargetEvaluation",
    "TypeIControlTarget",
    "build_power_curves",
    "build_specificity_matrix",
    "build_type_i_table",
    "dump_study_config",
    "enumerate_study",
    "enumerate_units",
    "evaluate_targets",
    "load_study_config",
    "merge_shards",
    "partition_unit",
    "render_power_curves",
    "render_specificity_matrix",
    "render_type_i_plot",
    "run_shard",
    "summarize_combined_rule",
    "summarize_study",
    "write_report_csvs",
    "write_target_report",
]
