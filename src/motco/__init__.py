"""MOTCO package.

Multi-omics Trajectory Comparison utilities:
- Latent space generation (PLS-DA, SNF)
- Group differences on multivariate trajectories
"""

from motco.viz import plot_trajectories, plot_trajectory_from_data, plot_trajectory_from_plsr

__all__ = [
    "simulations",
    "stats",
    "plot_trajectories",
    "plot_trajectory_from_data",
    "plot_trajectory_from_plsr",
]

__version__ = "0.6.0"
