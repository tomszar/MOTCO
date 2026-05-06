"""MOTCO package.

Multi-omics Trajectory Comparison utilities:
- Latent space generation (PLS-DA, SNF)
- Group differences on multivariate trajectories
"""

from motco.viz import plot_trajectories, plot_trajectory_from_data

__all__ = [
    "simulations",
    "stats",
    "plot_trajectories",
    "plot_trajectory_from_data",
]

__version__ = "0.4.0"
