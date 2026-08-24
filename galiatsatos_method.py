"""Public import surface for the GALIATSATOS machine-learning method."""

from plabn import (
    GaliatsatosMethod,
    coarsening_rank,
    make_ordered_threshold_operator,
    sample_observed_labels,
)

__all__ = [
    "GaliatsatosMethod",
    "coarsening_rank",
    "make_ordered_threshold_operator",
    "sample_observed_labels",
]
