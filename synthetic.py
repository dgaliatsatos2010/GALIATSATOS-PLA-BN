"""Synthetic ordered-class TAN data for the PLA-BN prototype."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


TRUE_PARENTS = np.array([-1, 0, 0, 1, 1, 2, 2, 4], dtype=int)
TRUE_PRIOR = np.array([0.34, 0.28, 0.23, 0.15], dtype=float)


def simulate_ordered_tan(
    n_samples: int,
    rng: np.random.Generator,
    parents: np.ndarray = TRUE_PARENTS,
    class_prior: np.ndarray = TRUE_PRIOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate eight binary features from a four-class TAN."""
    n_features = len(parents)
    latent = rng.choice(len(class_prior), size=n_samples, p=class_prior)
    features = np.zeros((n_samples, n_features), dtype=int)
    base = np.array([-0.15, 0.25, -0.35, 0.10, -0.20, 0.30, -0.10, 0.20])
    slope = np.array([0.95, -0.70, 0.85, -0.80, 0.72, -0.76, 0.64, -0.68])
    parent_effect = np.array([0.0, 1.05, -1.00, 1.10, -0.95, 1.00, -1.08, 0.98])

    for feature_index, parent_index in enumerate(parents):
        centered_class = latent - 1.5
        linear_predictor = base[feature_index] + slope[feature_index] * centered_class
        if parent_index >= 0:
            linear_predictor += parent_effect[feature_index] * (
                2 * features[:, parent_index] - 1
            )
        probability = np.clip(expit(linear_predictor), 0.03, 0.97)
        features[:, feature_index] = rng.binomial(1, probability)
    return features, latent

