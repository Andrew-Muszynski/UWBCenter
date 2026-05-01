"""Dual-pipeline distance ensemble used by combined .pkl files.

Wraps two pre-trained sklearn distance pipelines (short-range + long-range)
that may have different `dist_feats` lists.  Exposes a single .predict(X)
method where X is a numpy array aligned to the UNION of both feature lists
(same shape that uwb_ble_calibration.safe_build_X(df, union_feats) returns).

Predictions are combined by raw-distance routing: rows with raw distance
below `route_threshold_m` use the short-range model, the rest use the
long-range model.  Both models can also be blended (weighted average) by
setting `blend="weighted"`.

This module must be importable both when the pickle is *created* and when
it is *loaded*, so it lives at the project root and is imported by
uwb_ble_calibration.py at startup.
"""
from __future__ import annotations

import numpy as np


class DualPipelineEnsemble:
    def __init__(
        self,
        model_short,
        feats_short: list,
        model_long,
        feats_long: list,
        union_feats: list,
        raw_dist_idx: int,
        route_threshold_m: float = 0.5,
        blend: str = "route",
        weight_short: float = 0.5,
    ):
        self.model_short = model_short
        self.model_long = model_long
        self.feats_short = list(feats_short)
        self.feats_long = list(feats_long)
        self.union_feats = list(union_feats)
        self.idx_short = [self.union_feats.index(f) for f in self.feats_short]
        self.idx_long = [self.union_feats.index(f) for f in self.feats_long]
        self.raw_dist_idx = int(raw_dist_idx)
        self.route_threshold_m = float(route_threshold_m)
        self.blend = blend
        self.weight_short = float(weight_short)

    def _predict_pair(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = X[:, self.idx_short]
        Xl = X[:, self.idx_long]
        return self.model_short.predict(Xs), self.model_long.predict(Xl)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ps, pl = self._predict_pair(X)
        if self.blend == "route":
            raw = X[:, self.raw_dist_idx]
            return np.where(raw < self.route_threshold_m, ps, pl)
        if self.blend == "weighted":
            w = self.weight_short
            return w * ps + (1.0 - w) * pl
        raise ValueError(f"Unknown blend mode: {self.blend!r}")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "DualPipelineEnsemble("
            f"blend={self.blend!r}, threshold={self.route_threshold_m}, "
            f"feats_short={len(self.feats_short)}, feats_long={len(self.feats_long)}, "
            f"union={len(self.union_feats)})"
        )
