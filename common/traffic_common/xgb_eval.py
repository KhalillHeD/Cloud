"""Pure-Python evaluator for an XGBoost model saved as JSON (reg:squarederror, gbtree).

Lets the forecast Lambda run the trained model without numpy/xgboost (small package, no
endpoint cost). tests/test_xgb_eval.py checks it against xgboost's own predictions.
"""
from __future__ import annotations

import math
import struct


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def _parse_base_score(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(str(raw).strip("[]").split(",")[0])   # xgboost >= 3 writes "[6.1E1]"


class JsonBooster:
    def __init__(self, model: dict):
        learner = model["learner"]
        objective = learner["objective"]["name"]
        if objective not in ("reg:squarederror", "reg:linear", "reg:absoluteerror", "reg:pseudohubererror"):
            raise ValueError(f"unsupported objective {objective}")
        self.base_score = _parse_base_score(learner["learner_model_param"]["base_score"])
        self.feature_names = learner.get("feature_names") or []
        self.trees = []
        for t in learner["gradient_booster"]["model"]["trees"]:
            self.trees.append((t["left_children"], t["right_children"], t["split_indices"],
                               [_f32(c) for c in t["split_conditions"]], [bool(d) for d in t["default_left"]]))

    def predict_row(self, x: list[float]) -> float:
        xs = [x_ if math.isnan(x_) else _f32(x_) for x_ in x]
        total = self.base_score
        for left, right, split, cond, default_left in self.trees:
            i = 0
            while left[i] != -1:
                v = xs[split[i]]
                if math.isnan(v):
                    i = left[i] if default_left[i] else right[i]
                else:
                    i = left[i] if v < cond[i] else right[i]
            total += cond[i]
        return total

    def predict(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_row(r) for r in rows]
