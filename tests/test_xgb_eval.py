"""The pure-Python evaluator used by the forecast Lambda must match xgboost itself."""
import json

import numpy as np
import pytest
from traffic_common.xgb_eval import JsonBooster, _parse_base_score

xgb = pytest.importorskip("xgboost")


def test_matches_xgboost_with_missing_values(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8)).astype("float32")
    y = (X[:, 0] * 3 + np.sin(X[:, 1]) * 5 + 60).astype("float32")
    X[rng.random(X.shape) < 0.15] = np.nan
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 5, "eta": 0.3, "tree_method": "hist"},
                        xgb.DMatrix(X, label=y, missing=np.nan), num_boost_round=40)
    path = tmp_path / "m.json"
    booster.save_model(str(path))
    ours = JsonBooster(json.loads(path.read_text()))
    Xt = rng.normal(size=(300, 8)).astype("float32")
    Xt[rng.random(Xt.shape) < 0.2] = np.nan
    ref = booster.predict(xgb.DMatrix(Xt, missing=np.nan))
    got = ours.predict([[float(v) for v in row] for row in Xt])
    assert np.allclose(got, ref, atol=1e-3)


def test_base_score_formats():
    assert _parse_base_score("5E-1") == 0.5
    assert _parse_base_score("[6.1E1]") == 61.0
    assert _parse_base_score(3) == 3.0
