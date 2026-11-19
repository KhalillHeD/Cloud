"""Training + evaluation, shared by local training and the SageMaker training job.

Input: training.npz written by scripts/precompute_freeflow.py
  speeds (T,S) NaN=missing, ts (T,) ISO strings, hour/dow/minute (T,), free_flow (S,),
  expected (S,2016) median profile, mean_profile (S,2016), neighbor_idx (S,k), sensor_ids (S,)
Output in out_dir: model.json (XGBoost JSON), metrics.json

One model covers every horizon ("horizon" is a feature), so a single endpoint or JSON
model serves +15, +30 and +60 min.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from ml.evaluate import metrics
from ml.features import FEATURE_NAMES, builder_from_npz

DEFAULT_PARAMS = {
    "objective": "reg:squarederror", "eta": 0.1, "max_depth": 7, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "mae",
}


def _rows(T: int, S: int, t_lo: int, t_hi: int, steps: list[int]):
    t = np.arange(t_lo, t_hi)
    tt, ss, hh = np.meshgrid(t, np.arange(S), np.array(steps), indexing="ij")
    return tt.ravel(), ss.ravel(), hh.ravel()


def train(npz_path: str | Path, out_dir: str | Path, horizons_min=(15, 30, 60), num_round: int = 300,
          max_train_rows: int = 1_500_000, params: dict | None = None, trained_on: str = "local",
          seed: int = 7) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(npz_path))
    fb = builder_from_npz(d)
    T, S = d["speeds"].shape
    steps = [h // 5 for h in horizons_min]
    hmax = max(steps)
    n_train, n_val = int(T * 0.7), int(T * 0.8)
    rng = np.random.default_rng(seed)

    def split(lo, hi, cap):
        t, s, h = _rows(T, S, lo, hi, steps)
        y = fb.target(t, s, h)
        keep = np.isfinite(y)
        t, s, h, y = t[keep], s[keep], h[keep], y[keep]
        if cap and len(t) > cap:
            pick = np.sort(rng.choice(len(t), cap, replace=False))
            t, s, h, y = t[pick], s[pick], h[pick], y[pick]
        return fb.build(t, s, h), y, t, s, h

    # 12-step warm-up at the start; no target may cross into the next split
    Xtr, ytr, *_ = split(12, n_train - hmax, max_train_rows)
    Xva, yva, *_ = split(n_train, n_val - hmax, max_train_rows // 5)
    Xte, yte, tte, ste, hte = split(n_val, T - hmax, None)
    print(f"rows: train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}  ({time.time() - t0:.0f}s)", flush=True)

    p = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES, missing=np.nan)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=FEATURE_NAMES, missing=np.nan)
    booster = xgb.train(p, dtr, num_boost_round=num_round, evals=[(dtr, "train"), (dva, "val")],
                        early_stopping_rounds=30, verbose_eval=50)
    best = booster.best_iteration + 1
    booster = booster[:best]                         # keep only the best trees
    pred = booster.predict(xgb.DMatrix(Xte, feature_names=FEATURE_NAMES, missing=np.nan))

    # baselines on exactly the same test rows
    lags = Xte[:, :12]
    first_obs = np.argmax(np.isfinite(lags), axis=1)
    last_value = lags[np.arange(len(lags)), first_obs]
    exp_target = Xte[:, FEATURE_NAMES.index("expected_target")]
    last_value = np.where(np.isfinite(last_value), last_value, exp_target)
    wslot = (d["dow"][tte] * 288 + (d["hour"][tte] * 60 + d["minute"][tte]) // 5 + hte) % 2016
    hist_avg = d["mean_profile"][ste, wslot]

    result = {"horizons": {}}
    for hm, st in zip(horizons_min, steps):
        m = hte == st
        result["horizons"][str(hm)] = {
            "last_value": metrics(yte[m], last_value[m]),
            "historical_average": metrics(yte[m], hist_avg[m]),
            "xgboost": metrics(yte[m], pred[m]),
        }
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    result.update({
        "n_sensors": int(S), "n_trees": int(best), "n_train_rows": int(len(ytr)), "trained_on": trained_on,
        "split": {"train_end": str(d["ts"][n_train - 1]), "val_end": str(d["ts"][n_val - 1]),
                  "test_start": str(d["ts"][n_val])},
        "feature_names": FEATURE_NAMES, "horizons_min": list(horizons_min), "params": p,
        "feature_importance": {k: v / total for k, v in gain.items()},
        "train_seconds": round(time.time() - t0, 1),
    })
    booster.save_model(str(out / "model.json"))
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for hm, per in result["horizons"].items():
        print(f"+{hm:>2} min  MAE  last={per['last_value']['mae']:.3f}  hist={per['historical_average']['mae']:.3f}"
              f"  xgb={per['xgboost']['mae']:.3f}")
    return result
