"""Metrics and the results table (docs/results.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE and MAPE; rows with a missing/zero truth are ignored (METR-LA zeros = missing)."""
    ok = np.isfinite(y_true) & (y_true > 0) & np.isfinite(y_pred)
    t, p = y_true[ok].astype("float64"), y_pred[ok].astype("float64")
    err = p - t
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / t) * 100),
        "n": int(ok.sum()),
    }


def results_markdown(m: dict) -> str:
    lines = [
        "# Results",
        "",
        f"Forecast of speed (mph) on METR-LA, {m['n_sensors']} sensors, chronological split "
        f"70/10/20 (train until {m['split']['train_end']}, test from {m['split']['test_start']}).",
        f"Model: XGBoost ({m['n_trees']} trees, trained on {m['n_train_rows']:,} rows, "
        f"where: {m['trained_on']}). MAPE ignores missing (zero) readings.",
        "",
        "| Horizon | Method | MAE (mph) | RMSE (mph) | MAPE (%) | Rows |",
        "|---|---|---|---|---|---|",
    ]
    for h, per in m["horizons"].items():
        best = min(per, key=lambda k: per[k]["mae"])
        for name in ("last_value", "historical_average", "xgboost"):
            r = per[name]
            label = {"last_value": "Last value", "historical_average": "Historical average (same slot)",
                     "xgboost": "**XGBoost**"}[name]
            mark = " ✓" if name == best else ""
            lines.append(f"| +{h} min | {label}{mark} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['mape']:.2f} | {r['n']:,} |")
    lines += ["", "✓ = lowest MAE for that horizon.", ""]
    beats = all(per["xgboost"]["mae"] < min(per["last_value"]["mae"], per["historical_average"]["mae"])
                for h, per in m["horizons"].items() if h in ("15", "30"))
    lines.append(f"XGBoost beats both baselines at +15 and +30 min: **{'yes' if beats else 'no'}**.")
    if m.get("feature_importance"):
        top = sorted(m["feature_importance"].items(), key=lambda kv: -kv[1])[:8]
        lines += ["", "Most useful features (total gain share): " +
                  ", ".join(f"`{k}` {v:.0%}" for k, v in top) + "."]
    return "\n".join(lines) + "\n"


def write_results(metrics_path: str | Path, out_md: str | Path) -> str:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    text = results_markdown(m)
    out = Path(out_md)
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    marker = "<!-- end of generated metrics -->"
    tail = existing.split(marker, 1)[1] if marker in existing else ""
    out.write_text(text + "\n" + marker + tail, encoding="utf-8")
    return text
