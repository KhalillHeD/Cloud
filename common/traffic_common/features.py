"""Forecast features, pure Python (used by the forecast Lambda, no numpy needed).

ml/features.py builds exactly the same columns with numpy for training; a test checks the
two agree. Missing speeds are None here and NaN there; XGBoost routes them with its
learned default direction.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from .config import SLOT_MINUTES
from .congestion import SLOTS_PER_DAY, parse_ts, week_slot

N_LAGS = 12  # one hour of history
FEATURE_NAMES = (
    [f"lag_{i}" for i in range(N_LAGS)]
    + ["mean_3", "mean_12", "std_12", "delta_1", "delta_3", "nbr_mean", "nbr_delta",
       "free_flow", "expected_now", "expected_target", "hour", "dow", "slot_sin", "slot_cos",
       "horizon", "sensor_idx"]
)
NAN = float("nan")


def _vals(xs):
    return [x for x in xs if x is not None and not math.isnan(x)]


def nanmean(xs) -> float:
    v = _vals(xs)
    return sum(v) / len(v) if v else NAN


def nanstd(xs) -> float:
    v = _vals(xs)
    if not v:
        return NAN
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _f(x) -> float:
    return NAN if x is None else float(x)


def feature_row(lags: list, nbr_lag0: list, nbr_lag3: list, free_flow: float, expected: list,
                ts: str | datetime, horizon_steps: int, sensor_idx: int) -> list[float]:
    """lags: speeds newest first (lags[0] = speed at ts), length N_LAGS, None = missing.
    nbr_lag0 / nbr_lag3: neighbours' speeds at ts and 3 slots earlier."""
    lags = [_f(x) for x in (list(lags) + [None] * N_LAGS)[:N_LAGS]]
    t = parse_ts(ts)
    target = t + timedelta(minutes=SLOT_MINUTES * horizon_steps)
    day_slot = (t.hour * 60 + t.minute) // SLOT_MINUTES
    angle = 2 * math.pi * day_slot / SLOTS_PER_DAY
    nm0, nm3 = nanmean(nbr_lag0), nanmean(nbr_lag3)
    return lags + [
        nanmean(lags[:3]), nanmean(lags), nanstd(lags),
        lags[0] - lags[1], lags[0] - lags[3],
        nm0, nm0 - nm3,
        float(free_flow), float(expected[week_slot(t)]), float(expected[week_slot(target)]),
        float(t.hour), float(t.weekday()), math.sin(angle), math.cos(angle),
        float(horizon_steps), float(sensor_idx),
    ]
