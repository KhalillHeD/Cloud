"""Vectorised feature builder for training (numpy only, so it also runs in the SageMaker
XGBoost container). Same columns, same order, same NaN rules as
common/traffic_common/features.py::feature_row, which the forecast Lambda uses.
"""
from __future__ import annotations

import warnings

import numpy as np

N_LAGS = 12
FEATURE_NAMES = (
    [f"lag_{i}" for i in range(N_LAGS)]
    + ["mean_3", "mean_12", "std_12", "delta_1", "delta_3", "nbr_mean", "nbr_delta",
       "free_flow", "expected_now", "expected_target", "hour", "dow", "slot_sin", "slot_cos",
       "horizon", "sensor_idx"]
)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Value k steps earlier (NaN before the start)."""
    if k == 0:
        return a
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _nanmean(stack, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=axis)


def _nanstd(stack, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(stack, axis=axis)


class FeatureBuilder:
    """speeds: (T, S) float array, NaN = missing. minutes: (T,) minutes since Monday 00:00 of
    the week of each timestamp + absolute info (hour, dow) passed separately."""

    def __init__(self, speeds: np.ndarray, hour: np.ndarray, dow: np.ndarray, minute: np.ndarray,
                 free_flow: np.ndarray, expected: np.ndarray, neighbor_idx: np.ndarray):
        self.speeds = speeds.astype("float32")
        self.T, self.S = speeds.shape
        self.hour, self.dow, self.minute = hour, dow, minute
        self.day_slot = (hour * 60 + minute) // 5
        self.wslot = dow * 288 + self.day_slot
        self.free_flow = free_flow
        self.expected = expected                      # (S, 2016)
        self.lags = np.stack([_shift(self.speeds, k) for k in range(N_LAGS)])   # (12, T, S)
        self.mean3 = _nanmean(self.lags[:3], 0)
        self.mean12 = _nanmean(self.lags, 0)
        self.std12 = _nanstd(self.lags, 0)
        nb = neighbor_idx                              # (S, k), -1 = none
        l0, l3 = self.lags[0], self.lags[3]
        nm0 = np.full((self.T, self.S), np.nan, dtype="float32")
        nm3 = np.full((self.T, self.S), np.nan, dtype="float32")
        for s in range(self.S):
            idx = [j for j in nb[s] if j >= 0]
            if idx:
                nm0[:, s] = _nanmean(l0[:, idx], 1)
                nm3[:, s] = _nanmean(l3[:, idx], 1)
        self.nm0, self.nm3 = nm0, nm3

    def target(self, t: np.ndarray, s: np.ndarray, h: np.ndarray) -> np.ndarray:
        tt = t + h
        y = np.full(len(t), np.nan, dtype="float32")
        ok = tt < self.T
        y[ok] = self.speeds[tt[ok], s[ok]]
        return y

    def build(self, t: np.ndarray, s: np.ndarray, h: np.ndarray) -> np.ndarray:
        n = len(t)
        X = np.empty((n, len(FEATURE_NAMES)), dtype="float32")
        X[:, :N_LAGS] = self.lags[:, t, s].T
        lag0, lag1, lag3 = X[:, 0], X[:, 1], X[:, 3]
        c = N_LAGS
        X[:, c] = self.mean3[t, s]
        X[:, c + 1] = self.mean12[t, s]
        X[:, c + 2] = self.std12[t, s]
        X[:, c + 3] = lag0 - lag1
        X[:, c + 4] = lag0 - lag3
        X[:, c + 5] = self.nm0[t, s]
        X[:, c + 6] = self.nm0[t, s] - self.nm3[t, s]
        X[:, c + 7] = self.free_flow[s]
        X[:, c + 8] = self.expected[s, self.wslot[t]]
        # target slot computed from wall-clock arithmetic (week wraps), not from the index
        tgt_wslot = (self.wslot[t] + h) % 2016
        X[:, c + 9] = self.expected[s, tgt_wslot]
        X[:, c + 10] = self.hour[t]
        X[:, c + 11] = self.dow[t]
        ang = 2 * np.pi * self.day_slot[t] / 288
        X[:, c + 12] = np.sin(ang)
        X[:, c + 13] = np.cos(ang)
        X[:, c + 14] = h
        X[:, c + 15] = s
        return X


def builder_from_npz(d) -> FeatureBuilder:
    return FeatureBuilder(d["speeds"], d["hour"], d["dow"], d["minute"], d["free_flow"], d["expected"],
                          d["neighbor_idx"])
