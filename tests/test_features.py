"""The training features (numpy) and the Lambda features (pure Python) must be identical."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from traffic_common import features as pure

from ml import features as vec


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    T, S = 400, 5
    speeds = (55 + 10 * rng.standard_normal((T, S))).astype("float32")
    speeds[rng.random((T, S)) < 0.1] = np.nan            # missing readings
    speeds[:20, 2] = np.nan                               # a long gap
    t0 = datetime(2012, 3, 4, 22, 0)                      # crosses Sunday -> Monday (week wrap)
    times = [t0 + timedelta(minutes=5 * i) for i in range(T)]
    hour = np.array([t.hour for t in times])
    dow = np.array([t.weekday() for t in times])
    minute = np.array([t.minute for t in times])
    free_flow = np.array([65, 60, 68, 62, 66], dtype="float32")
    expected = (50 + 10 * rng.random((S, 2016))).astype("float32").round(1)
    nbr = np.array([[1, 2, 3], [0, 2, -1], [1, 3, 4], [2, 4, 0], [3, 2, 1]])
    fb = vec.FeatureBuilder(speeds, hour, dow, minute, free_flow, expected, nbr)
    return speeds, times, free_flow, expected, nbr, fb


def test_same_feature_names():
    assert list(vec.FEATURE_NAMES) == list(pure.FEATURE_NAMES)


@pytest.mark.parametrize("t,s,h", [(0, 0, 3), (5, 2, 6), (15, 2, 3), (100, 1, 12), (287, 4, 6), (399, 3, 3),
                                   (26, 0, 12)])
def test_vectorised_equals_pure(data, t, s, h):
    speeds, times, free_flow, expected, nbr, fb = data
    X = fb.build(np.array([t]), np.array([s]), np.array([h]))[0]

    def val(tt, ss):
        if tt < 0:
            return None
        v = speeds[tt, ss]
        return None if np.isnan(v) else float(v)

    lags = [val(t - k, s) for k in range(12)]
    nb = [j for j in nbr[s] if j >= 0]
    row = pure.feature_row(lags, [val(t, j) for j in nb], [val(t - 3, j) for j in nb], float(free_flow[s]),
                           [float(x) for x in expected[s]], times[t], h, s)
    assert len(row) == len(X)
    for name, a, b in zip(pure.FEATURE_NAMES, row, X, strict=True):
        if math.isnan(a):
            assert np.isnan(b), name
        else:
            assert a == pytest.approx(float(b), rel=1e-4, abs=1e-4), name


def test_target_is_future_speed(data):
    speeds, *_, fb = data
    y = fb.target(np.array([10, 399]), np.array([1, 1]), np.array([3, 3]))
    assert y[0] == speeds[13, 1] or (np.isnan(y[0]) and np.isnan(speeds[13, 1]))
    assert np.isnan(y[1])                                  # beyond the end
