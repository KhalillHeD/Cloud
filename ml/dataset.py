"""Load METR-LA and precompute per-sensor statistics (Phase 1).

Everything statistical is computed on the training period only (first 70 % of time), so
the free-flow speed, the expected-time profile and the "historical average" baseline never
see the test period.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = 2016
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1
DEFAULT_SEGMENT_M = 800.0


def read_metr_la_h5(path: str | Path) -> pd.DataFrame:
    """Read the DCRNN pandas 'fixed' HDF5 file with h5py (no PyTables dependency)."""
    with h5py.File(path, "r") as f:
        key = "df" if "df" in f else next(iter(f.keys()))
        g = f[key]
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in g["block0_items"][:]]
        index = pd.to_datetime(g["axis1"][:])
        values = g["block0_values"][:]
    return pd.DataFrame(values.astype("float32"), index=index, columns=cols)


def read_locations(path: str | Path) -> pd.DataFrame:
    loc = pd.read_csv(path)
    loc.columns = [c.strip().lower() for c in loc.columns]
    loc = loc.rename(columns={"latitude": "lat", "longitude": "lon"})
    loc["sensor_id"] = loc["sensor_id"].astype(str)
    return loc[["sensor_id", "lat", "lon"]].set_index("sensor_id")


def read_distances(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"from": str, "to": str})
    return d[d["from"] != d["to"]]


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rank_sensors(loc: pd.DataFrame) -> list[str]:
    """Order sensors by distance to the network centroid: any prefix of length N is a compact area."""
    clat, clon = loc["lat"].median(), loc["lon"].median()
    dist = {s: haversine_m(r.lat, r.lon, clat, clon) for s, r in loc.iterrows()}
    return sorted(dist, key=lambda s: (dist[s], s))


def week_slots(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.day_of_week * SLOTS_PER_DAY + (index.hour * 60 + index.minute) // 5).to_numpy()


def split_points(n: int) -> tuple[int, int]:
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def prepare(data_dir: str | Path, n_sensors: int, freeflow_pct: float = 85.0, k_neighbors: int = 3) -> dict:
    data_dir = Path(data_dir)
    df = read_metr_la_h5(data_dir / "metr-la.h5")
    loc = read_locations(data_dir / "graph_sensor_locations.csv")
    dist = read_distances(data_dir / "distances_la_2012.csv")
    ranked = [s for s in rank_sensors(loc) if s in df.columns]
    selected = ranked[:n_sensors]
    speeds_all = df.replace(0.0, np.nan)              # METR-LA zeros = missing readings
    speeds = speeds_all[selected]
    n_train, _ = split_points(len(speeds))
    train = speeds.iloc[:n_train]
    slots = week_slots(train.index)

    free_flow = np.nanpercentile(train.to_numpy(), freeflow_pct, axis=0)
    by_slot = train.groupby(slots)
    med = by_slot.median().reindex(range(SLOTS_PER_WEEK))
    mean = by_slot.mean().reindex(range(SLOTS_PER_WEEK))
    med = med.fillna(train.median())                  # slots never observed: overall median
    mean = mean.fillna(train.mean())

    # segment length: road distance to the nearest downstream sensor (clipped to a sane range)
    out_min = dist.groupby("from")["cost"].min()
    pair = {}
    for r in dist.itertuples(index=False):
        key = tuple(sorted((r[0], r[1])))
        pair[key] = min(pair.get(key, math.inf), r[2])
    sensors = []
    for i, s in enumerate(selected):
        cand = []
        for j, o in enumerate(selected):
            if o == s:
                continue
            d = pair.get(tuple(sorted((s, o))))
            if d is None:   # no road path recorded: straight-line distance, penalised
                d = 1.5 * haversine_m(loc.at[s, "lat"], loc.at[s, "lon"], loc.at[o, "lat"], loc.at[o, "lon"])
            cand.append((d, j))
        neighbors = [j for _, j in sorted(cand)[:k_neighbors]]
        seg = float(np.clip(out_min.get(s, DEFAULT_SEGMENT_M), 200.0, 3000.0))
        sensors.append({
            "sensor_id": s, "idx": i, "rank": i,
            "lat": float(loc.at[s, "lat"]), "lon": float(loc.at[s, "lon"]),
            "free_flow_mph": round(float(free_flow[i]), 2),
            "segment_length_m": round(seg, 1),
            "neighbors": [selected[j] for j in neighbors],
            "neighbor_idx": neighbors,
            "expected_speed": [round(float(v), 1) for v in med[s].to_numpy()],
            "mean_speed": [round(float(v), 1) for v in mean[s].to_numpy()],
        })
    return {"speeds": speeds, "speeds_all": speeds_all, "sensors": sensors, "ranked": ranked, "loc": loc,
            "n_train": n_train}
