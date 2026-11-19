#!/usr/bin/env python3
"""Phase 1 precomputation, from data/ to data/processed/.

  sensor_meta.json   the N_SENSORS selected sensors: free-flow speed (85th percentile of the
                     training period), segment length, 3 nearest neighbours, weekly median
                     profile (expected speed per weekday x 5-min slot) and mean profile
  sensors.json       all 207 sensors ranked by distance to the network centre (meta/sensors.json in S3)
  s3/replay/DAY.json one file per day, 288 slots x 207 sensors, null = missing (replay source)
  training.npz       arrays for ml/train_core.py (local or SageMaker)

Usage: python scripts/precompute_freeflow.py [--n-sensors 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]

from traffic_common.config import Settings, load_dotenv  # noqa: E402

from ml.dataset import prepare  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "processed"
MIRROR = OUT / "s3"   # same layout as the lake bucket: meta/sensors.json, replay/DAY.json


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    st = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sensors", type=int, default=st.n_sensors)
    args = ap.parse_args(argv)
    if not (DATA / "metr-la.h5").exists():
        print("data/metr-la.h5 missing: run `python tasks.py download` first.")
        return 1
    print(f"Preparing {args.n_sensors} sensors ...")
    p = prepare(DATA, args.n_sensors, st.freeflow_percentile)
    (MIRROR / "replay").mkdir(parents=True, exist_ok=True)
    (MIRROR / "meta").mkdir(exist_ok=True)

    meta = p["sensors"]
    (OUT / "sensor_meta.json").write_text(json.dumps({"n_sensors": args.n_sensors, "sensors": meta}),
                                          encoding="utf-8")
    loc = p["loc"]
    ranked = [{"sensor_id": s, "lat": float(loc.at[s, "lat"]), "lon": float(loc.at[s, "lon"]), "rank": i}
              for i, s in enumerate(p["ranked"])]
    (OUT / "sensors.json").write_text(json.dumps(ranked), encoding="utf-8")
    (MIRROR / "meta" / "sensors.json").write_text(json.dumps(ranked), encoding="utf-8")

    allsp = p["speeds_all"][p["ranked"]]
    ids = list(allsp.columns)
    n_days = 0
    for day, frame in allsp.groupby(allsp.index.date):
        frame = frame.reindex(
            frame.index[0].normalize() + np.arange(288) * np.timedelta64(5, "m"))  # full 288 slots
        rows = [[None if np.isnan(v) else round(float(v), 1) for v in r] for r in frame.to_numpy()]
        (MIRROR / "replay" / f"{day}.json").write_text(
            json.dumps({"date": str(day), "sensors": ids, "speeds": rows}, separators=(",", ":")),
            encoding="utf-8")
        n_days += 1

    sp = p["speeds"]
    idx = sp.index
    np.savez_compressed(
        OUT / "training.npz",
        speeds=sp.to_numpy(dtype="float32"),
        ts=np.array(list(idx.strftime("%Y-%m-%dT%H:%M:%S")), dtype="U19"),
        hour=idx.hour.to_numpy(), dow=idx.day_of_week.to_numpy(), minute=idx.minute.to_numpy(),
        free_flow=np.array([m["free_flow_mph"] for m in meta], dtype="float32"),
        expected=np.array([m["expected_speed"] for m in meta], dtype="float32"),
        mean_profile=np.array([m["mean_speed"] for m in meta], dtype="float32"),
        neighbor_idx=np.array([m["neighbor_idx"] for m in meta], dtype="int64"),
        sensor_ids=np.array([m["sensor_id"] for m in meta], dtype="U16"),
    )
    ff = [m["free_flow_mph"] for m in meta]
    missing = float(np.isnan(sp.to_numpy()).mean() * 100)
    print(f"  {len(meta)} sensors, free-flow {min(ff):.1f}-{max(ff):.1f} mph (median {np.median(ff):.1f}), "
          f"{missing:.1f}% missing readings, {n_days} replay days")
    print(f"Written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
