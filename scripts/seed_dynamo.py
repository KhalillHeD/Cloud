#!/usr/bin/env python3
"""Load the precomputed data into the deployed (or local) resources.

  meta      sensor_meta table + meta/sensors.json, meta/sensor_meta.json in S3
  replay    replay/DAY.json files to S3 (the producer's data source)
  lake      curated/ Parquet backfill of the whole METR-LA history (Athena history from day one)
  model     latest local model (ml/artifacts/latest) published if none is active
  reset     replay cursor back to REPLAY_START and 6 h of warm-up history, so the charts
            and the forecasts have data at the first tick

Usage: python scripts/seed_dynamo.py [meta replay lake model reset]   (default: all)
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common"), str(ROOT / "scripts")]

from traffic_common import store  # noqa: E402
from traffic_common.config import SLOT_MINUTES, Settings, load_dotenv  # noqa: E402
from traffic_common.congestion import fmt_ts, parse_ts  # noqa: E402

PROC = ROOT / "data" / "processed"
MIRROR = PROC / "s3"
STEPS = ("meta", "replay", "lake", "model", "reset")


def log(msg: str) -> None:
    print(msg, flush=True)


def require_processed() -> dict:
    f = PROC / "sensor_meta.json"
    if not f.exists():
        raise SystemExit("data/processed missing: run `python tasks.py download` and `python tasks.py eda`.")
    return json.loads(f.read_text(encoding="utf-8"))


def seed_meta(st: Settings) -> None:
    meta = require_processed()
    if meta["n_sensors"] != st.n_sensors:
        raise SystemExit(f"processed data is for {meta['n_sensors']} sensors but N_SENSORS={st.n_sensors}: "
                         "rerun `python tasks.py eda` (and `train`).")
    items = [store.encode_meta({k: v for k, v in m.items() if k != "neighbor_idx"} | {"idx": m["idx"]})
             for m in meta["sensors"]]
    store.batch_put("sensor_meta", items)
    s3 = store.client("s3")
    s3.upload_file(str(MIRROR / "meta" / "sensors.json"), store.lake_bucket(), "meta/sensors.json")
    light = [{k: m[k] for k in ("sensor_id", "lat", "lon", "free_flow_mph", "segment_length_m")}
             for m in meta["sensors"]]
    s3.put_object(Bucket=store.lake_bucket(), Key="meta/sensor_meta.json",
                  Body="\n".join(json.dumps(x) for x in light).encode())
    log(f"  sensor_meta: {len(items)} sensors")


def upload_replay(st: Settings) -> None:
    from concurrent.futures import ThreadPoolExecutor
    files = sorted((MIRROR / "replay").glob("*.json"))
    s3 = store.client("s3")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda f: s3.upload_file(str(f), store.lake_bucket(), f"replay/{f.name}"), files))
    log(f"  replay: {len(files)} day files")


def backfill_lake(st: Settings) -> None:
    """curated/dt=YYYY-MM-DD/part-backfill.parquet, same columns as the Glue ETL output."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ml.dataset import read_metr_la_h5
    meta = require_processed()["sensors"]
    df = read_metr_la_h5(ROOT / "data" / "metr-la.h5")[[m["sensor_id"] for m in meta]]
    ff = pd.Series({m["sensor_id"]: m["free_flow_mph"] for m in meta})
    seg = pd.Series({m["sensor_id"]: m["segment_length_m"] for m in meta})
    long = df.replace(0.0, np.nan).stack(future_stack=True).rename("speed_mph").reset_index()
    long.columns = ["ts", "sensor_id", "speed_mph"]
    long["free_flow_mph"] = long["sensor_id"].map(ff)
    ci = (1 - long["speed_mph"] / long["free_flow_mph"]).clip(lower=0)
    long["congestion_index"] = ci
    bins = [-1, st.level_moderate, st.level_heavy, st.level_severe, 10]
    long["level"] = pd.cut(ci, bins=bins, labels=["free", "moderate", "heavy", "severe"], right=False).astype(str)
    long.loc[ci.isna(), "level"] = "unknown"
    long["travel_time_s"] = long["sensor_id"].map(seg) / (long["speed_mph"] * 0.44704)
    long["incident_flag"] = False
    long["source"] = "metr-la-backfill"
    long["hour"] = long["ts"].dt.hour.astype("int32")
    long["dow"] = long["ts"].dt.dayofweek.astype("int32")
    long["dt"] = long["ts"].dt.strftime("%Y-%m-%d")
    long["ts"] = long["ts"].astype("datetime64[ms]")           # Athena cannot read nanosecond timestamps
    long["speed_filled"] = long["speed_mph"]
    for col in ("speed_mph", "speed_filled", "free_flow_mph", "congestion_index", "travel_time_s"):
        long[col] = long[col].astype("float64")                 # Glue table declares double
    s3 = store.client("s3")
    n = 0
    for dt, part in long.groupby("dt"):
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(part.drop(columns="dt"), preserve_index=False), buf, compression="snappy")
        s3.put_object(Bucket=store.lake_bucket(), Key=f"curated/dt={dt}/part-backfill.parquet", Body=buf.getvalue())
        n += 1
    log(f"  lake: {len(long):,} rows in {n} curated partitions")


def publish_local_model(st: Settings, force: bool = False) -> None:
    latest = ROOT / "ml" / "artifacts" / "latest"
    if not (latest / "model.json").exists():
        log("  model: no local model yet (run `python tasks.py train`); forecasts use the baseline")
        return
    if store.get_state("model", "active") and not force:
        log("  model: an active model is already published")
        return
    from ml.train import publish
    publish(st, latest, "local-" + time.strftime("%Y%m%d-%H%M%S"), None)


def reset_replay(st: Settings, warm_hours: float = 6.0) -> str:
    from components import load
    producer, processing = load("producer"), load("processing")
    from traffic_common.replay import clock_replay_ts, la_now
    start = parse_ts(st.replay_start) if st.replay_mode == "cursor" else clock_replay_ts(la_now())
    src = producer.source()
    steps = int(warm_hours * 60 / SLOT_MINUTES)
    for k in range(steps, 0, -1):
        ts = start - timedelta(minutes=SLOT_MINUTES * k)
        processing.process(producer.build_records(ts, st, src), st, publish=False)
    store.put_state("replay", "cursor", {"ts": fmt_ts(start)})
    log(f"  reset: cursor at {fmt_ts(start)} with {warm_hours:g} h of warm-up history")
    return fmt_ts(start)


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    st = Settings.from_env()
    steps = [s for s in (argv if argv is not None else sys.argv[1:]) if s in STEPS] or list(STEPS)
    log(f"Seeding {', '.join(steps)} into bucket {store.lake_bucket()} / tables {store.table_name('*')}")
    if "meta" in steps:
        seed_meta(st)
    if "replay" in steps:
        upload_replay(st)
    if "lake" in steps:
        backfill_lake(st)
    if "model" in steps:
        publish_local_model(st)
    if "reset" in steps:
        reset_replay(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
