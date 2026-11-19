"""Block 2: replay producer Lambda (EventBridge schedule -> Kinesis).

Each invocation publishes one 5-minute slice per tick. EventBridge cannot fire more often
than once a minute, so in demo mode (TICK_SECONDS < 60) one invocation publishes
60 / TICK_SECONDS slices, waiting TICK_SECONDS between them. TICK_SECONDS must be 300
(real time, schedule every 5 min) or a divisor of 60 (10, 15, 20, 30, 60).

Record schema: {sensor_id, ts, speed_mph, lat, lon, incident_flag, source}
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from traffic_common import store
from traffic_common.config import Settings
from traffic_common.congestion import fmt_ts, parse_ts
from traffic_common.replay import MetrLaReplay, clock_replay_ts, incident_factor, la_now, next_cursor

_SOURCE = None


def source():
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = MetrLaReplay()
    return _SOURCE


def take_replay_ts(st: Settings) -> datetime:
    """The dataset slot to publish now (cursor mode advances the stored cursor)."""
    if st.replay_mode == "clock":
        return clock_replay_ts(la_now())
    cur = store.get_state("replay", "cursor")
    ts = parse_ts(cur["ts"]) if cur else parse_ts(st.replay_start)
    store.put_state("replay", "cursor", {"ts": fmt_ts(next_cursor(ts))})
    return ts


def build_records(ts: datetime, st: Settings, src=None) -> list[dict]:
    src = src or source()
    sensors = src.sensors()[: st.n_sensors]
    speeds = src.get_slice(ts)
    out = []
    for s in sensors:
        sid = s["sensor_id"]
        speed = speeds.get(sid)
        speed = None if speed is None or speed <= 0 else float(speed)
        factor = incident_factor(sid, ts, st.synthetic_incident_rate)
        incident = factor is not None and speed is not None
        if incident:
            speed = round(speed * factor, 1)
        out.append({
            "sensor_id": sid, "ts": fmt_ts(ts), "speed_mph": speed,
            "lat": s["lat"], "lon": s["lon"], "incident_flag": incident,
            "source": "synthetic-incident" if incident else "metr-la-replay",
        })
    return out


def put_records(records: list[dict]) -> int:
    kin = store.client("kinesis")
    sent = 0
    for i in range(0, len(records), 500):
        batch = [{"Data": (json.dumps(r) + "\n").encode(), "PartitionKey": r["sensor_id"]}
                 for r in records[i:i + 500]]
        for _attempt in range(3):
            resp = kin.put_records(StreamName=store.stream_name(), Records=batch)
            failed = [b for b, r in zip(batch, resp["Records"], strict=True) if "ErrorCode" in r]
            sent += len(batch) - len(failed)
            if not failed:
                break
            batch = failed
            time.sleep(0.5)
    return sent


def handler(event, context):
    st = Settings.from_env()
    event = event or {}
    ticks = int(event.get("ticks") or (60 // st.tick_seconds if st.tick_seconds < 60 else 1))
    published = []
    for i in range(ticks):
        start = time.monotonic()
        ts = take_replay_ts(st)
        records = build_records(ts, st)
        n = put_records(records)
        published.append({"ts": fmt_ts(ts), "records": n,
                          "incidents": sum(r["incident_flag"] for r in records)})
        print(json.dumps(published[-1]))
        if i < ticks - 1:
            time.sleep(max(0.0, st.tick_seconds - (time.monotonic() - start)))
    return {"published": published}
