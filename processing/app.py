"""Block 4: congestion processor Lambda (Kinesis trigger).

For each record: congestion index, level, travel time, expected travel time, delay
(formula in traffic_common/congestion.py). Writes traffic_latest and traffic_history, and
publishes an alert event when a segment escalates to heavy or severe.
"""
from __future__ import annotations

import base64
import json

from traffic_common import store
from traffic_common.config import Settings
from traffic_common.congestion import evaluate, is_escalation

HISTORY_TTL_DAYS = 3
_META: dict[str, dict] = {}


def meta_for(ids: set[str]) -> dict[str, dict]:
    if not _META or not ids.issubset(_META):
        _META.update(store.all_meta())
    return _META


def process(records: list[dict], st: Settings, publish: bool = True) -> dict:
    meta = meta_for({str(r["sensor_id"]) for r in records})
    states = [evaluate(r, meta[str(r["sensor_id"])], st) for r in records if str(r["sensor_id"]) in meta]
    skipped = len(records) - len(states)

    newest: dict[str, dict] = {}
    for s in states:
        if s["sensor_id"] not in newest or s["ts"] > newest[s["sensor_id"]]["ts"]:
            newest[s["sensor_id"]] = s
    prev = store.latest_for(list(newest))
    alerts, latest = [], []
    for sid, s in newest.items():
        p = prev.get(sid)
        if p and p.get("ts", "") > s["ts"]:
            continue                       # late/out-of-order record: keep the newer state
        s = {**s, "prev_level": (p or {}).get("level")}
        latest.append(s)
        if is_escalation(s["prev_level"], s["level"]):
            alerts.append({"kind": "current", "sensor_id": sid, "ts": s["ts"], "level": s["level"],
                           "prev_level": s["prev_level"], "speed_mph": s["speed_mph"],
                           "free_flow_mph": s["free_flow_mph"], "delay_s": s["delay_s"],
                           "incident_flag": s["incident_flag"], "lat": s["lat"], "lon": s["lon"]})

    exp = store.expire_at(HISTORY_TTL_DAYS)
    store.batch_put("traffic_history", [{**s, "expire_at": exp} for s in states])
    store.batch_put("traffic_latest", latest)
    if publish and alerts:
        store.publish_alert_event("current", alerts)
    levels: dict[str, int] = {}
    for s in latest:
        levels[s["level"]] = levels.get(s["level"], 0) + 1
    return {"processed": len(states), "skipped_unknown_sensor": skipped, "levels": levels, "alerts": alerts}


def handler(event, context):
    records = [json.loads(base64.b64decode(r["kinesis"]["data"])) for r in event.get("Records", [])]
    result = process(records, Settings.from_env())
    print(json.dumps({k: v for k, v in result.items() if k != "alerts"} | {"alerts": len(result["alerts"])}))
    return {k: v for k, v in result.items() if k != "alerts"} | {"alerts": len(result["alerts"])}
