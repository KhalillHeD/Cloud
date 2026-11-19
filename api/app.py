"""Block 10 backend: HTTP API (API Gateway v2) Lambda for the dashboard.

  GET /latest                          latest state of every sensor + summary
  GET /sensors                         sensor metadata (position, free-flow, segment length)
  GET /forecast?horizon=15|30|60       latest forecast snapshot for that horizon
  GET /history?sensor_id=&from=&to=    one sensor over a time range (default: last 6 h)
  GET /history?ts=                     every sensor at one past slot (timeline slider)
  GET /alerts?limit=                   alerts feed
  GET /model                           active model + accuracy
  POST /agent/chat                     routed straight to the agent Lambda by API Gateway
"""
from __future__ import annotations

import json
import os
import re
from datetime import timedelta

from boto3.dynamodb.conditions import Key
from traffic_common import store
from traffic_common.congestion import LEVEL_RANK, fmt_ts, parse_ts

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")
SENSOR_RE = re.compile(r"^\d{3,10}$")
_META: dict = {}


class BadRequest(Exception):
    pass


def _meta() -> dict:
    if not _META:
        _META.update(store.all_meta())
    return _META


def _ts(value: str, name: str) -> str:
    if not value or not TS_RE.match(value):
        raise BadRequest(f"{name} must look like 2012-03-05T08:00:00")
    return fmt_ts(parse_ts(value))


def summary(items: list[dict]) -> dict:
    total_m = sum(_meta().get(i["sensor_id"], {}).get("segment_length_m", 0) for i in items) or 1
    by_level: dict[str, dict] = {}
    for i in items:
        lv = i.get("level", "unknown")
        d = by_level.setdefault(lv, {"count": 0, "km": 0.0})
        d["count"] += 1
        d["km"] += _meta().get(i["sensor_id"], {}).get("segment_length_m", 0) / 1000
    congested = sum(v["km"] for k, v in by_level.items() if k in ("heavy", "severe"))
    return {"by_level": {k: {"count": v["count"], "km": round(v["km"], 2)} for k, v in by_level.items()},
            "total_km": round(total_m / 1000, 2), "congested_km": round(congested, 2),
            "pct_congested": round(congested / (total_m / 1000) * 100, 2),
            "incidents": sum(bool(i.get("incident_flag")) for i in items)}


def get_latest(q: dict) -> dict:
    items = store.latest_all()
    items.sort(key=lambda s: (-LEVEL_RANK.get(s.get("level"), -1), -(s.get("delay_s") or 0)))
    return {"as_of": max((s["ts"] for s in items), default=None), "tick_seconds": int(os.environ.get("TICK_SECONDS", 300)),
            "summary": summary(items), "sensors": items}


def get_sensors(q: dict) -> dict:
    keys = ("sensor_id", "lat", "lon", "free_flow_mph", "segment_length_m", "neighbors", "rank")
    return {"sensors": sorted(({k: m.get(k) for k in keys} for m in _meta().values()), key=lambda m: m["rank"])}


def get_forecast(q: dict) -> dict:
    try:
        h = int(q.get("horizon", 30))
    except ValueError as exc:
        raise BadRequest("horizon must be 15, 30 or 60") from exc
    snap = store.get_state("forecast_latest", str(h))
    if not snap:
        return {"horizon_min": h, "items": [], "available": False}
    return {**snap, "available": True, "summary": summary(snap["items"])}


def get_history(q: dict) -> dict:
    if q.get("ts"):
        ts = _ts(q["ts"], "ts")
        items = store.history_at(ts)
        return {"ts": ts, "summary": summary(items), "sensors": items}
    sid = q.get("sensor_id", "")
    if not SENSOR_RE.match(sid):
        raise BadRequest("sensor_id (digits) or ts is required")
    if q.get("to"):
        t_to = _ts(q["to"], "to")
    else:
        latest = store.latest_for([sid]).get(sid)
        if not latest:
            return {"sensor_id": sid, "points": []}
        t_to = latest["ts"]
    t_from = _ts(q["from"], "from") if q.get("from") else fmt_ts(parse_ts(t_to) - timedelta(hours=6))
    if parse_ts(t_to) - parse_ts(t_from) > timedelta(days=3):
        raise BadRequest("range limited to 3 days (older data: Athena)")
    pts = store.history_for_sensor(sid, t_from, t_to)
    keys = ("ts", "speed_mph", "level", "travel_time_s", "expected_tt_s", "free_flow_tt_s", "delay_s",
            "incident_flag")
    fc = store.table("forecasts").query(
        KeyConditionExpression=Key("sensor_id").eq(sid) & Key("target_ts").gt(t_to))["Items"]
    fc = [f for f in store.from_dynamo(fc) if f.get("issued_ts") == t_to]
    return {"sensor_id": sid, "from": t_from, "to": t_to,
            "meta": {k: _meta().get(sid, {}).get(k) for k in ("lat", "lon", "free_flow_mph", "segment_length_m")},
            "points": [{k: p.get(k) for k in keys} for p in pts],
            "forecast": sorted(({k: f.get(k) for k in ("target_ts", "horizon_min", "speed_mph", "level",
                                                       "travel_time_s", "expected_tt_s")} for f in fc),
                               key=lambda f: f["target_ts"])}


def get_alerts(q: dict) -> dict:
    limit = max(1, min(int(q.get("limit", 30)), 100))
    return {"alerts": store.recent_alerts(limit)}


def get_model(q: dict) -> dict:
    return {"model": store.get_state("model", "active")}


ROUTES = {"/latest": get_latest, "/sensors": get_sensors, "/forecast": get_forecast, "/history": get_history,
          "/alerts": get_alerts, "/model": get_model}


def respond(status: int, body) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*",
                                              "Cache-Control": "no-store"},
            "body": json.dumps(body, default=str)}


def handler(event, context):
    path = event.get("rawPath") or event.get("path") or "/"
    stage = (event.get("requestContext") or {}).get("stage")
    if stage and stage != "$default" and path.startswith(f"/{stage}/"):
        path = path[len(stage) + 1:]
    fn = ROUTES.get(path.rstrip("/") or "/")
    if not fn:
        return respond(404, {"error": f"no route {path}", "routes": sorted(ROUTES)})
    try:
        return respond(200, fn(event.get("queryStringParameters") or {}))
    except BadRequest as exc:
        return respond(400, {"error": str(exc)})
