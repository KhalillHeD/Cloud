"""Agent tools (block 9). Each function is plain Python (testable, usable without an LLM);
`langchain_tools()` wraps them for tool calling.

  get_current_congestion  -> DynamoDB traffic_latest
  get_forecast            -> DynamoDB forecasts snapshot
  query_history           -> Athena, allow-listed templates only
  send_alert              -> SNS + alerts feed
"""
from __future__ import annotations

import functools
import json
import os
import time

from traffic_common import history_sql, store
from traffic_common.config import Settings
from traffic_common.congestion import LEVEL_RANK, congestion_index, level_of, parse_ts, week_slot

_META: dict = {}


def _meta() -> dict:
    if not _META:
        _META.update(store.all_meta())
    return _META


def _in_bbox(s: dict, bbox) -> bool:
    if not bbox:
        return True
    south, west, north, east = bbox
    return south <= s["lat"] <= north and west <= s["lon"] <= east


def get_current_congestion(sensor_ids: list[str] | None = None, bbox: list[float] | None = None,
                           only_congested: bool = False, limit: int = 20) -> dict:
    """Current congestion per sensor, worst first. bbox = [south, west, north, east]."""
    rows = [s for s in store.latest_all()
            if (not sensor_ids or s["sensor_id"] in sensor_ids) and _in_bbox(s, bbox)]
    if only_congested:
        rows = [s for s in rows if s.get("level") in ("heavy", "severe")]
    rows.sort(key=lambda s: -(s.get("congestion_index") or 0))
    counts: dict[str, int] = {}
    for s in rows:
        counts[s["level"]] = counts.get(s["level"], 0) + 1
    keys = ("sensor_id", "ts", "level", "speed_mph", "free_flow_mph", "congestion_index", "delay_s",
            "incident_flag", "lat", "lon")
    return {"as_of": max((s["ts"] for s in rows), default=None), "level_counts": counts,
            "sensors": [{k: s.get(k) for k in keys} for s in rows[: max(1, min(int(limit), 100))]]}


def get_forecast(horizon: int = 30, sensor_id: str | None = None, only_congested: bool = True,
                 limit: int = 15) -> dict:
    """Forecast at +horizon minutes (15, 30 or 60) with the context needed to explain it."""
    st = Settings.from_env()
    snap = store.get_state("forecast_latest", str(int(horizon)))
    if not snap:
        return {"error": f"no forecast for horizon {horizon} yet"}
    latest = {s["sensor_id"]: s for s in store.latest_all()}
    meta = _meta()
    target = parse_ts(snap["target_ts"])
    out = []
    for f in snap["items"]:
        sid = f["sensor_id"]
        if sensor_id and sid != str(sensor_id):
            continue
        if only_congested and not sensor_id and f["level"] not in ("heavy", "severe"):
            continue
        m = meta.get(sid, {})
        cur = latest.get(sid, {})
        typical = m["expected_speed"][week_slot(target)] if m else None
        hist = store.last_n(sid, 4)
        trend = None
        if len(hist) == 4 and hist[0].get("speed_mph") and hist[3].get("speed_mph"):
            trend = round(hist[0]["speed_mph"] - hist[3]["speed_mph"], 1)
        out.append({
            "sensor_id": sid, "lat": m.get("lat"), "lon": m.get("lon"),
            "predicted_level": f["level"], "predicted_speed_mph": f["speed_mph"],
            "free_flow_mph": m.get("free_flow_mph"), "predicted_delay_s": f.get("delay_s"),
            "current_level": cur.get("level"), "current_speed_mph": cur.get("speed_mph"),
            "trend_mph_15min": trend, "incident_flag": cur.get("incident_flag", False),
            "typical_speed_mph": typical,
            "typical_level": level_of(congestion_index(typical, m.get("free_flow_mph", 0)), st) if m else None,
        })
    out.sort(key=lambda r: (-LEVEL_RANK.get(r["predicted_level"], -1), r["predicted_speed_mph"]))
    return {"issued_at": snap["issued_ts"], "target_time": snap["target_ts"], "horizon_min": int(horizon),
            "model": snap.get("model"), "count": len(out), "sensors": out[: max(1, min(int(limit), 100))]}


def query_history(template: str, params: dict | None = None) -> dict:
    """Run one allow-listed Athena template over the curated data lake table."""
    sql = history_sql.render(template, params or {})
    ath = store.client("athena")
    qid = ath.start_query_execution(
        QueryString=sql, WorkGroup=os.environ.get("ATHENA_WORKGROUP", f"{store.project()}-wg"),
        QueryExecutionContext={"Database": os.environ.get("GLUE_DATABASE", store.project().replace("-", "_"))},
    )["QueryExecutionId"]
    deadline = time.time() + 25
    while time.time() < deadline:
        q = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if q["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.7)
    if q["State"] != "SUCCEEDED":
        return {"error": f"query {q['State']}: {q.get('StateChangeReason', 'timeout')}"}
    res = ath.get_query_results(QueryExecutionId=qid, MaxResults=60)
    rows = res["ResultSet"]["Rows"]
    header = [c.get("VarCharValue") for c in rows[0]["Data"]] if rows else []
    data = [dict(zip(header, [c.get("VarCharValue") for c in r["Data"]], strict=False)) for r in rows[1:]]
    return {"template": template, "rows": data}


def send_alert(message: str, severity: str = "heavy", sensor_ids: list[str] | None = None) -> dict:
    """Email operators through SNS and add the alert to the dashboard feed."""
    topic = os.environ.get("ALERT_TOPIC_ARN")
    if topic:
        store.client("sns").publish(TopicArn=topic, Subject=f"[Traffic] {severity.upper()} congestion alert"[:100],
                                    Message=message)
    store.put_alert({"message": message, "severity": severity, "sensor_ids": sensor_ids or [],
                     "sensor_id": (sensor_ids or ["-"])[0], "sent": bool(topic)})
    return {"sent": bool(topic), "severity": severity}


def langchain_tools():
    """The four tools as LangChain StructuredTools (schemas inferred from the signatures)."""
    from langchain_core.tools import StructuredTool

    def as_json_tool(fn):
        @functools.wraps(fn)
        def run(**kwargs):
            try:
                return json.dumps(fn(**kwargs), default=str)
            except Exception as exc:  # noqa: BLE001 - errors go back to the model, not up the stack
                return json.dumps({"error": str(exc)})
        return StructuredTool.from_function(func=run, name=fn.__name__, description=_doc(fn))

    return [as_json_tool(f) for f in TOOL_FUNCS]


def _doc(fn) -> str:
    extra = ""
    if fn is query_history:
        extra = " Templates: " + "; ".join(f"{k} (params {list(v['params'])}): {v['description']}"
                                           for k, v in history_sql.TEMPLATES.items())
    return (fn.__doc__ or "").strip() + extra


TOOL_FUNCS = (get_current_congestion, get_forecast, query_history, send_alert)
