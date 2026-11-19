"""Block 9: AI agent Lambda (Bedrock + LangChain), three modes.

  (a) alert   EventBridge "CongestionAlert" events from the processor / forecast Lambda.
              De-duplicates per sensor (ALERT_DEDUP_MINUTES), writes a short alert with the
              LLM (template if none), emails it via SNS, adds it to the dashboard feed.
  (b) report  EventBridge daily schedule ({"mode": "daily_report"}): worst segments, peak
              hours, hotspots (Athena), forecast accuracy (DynamoDB), alerts; stored in
              S3 reports/ and emailed.
  (c) chat    POST /agent/chat {"message", "history"?}: tool-calling loop; the answer lists
              the tool calls actually made.
"""
from __future__ import annotations

import base64
import json
import re
import time
from datetime import timedelta

import prompts
import tools
from llm import LLM, text_of
from traffic_common import store
from traffic_common.config import Settings
from traffic_common.congestion import LEVEL_RANK, fmt_ts, parse_ts

MAX_STEPS = 6
CHAT_TOOLS = ("get_current_congestion", "get_forecast", "query_history")


# ----------------------------------------------------------------------------- (a) alerts

def alert_mode(detail: dict, st: Settings, llm: LLM | None = None) -> dict:
    events = [a for a in detail.get("alerts", []) if store.claim_alert(a["sensor_id"], st.alert_dedup_minutes)]
    if not events:
        return {"sent": 0, "deduplicated": len(detail.get("alerts", []))}
    severity = max((a["level"] for a in events), key=lambda lv: LEVEL_RANK.get(lv, 0))
    llm = llm or LLM(st)
    message = None
    if llm.available:
        try:
            message = text_of(llm.invoke(prompts.ALERT.format(events=json.dumps(events, default=str)))).strip()
        except Exception as exc:  # noqa: BLE001 - never lose an alert because the LLM failed
            print(f"LLM alert text failed: {exc}")
    message = message or template_alert(events)
    res = tools.send_alert(message, severity, [a["sensor_id"] for a in events])
    return {"sent": len(events), "severity": severity, "message": message, "email": res["sent"]}


def template_alert(events: list[dict]) -> str:
    parts = []
    for a in events[:8]:
        when = f"forecast in {a['horizon_min']} min" if a.get("kind") == "forecast" else "now"
        inc = " (SYNTHETIC incident)" if a.get("incident_flag") else ""
        parts.append(f"sensor {a['sensor_id']} {a['level'].upper()} {when}: {a.get('speed_mph')} mph "
                     f"vs free-flow {a.get('free_flow_mph')} mph{inc}")
    more = f" (+{len(events) - 8} more)" if len(events) > 8 else ""
    return "Congestion alert at " + str(events[0].get("ts")) + ": " + "; ".join(parts) + more


# ----------------------------------------------------------------------------- (b) daily report

def forecast_accuracy(now_ts: str, hours: int = 24) -> dict:
    t1 = parse_ts(now_ts)
    t0 = fmt_ts(t1 - timedelta(hours=hours))
    from boto3.dynamodb.conditions import Key
    errs: dict[int, list[float]] = {}
    for sid in tools._meta():
        fc = store.table("forecasts").query(
            KeyConditionExpression=Key("sensor_id").eq(sid) & Key("target_ts").between(t0, now_ts))["Items"]
        if not fc:
            continue
        actual = {h["ts"]: h.get("speed_mph") for h in store.history_for_sensor(sid, t0, now_ts)}
        for f in store.from_dynamo(fc):
            a = actual.get(f["target_ts"])
            if a:
                errs.setdefault(int(f["horizon_min"]), []).append(abs(f["speed_mph"] - a))
    return {str(h): {"mae_mph": round(sum(v) / len(v), 2), "n": len(v)} for h, v in sorted(errs.items())}


def daily_report(st: Settings, llm: LLM | None = None) -> dict:
    latest = store.latest_all()
    now_ts = max((s["ts"] for s in latest), default=None)
    data: dict = {"replay_time": now_ts, "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    for name, params in (("most_congested", {"days": 1, "limit": 5}), ("by_hour", {}),
                         ("hotspots", {"min_pct": 20, "limit": 5}), ("daily_summary", {"days": 7})):
        try:
            data[name] = tools.query_history(name, params).get("rows")
        except Exception as exc:  # noqa: BLE001
            data[name] = {"error": str(exc)}
    if isinstance(data.get("by_hour"), list):
        rows = sorted(data["by_hour"], key=lambda r: -float(r.get("avg_congestion") or 0))
        data["peak_hours"] = rows[:3]
        del data["by_hour"]
    data["forecast_accuracy_last_24h"] = forecast_accuracy(now_ts) if now_ts else {}
    alerts = store.recent_alerts(100)
    data["alerts"] = {"count": len(alerts),
                      "by_severity": {s: sum(a.get("severity") == s for a in alerts) for s in ("heavy", "severe")}}
    data["current"] = tools.get_current_congestion(limit=5)
    llm = llm or LLM(st)
    report = None
    if llm.available:
        try:
            report = text_of(llm.invoke(prompts.REPORT.format(data=json.dumps(data, default=str))))
        except Exception as exc:  # noqa: BLE001
            print(f"LLM report failed: {exc}")
    report = report or "# Daily traffic report\n\n```json\n" + json.dumps(data, indent=2, default=str) + "\n```\n"
    day = (now_ts or "unknown")[:10]
    key = f"reports/{day}-{time.strftime('%H%M%S')}.md"
    store.client("s3").put_object(Bucket=store.lake_bucket(), Key=key, Body=report.encode(),
                                  ContentType="text/markdown")
    import os
    if os.environ.get("ALERT_TOPIC_ARN"):
        store.client("sns").publish(TopicArn=os.environ["ALERT_TOPIC_ARN"], Subject=f"[Traffic] Daily report {day}",
                                    Message=report[:200_000])
    return {"report_key": key, "report": report}


# ----------------------------------------------------------------------------- (c) chat

def chat(message: str, history: list[dict] | None = None, llm: LLM | None = None) -> dict:
    llm = llm or LLM()
    if not llm.available:
        return offline_chat(message)
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    lc_tools = [t for t in tools.langchain_tools() if t.name in CHAT_TOOLS]
    by_name = {t.name: t for t in lc_tools}
    msgs = [SystemMessage(prompts.SYSTEM)]
    for h in (history or [])[-6:]:
        msgs.append(HumanMessage(h["content"]) if h.get("role") == "user" else AIMessage(h.get("content", "")))
    msgs.append(HumanMessage(message))
    calls = []
    for _ in range(MAX_STEPS):
        ai = llm.invoke(msgs, lc_tools)
        msgs.append(ai)
        if not ai.tool_calls:
            return {"answer": text_of(ai), "tool_calls": calls, "provider": llm.provider}
        for tc in ai.tool_calls:
            out = by_name[tc["name"]].invoke(tc["args"]) if tc["name"] in by_name else '{"error":"unknown tool"}'
            calls.append({"name": tc["name"], "args": tc["args"]})
            msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"answer": "I needed too many steps; please ask a narrower question.", "tool_calls": calls,
            "provider": llm.provider}


def offline_chat(message: str) -> dict:
    """No LLM available: answer the common questions from real tool calls with fixed wording."""
    m = re.search(r"\b(15|30|60)\s*(min|minutes)?\b", message)
    horizon = int(m.group(1)) if m else 30
    calls = [{"name": "get_forecast", "args": {"horizon": horizon}}]
    fc = tools.get_forecast(horizon=horizon)
    if "error" in fc:
        return {"answer": fc["error"], "tool_calls": calls, "provider": "none"}
    if not fc["sensors"]:
        ans = f"No segment is forecast to be heavy or severe at {fc['target_time']} (+{horizon} min)."
    else:
        lines = [f"Forecast for {fc['target_time']} (+{horizon} min, model {fc['model']}):"]
        for s in fc["sensors"][:8]:
            lines.append(f"- sensor {s['sensor_id']}: {s['predicted_level']} "
                         f"({s['predicted_speed_mph']} mph vs {s['free_flow_mph']} free-flow) - {reason(s)}")
        ans = "\n".join(lines)
    return {"answer": ans + "\n\n(offline mode: no LLM configured, answer built from tool results)",
            "tool_calls": calls, "provider": "none"}


def reason(s: dict) -> str:
    why = []
    if s.get("incident_flag"):
        why.append("synthetic incident in progress")
    if s.get("typical_level") in ("heavy", "severe"):
        why.append(f"recurring: usually {s['typical_level']} at this time ({s['typical_speed_mph']} mph)")
    if s.get("trend_mph_15min") is not None and s["trend_mph_15min"] <= -3:
        why.append(f"speed falling {s['trend_mph_15min']} mph in 15 min")
    if s.get("current_level") in ("heavy", "severe"):
        why.append(f"already {s['current_level']} now")
    return "; ".join(why) or "model expects speeds to drop"


# ----------------------------------------------------------------------------- entry point

def _http(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json",
                                              "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(body, default=str)}


def handler(event, context):
    st = Settings.from_env()
    if event.get("detail-type") == "CongestionAlert":
        res = alert_mode(event["detail"], st)
        print(json.dumps(res, default=str))
        return res
    if event.get("mode") == "daily_report":
        res = daily_report(st)
        print(res["report_key"])
        return {"report_key": res["report_key"]}
    if "requestContext" in event:
        raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode()
        try:
            body = json.loads(raw)
            msg = str(body.get("message", "")).strip()[:2000]
        except (ValueError, AttributeError):
            return _http(400, {"error": "body must be JSON {message}"})
        if not msg:
            return _http(400, {"error": "message is required"})
        try:
            return _http(200, chat(msg, body.get("history")))
        except Exception as exc:  # noqa: BLE001
            print(f"chat failed: {exc}")
            return _http(502, {"error": f"agent failed: {exc}"})
    return {"error": "unknown event"}
