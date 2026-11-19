"""Block 8: scheduled forecast Lambda (every 5 min, every minute in demo mode).

Reads the last hour per sensor from traffic_history, builds features (pure Python), and
predicts speed at +15/+30/+60 min with the active model:
  mode "lambda"    model.json from S3, evaluated in-process (traffic_common.xgb_eval)
  mode "sagemaker" the Serverless Inference endpoint (CSV)
  no model yet     historical-average baseline, labelled as such
Writes the forecasts table, a per-horizon snapshot for the dashboard, and forecast alert
events when a segment is predicted to escalate to heavy/severe.
"""
from __future__ import annotations

import json
import math
from datetime import timedelta

from traffic_common import store
from traffic_common.config import SLOT_MINUTES, Settings
from traffic_common.congestion import LEVEL_RANK, congestion_index, fmt_ts, level_of, parse_ts, travel_time_s, week_slot
from traffic_common.features import N_LAGS, feature_row
from traffic_common.xgb_eval import JsonBooster

FORECAST_TTL_DAYS = 2
_MODEL: dict = {"key": None, "booster": None}
_META: dict = {}


def active_model() -> dict | None:
    return store.get_state("model", "active")


def predictor(model: dict | None):
    if not model:
        return None, "historical-average (no model published yet)"
    if model.get("mode") == "sagemaker" and model.get("endpoint"):
        rt = store.client("sagemaker-runtime")

        def call(rows):
            body = "\n".join(",".join("nan" if math.isnan(v) else f"{v:.5g}" for v in r) for r in rows)
            resp = rt.invoke_endpoint(EndpointName=model["endpoint"], ContentType="text/csv", Body=body)
            text = resp["Body"].read().decode().strip()
            return [float(x) for x in text.replace("\n", ",").split(",") if x.strip()]
        return call, f"xgboost@sagemaker:{model['endpoint']}"
    if _MODEL["key"] != model["s3_key"]:
        _MODEL["booster"] = JsonBooster(store.s3_get_json(model["s3_key"]))
        _MODEL["key"] = model["s3_key"]
    return _MODEL["booster"].predict, f"xgboost@lambda:{model.get('run')}"


def run(st: Settings, publish: bool = True) -> dict:
    if not _META:
        _META.update(store.all_meta())
    latest = {s["sensor_id"]: s for s in store.latest_all() if s["sensor_id"] in _META}
    if not latest:
        return {"status": "no data yet"}
    now_ts = max(s["ts"] for s in latest.values())
    t_now = parse_ts(now_ts)
    window = [fmt_ts(t_now - timedelta(minutes=SLOT_MINUTES * k)) for k in range(N_LAGS)]
    hist: dict[str, dict[str, float | None]] = {}
    for sid in _META:
        rows = store.last_n(sid, N_LAGS, until_ts=now_ts)
        hist[sid] = {r["ts"]: r.get("speed_mph") for r in rows}

    def lag_list(sid):
        return [hist.get(sid, {}).get(ts) for ts in window]

    model = active_model()
    predict, model_label = predictor(model)
    steps = {h: h // SLOT_MINUTES for h in st.forecast_horizons_min}
    if model and model.get("horizons_min"):
        steps = {h: s for h, s in steps.items() if h in model["horizons_min"]}
    jobs, rows = [], []
    for sid, m in _META.items():
        lags = lag_list(sid)
        nb = m.get("neighbors", [])
        nb0 = [hist.get(n, {}).get(window[0]) for n in nb]
        nb3 = [hist.get(n, {}).get(window[3]) for n in nb]
        for h, k in steps.items():
            jobs.append((sid, h, k))
            rows.append(feature_row(lags, nb0, nb3, m["free_flow_mph"], m["expected_speed"], t_now, k, m["idx"]))
    if predict:
        preds = predict(rows)
    else:  # baseline: mean speed of the same weekday/slot in the training period
        preds = [_META[sid]["mean_speed"][week_slot(t_now + timedelta(minutes=SLOT_MINUTES * k))]
                 for sid, _h, k in jobs]

    exp = store.expire_at(FORECAST_TTL_DAYS)
    items, snapshot, alerts = [], {h: [] for h in steps}, []
    for (sid, h, k), p in zip(jobs, preds, strict=True):
        m = _META[sid]
        target = t_now + timedelta(minutes=SLOT_MINUTES * k)
        speed = min(max(float(p), 1.0), 90.0)
        ci = congestion_index(speed, m["free_flow_mph"])
        lvl = level_of(ci, st)
        tt = travel_time_s(m["segment_length_m"], speed)
        ett = travel_time_s(m["segment_length_m"], m["expected_speed"][week_slot(target)])
        it = {"sensor_id": sid, "target_ts": fmt_ts(target), "horizon_min": h, "issued_ts": now_ts,
              "speed_mph": round(speed, 2), "congestion_index": round(ci, 4), "level": lvl,
              "travel_time_s": round(tt, 1), "expected_tt_s": round(ett, 1), "delay_s": round(tt - ett, 1),
              "model": model_label}
        items.append({**it, "expire_at": exp})
        snapshot[h].append({k2: it[k2] for k2 in ("sensor_id", "speed_mph", "congestion_index", "level",
                                                     "travel_time_s", "expected_tt_s", "delay_s")})
        cur = latest.get(sid, {}).get("level")
        if lvl in ("heavy", "severe") and LEVEL_RANK[lvl] > LEVEL_RANK.get(cur or "unknown", -1) and h <= 30:
            alerts.append({"kind": "forecast", "sensor_id": sid, "ts": now_ts, "target_ts": fmt_ts(target),
                           "horizon_min": h, "level": lvl, "current_level": cur, "speed_mph": round(speed, 1),
                           "free_flow_mph": m["free_flow_mph"], "lat": m["lat"], "lon": m["lon"]})
    store.batch_put("forecasts", items)
    for h, lst in snapshot.items():
        store.put_state("forecast_latest", str(h), {"issued_ts": now_ts, "horizon_min": h, "model": model_label,
                                                     "target_ts": fmt_ts(t_now + timedelta(minutes=h)),
                                                     "items": lst})
    # keep only the earliest horizon per sensor in alerts
    first: dict[str, dict] = {}
    for a in alerts:
        if a["sensor_id"] not in first or a["horizon_min"] < first[a["sensor_id"]]["horizon_min"]:
            first[a["sensor_id"]] = a
    alerts = list(first.values())
    if publish and alerts:
        store.publish_alert_event("forecast", alerts)
    return {"issued_ts": now_ts, "forecasts": len(items), "model": model_label, "alerts": alerts}


def handler(event, context):
    """Real time: one forecast per invocation (every 5 min). Demo (TICK_SECONDS < 60): the
    schedule fires every minute, so run once per tick, a few seconds after each producer tick."""
    import time
    st = Settings.from_env()
    ticks = 60 // st.tick_seconds if st.tick_seconds < 60 else 1
    out, last = [], None
    for i in range(ticks):
        start = time.monotonic()
        if ticks > 1:
            time.sleep(min(4.0, st.tick_seconds / 2))       # let the processor store the tick first
        res = run(st)
        summary = {k: v for k, v in res.items() if k != "alerts"} | {"alerts": len(res.get("alerts", []))}
        if summary.get("issued_ts") != last:
            print(json.dumps(summary))
        last = summary.get("issued_ts")
        out.append(summary)
        if i < ticks - 1:
            time.sleep(max(0.0, st.tick_seconds - (time.monotonic() - start)))
    return {"runs": out}
