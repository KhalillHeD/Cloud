"""Resource names and DynamoDB / S3 access shared by every Lambda.

Names derive from PROJECT_NAME (the SAM template uses the same patterns), so no table
name has to be passed around. boto3 honours AWS_ENDPOINT_URL, which is how local mode
points everything at the moto server.

Tables (block 5):
  sensor_meta      PK sensor_id            free-flow speed, coordinates, segment length, weekly profile
  traffic_latest   PK sensor_id            latest state per sensor
  traffic_history  PK sensor_id, SK ts     recent states (TTL 3 days); GSI by_ts for the timeline slider
  forecasts        PK sensor_id, SK target_ts  (TTL 2 days)
  app_state        PK pk, SK sk            replay cursor, alert de-duplication, alerts feed,
                                           latest forecast snapshot, active model
"""
from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from functools import cache

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

TABLES = ("sensor_meta", "traffic_latest", "traffic_history", "forecasts", "app_state")


def project() -> str:
    return os.environ.get("PROJECT_NAME", "traffic-analytics")


def table_name(logical: str) -> str:
    return f"{project()}-{logical}"


def stream_name() -> str:
    return f"{project()}-stream"


def lake_bucket() -> str:
    return os.environ.get("LAKE_BUCKET") or f"{project()}-lake-local"


@cache
def resource():
    return boto3.resource("dynamodb")


@cache
def client(name: str):
    return boto3.client(name)


def table(logical: str):
    return resource().Table(table_name(logical))


# ----------------------------------------------------------------------------- (de)serialisation

def to_dynamo(obj):
    """floats -> Decimal, drop NaN (DynamoDB refuses it)."""
    if isinstance(obj, float):
        return None if obj != obj else Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: to_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dynamo(v) for v in obj]
    return obj


def from_dynamo(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() and abs(obj) < 1e15 else float(obj)
    if isinstance(obj, dict):
        return {k: from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_dynamo(v) for v in obj]
    return obj


def expire_at(days: float) -> int:
    return int(time.time() + days * 86400)


def _scan_all(tbl, **kw) -> list[dict]:
    items, start = [], None
    while True:
        if start:
            kw["ExclusiveStartKey"] = start
        page = tbl.scan(**kw)
        items += page["Items"]
        start = page.get("LastEvaluatedKey")
        if not start:
            return [from_dynamo(i) for i in items]


def _query_all(tbl, **kw) -> list[dict]:
    items, start = [], None
    while True:
        if start:
            kw["ExclusiveStartKey"] = start
        page = tbl.query(**kw)
        items += page["Items"]
        start = page.get("LastEvaluatedKey")
        if not start or ("Limit" in kw and len(items) >= kw["Limit"]):
            return [from_dynamo(i) for i in items]


def batch_put(logical: str, items: list[dict]) -> None:
    with table(logical).batch_writer(overwrite_by_pkeys=_pkeys(logical)) as bw:
        for it in items:
            bw.put_item(Item=to_dynamo(it))


def _pkeys(logical: str) -> list[str]:
    return {"traffic_history": ["sensor_id", "ts"], "forecasts": ["sensor_id", "target_ts"],
            "app_state": ["pk", "sk"]}.get(logical, ["sensor_id"])


# ----------------------------------------------------------------------------- sensor_meta

def decode_meta(item: dict) -> dict:
    item = dict(item)
    if isinstance(item.get("expected_speed"), str):
        item["expected_speed"] = [float(x) for x in item["expected_speed"].split(",")]
    if isinstance(item.get("mean_speed"), str):
        item["mean_speed"] = [float(x) for x in item["mean_speed"].split(",")]
    return item


def encode_meta(item: dict) -> dict:
    item = dict(item)
    for k in ("expected_speed", "mean_speed"):
        if isinstance(item.get(k), list):
            item[k] = ",".join(f"{v:.1f}" for v in item[k])
    return item


def all_meta() -> dict[str, dict]:
    return {m["sensor_id"]: decode_meta(m) for m in _scan_all(table("sensor_meta"))}


# ----------------------------------------------------------------------------- traffic state

def latest_all() -> list[dict]:
    return _scan_all(table("traffic_latest"))


def latest_for(sensor_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ids = list(dict.fromkeys(sensor_ids))
    for i in range(0, len(ids), 100):
        keys = [{"sensor_id": s} for s in ids[i:i + 100]]
        req = {table_name("traffic_latest"): {"Keys": keys}}
        while req:
            resp = resource().batch_get_item(RequestItems=req)
            for it in resp["Responses"].get(table_name("traffic_latest"), []):
                out[it["sensor_id"]] = from_dynamo(it)
            req = resp.get("UnprocessedKeys") or None
    return out


def history_for_sensor(sensor_id: str, ts_from: str, ts_to: str) -> list[dict]:
    return _query_all(table("traffic_history"),
                      KeyConditionExpression=Key("sensor_id").eq(sensor_id) & Key("ts").between(ts_from, ts_to))


def last_n(sensor_id: str, n: int, until_ts: str | None = None) -> list[dict]:
    """Newest first."""
    cond = Key("sensor_id").eq(sensor_id)
    if until_ts:
        cond = cond & Key("ts").lte(until_ts)
    page = table("traffic_history").query(KeyConditionExpression=cond, ScanIndexForward=False, Limit=n)
    return [from_dynamo(i) for i in page["Items"]]


def history_at(ts: str) -> list[dict]:
    return _query_all(table("traffic_history"), IndexName="by_ts", KeyConditionExpression=Key("ts").eq(ts))


# ----------------------------------------------------------------------------- app_state

def get_state(pk: str, sk: str) -> dict | None:
    it = table("app_state").get_item(Key={"pk": pk, "sk": sk}).get("Item")
    return from_dynamo(it) if it else None


def put_state(pk: str, sk: str, data: dict, ttl_days: float | None = None) -> None:
    item = {"pk": pk, "sk": sk, **data}
    if ttl_days:
        item["expire_at"] = expire_at(ttl_days)
    table("app_state").put_item(Item=to_dynamo(item))


def claim_alert(sensor_id: str, window_minutes: int, now: float | None = None) -> bool:
    """Atomically take the right to alert for a sensor; False if it alerted within the window."""
    now = time.time() if now is None else now
    try:
        table("app_state").put_item(
            Item=to_dynamo({"pk": "dedup", "sk": sensor_id, "last": int(now), "expire_at": int(now + 86400)}),
            ConditionExpression=Attr("last").not_exists() | Attr("last").lt(int(now - window_minutes * 60)))
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def put_alert(alert: dict) -> None:
    put_state("alert", f"{int(time.time() * 1000):015d}#{alert.get('sensor_id', '-')}", alert, ttl_days=7)


def recent_alerts(limit: int = 50) -> list[dict]:
    page = table("app_state").query(KeyConditionExpression=Key("pk").eq("alert"), ScanIndexForward=False,
                                    Limit=limit)
    return [from_dynamo(i) for i in page["Items"]]


def publish_alert_event(kind: str, alerts: list[dict]) -> None:
    """Alert events (current congestion or forecast) go to the default EventBridge bus; a rule
    routes them to the agent Lambda (alert mode)."""
    if not alerts:
        return
    for i in range(0, len(alerts), 25):   # keep each event far below the 256 KB limit
        client("events").put_events(Entries=[{
            "Source": "traffic.analytics", "DetailType": "CongestionAlert",
            "Detail": json.dumps({"kind": kind, "alerts": alerts[i:i + 25]}, default=str),
        }])


# ----------------------------------------------------------------------------- S3 helpers

def s3_get_json(key: str, bucket: str | None = None):
    body = client("s3").get_object(Bucket=bucket or lake_bucket(), Key=key)["Body"].read()
    return json.loads(body)


def s3_put_json(key: str, data, bucket: str | None = None) -> None:
    client("s3").put_object(Bucket=bucket or lake_bucket(), Key=key, Body=json.dumps(data).encode(),
                            ContentType="application/json")
