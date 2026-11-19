#!/usr/bin/env python3
"""`make local`: the whole pipeline on this machine, no AWS account needed.

  - starts an in-process AWS mock (moto) and creates the same tables/stream/bucket as the stack
  - seeds metadata, publishes the local model (ml/artifacts/latest) and warms up 6 h of history
  - every tick: producer -> Kinesis -> congestion processor -> forecast -> agent (alerts)
    (the same handler code the Lambdas run)
  - serves the dashboard and the API on http://localhost:8080 (POST /api/agent/chat included)

Options: --ticks N (stop after N ticks), --tick-seconds S (default 10), --port P, --no-server
LLM: LLM_PROVIDER=none by default locally; set LOCAL_LLM_PROVIDER=anthropic|openai and LLM_API_KEY
to chat with a real model.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common"), str(ROOT / "scripts")]
from traffic_common.config import load_dotenv  # noqa: E402

MOTO_PORT = 5055


def setup_env(tick_seconds: int) -> None:
    load_dotenv(ROOT / ".env")
    os.environ.update({
        "AWS_ENDPOINT_URL": f"http://127.0.0.1:{MOTO_PORT}", "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
        "REPLAY_DIR": str(ROOT / "data" / "processed" / "s3"), "REPLAY_MODE": "cursor",
        "TICK_SECONDS": str(tick_seconds), "LLM_PROVIDER": os.environ.get("LOCAL_LLM_PROVIDER", "none"),
    })
    os.environ.setdefault("LAKE_BUCKET", f"{os.environ.get('PROJECT_NAME', 'traffic-analytics')}-lake-local")
    os.environ.pop("AWS_PROFILE", None)
    os.environ.pop("AWS_SESSION_TOKEN", None)


def start_moto():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    from moto.server import ThreadedMotoServer
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=MOTO_PORT, verbose=False)
    server.start()
    return server


def create_resources() -> None:
    """Mirror of the stack's tables, stream and bucket (infra/template.yaml)."""
    from traffic_common import store
    ddb = store.client("dynamodb")
    S, pay = "S", "PAY_PER_REQUEST"

    def table(name, keys, gsi=None):
        attrs = {k for k, _ in keys} | ({k for k, _ in gsi[1]} if gsi else set())
        kw = {"TableName": store.table_name(name), "BillingMode": pay,
              "AttributeDefinitions": [{"AttributeName": a, "AttributeType": S} for a in sorted(attrs)],
              "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in keys]}
        if gsi:
            kw["GlobalSecondaryIndexes"] = [{"IndexName": gsi[0], "Projection": {"ProjectionType": "ALL"},
                                             "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in gsi[1]]}]
        ddb.create_table(**kw)

    table("sensor_meta", [("sensor_id", "HASH")])
    table("traffic_latest", [("sensor_id", "HASH")])
    table("traffic_history", [("sensor_id", "HASH"), ("ts", "RANGE")], ("by_ts", [("ts", "HASH"), ("sensor_id", "RANGE")]))
    table("forecasts", [("sensor_id", "HASH"), ("target_ts", "RANGE")])
    table("app_state", [("pk", "HASH"), ("sk", "RANGE")])
    store.client("kinesis").create_stream(StreamName=store.stream_name(), ShardCount=1)
    store.client("s3").create_bucket(Bucket=store.lake_bucket())


class Pipeline:
    def __init__(self):
        from components import load
        from traffic_common import store
        from traffic_common.config import Settings
        self.store, self.st = store, Settings.from_env()
        self.producer, self.processing = load("producer"), load("processing")
        self.forecast, self.agent = load("forecast"), load("agent")
        kin = store.client("kinesis")
        shard = kin.describe_stream(StreamName=store.stream_name())["StreamDescription"]["Shards"][0]["ShardId"]
        self.iterator = kin.get_shard_iterator(StreamName=store.stream_name(), ShardId=shard,
                                               ShardIteratorType="LATEST")["ShardIterator"]
        self.lock = threading.Lock()

    def tick(self) -> dict:
        with self.lock:
            ts = self.producer.take_replay_ts(self.st)
            self.producer.put_records(self.producer.build_records(ts, self.st))
            kin = self.store.client("kinesis")
            resp = kin.get_records(ShardIterator=self.iterator, Limit=1000)
            self.iterator = resp["NextShardIterator"]
            event = {"Records": [{"kinesis": {"data": base64.b64encode(r["Data"]).decode()}} for r in resp["Records"]]}
            records = [json.loads(base64.b64decode(r["kinesis"]["data"])) for r in event["Records"]]
            res = self.processing.process(records, self.st, publish=False)
            fc = self.forecast.run(self.st, publish=False)
            sent = 0
            for kind, alerts in (("current", res["alerts"]), ("forecast", fc.get("alerts", []))):
                if alerts:
                    sent += self.agent.alert_mode({"kind": kind, "alerts": alerts}, self.st).get("sent", 0)
            return {"ts": ts.isoformat(), "levels": res["levels"], "forecast_model": fc.get("model"),
                    "alerts_sent": sent}


def make_handler(pipeline: Pipeline, api, agent):
    dash = ROOT / "dashboard"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dash), **kw)

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, status, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/config.js":
                cfg = {"apiBase": "/api", "mode": "local", "tickSeconds": pipeline.st.tick_seconds}
                return self._send(200, f"window.APP_CONFIG = {json.dumps(cfg)};", "application/javascript")
            if u.path.startswith("/api/"):
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                with pipeline.lock:
                    r = api.handler({"rawPath": u.path[4:], "queryStringParameters": q}, None)
                return self._send(r["statusCode"], r["body"])
            return super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path == "/api/agent/chat":
                body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
                with pipeline.lock:
                    r = agent.handler({"requestContext": {}, "body": body}, None)
                return self._send(r["statusCode"], r["body"])
            self._send(404, '{"error":"not found"}')

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=0, help="stop after N ticks (0 = run until Ctrl+C)")
    ap.add_argument("--tick-seconds", type=int, default=10)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-server", action="store_true")
    args = ap.parse_args(argv)
    setup_env(args.tick_seconds)
    if not (ROOT / "data" / "processed" / "sensor_meta.json").exists():
        print("Run `python tasks.py download` and `python tasks.py eda` first.")
        return 1
    moto = start_moto()
    try:
        import seed_dynamo
        from traffic_common.config import Settings
        st = Settings.from_env()
        print("Local AWS mock started; creating resources and seeding ...")
        create_resources()
        seed_dynamo.seed_meta(st)
        seed_dynamo.publish_local_model(st)
        seed_dynamo.reset_replay(st)
        pipe = Pipeline()
        from components import load
        if not args.no_server:
            httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(pipe, load("api"), load("agent")))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            print(f"Dashboard: http://localhost:{args.port}   (Ctrl+C to stop)")
        n = 0
        while not args.ticks or n < args.ticks:
            t0 = time.monotonic()
            print(json.dumps(pipe.tick()), flush=True)
            n += 1
            if not args.ticks or n < args.ticks:
                time.sleep(max(0.0, args.tick_seconds - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        moto.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
