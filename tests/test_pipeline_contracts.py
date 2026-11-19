"""End-to-end contracts on an in-process AWS mock (moto): processor -> DynamoDB -> forecast
-> API routes -> agent (alert de-duplication, chat tool loop)."""
import base64
import json
from datetime import datetime, timedelta

import components
import pytest

moto = pytest.importorskip("moto")

SENSORS = ["100", "200", "300"]


@pytest.fixture
def aws(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                 "PROJECT_NAME": "tt", "LAKE_BUCKET": "tt-lake", "LLM_PROVIDER": "none", "TICK_SECONDS": "300"}.items():
        monkeypatch.setenv(k, v)
    for k in ("AWS_ENDPOINT_URL", "AWS_PROFILE", "AWS_SESSION_TOKEN", "ALERT_TOPIC_ARN", "N_SENSORS"):
        monkeypatch.delenv(k, raising=False)
    from traffic_common import store
    with moto.mock_aws():
        store.client.cache_clear()
        store.resource.cache_clear()
        import local_run
        local_run.create_resources()
        meta = [{"sensor_id": s, "idx": i, "rank": i, "lat": 34.1 + i / 100, "lon": -118.3, "free_flow_mph": 65.0,
                 "segment_length_m": 1000.0, "neighbors": [x for x in SENSORS if x != s],
                 "expected_speed": [60.0] * 2016, "mean_speed": [58.0] * 2016} for i, s in enumerate(SENSORS)]
        store.batch_put("sensor_meta", [store.encode_meta(m) for m in meta])
        for name in ("processing", "forecast", "api", "agent"):
            mod = components.load(name)
            getattr(mod, "_META", {}).clear()
        components.load("agent").tools._META.clear()
        yield store
        store.client.cache_clear()
        store.resource.cache_clear()


def kinesis_event(records):
    return {"Records": [{"kinesis": {"data": base64.b64encode(json.dumps(r).encode()).decode()}} for r in records]}


def feed(n_slots, speed_of):
    proc = components.load("processing")
    t0 = datetime(2012, 3, 5, 7, 0)
    out = []
    for k in range(n_slots):
        ts = (t0 + timedelta(minutes=5 * k)).strftime("%Y-%m-%dT%H:%M:%S")
        recs = [{"sensor_id": s, "ts": ts, "speed_mph": speed_of(s, k), "lat": 34.1, "lon": -118.3,
                 "incident_flag": False, "source": "metr-la-replay"} for s in SENSORS]
        out.append(proc.handler(kinesis_event(recs), None))
    return out


def test_processor_writes_state_and_detects_escalation(aws):
    proc = components.load("processing")
    feed(3, lambda s, k: 62.0)
    latest = {x["sensor_id"]: x for x in aws.latest_all()}
    assert set(latest) == set(SENSORS) and latest["100"]["level"] == "free"
    from traffic_common.config import Settings
    res = proc.process([{"sensor_id": "100", "ts": "2012-03-05T07:15:00", "speed_mph": 20.0}], Settings(),
                       publish=False)
    assert [a["sensor_id"] for a in res["alerts"]] == ["100"] and res["alerts"][0]["level"] == "severe"
    again = proc.process([{"sensor_id": "100", "ts": "2012-03-05T07:20:00", "speed_mph": 21.0}], Settings(),
                         publish=False)
    assert again["alerts"] == []                               # still severe: no new escalation
    assert len(aws.history_for_sensor("100", "2012-03-05T00:00:00", "2012-03-05T23:00:00")) == 5


def test_forecast_then_api_contracts(aws):
    feed(13, lambda s, k: 60.0 - k * (3 if s == "300" else 0))
    fc = components.load("forecast")
    from traffic_common.config import Settings
    res = fc.run(Settings(), publish=False)
    assert res["forecasts"] == 9 and res["model"].startswith("historical-average")
    api = components.load("api")

    def get(path, **q):
        r = api.handler({"rawPath": path, "queryStringParameters": q or None}, None)
        return r["statusCode"], json.loads(r["body"])

    code, latest = get("/latest")
    assert code == 200 and latest["as_of"] == "2012-03-05T08:00:00" and len(latest["sensors"]) == 3
    assert {"by_level", "total_km", "congested_km", "pct_congested", "incidents"} <= set(latest["summary"])
    assert latest["sensors"][0]["sensor_id"] == "300"         # worst first
    for h in (15, 30, 60):
        code, f = get("/forecast", horizon=str(h))
        assert code == 200 and f["available"] and len(f["items"]) == 3
        assert f["target_ts"] == (datetime(2012, 3, 5, 8, 0) + timedelta(minutes=h)).strftime("%Y-%m-%dT%H:%M:%S")
    code, h = get("/history", sensor_id="300")
    assert code == 200 and len(h["points"]) == 13 and len(h["forecast"]) == 3
    assert {"ts", "speed_mph", "travel_time_s", "expected_tt_s", "delay_s"} <= set(h["points"][0])
    code, snap = get("/history", ts="2012-03-05T07:30:00")
    assert code == 200 and len(snap["sensors"]) == 3
    assert get("/history", sensor_id="x' OR 1")[0] == 400
    assert get("/history", ts="yesterday")[0] == 400
    assert get("/forecast", horizon="abc")[0] == 400
    assert get("/nope")[0] == 404
    code, sensors = get("/sensors")
    assert code == 200 and [s["sensor_id"] for s in sensors["sensors"]] == SENSORS
    assert get("/alerts")[1] == {"alerts": []}


def test_alert_dedup_window(aws):
    agent = components.load("agent")
    from traffic_common.config import Settings
    alert = {"kind": "current", "sensor_id": "100", "ts": "2012-03-05T08:00:00", "level": "severe",
             "speed_mph": 20, "free_flow_mph": 65}
    first = agent.alert_mode({"alerts": [alert]}, Settings())
    second = agent.alert_mode({"alerts": [alert]}, Settings())
    assert first["sent"] == 1 and "sensor 100" in first["message"]
    assert second == {"sent": 0, "deduplicated": 1}
    assert len(aws.recent_alerts()) == 1
    assert aws.claim_alert("100", 30, now=10**10)              # allowed again once the window has passed


def test_chat_uses_real_tool_calls(aws):
    """A scripted chat model asks for get_forecast; the agent must execute it and return the answer."""
    from langchain_core.messages import AIMessage
    feed(13, lambda s, k: 60.0 - k * 3)
    components.load("forecast").run(__import__("traffic_common.config", fromlist=["Settings"]).Settings(),
                                    publish=False)
    agent = components.load("agent")

    class ScriptedLLM:
        available, provider = True, "scripted"

        def __init__(self):
            self.seen = []

        def invoke(self, msgs, tools=None):
            self.seen.append(msgs[-1])
            if len(self.seen) == 1:
                assert {t.name for t in tools} == {"get_current_congestion", "get_forecast", "query_history"}
                return AIMessage(content="", tool_calls=[{"name": "get_forecast", "args": {"horizon": 30},
                                                          "id": "call_1"}])
            data = json.loads(msgs[-1].content)
            return AIMessage(content=f"{data['count']} segments at {data['target_time']}")

    llm = ScriptedLLM()
    out = agent.chat("Which roads will be congested in 30 minutes, and why?", llm=llm)
    assert out["tool_calls"] == [{"name": "get_forecast", "args": {"horizon": 30}}]
    assert out["answer"].endswith("2012-03-05T08:30:00")
    offline = agent.offline_chat("what about in 15 minutes?")
    assert offline["tool_calls"][0]["args"] == {"horizon": 15}


def test_bedrock_and_anthropic_models_accept_the_tool_schemas(monkeypatch):
    """Binding the tools converts their schemas; this must work for both providers (no network call)."""
    pytest.importorskip("langchain_aws")
    agent_tools = components.load("agent").tools
    from llm import make_model
    from traffic_common.config import Settings
    tools = agent_tools.langchain_tools()
    bedrock = make_model("bedrock", Settings(bedrock_model_id="us.anthropic.claude-haiku-test"))
    bedrock.bind_tools(tools)
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    make_model("anthropic", Settings()).bind_tools(tools)
