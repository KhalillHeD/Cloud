from datetime import datetime, timedelta

import components
from traffic_common.config import Settings
from traffic_common.replay import DATA_END, DATA_START, FIRST_MONDAY, clock_replay_ts, incident_factor, next_cursor


def test_clock_mapping_keeps_weekday_and_time():
    for now in (datetime(2026, 11, 19, 8, 7), datetime(2027, 1, 3, 23, 59), datetime(2026, 2, 2, 0, 0)):
        t = clock_replay_ts(now)
        assert t.weekday() == now.weekday()
        assert (t.hour, t.minute) == (now.hour, now.minute - now.minute % 5)
        assert FIRST_MONDAY <= t < FIRST_MONDAY + timedelta(weeks=16)


def test_cursor_advances_and_wraps():
    assert next_cursor(datetime(2012, 3, 5, 8, 0)) == datetime(2012, 3, 5, 8, 5)
    assert next_cursor(DATA_END) == DATA_START


def test_incidents_are_deterministic_and_roughly_at_rate():
    ts = datetime(2012, 3, 5, 8, 0)
    assert incident_factor("773869", ts, 0.01) == incident_factor("773869", ts + timedelta(minutes=5), 0.01)
    assert incident_factor("773869", ts, 0) is None
    hits = sum(incident_factor(str(s), ts + timedelta(minutes=30 * k), 0.002) is not None
               for s in range(300) for k in range(40))
    expected = 300 * 40 * 0.002 * 6                          # rate per tick x 6 ticks per window
    assert 0.6 * expected < hits < 1.4 * expected
    f = incident_factor("1", ts, 1.0)
    assert 0.15 <= f <= 0.40


class FakeSource:
    def sensors(self):
        return [{"sensor_id": str(i), "lat": 34.0 + i / 100, "lon": -118.0} for i in range(5)]

    def get_slice(self, ts):
        return {"0": 60.0, "1": 0.0, "2": None, "3": 55.5, "4": 61.0}


def test_producer_records_follow_the_schema():
    producer = components.load("producer")
    st = Settings(n_sensors=3, synthetic_incident_rate=0.0)
    recs = producer.build_records(datetime(2012, 3, 5, 8, 0), st, FakeSource())
    assert len(recs) == 3                                     # N_SENSORS prefix
    assert set(recs[0]) == {"sensor_id", "ts", "speed_mph", "lat", "lon", "incident_flag", "source"}
    assert recs[0]["ts"] == "2012-03-05T08:00:00" and recs[0]["source"] == "metr-la-replay"
    assert recs[1]["speed_mph"] is None and recs[2]["speed_mph"] is None   # zero and missing


def test_synthetic_incidents_are_labelled():
    producer = components.load("producer")
    recs = producer.build_records(datetime(2012, 3, 5, 8, 0), Settings(n_sensors=5, synthetic_incident_rate=1.0),
                                  FakeSource())
    hit = [r for r in recs if r["incident_flag"]]
    assert hit and all(r["source"] == "synthetic-incident" for r in hit)
    assert all(r["speed_mph"] < 61 * 0.41 for r in hit)
