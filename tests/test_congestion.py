import pytest
from traffic_common.config import Settings
from traffic_common.congestion import congestion_index, evaluate, is_escalation, level_of, travel_time_s, week_slot

S = Settings()


def test_congestion_index_formula():
    assert congestion_index(60, 60) == 0
    assert congestion_index(30, 60) == pytest.approx(0.5)
    assert congestion_index(70, 60) == 0          # faster than free flow -> clipped at 0
    assert congestion_index(None, 60) is None
    assert congestion_index(30, 0) is None


@pytest.mark.parametrize("ci,level", [(0.0, "free"), (0.1999, "free"), (0.2, "moderate"), (0.3999, "moderate"),
                                      (0.4, "heavy"), (0.5999, "heavy"), (0.6, "severe"), (1.0, "severe"),
                                      (None, "unknown")])
def test_level_boundaries(ci, level):
    assert level_of(ci, S) == level


def test_thresholds_are_configurable():
    s = Settings(level_moderate=0.1, level_heavy=0.2, level_severe=0.3)
    assert level_of(0.25, s) == "heavy"


def test_travel_time():
    assert travel_time_s(1000, 50) == pytest.approx(1000 / (50 * 0.44704))
    assert travel_time_s(1000, 0) is None
    assert travel_time_s(1000, None) is None


def test_week_slot():
    assert week_slot("2012-03-05T00:00:00") == 0            # Monday
    assert week_slot("2012-03-05T08:05:00") == 97
    assert week_slot("2012-03-11T23:55:00") == 2015         # Sunday last slot


def test_escalation():
    assert is_escalation("free", "heavy")
    assert is_escalation("heavy", "severe")
    assert is_escalation(None, "severe")
    assert not is_escalation("severe", "heavy")
    assert not is_escalation("heavy", "heavy")
    assert not is_escalation("free", "moderate")


def _meta(expected=50.0):
    return {"free_flow_mph": 65.0, "segment_length_m": 1000.0, "expected_speed": [expected] * 2016,
            "lat": 34.1, "lon": -118.3}


def test_evaluate_full_record():
    st = evaluate({"sensor_id": "1", "ts": "2012-03-05T08:00:00", "speed_mph": 26.0, "incident_flag": True,
                   "source": "synthetic-incident"}, _meta(), S)
    assert st["level"] == "severe"
    assert st["congestion_index"] == pytest.approx(0.6)
    tt, ett = 1000 / (26 * 0.44704), 1000 / (50 * 0.44704)
    assert st["travel_time_s"] == pytest.approx(tt, abs=0.1)
    assert st["expected_tt_s"] == pytest.approx(ett, abs=0.1)
    assert st["delay_s"] == pytest.approx(tt - ett, abs=0.2)
    assert st["incident_flag"] is True


@pytest.mark.parametrize("speed", [0, None, -3])
def test_zero_speed_is_missing(speed):
    st = evaluate({"sensor_id": "1", "ts": "2012-03-05T08:00:00", "speed_mph": speed}, _meta(), S)
    assert st["level"] == "unknown" and st["speed_mph"] is None and st["delay_s"] is None
