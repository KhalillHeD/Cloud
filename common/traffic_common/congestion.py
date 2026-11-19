"""Block 4: the congestion formula, shared by the processor, the backfill and the tests.

congestion_index = max(0, 1 - speed / free_flow_speed)
level:            free < LEVEL_MODERATE <= moderate < LEVEL_HEAVY <= heavy < LEVEL_SEVERE <= severe
travel time:      segment_length / speed
expected time:    segment_length / median speed of the same sensor, weekday and 5-min slot
                  (the median of length/speed equals length/median(speed), as the map is monotone)
delay:            actual - expected
"""
from __future__ import annotations

from datetime import datetime

from .config import SLOT_MINUTES, Settings

MPH_TO_MPS = 0.44704
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES     # 288
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY          # 2016
LEVEL_RANK = {"unknown": -1, "free": 0, "moderate": 1, "heavy": 2, "severe": 3}
ALERT_LEVELS = ("heavy", "severe")


def parse_ts(ts: str | datetime) -> datetime:
    return ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)


def fmt_ts(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def week_slot(ts: str | datetime) -> int:
    """Index 0..2015 of the 5-minute slot in the week (Monday 00:00 = 0)."""
    t = parse_ts(ts)
    return t.weekday() * SLOTS_PER_DAY + (t.hour * 60 + t.minute) // SLOT_MINUTES


def congestion_index(speed_mph: float | None, free_flow_mph: float) -> float | None:
    if speed_mph is None or free_flow_mph is None or free_flow_mph <= 0:
        return None
    return max(0.0, 1.0 - speed_mph / free_flow_mph)


def level_of(ci: float | None, s: Settings) -> str:
    if ci is None:
        return "unknown"
    if ci < s.level_moderate:
        return "free"
    if ci < s.level_heavy:
        return "moderate"
    if ci < s.level_severe:
        return "heavy"
    return "severe"


def travel_time_s(length_m: float, speed_mph: float | None) -> float | None:
    if speed_mph is None or speed_mph <= 0:
        return None
    return length_m / (speed_mph * MPH_TO_MPS)


def is_escalation(prev_level: str | None, new_level: str) -> bool:
    """True when a segment enters heavy/severe or goes from heavy to severe."""
    return new_level in ALERT_LEVELS and LEVEL_RANK[new_level] > LEVEL_RANK.get(prev_level or "unknown", -1)


def evaluate(record: dict, meta: dict, s: Settings) -> dict:
    """Turn one stream record into the traffic_latest / traffic_history state."""
    speed = record.get("speed_mph")
    speed = None if speed is None or speed <= 0 else float(speed)   # METR-LA zeros = missing
    ff = float(meta["free_flow_mph"])
    length = float(meta["segment_length_m"])
    expected_speed = meta["expected_speed"][week_slot(record["ts"])]
    ci = congestion_index(speed, ff)
    tt = travel_time_s(length, speed)
    ett = travel_time_s(length, expected_speed)
    return {
        "sensor_id": str(record["sensor_id"]),
        "ts": record["ts"],
        "speed_mph": None if speed is None else round(speed, 2),
        "free_flow_mph": round(ff, 2),
        "congestion_index": None if ci is None else round(ci, 4),
        "level": level_of(ci, s),
        "segment_length_m": round(length, 1),
        "free_flow_tt_s": round(travel_time_s(length, ff), 1),
        "travel_time_s": None if tt is None else round(tt, 1),
        "expected_tt_s": None if ett is None else round(ett, 1),
        "delay_s": None if tt is None or ett is None else round(tt - ett, 1),
        "lat": record.get("lat", meta.get("lat")),
        "lon": record.get("lon", meta.get("lon")),
        "incident_flag": bool(record.get("incident_flag", False)),
        "source": record.get("source", "unknown"),
    }
