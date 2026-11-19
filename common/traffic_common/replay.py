"""Block 1: traffic sources. The METR-LA replay makes 2012 data look live.

Replay files (written by scripts/precompute_freeflow.py, uploaded to S3 replay/ by the seed):
  replay/YYYY-MM-DD.json  {"date", "sensors": [ids], "speeds": [[mph or null] * n_sensors] * 288}
  meta/sensors.json       [{"sensor_id", "lat", "lon", "rank"}] ordered so the first N form a
                          compact area (N_SENSORS picks a prefix)

Time mapping ("which 2012 slot is now?"):
  clock  : same weekday and time of day as now in Los Angeles, cycling over the 16 full weeks
  cursor : starts at REPLAY_START, advances one 5-minute slot per tick (demo mode)

A real API can replace the replay by implementing TrafficSource.get_slice().
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from .config import SLOT_MINUTES

DATA_START = datetime(2012, 3, 1)
DATA_END = datetime(2012, 6, 27, 23, 55)
FIRST_MONDAY = datetime(2012, 3, 5)
FULL_WEEKS = 16


def la_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001 - no tz database (e.g. Windows without tzdata)
        return (datetime.now(UTC) - timedelta(hours=8)).replace(tzinfo=None)


def floor_slot(t: datetime) -> datetime:
    return t.replace(minute=t.minute - t.minute % SLOT_MINUTES, second=0, microsecond=0)


def clock_replay_ts(now: datetime) -> datetime:
    now = floor_slot(now)
    monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0)
    week = ((monday - FIRST_MONDAY).days // 7) % FULL_WEEKS
    return FIRST_MONDAY + timedelta(weeks=week) + (now - monday)


def next_cursor(ts: datetime) -> datetime:
    nxt = ts + timedelta(minutes=SLOT_MINUTES)
    return DATA_START if nxt > DATA_END else nxt


def incident_factor(sensor_id: str, ts: datetime, rate_per_tick: float) -> float | None:
    """Synthetic incident: deterministic per sensor and 30-minute window, so every tick of the
    window agrees without any state. Returns the speed multiplier, or None if no incident."""
    if rate_per_tick <= 0:
        return None
    bucket = int((ts - DATA_START).total_seconds() // 1800)
    h = hashlib.sha256(f"{sensor_id}|{bucket}".encode()).digest()
    u = int.from_bytes(h[:4], "big") / 2**32
    p = min(1.0, rate_per_tick * 1800 / (SLOT_MINUTES * 60))   # probability per 30-min window
    if u >= p:
        return None
    return 0.15 + 0.25 * (int.from_bytes(h[4:6], "big") / 2**16)  # speed drops to 15-40 %


class TrafficSource(Protocol):
    def sensors(self) -> list[dict]: ...
    def get_slice(self, ts: datetime) -> dict[str, float | None]: ...


class MetrLaReplay:
    """Reads replay files from a local directory (REPLAY_DIR) or from the S3 lake bucket."""

    def __init__(self, local_dir: str | None = None):
        self.local_dir = Path(local_dir) if local_dir else (Path(os.environ["REPLAY_DIR"])
                                                            if os.environ.get("REPLAY_DIR") else None)
        self._days: dict[str, dict] = {}
        self._sensors: list[dict] | None = None

    def _load(self, key: str):
        if self.local_dir:
            return json.loads((self.local_dir / key).read_text(encoding="utf-8"))
        from .store import s3_get_json
        return s3_get_json(key)

    def sensors(self) -> list[dict]:
        if self._sensors is None:
            self._sensors = self._load("meta/sensors.json")
        return self._sensors

    def get_slice(self, ts: datetime) -> dict[str, float | None]:
        day = ts.strftime("%Y-%m-%d")
        if day not in self._days:
            if len(self._days) > 3:
                self._days.clear()
            self._days[day] = self._load(f"replay/{day}.json")
        d = self._days[day]
        row = d["speeds"][(ts.hour * 60 + ts.minute) // SLOT_MINUTES]
        return dict(zip(d["sensors"], row, strict=True))


class RealApiSource:
    """Placeholder for a live provider (e.g. a city open-data feed or a commercial traffic API).
    Implement sensors() and get_slice(ts) returning {sensor_id: speed_mph}; nothing else changes."""

    def sensors(self) -> list[dict]:
        raise NotImplementedError("configure a real traffic API here")

    def get_slice(self, ts: datetime) -> dict[str, float | None]:
        raise NotImplementedError("configure a real traffic API here")
