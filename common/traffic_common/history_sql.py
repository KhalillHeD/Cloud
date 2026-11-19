"""Allow-listed Athena queries over the curated table (block 7).

The agent's query_history tool may only run these templates; parameters are validated
and typed before being substituted, so no free SQL ever reaches Athena. The same
templates are created as Athena named queries at deploy time and exported to
glue/athena/*.sql.
"""
from __future__ import annotations

import re

TABLE = "curated_traffic"
_SENSOR = re.compile(r"^\d{3,10}$")

TEMPLATES: dict[str, dict] = {
    "most_congested": {
        "description": "Segments with the highest average congestion index over the last N days of data",
        "params": {"days": (int, 1, 60, 7), "limit": (int, 1, 50, 10)},
        "sql": """
WITH bounds AS (SELECT max(dt) AS last_dt FROM {table})
SELECT sensor_id,
       round(avg(congestion_index), 3)                                     AS avg_congestion,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_heavy_or_severe,
       round(avg(speed_mph), 1)                                            AS avg_speed_mph,
       count(*)                                                            AS readings
FROM {table}, bounds
WHERE dt > date_format(date_add('day', -{days}, date_parse(last_dt, '%Y-%m-%d')), '%Y-%m-%d')
  AND congestion_index IS NOT NULL
GROUP BY sensor_id
ORDER BY avg_congestion DESC
LIMIT {limit}""",
    },
    "by_hour": {
        "description": "Average speed and congestion by hour of day (optionally one sensor)",
        "params": {"sensor_id": (str, None, None, None)},
        "sql": """
SELECT hour, round(avg(speed_mph), 1) AS avg_speed_mph, round(avg(congestion_index), 3) AS avg_congestion,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_heavy_or_severe
FROM {table}
WHERE congestion_index IS NOT NULL {sensor_filter}
GROUP BY hour ORDER BY hour""",
    },
    "by_weekday": {
        "description": "Average congestion by weekday (0 = Monday) and hour (optionally one sensor)",
        "params": {"sensor_id": (str, None, None, None)},
        "sql": """
SELECT dow, hour, round(avg(congestion_index), 3) AS avg_congestion, round(avg(speed_mph), 1) AS avg_speed_mph
FROM {table}
WHERE congestion_index IS NOT NULL {sensor_filter}
GROUP BY dow, hour ORDER BY dow, hour""",
    },
    "hotspots": {
        "description": "Recurring hotspots: segments heavy/severe in at least min_pct % of weekday peak slots "
                       "(7-10 am, 3-7 pm)",
        "params": {"min_pct": (int, 1, 100, 20), "limit": (int, 1, 50, 15)},
        "sql": """
SELECT sensor_id,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_peak_congested,
       count(DISTINCT dt) AS days
FROM {table}
WHERE dow < 5 AND (hour BETWEEN 7 AND 9 OR hour BETWEEN 15 AND 18) AND congestion_index IS NOT NULL
GROUP BY sensor_id
HAVING avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100 >= {min_pct}
ORDER BY pct_peak_congested DESC
LIMIT {limit}""",
    },
    "daily_summary": {
        "description": "Per-day share of congested readings and the worst hour, last N days",
        "params": {"days": (int, 1, 60, 7)},
        "sql": """
WITH bounds AS (SELECT max(dt) AS last_dt FROM {table}),
hourly AS (
  SELECT dt, hour, avg(congestion_index) AS ci FROM {table}, bounds
  WHERE dt > date_format(date_add('day', -{days}, date_parse(last_dt, '%Y-%m-%d')), '%Y-%m-%d')
  GROUP BY dt, hour)
SELECT dt, round(max(ci), 3) AS worst_hour_congestion, max_by(hour, ci) AS worst_hour,
       round(avg(ci), 3) AS avg_congestion
FROM hourly GROUP BY dt ORDER BY dt""",
    },
}


def render(name: str, params: dict | None = None, table: str = TABLE) -> str:
    if name not in TEMPLATES:
        raise ValueError(f"unknown template '{name}'; allowed: {', '.join(TEMPLATES)}")
    spec = TEMPLATES[name]
    params = dict(params or {})
    unknown = set(params) - set(spec["params"])
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)} for {name}")
    values: dict[str, str] = {"table": table, "sensor_filter": ""}
    for key, (typ, lo, hi, default) in spec["params"].items():
        v = params.get(key, default)
        if typ is int:
            v = int(v)
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be between {lo} and {hi}")
            values[key] = str(v)
        elif key == "sensor_id":
            if v is None or v == "":
                continue
            v = str(v)
            if not _SENSOR.match(v):
                raise ValueError("sensor_id must be digits")
            values["sensor_filter"] = f"AND sensor_id = '{v}'"
    return spec["sql"].format(**values).strip()
