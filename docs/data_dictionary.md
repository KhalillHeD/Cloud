# Data dictionary

All timestamps are **replay time**: METR-LA local Los Angeles time in 2012, ISO format
`YYYY-MM-DDTHH:MM:SS`, aligned to 5-minute slots. Speeds are mph.

## Source files (data/, gitignored)

| File | Content |
|---|---|
| `metr-la.h5` | pandas HDF5, key `df`: 34272 rows (5-min slots, 2012-03-01 00:00 → 2012-06-27 23:55) × 207 sensor columns, speed mph, **0 = missing** (8.1 %) |
| `graph_sensor_locations.csv` | `index, sensor_id, latitude, longitude` |
| `distances_la_2012.csv` | `from, to, cost` road distance in metres between sensors (directional) |

## Derived (data/processed/)

| File | Content |
|---|---|
| `sensor_meta.json` | selected `N_SENSORS` sensors (see below) |
| `sensors.json`, `s3/meta/sensors.json` | all 207 sensors ranked by distance to the network centre (the first N are used) |
| `s3/replay/YYYY-MM-DD.json` | `{date, sensors[207], speeds[288][207]}` (null = missing) |
| `training.npz` | arrays for training: speeds (T×N, NaN missing), ts, hour, dow, minute, free_flow, expected, mean_profile, neighbor_idx, sensor_ids |

## Stream record (blocks 2 → 3 → 4 and 6)

| Field | Type | Meaning |
|---|---|---|
| `sensor_id` | string | METR-LA detector id |
| `ts` | string | slot start (replay time) |
| `speed_mph` | number or null | null when the reading is missing (METR-LA zero) |
| `lat`, `lon` | number | detector position |
| `incident_flag` | bool | **true only for synthetic incidents** |
| `source` | string | `metr-la-replay` or `synthetic-incident` (a real API would use its own name) |

Synthetic incidents: per sensor and 30-minute window, with probability
`SYNTHETIC_INCIDENT_RATE × 6`, the speed is multiplied by 0.15–0.40. Deterministic (hash of
sensor and window), so replays are reproducible.

## DynamoDB (block 5)

| Table | Keys | Attributes |
|---|---|---|
| `<p>-sensor_meta` | `sensor_id` | `idx`, `rank`, `lat`, `lon`, `free_flow_mph` (85th percentile, training period), `segment_length_m` (road distance to the nearest downstream sensor, clipped 200–3000 m), `neighbors` (3 nearest by road), `expected_speed` / `mean_speed` (2016 comma-separated values: median / mean speed per weekday × 5-min slot, training period) |
| `<p>-traffic_latest` | `sensor_id` | `ts`, `speed_mph`, `free_flow_mph`, `congestion_index`, `level`, `segment_length_m`, `free_flow_tt_s`, `travel_time_s`, `expected_tt_s`, `delay_s`, `lat`, `lon`, `incident_flag`, `source`, `prev_level` |
| `<p>-traffic_history` | `sensor_id`, `ts` (GSI `by_ts`: `ts`, `sensor_id`) | same as latest + `expire_at` (TTL 3 days) |
| `<p>-forecasts` | `sensor_id`, `target_ts` | `horizon_min`, `issued_ts`, `speed_mph`, `congestion_index`, `level`, `travel_time_s`, `expected_tt_s`, `delay_s`, `model`, `expire_at` (TTL 2 days) |
| `<p>-app_state` | `pk`, `sk` | `replay/cursor` (next slot), `dedup/<sensor>` (last alert time), `alert/<ms>#<sensor>` (alerts feed, TTL 7 days), `forecast_latest/<h>` (snapshot for the dashboard), `model/active` (`mode`, `s3_key`, `endpoint`, `mae`) |

Congestion: `congestion_index = max(0, 1 − speed / free_flow)`; level free < 0.2 ≤ moderate
< 0.4 ≤ heavy < 0.6 ≤ severe. Travel time = length / speed; expected = length / median speed
of the same weekday and slot; delay = actual − expected.

## Data lake (block 7, bucket `<p>-lake-<account>-<region>`)

| Prefix | Format | Written by |
|---|---|---|
| `raw/dt=YYYY-MM-DD/hour=HH/` | JSON lines (stream records), arrival time | Firehose |
| `curated/dt=YYYY-MM-DD/` | Parquet: `ts, sensor_id, speed_mph, speed_filled, free_flow_mph, congestion_index, level, travel_time_s, incident_flag, source, hour, dow` (replay date) | Glue ETL (streamed data) + `seed` backfill (full METR-LA history) |
| `features/dt=YYYY-MM-DD/` | Parquet: hourly `avg_speed_mph, avg_congestion, share_congested, had_incident, readings` per sensor | Glue ETL |
| `replay/`, `meta/` | JSON | `seed` |
| `models/<run>/` | `model.json` (XGBoost JSON), `metrics.json` | `train` |
| `ml/<run>/` | SageMaker input, code and output | `train --where sagemaker` |
| `reports/` | daily reports (Markdown) | agent |
| `athena-results/` | query results (expire after 7 days) | Athena |

Glue catalog (`traffic_analytics`): `raw` (crawler), `curated_traffic` and `features_hourly`
(defined in the template, partition projection on `dt`).

## Forecast features (block 8)

`lag_0 … lag_11` (speed now and the previous hour), `mean_3`, `mean_12`, `std_12`,
`delta_1`, `delta_3`, `nbr_mean` (3 road neighbours now), `nbr_delta`, `free_flow`,
`expected_now`, `expected_target` (median profile at the target slot), `hour`, `dow`,
`slot_sin`, `slot_cos`, `horizon` (steps: 3, 6, 12), `sensor_idx`. Target: speed at
t + horizon. Identical in `ml/features.py` (training, numpy) and
`common/traffic_common/features.py` (Lambda, pure Python); a test enforces it.
