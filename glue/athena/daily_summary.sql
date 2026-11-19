-- Per-day share of congested readings and the worst hour, last N days
-- defaults: {'days': 7}
WITH bounds AS (SELECT max(dt) AS last_dt FROM curated_traffic),
hourly AS (
  SELECT dt, hour, avg(congestion_index) AS ci FROM curated_traffic, bounds
  WHERE dt > date_format(date_add('day', -7, date_parse(last_dt, '%Y-%m-%d')), '%Y-%m-%d')
  GROUP BY dt, hour)
SELECT dt, round(max(ci), 3) AS worst_hour_congestion, max_by(hour, ci) AS worst_hour,
       round(avg(ci), 3) AS avg_congestion
FROM hourly GROUP BY dt ORDER BY dt;
