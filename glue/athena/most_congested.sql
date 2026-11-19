-- Segments with the highest average congestion index over the last N days of data
-- defaults: {'days': 7, 'limit': 10}
WITH bounds AS (SELECT max(dt) AS last_dt FROM curated_traffic)
SELECT sensor_id,
       round(avg(congestion_index), 3)                                     AS avg_congestion,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_heavy_or_severe,
       round(avg(speed_mph), 1)                                            AS avg_speed_mph,
       count(*)                                                            AS readings
FROM curated_traffic, bounds
WHERE dt > date_format(date_add('day', -7, date_parse(last_dt, '%Y-%m-%d')), '%Y-%m-%d')
  AND congestion_index IS NOT NULL
GROUP BY sensor_id
ORDER BY avg_congestion DESC
LIMIT 10;
