-- Average speed and congestion by hour of day (optionally one sensor)
-- defaults: {'sensor_id': None}
SELECT hour, round(avg(speed_mph), 1) AS avg_speed_mph, round(avg(congestion_index), 3) AS avg_congestion,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_heavy_or_severe
FROM curated_traffic
WHERE congestion_index IS NOT NULL 
GROUP BY hour ORDER BY hour;
