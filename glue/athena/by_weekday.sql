-- Average congestion by weekday (0 = Monday) and hour (optionally one sensor)
-- defaults: {'sensor_id': None}
SELECT dow, hour, round(avg(congestion_index), 3) AS avg_congestion, round(avg(speed_mph), 1) AS avg_speed_mph
FROM curated_traffic
WHERE congestion_index IS NOT NULL 
GROUP BY dow, hour ORDER BY dow, hour;
