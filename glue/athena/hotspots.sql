-- Recurring hotspots: segments heavy/severe in at least min_pct % of weekday peak slots (7-10 am, 3-7 pm)
-- defaults: {'min_pct': 20, 'limit': 15}
SELECT sensor_id,
       round(avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100, 1) AS pct_peak_congested,
       count(DISTINCT dt) AS days
FROM curated_traffic
WHERE dow < 5 AND (hour BETWEEN 7 AND 9 OR hour BETWEEN 15 AND 18) AND congestion_index IS NOT NULL
GROUP BY sensor_id
HAVING avg(CASE WHEN level IN ('heavy','severe') THEN 1.0 ELSE 0 END) * 100 >= 20
ORDER BY pct_peak_congested DESC
LIMIT 15;
