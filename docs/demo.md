# 5-minute demo script

**Before the presentation (15 min earlier):**

```powershell
python tasks.py deploy      # if not deployed yet
python tasks.py demo        # starts the replay at Monday 06:00, 1 tick (5 replay-minutes) every 10 s
```

The morning rush builds from about 07:00 replay time, about 2 minutes after `demo`
starts. Confirm the SNS subscription email beforehand so alert emails arrive. If the
network is unreliable, `python tasks.py local` gives the same dashboard on the laptop.

| Time | Show | Say |
|---|---|---|
| 0:00–0:45 | Slide "How the AWS services connect" | "Ten blocks, all serverless. A replay producer makes 2012 METR-LA data look live: every tick, one 5-minute slice of 50 loop detectors goes into Kinesis." |
| 0:45–1:45 | Dashboard, **LIVE** | Map colours = congestion index `1 − speed / free-flow` (free/moderate/heavy/severe). Point at the KPI: "x % of the monitored km is heavy or severe". Watch roads turn red as the rush builds; the replay clock moves 5 minutes every 10 seconds. |
| 1:45–2:30 | Congested roads table + travel-time chart | Click the worst segment: free-flow vs expected vs actual travel time; the delay badge is actual − expected (expected = the usual median for that weekday and time). The dashed line is the forecast. |
| 2:30–3:15 | Timeline slider | Drag left: measured history (-60 → now). Drag right: **FORECAST** badge, dashed map, +15/+30/+60 min. "XGBoost: 2.9 mph MAE at +15 min against 3.4 for last-value; it beats both baselines at every horizon" (docs/results.md). |
| 3:15–4:15 | Chat box | Ask: *"Which roads will be congested in 30 minutes, and why?"* Show the answer and the **tool calls** line under it (get_forecast, get_current_congestion): the agent reads DynamoDB, it does not guess. Optional: *"What are the recurring hotspots?"* → Athena template. |
| 4:15–4:45 | Alerts feed + email on the phone | The agent wrote the alert; de-duplicated to one per segment per 30 min. Purple "synthetic incident" tags are injected test events, clearly labelled. |
| 4:45–5:00 | Terminal | `python tasks.py destroy` removes every billable resource: "one command to deploy, one to destroy". |

**Backup commands**

```powershell
python tasks.py ask "Which roads will be congested in 30 minutes, and why?"
python tasks.py report      # daily report into S3 reports/ + email
python tasks.py status
```
