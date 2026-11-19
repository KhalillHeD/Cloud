"""Single source of configuration, read from environment variables (and .env locally).

Every Lambda, script and notebook reads its settings through `Settings.from_env()`, so a
value such as N_SENSORS or the congestion thresholds is defined in exactly one place.
Defaults are documented in docs/architecture.md and .env.example.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MAX_SENSORS = 207          # METR-LA has 207 loop detectors
SLOT_MINUTES = 5           # METR-LA sampling interval
LEVELS = ("free", "moderate", "heavy", "severe")


def load_dotenv(path: str | os.PathLike = ".env", override: bool = False) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE, # comments, optional quotes). Returns what it read.

    Existing environment variables win unless override=True, so a real deployment
    environment is never shadowed by a stale local file.
    """
    values: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    if os.environ.get("AWS_REGION") and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"]   # boto3 reads AWS_DEFAULT_REGION
    return values


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    return int(_get(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_get(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # --- AWS / deployment
    project_name: str = "traffic-analytics"
    region: str = "us-east-1"
    stack_name: str = "traffic-analytics"
    lab_role_arn: str | None = None          # None -> arn:aws:iam::<account>:role/LabRole
    aws_endpoint_url: str | None = None      # set in local mode (moto / LocalStack)

    # --- data / replay (blocks 1-2)
    n_sensors: int = 50
    tick_seconds: int = 300                  # real seconds between two 5-min slices (demo: 10)
    replay_mode: str = "clock"               # clock: same weekday+time as now | cursor: from REPLAY_START
    replay_start: str = "2012-03-05T06:00:00"
    synthetic_incident_rate: float = 0.002   # probability per sensor per tick

    # --- congestion (block 4)
    freeflow_percentile: float = 85.0
    level_moderate: float = 0.2
    level_heavy: float = 0.4
    level_severe: float = 0.6

    # --- forecasting (block 8)
    forecast_horizons_min: tuple[int, ...] = (15, 30, 60)

    # --- alerts / agent (block 9)
    alert_dedup_minutes: int = 30
    alert_email: str | None = None
    llm_provider: str = "bedrock"            # bedrock | anthropic | openai (key from Secrets Manager/env) | none
    bedrock_model_id: str | None = None      # None -> chosen by scripts/check_access.py

    @classmethod
    def from_env(cls) -> Settings:
        horizons = tuple(int(h) for h in _get("FORECAST_HORIZONS", "15,30,60").split(",") if h.strip())
        s = cls(
            project_name=_get("PROJECT_NAME", cls.project_name),
            region=_get("AWS_REGION", _get("AWS_DEFAULT_REGION", cls.region)),
            stack_name=_get("STACK_NAME", cls.stack_name),
            lab_role_arn=_get("LAB_ROLE_ARN"),
            aws_endpoint_url=_get("AWS_ENDPOINT_URL"),
            n_sensors=_int("N_SENSORS", cls.n_sensors),
            tick_seconds=_int("TICK_SECONDS", cls.tick_seconds),
            replay_mode=_get("REPLAY_MODE", cls.replay_mode),
            replay_start=_get("REPLAY_START", cls.replay_start),
            synthetic_incident_rate=_float("SYNTHETIC_INCIDENT_RATE", cls.synthetic_incident_rate),
            freeflow_percentile=_float("FREEFLOW_PERCENTILE", cls.freeflow_percentile),
            level_moderate=_float("LEVEL_MODERATE", cls.level_moderate),
            level_heavy=_float("LEVEL_HEAVY", cls.level_heavy),
            level_severe=_float("LEVEL_SEVERE", cls.level_severe),
            forecast_horizons_min=horizons,
            alert_dedup_minutes=_int("ALERT_DEDUP_MINUTES", cls.alert_dedup_minutes),
            alert_email=_get("ALERT_EMAIL"),
            llm_provider=_get("LLM_PROVIDER", cls.llm_provider),
            bedrock_model_id=_get("BEDROCK_MODEL_ID"),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not 1 <= self.n_sensors <= MAX_SENSORS:
            raise ValueError(f"N_SENSORS must be between 1 and {MAX_SENSORS}, got {self.n_sensors}")
        if not 0 < self.level_moderate < self.level_heavy < self.level_severe <= 1:
            raise ValueError("thresholds must satisfy 0 < LEVEL_MODERATE < LEVEL_HEAVY < LEVEL_SEVERE <= 1")
        if not 0 < self.freeflow_percentile <= 100:
            raise ValueError("FREEFLOW_PERCENTILE must be in (0, 100]")
        if self.tick_seconds < 1:
            raise ValueError("TICK_SECONDS must be >= 1")
        if self.replay_mode not in ("clock", "cursor"):
            raise ValueError("REPLAY_MODE must be 'clock' or 'cursor'")
        if not 0 <= self.synthetic_incident_rate <= 1:
            raise ValueError("SYNTHETIC_INCIDENT_RATE must be a probability")
        if not self.forecast_horizons_min or any(h <= 0 or h % SLOT_MINUTES for h in self.forecast_horizons_min):
            raise ValueError(f"FORECAST_HORIZONS must be positive multiples of {SLOT_MINUTES} minutes")
        if self.llm_provider not in ("bedrock", "anthropic", "openai", "none"):
            raise ValueError("LLM_PROVIDER must be bedrock, anthropic, openai or none")

    @property
    def horizon_steps(self) -> dict[int, int]:
        """Horizon in minutes -> number of 5-minute steps ahead (15 -> 3)."""
        return {h: h // SLOT_MINUTES for h in self.forecast_horizons_min}

    def resolve_lab_role_arn(self, account_id: str) -> str:
        return self.lab_role_arn or f"arn:aws:iam::{account_id}:role/LabRole"
