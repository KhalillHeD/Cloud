import pytest
from traffic_common.config import MAX_SENSORS, Settings, load_dotenv

ENV_KEYS = ["N_SENSORS", "LEVEL_MODERATE", "LEVEL_HEAVY", "LEVEL_SEVERE", "FORECAST_HORIZONS",
            "REPLAY_MODE", "LAB_ROLE_ARN", "AWS_REGION", "AWS_DEFAULT_REGION", "LLM_PROVIDER", "TICK_SECONDS"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults_match_the_spec():
    s = Settings.from_env()
    assert s.n_sensors == 50
    assert (s.level_moderate, s.level_heavy, s.level_severe) == (0.2, 0.4, 0.6)
    assert s.freeflow_percentile == 85
    assert s.region == "us-east-1"
    assert s.alert_dedup_minutes == 30
    assert s.horizon_steps == {15: 3, 30: 6, 60: 12}


@pytest.mark.parametrize("n", [0, MAX_SENSORS + 1, -5])
def test_n_sensors_bounds(monkeypatch, n):
    monkeypatch.setenv("N_SENSORS", str(n))
    with pytest.raises(ValueError, match="N_SENSORS"):
        Settings.from_env()


def test_whole_city_allowed(monkeypatch):
    monkeypatch.setenv("N_SENSORS", "207")
    assert Settings.from_env().n_sensors == 207


def test_thresholds_must_increase(monkeypatch):
    monkeypatch.setenv("LEVEL_HEAVY", "0.1")
    with pytest.raises(ValueError, match="thresholds"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["15,32", "0", ""])
def test_horizons_must_be_5_min_multiples(monkeypatch, value):
    monkeypatch.setenv("FORECAST_HORIZONS", value if value else ",")
    with pytest.raises(ValueError, match="FORECAST_HORIZONS"):
        Settings.from_env()


def test_lab_role_default_and_override(monkeypatch):
    assert Settings.from_env().resolve_lab_role_arn("123") == "arn:aws:iam::123:role/LabRole"
    monkeypatch.setenv("LAB_ROLE_ARN", "arn:aws:iam::1:role/Other")
    assert Settings.from_env().resolve_lab_role_arn("123") == "arn:aws:iam::1:role/Other"


def test_empty_value_means_default(monkeypatch):
    monkeypatch.setenv("N_SENSORS", "  ")
    assert Settings.from_env().n_sensors == 50


def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nN_SENSORS=80\nexport REPLAY_MODE="cursor"\nTICK_SECONDS=10 # demo\nnoequals\n')
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    values = load_dotenv(f)
    assert values == {"N_SENSORS": "80", "REPLAY_MODE": "cursor", "TICK_SECONDS": "10"}
    s = Settings.from_env()
    assert (s.n_sensors, s.replay_mode, s.tick_seconds, s.region) == (80, "cursor", 10, "us-west-2")


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("N_SENSORS=80\n")
    monkeypatch.setenv("N_SENSORS", "20")
    load_dotenv(f)
    assert Settings.from_env().n_sensors == 20
