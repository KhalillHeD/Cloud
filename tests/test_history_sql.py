import pytest
from traffic_common import history_sql


def test_every_template_renders_with_defaults():
    for name in history_sql.TEMPLATES:
        sql = history_sql.render(name)
        assert "curated_traffic" in sql and "{" not in sql


def test_parameters_are_typed_and_bounded():
    assert "LIMIT 5" in history_sql.render("most_congested", {"days": 3, "limit": 5})
    with pytest.raises(ValueError):
        history_sql.render("most_congested", {"limit": 500})
    with pytest.raises(ValueError):
        history_sql.render("most_congested", {"days": "1; DROP TABLE x"})


def test_no_free_sql():
    with pytest.raises(ValueError, match="unknown template"):
        history_sql.render("SELECT * FROM secrets")
    with pytest.raises(ValueError, match="unknown parameter"):
        history_sql.render("by_hour", {"where": "1=1"})
    with pytest.raises(ValueError, match="digits"):
        history_sql.render("by_hour", {"sensor_id": "1' OR '1'='1"})
    assert "sensor_id = '773869'" in history_sql.render("by_hour", {"sensor_id": "773869"})
