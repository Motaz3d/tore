"""Tests for `tx reproduce` — replaying a saved TxResult envelope.

Network-free: the CLI's engine factory is monkeypatched with the same fake
hazard modules used in tests/test_tx_core.py. The contract under test:
re-run the recorded request, compare honestly (per-hazard status is the
substance; analysis_id is day-scoped), exit 0 reproduced / 1 diverged /
2 usage error — never a fabricated match.
"""

from __future__ import annotations

import json

import pytest

from tx_core import cli

from fakes import FakeHazardModule, make_engine


def _envelope(engine, **kw) -> dict:
    """A real TxResult envelope produced by the given (fake) engine."""
    return engine.analyze(**kw).to_dict()


def _write(tmp_path, payload) -> str:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture()
def engine(monkeypatch):
    fake = make_engine({"flood": FakeHazardModule("flood")})
    monkeypatch.setattr(cli, "TXEngine", lambda: fake)
    return fake


def test_reproduce_identical_result(engine, tmp_path, capsys):
    env = _envelope(engine, lat=40.5, lon=-8.1, hazards=["flood"])
    path = _write(tmp_path, env)
    assert cli.main(["reproduce", path, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["reproduced"] is True
    assert report["same_analysis_id"] is True  # same day + same inputs
    assert report["same_status"] is True
    assert report["hazard_status_diffs"] == {}
    assert report["new_analysis_id"] == env["analysis_id"]


def test_reproduce_text_output(engine, tmp_path, capsys):
    env = _envelope(engine, lat=40.5, lon=-8.1, hazards=["flood"])
    path = _write(tmp_path, env)
    assert cli.main(["reproduce", path]) == 0
    out = capsys.readouterr().out
    assert env["analysis_id"] in out
    assert "REPRODUCED" in out


def test_reproduce_diverged_status(engine, tmp_path, capsys):
    env = _envelope(engine, lat=40.5, lon=-8.1, hazards=["flood"])
    # The world changed: the same hazard now reports unavailable.
    env2 = dict(env)
    env2["status"] = "ok"
    env2["results"] = [dict(r) for r in env["results"]]
    env2["results"][0]["status"] = "ok"
    engine._registry = lambda hid: FakeHazardModule(  # re-point the fake
        hid, status="unavailable", unavailable_reason="source offline")
    path = _write(tmp_path, env2)
    assert cli.main(["reproduce", path, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["reproduced"] is False
    assert report["same_status"] is False
    assert report["hazard_status_diffs"] == {
        "flood": {"was": "ok", "now": "unavailable"}
    }


def test_reproduce_text_reports_divergence(engine, tmp_path, capsys):
    env = _envelope(engine, lat=40.5, lon=-8.1, hazards=["flood"])
    env["results"][0]["status"] = "partial"  # recorded as partial…
    path = _write(tmp_path, env)             # …engine now says ok
    assert cli.main(["reproduce", path]) == 1
    out = capsys.readouterr().out
    assert "DIVERGED" in out
    assert "flood" in out


def test_reproduce_missing_file(capsys):
    assert cli.main(["reproduce", "/no/such/file.json"]) == 2
    assert "error" in capsys.readouterr().err


def test_reproduce_invalid_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json {", encoding="utf-8")
    assert cli.main(["reproduce", str(path)]) == 2
    assert "error" in capsys.readouterr().err


def test_reproduce_not_an_envelope(tmp_path, capsys):
    path = _write(tmp_path, {"job_id": "TXJ-x", "status": "queued"})
    assert cli.main(["reproduce", path]) == 2
    err = capsys.readouterr().err
    assert "not a TX result envelope" in err


def test_reproduce_bad_coordinates(tmp_path, capsys):
    path = _write(tmp_path, {"location": {"lat": "north", "lon": 2}})
    assert cli.main(["reproduce", path]) == 2
    assert "error" in capsys.readouterr().err


def test_reproduce_engine_version_stamped(engine, tmp_path, capsys):
    env = _envelope(engine, lat=1.0, lon=2.0, hazards=["flood"])
    path = _write(tmp_path, env)
    assert cli.main(["reproduce", path, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["engine_version"] == env["engine_version"]
