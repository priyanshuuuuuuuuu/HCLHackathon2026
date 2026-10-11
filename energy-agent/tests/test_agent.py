"""Agent paths with a fake LLM: invented numbers must never reach the answer."""

import pytest

from app import agent, config, llm
from app.ingest import ingest


@pytest.fixture(scope="module", autouse=True)
def data(tmp_path_factory):
    import subprocess
    import sys
    folder = tmp_path_factory.mktemp("data")
    store = tmp_path_factory.mktemp("store")
    config.STORE, config.DB_PATH, config.SLOTS_PATH = store, store / "app.db", store / "meter_clean.parquet"
    subprocess.run([sys.executable, "scripts/generate_data.py", str(folder)], check=True, capture_output=True)
    for t in ["tariff", "production", "weather", "meter_15min"]:
        ingest(str(folder / f"{t}.csv"), t)


def test_mock_answers():
    config.MOCK_LLM = True
    r = agent.answer("Why did the bill go up?")
    assert r["answer_type"] == "decomposition" and r["adds_up"]
    assert agent.answer("What will the tariff be next year?")["answer_type"] == "not_in_data"
    assert agent.answer("Why?", period="2030-Q1")["answer_type"] == "not_in_data"


def test_llm_invented_number_is_replaced(monkeypatch):
    config.MOCK_LLM = False

    def fake_call(system, user, max_tokens=500):
        if "classify" in system:
            return '{"intent": "decomposition", "period": "2026-Q3", "compare_to": "2026-Q2"}', 10
        return "The bill rose 99.9% because of old machines.", 10

    monkeypatch.setattr(llm, "call", fake_call)
    r = agent.answer("Why did the bill go up?")
    trace = agent.get_trace(r["run_id"])
    assert "99.9" not in r["summary"] and trace["checker_passed"] is False
    assert trace["llm_calls"] == 3      # router + writer + one retry after the checker rejected it
    config.MOCK_LLM = True
