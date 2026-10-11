from app.checker import invented_numbers

RESULTS = {"total_pct": 15.0, "bridge": [{"cause": "tariff", "pct_points": 8.1}], "total_change": 150000,
           "actions": [{"saving_money_per_year": [14400, 19200], "night_vs_day": 0.38}]}


def test_numbers_from_tools_pass():
    text = "The bill rose 15.0% (Rs 150,000). Tariff added 8.1 points. Save Rs 14,400-19,200; night load is 38%."
    assert invented_numbers(text, RESULTS) == []


def test_invented_numbers_are_caught():
    assert invented_numbers("The industry average is 20.5% and you could save Rs 99,999.", RESULTS) == ["20.5", "99,999"]


def test_dates_and_small_counts_ignored():
    assert invented_numbers("Between 2026-07-01 and 2026-09-30 the top 3 meters peaked at 02:00", RESULTS) == []
