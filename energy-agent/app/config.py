"""All settings and assumptions in one place. Values marked CONFIRM come from the mentors / data pack."""

import os
from pathlib import Path

STORE = Path(os.getenv("STORE_DIR", "store"))
DB_PATH = STORE / "app.db"                    # SQLite: reference tables, daily tables, logs, traces
SLOTS_PATH = STORE / "meter_clean.parquet"    # clean, priced 15-min meter data

PLANT_TZ = os.getenv("PLANT_TZ", "Asia/Kolkata")                 # CONFIRM: local time for night/peak
PEAK_HOURS = (18, 22)                                            # CONFIRM: local hours [start, end)
NIGHT_HOURS = (0, 5)                                             # local hours used for leak detection
EMISSION_FACTOR = float(os.getenv("EMISSION_FACTOR", "0.4"))     # CONFIRM: kg CO2e per kWh
SUBMETER_TOLERANCE = 0.05                                        # allowed |main - sum(sub)| / main
MAX_GAP_SLOTS = 4                                                # interpolate gaps up to 1 hour only
REPAIR_COST = (20000, 50000)                                     # ASSUMED Rs range (not in data)
# Rows per file as stated in the data card (validation check "row_count"). None = no expectation.
EXPECTED_ROWS = {"meter_15min": 158022, "tariff": 5, "production": 553, "weather": 184}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")            # anthropic | openai | gemini | groq | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() == "true"       # true = run without any model

REQUIRED = {
    "meter_15min": ["meter_id", "ts", "kwh"],
    "tariff": ["valid_from", "valid_to", "period", "price_per_kwh"],
    "production": ["date", "product", "units", "shift_pattern", "is_holiday"],
    "weather": ["date", "avg_temp_c", "cooling_degree_days", "heating_degree_days"],
}
