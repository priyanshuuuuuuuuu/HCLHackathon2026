"""Storage. SQLite for tables, one Parquet file for the 15-min meter series.

Tables:
  tariff, production, weather   - copies of the uploaded reference files
  daily_energy                  - one row per day per meter (kwh, cost, idle_kwh, night_kwh)
  daily_context                 - one row per day (units per product, degree-days, flags)
  ingest_log                    - every upload: rows loaded/rejected + validation report
  runs                          - every /analyze call with its full trace (R12)
The data version = number of successful uploads, so every answer says which data it used.
"""

import json
import sqlite3

import pandas as pd

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_log (
    version INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, table_name TEXT, file_name TEXT,
    rows_loaded INTEGER, rows_rejected INTEGER, status TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT, question TEXT, answer_type TEXT, data_version INTEGER,
    llm_calls INTEGER, tokens INTEGER, latency_ms INTEGER, checker_passed INTEGER,
    trace TEXT, response TEXT);
"""


def connect():
    config.STORE.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def init():
    with connect() as conn:
        conn.executescript(SCHEMA)


def read(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def write_table(name: str, df: pd.DataFrame) -> None:
    with connect() as conn:
        df.to_sql(name, conn, if_exists="replace", index=False)


def table_exists(name: str) -> bool:
    return not read("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).empty


def execute(sql: str, params: tuple = ()) -> None:
    with connect() as conn:
        conn.execute(sql, params)


def data_version() -> int:
    init()
    return int(read("SELECT COUNT(*) AS n FROM ingest_log WHERE status != 'rejected'")["n"][0])


def to_json(obj) -> str:
    return json.dumps(obj, default=str)
