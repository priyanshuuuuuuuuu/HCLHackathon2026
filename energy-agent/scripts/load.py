"""Load a folder of CSVs (judges use this): python -m scripts.load data/synthetic
Reference tables first, then meters. Prints the validation report."""

import sys
from pathlib import Path

from app.ingest import ingest

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
for table in ["tariff", "production", "weather", "meter_15min"]:
    path = folder / f"{table}.csv"
    if not path.exists():
        print(f"{table}: not found, skipped")
        continue
    r = ingest(str(path), table, path.name)
    print(f"{table}: {r['status']}  loaded={r['rows_loaded']}  rejected={r['rows_rejected']}  version={r['data_version']}")
    for c in r["validation"].get("checks", []):
        print(f"   [{'PASS' if c['passed'] else 'WARN'}] {c['name']}: {c['detail']}")
    for key in ("fixed_while_parsing", "fixes"):
        if r["validation"].get(key):
            print(f"   {key}: {r['validation'][key]}")
    if r.get("error") or r["validation"].get("error"):
        print("   ERROR:", r["validation"].get("error"))
