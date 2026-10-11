"""Evaluation (section 7 of the guide).

    python -m eval.run_eval --label mock          # MOCK_LLM=true  (keyword router + template writer)
    python -m eval.run_eval --label claude-haiku  # MOCK_LLM=false (real LLM router + writer)

Runs every case in cases.json through the real API (in-process), checks the expected result,
and rewrites eval/report.md comparing all configurations run so far. Also saves sample traces.
Method: exact match on answer_type / flagged asset / top cause, plus the number checker for faithfulness.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import app

HERE = Path(__file__).parent
client = TestClient(app)


def passed(case: dict, r: dict, t: dict) -> bool:
    e = case["expect"]
    top = max(r["bridge"], key=lambda b: b["pct_points"])["cause"] if r["bridge"] else None
    checks = [
        r["answer_type"] == e.get("answer_type", r["answer_type"]),
        not e.get("adds_up") or r["adds_up"] is True,
        not e.get("waste_not_top") or (top is not None and top != "waste"),
        e.get("mentions", "").lower() in r["summary"].lower(),
        e.get("must_not_contain", "\0") not in r["summary"],
        not e.get("faithful") or t["checker_passed"] is not False,
    ]
    return all(checks)


def main(label: str):
    cases = json.loads((HERE / "cases.json").read_text())
    rows, saved_types = [], set()
    for c in cases:
        t0 = time.time()
        r = client.post("/analyze", json={k: c.get(k) for k in ("question", "period", "compare_to")}).json()
        latency = int((time.time() - t0) * 1000)
        t = client.get(f"/trace/{r['run_id']}").json()
        rows.append({"id": c["id"], "category": c["category"], "pass": passed(c, r, t), "latency_ms": latency,
                     "llm_calls": t["llm_calls"], "tokens": t["tokens"], "checker_passed": t["checker_passed"]})
        print(f"{c['id']} {'PASS' if rows[-1]['pass'] else 'FAIL'}  {c['category']:<16} {r['answer_type']}")
        if r["answer_type"] not in saved_types and len(saved_types) < 3:      # 3 sample traces for submission
            saved_types.add(r["answer_type"])
            (HERE.parent / "docs" / "traces" / f"{r['answer_type']}.json").write_text(json.dumps(t, indent=2))

    lat = sorted(r["latency_ms"] for r in rows)
    summary = {
        "cases": len(rows), "passed": sum(r["pass"] for r in rows),
        "by_category": {k: f"{sum(r['pass'] for r in rows if r['category'] == k)}/{sum(r['category'] == k for r in rows)}"
                        for k in sorted({r["category"] for r in rows})},
        "faithfulness": f"{sum(r['checker_passed'] is not False for r in rows)}/{len(rows)}",
        "p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(0.95 * (len(lat) - 1))],
        "llm_calls_per_request": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "tokens_per_request": round(statistics.mean(r["tokens"] for r in rows)),
    }
    results_file = HERE / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    results[label] = {"summary": summary, "rows": rows}
    results_file.write_text(json.dumps(results, indent=2))
    write_report(results)
    print(json.dumps(summary, indent=2))


def write_report(results: dict):
    keys = ["passed", "faithfulness", "p50_ms", "p95_ms", "llm_calls_per_request", "tokens_per_request"]
    lines = ["# Evaluation report", "",
             "Method: exact match on answer type, flagged asset, top cause and additivity; the number checker "
             "verifies every number in the summary comes from a tool. Data: synthetic pack (docs/data_card.md).", "",
             "| metric | " + " | ".join(results) + " |", "|---" * (len(results) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(v["summary"][k]) for v in results.values()) + " |")
    for cat in next(iter(results.values()))["summary"]["by_category"]:
        lines.append(f"| {cat} | " + " | ".join(v["summary"]["by_category"].get(cat, "-") for v in results.values()) + " |")
    lines += ["", "Choice: (fill in after running both configurations - which one and why, using the numbers above)."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="mock")
    main(ap.parse_args().label)
