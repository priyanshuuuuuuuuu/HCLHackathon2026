# Factory Energy Cost Decomposition & Savings Agent

HCLTech Future Ready AI Engineer Hackathon · Team: _<name>_

**The question:** the factory's electricity bill went up ~15% vs last quarter. Why?
**Our answer:** code splits the change into **tariff, volume, product mix, weather and waste** (the parts always add up),
finds the controllable waste (e.g. a compressor leaking at night), and ranks fixes with savings and payback as ranges.
An LLM only understands the question and words the answer; **every number comes from code**, and a checker rejects any number the LLM invents.

![architecture](docs/architecture.png)

## How it works

| Step | File | What happens |
|---|---|---|
| 1. Ingest (R1, R4) | `app/ingest.py` | load CSV → check columns → reject unparseable rows → validation report → clean (duplicates, gaps ≤ 1 h interpolated, meter resets, flags) → price every 15-min slot with the tariff valid at that moment → daily tables |
| 2. Baseline (R2) | `app/model.py` | least squares on daily data: `kWh = baseload + Σ kWh_per_unit × units + kWh_per_CDD × CDD` |
| 3. Tools (R3, R5–R9) | `app/tools.py` | 8 deterministic functions (below) |
| 4. Agent (R10, R12) | `app/agent.py` | LangGraph: route → run_tools → write → finish; every run saved as a trace |
| 5. API / UI | `app/api.py`, `ui.py` | the 4 required endpoints; Streamlit with the bill-bridge waterfall (R11) |

**Design choices (and why)**
- **One agent, not several.** All maths is deterministic, so it is code. The LLM is used only where language is needed: understanding the question and writing the summary. More agents would add cost, latency and failure points without adding accuracy.
- **The LLM never chooses dates or tools.** It returns an intent and period words; code converts them to dates, checks they exist in the data, and picks the tool from a fixed map.
- **Clean once at ingest, not per question:** fast, consistent answers, and the validation report comes for free. Batch today; in production the same `ingest()` would run every 15 minutes. Every upload is merged on its key (meters `meter_id+ts`, tariff `valid_from+period`, production `date+product`, weather `date`): new rows added, existing rows replaced, the same file twice changes nothing.
- **Linear regression, not ML:** we must *explain* the change, not forecast. Each coefficient means something (baseload, kWh per unit of each product, kWh per degree-day), and the bridge uses exactly those meanings. ~90 daily points would overfit anything complex.
- **SQLite + one Parquet file:** SQLite (required) for tables, logs and traces; the 15-min meter series in Parquet, so it isn't loaded into SQLite just to satisfy the rule.
- **No vector store:** there are no documents to retrieve; every answer comes from structured data.

### The bill bridge (`decompose_change`)

```
base price   P = Bill_B / kWh_B
tariff       = Bill_C − kWh_C × P                       (price effect — no model)
volume       = model(C volume, B mix, B weather) − model(B)
product mix  = model(C volume, C mix, B weather) − previous line
weather      = model(C volume, C mix, C weather) − previous line
waste        = (kWh_C − model(C)) − (kWh_B − model(B))   (energy the model can't explain)
kWh effects × P → Rs → percentage points of Bill_B
```
The kWh effects telescope to `kWh_C − kWh_B`, so `tariff + P × (kWh_C − kWh_B) = Bill_C − Bill_B`: **the parts always add up** (tested in `tests/test_bridge.py`).
Order follows Annex A: external causes first, waste last, so no machine is blamed before price, volume, mix and weather are removed.
Ranges: 5th–95th percentile over 200 bootstrap refits of the model (plus day-to-day noise for waste).

### Tools

| Tool | Req | Returns |
|---|---|---|
| `get_bill(start, end)` | R3 R4 | kWh, Rs, average price |
| `decompose_change(base, compare)` | R3 | the bridge above, with ranges and `adds_up` |
| `what_if_tariff(base, compare)` | what-if | compare period priced at the base price |
| `top_assets(start, end)` | R5 | kWh and share per sub-meter, plus `unmetered` = main − Σ sub |
| `idle_load(start, end)` | R6 | kWh used on non-production days and at night, per meter, with top hours |
| `detect_leaks(start, end)` | R7 | meters whose night/day load ratio rose clearly vs the previous period |
| `estimate_savings(start, end)` | R8 R9 | actions with kWh / Rs / CO₂e ranges, payback, ranked |
| `daily_detail(date)` | trap | actual vs expected kWh for a day; flags a holiday logged as a normal shift |

### Agent answer types
`decomposition`, `recommendation`, `calculated` (tool numbers), `not_in_data` (future tariff, periods outside the data), `clarification_needed` (vague question or unknown period), `refused` (asks to invent numbers).

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                        # MOCK_LLM=true works with no API key
python scripts/generate_data.py data/generated   # synthetic pack (+ data/generated/late_drop/)
python -m scripts.load data/generated       # loads + prints the validation report
uvicorn app.api:app --port 8000             # API
streamlit run ui.py                         # UI on :8501
pytest && ruff check .
python -m eval.run_eval --label mock        # evaluation -> eval/report.md
```
Docker: `docker compose up --build` (API :8000, UI :8501). To use a real model set `MOCK_LLM=false`, `LLM_PROVIDER`, `LLM_MODEL` and the key in `.env`.

### Sample requests
```bash
curl -X POST localhost:8000/ingest -F "file=@data/generated/late_drop/tariff.csv" -F 'meta={"table":"tariff"}'   # late drop
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
     -d '{"question": "Why did the bill go up?", "period": "2026-Q3", "compare_to": "2026-Q2"}'
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" -d '{"question": "What should we fix first?"}'
curl localhost:8000/trace/<run_id>
curl localhost:8000/health
```
Periods accepted: `2026-Q3`, `2026-08`, `2026-08-15`. Without a period, the latest complete quarter is compared with the one before.

## Data edge cases handled

| Trap in the data | How |
|---|---|
| Sub-meters ≠ main meter | reported in validation; the difference is kept as an `unmetered` meter |
| Missing intervals | gaps ≤ 1 hour interpolated and flagged; longer gaps stay missing and those days are excluded from training |
| Meter resets | negative or > 3× the 99.9th percentile → treated as missing → repaired and flagged |
| Duplicates | dropped on `(meter_id, ts)`, newest wins |
| Tariff change mid-quarter | every 15-min slot priced with the tariff valid at that moment |
| Mix shift raising kWh/unit | captured by per-product coefficients → shows as *product mix*, not waste |
| Hot period | cooling degree-days term → shows as *weather* |
| Holiday logged as a normal shift | shift recorded but 0 units produced (validation check 'shift calendar matches production') → flagged, excluded from training, explained by `daily_detail` |
| Late data drop (corrected tariff, recalibrated meter) | re-upload → rows overwritten, slots re-priced, daily tables rebuilt, model refit (cache keyed on data version). Tariff fix changes the tariff bar; the line_2 fix changes the asset breakdown (energy moves from 'unmetered' back to line_2) but not the bill, because the bill comes from the main meter |
| Dirty rows (bad numbers, extra fields, mixed time zones, name spelling, invalid dates, unclear holiday flags, weather outliers, conflicting duplicates) | parsed as text and normalised; unusable rows rejected with a reason; every fix counted in the validation report |

## Assumptions (confirm with mentors → `app/config.py`)
- Plant time zone `Asia/Kolkata` for night and peak hours; peak hours 18:00–22:00 local.
- Emission factor 0.4 kg CO₂e/kWh (replace with the data-pack value).
- Leak repair cost Rs 20,000–50,000 (not in the data) for payback.
- Energy effects in the bridge are valued at the base period's average price.
- Leak = night/day load ratio more than 1.5× the previous period and at least +0.1.

## Limitations
- The attribution order matters slightly (interaction effects go to the later step); Shapley/LMDI would be order-free.
- Short periods (a day, a week) give wide ranges; the weather effect is estimated from one season.
- Leak savings count night hours only (conservative) and assume the pattern continues all year.
- The leak rule cannot tell a leak from a deliberately added night process.

## Evaluation
`eval/cases.json` has 21 cases: blame price/mix not machines (3), numeric (3), data trap (1), waste (3),
not in data (3), what-if (2), invent-a-number (3), clarification (1), decomposition (2).
`python -m eval.run_eval --label <config>` writes `eval/report.md` comparing every configuration run (e.g. `mock` vs a real LLM):
pass rate per category, faithfulness (number checker), p50/p95 latency, LLM calls and tokens per request.
Sample traces for three answer types are in `docs/traces/`.

## Repository
```
app/config.py   settings and assumptions        app/tools.py    the 8 analysis tools
app/db.py       SQLite + Parquet storage        app/agent.py    LangGraph agent + trace
app/ingest.py   load/validate/clean/price       app/checker.py  number checker
app/model.py    baseline regression             app/llm.py      LLM provider switch
app/api.py      FastAPI endpoints               ui.py           Streamlit UI
scripts/        loader, data generator          eval/           cases, runner, report
tests/          bridge, checker, cleaning, agent
```

## Team contribution
| Member | Built | |
|---|---|---|
| _name_ | | |
| _name_ | | |
| _name_ | | |
| _name_ | | |

## AI usage disclosure
AI coding assistants (Claude) were used to draft the code structure, the synthetic data generator and these docs.
We verified it by: reading every file, unit tests (`pytest`: bridge additivity, checker, cleaning, tariff pricing, agent with a fake LLM),
the 21-case evaluation, and checking tool outputs by hand on the synthetic data whose true causes we know.

## Declaration
We declare this is our original work. Signed: _ · _ · _ · _
