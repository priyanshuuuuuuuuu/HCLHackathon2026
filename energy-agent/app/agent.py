"""The agent (LangGraph). One agent with a fixed shape:

    route --(answerable)--> run_tools --> write --> finish
          \\--(not_in_data / clarification / refused)----------> finish

- route:     the LLM picks the intent and period words; CODE turns them into dates and checks the data.
- run_tools: code picks and runs the tools for that intent (fixed map, no LLM choice).
- write:     the LLM words the summary from tool results only; the checker rejects invented numbers.
- finish:    code builds the response JSON and saves the trace.
Why one agent: all maths is deterministic code, so the LLM is only needed for language.
"""

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from app import checker, config, db, llm, tools

INTENTS = {  # intent -> answer_type
    "decomposition": "decomposition", "what_if_tariff": "calculated", "bill": "calculated",
    "assets": "calculated", "idle": "calculated", "day": "calculated", "leaks": "recommendation",
    "savings": "recommendation", "not_in_data": "not_in_data", "clarification": "clarification_needed",
    "help": "clarification_needed",
    "refused": "refused"}

# ---------------------------------------------------------------- periods (always relative to the DATA, not today)


def resolve(text: str, data_end: str) -> tuple[str, str]:
    """'2026-Q3' | '2026-08' | '2026-08-15' | 'latest_quarter' -> (start, end)."""
    t = text.strip().upper()
    if t == "LATEST_QUARTER":
        end = pd.Timestamp(data_end)
        q = pd.Period(end, "Q") if end.is_quarter_end else pd.Period(end, "Q") - 1   # last COMPLETE quarter
    elif re.fullmatch(r"\d{4}-Q[1-4]", t):
        q = pd.Period(t, "Q")
    elif re.fullmatch(r"\d{4}-\d{2}", t):
        q = pd.Period(t, "M")
    elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t, t
    else:
        raise ValueError(f"cannot understand the period '{text}'")
    return str(q.start_time.date()), str(q.end_time.date())


def previous(start: str, end: str) -> tuple[str, str]:
    """The period just before: previous quarter / month / same number of days."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for freq in ("Q", "M"):
        if s == pd.Period(s, freq).start_time and e.date() == pd.Period(s, freq).end_time.date():
            p = pd.Period(s, freq) - 1
            return str(p.start_time.date()), str(p.end_time.date())
    return str((s - (e - s) - pd.Timedelta(days=1)).date()), str((s - pd.Timedelta(days=1)).date())

# ---------------------------------------------------------------- routing


ROUTER_PROMPT = """You classify questions for a factory electricity-cost analysis. Never answer them.
The data covers {lo} to {hi}. Reply with ONE JSON object: {{"intent": ..., "period": ..., "compare_to": ...}}
intent is one of:
 decomposition (why the bill/cost changed), bill (how much in a period), assets (biggest energy users),
 idle (energy used when nothing is produced), leaks (leaks / night or weekend load), savings (what to fix,
 savings, payback, finance brief), day (one specific date), what_if_tariff (what if the price had not changed),
 not_in_data (future, forecasts, other sites, anything not in the data), clarification (too vague),
 refused (asks to invent or guess numbers), help (asks what you can do / how to use you).
period = the later/main period, compare_to = the earlier one, as 'YYYY-Qn', 'YYYY-MM' or 'YYYY-MM-DD', else null.
Each question is independent (there is no conversation history)."""

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
KEYWORDS = [  # checked in order; used in MOCK mode and when the LLM fails
    ("help", r"what can you|how can you help|what do you do|^help\b|capabilit"),
    ("refused", r"make up|invent|guess|fabricate|industry average"),
    ("not_in_data", r"next year|forecast|predict|future|competitor|other (plant|site)"),
    ("what_if_tariff", r"what if|old (price|tariff)|had not changed"),
    ("leaks", r"leak|night|weekend"),
    ("idle", r"idle|nothing (is )?produced|no production"),
    ("savings", r"fix|saving|payback|action|recommend|brief|first|what should we do|next step"),
    ("assets", r"most energy|biggest|which (asset|machine|meter)|use the most"),
    ("decomposition", r"why|went up|increase|rise|higher|more expensive|change|blame"),
    ("bill", r"bill|cost|how much|spend|kwh"),
]


def keyword_route(q: str, year: str) -> dict:
    t = q.lower()
    months = [MONTHS.index(m[:3]) + 1 for m in re.findall(r"\b(" + "|".join(MONTHS) + r")[a-z]*\b", t)]
    day = re.search(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r")", t)
    if day:
        return {"intent": "day", "period": f"{year}-{MONTHS.index(day[2]) + 1:02d}-{int(day[1]):02d}"}
    route = {"intent": "clarification",
             "period": f"{year}-{months[0]:02d}" if months else None,
             "compare_to": f"{year}-{months[1]:02d}" if len(months) > 1 else None}
    for intent, pattern in KEYWORDS:
        if re.search(pattern, t):
            route["intent"] = intent
            break
    return route


def llm_route(q: str, lo: str, hi: str) -> tuple[dict, int]:
    text, tokens = llm.call(ROUTER_PROMPT.format(lo=lo, hi=hi), q)
    route = json.loads(text[text.find("{"): text.rfind("}") + 1])
    if route.get("intent") not in INTENTS:
        raise llm.LLMError(f"bad intent {route.get('intent')}")
    return route, tokens

# ---------------------------------------------------------------- writing


WRITER_PROMPT = """You are an energy-cost analyst explaining results to a factory finance manager who is not an engineer.
You get the QUESTION, what the tool CALCULATED, a FIELD GUIDE, and TOOL_RESULTS (JSON).

Write a clear, explanatory answer in this markdown format:
**Answer:** one or two sentences that directly answer the question with the key number.
**Why:** 2-5 bullet points. Each bullet = one cause or finding, its number, and what it means in plain words
(e.g. "the price per kWh went up on 16 Aug, so the same energy cost more").
**What this means / what to do:** 1-3 bullets. Say what is controllable and what is not. If TOOL_RESULTS has
actions, give the first action with its savings range and payback, and say it needs manager approval.
**Note:** one short line with the most important assumption or limitation from TOOL_RESULTS.

STRICT RULES:
- Use ONLY numbers that appear in TOOL_RESULTS. Do not calculate new numbers, add up, convert units, or round
  to a different number. Do not use outside or industry figures. If a number is missing, say it is not available.
- For a bill change, explain tariff, volume, product mix and weather BEFORE waste; never blame a machine
  unless TOOL_RESULTS gives evidence for it.
- Money is Rs (write numbers like Rs 1,035,954, never lakh/crore). Give ranges as low-high.
- Never write placeholders such as "?" or "X". If a number is not in TOOL_RESULTS, leave it out.
- The waste amount in a bill bridge is the extra cost in that period, NOT a yearly saving. Only
  estimate_savings results are savings.
- This is a ONE-TIME answer, not a chat: it must be complete on its own. Never ask the user a question, never
  offer more help ("would you like...", "let me know..."), never suggest asking a follow-up question.
- 120-220 words. No tables, no code, no JSON."""

FIELD_GUIDE = {
    "decompose_change": "Bill bridge between a base and a compare period. total_pct = % change of the bill. Each bridge "
                        "item: cause, pct_points = its share of the change in percentage points of the base bill, range = "
                        "uncertainty (low, high), amount = Rs. Causes: tariff = price per kWh changed; volume = more/less "
                        "units produced; product_mix = more energy-heavy products; weather = hotter/colder days (cooling); "
                        "waste = energy the model cannot explain from price, production and weather (the controllable "
                        "part). adds_up = the parts sum to the total. model.coefficients = kWh per unit / per degree-day.",
    "what_if_tariff": "bill_at_base_price = what the compare period would have cost at the base period's average price; "
                      "difference = how much the price change added.",
    "get_bill": "kwh and cost (Rs) of the main meter in the period; avg_price = Rs per kWh.",
    "top_assets": "Energy per sub-meter: kwh, cost (Rs), share of the main meter. 'unmetered' = main minus all sub-meters.",
    "idle_load": "Energy used when nothing is produced (non-production days and night hours), per meter; top_hours = "
                 "hours of the day with most idle use.",
    "detect_leaks": "night_vs_day = typical night load divided by typical working-hours load; before = the same ratio in "
                    "the previous period; flag = true when it rose clearly (likely leak or equipment left running).",
    "estimate_savings": "Each action: yearly savings in kWh, Rs and tonnes CO2e as (low, high) ranges, payback_months "
                        "(low, high), ease, rationale, confidence, requires_approval.",
    "daily_detail": "actual_kwh vs expected_kwh (what the model expects for that day's production and weather); note "
                    "explains anomalies such as a holiday logged as a working shift.",
}
DROP_FOR_LLM = {"profile_15min", "evidence_refs", "boot"}   # long lists the writer does not need


def for_llm(results: dict) -> dict:
    return {tool: {k: v for k, v in out.items() if k not in DROP_FOR_LLM} for tool, out in results.items()}


def _rs(x) -> str:
    return f"Rs {x:,}"


def template(intent: str, r: dict) -> str:
    """Deterministic explanation: used in MOCK mode and whenever the LLM answer fails the number check."""
    if intent == "decomposition":
        d = r["decompose_change"]
        b = {x["cause"]: x for x in d["bridge"]}
        w = b["waste"]
        lines = [f"**Answer:** the bill went from {_rs(d['base']['bill'])} ({d['base']['start']} to {d['base']['end']}) "
                 f"to {_rs(d['compare']['bill'])} ({d['compare']['start']} to {d['compare']['end']}), a change of "
                 f"{d['total_pct']}% ({_rs(d['total_change'])}). The parts below add up to that total.",
                 "**Why** (percentage points of the old bill):",
                 f"- **Tariff {b['tariff']['pct_points']}**: the price per kWh changed, so the same energy cost more.",
                 f"- **Volume {b['volume']['pct_points']}** (range {b['volume']['range'][0]} to {b['volume']['range'][1]}): "
                 "a different amount was produced.",
                 f"- **Product mix {b['product_mix']['pct_points']}** (range {b['product_mix']['range'][0]} to "
                 f"{b['product_mix']['range'][1]}): the share of energy-heavy products changed.",
                 f"- **Weather {b['weather']['pct_points']}** (range {b['weather']['range'][0]} to {b['weather']['range'][1]}): "
                 "cooling needs changed with temperature.",
                 f"- **Waste {w['pct_points']}** (range {w['range'][0]} to {w['range'][1]}): energy not explained by price, "
                 "production or weather.",
                 "**What this means:** tariff, volume, mix and weather are not controllable; only the waste part "
                 f"({w['pct_points']} points, {_rs(w['amount'])}) can be reduced by fixing equipment or operations.",
                 f"**Note:** {d['limitations'][0]}"]
        return "\n".join(lines)
    if intent == "what_if_tariff":
        w = r["what_if_tariff"]
        return (f"**Answer:** at the earlier average price ({_rs(w['base_avg_price'])}/kWh) this period would have cost "
                f"{_rs(w['bill_at_base_price'])} instead of {_rs(w['actual_bill'])}.\n"
                f"- The price change alone added {_rs(w['difference'])}; the average price moved from "
                f"{w['base_avg_price']} to {w['compare_avg_price']} Rs/kWh.\n"
                f"**Note:** {w['limitations'][0]}.")
    if intent == "bill":
        b = r["get_bill"]
        return (f"**Answer:** from {b['start']} to {b['end']} ({b['days']} days) the factory used {b['kwh']:,} kWh, "
                f"costing {_rs(b['cost'])}.\n- Average price: {_rs(b['avg_price'])} per kWh (peak and off-peak combined).")
    if intent == "assets":
        a = r["top_assets"]
        rows = [f"- **{x['meter']}**: {x['kwh']:,} kWh, {x['share']:.1%} of the main meter ({_rs(x['cost'])})"
                for x in a["assets"][:5]]
        return (f"**Answer:** the biggest energy user from {a['start']} to {a['end']} is **{a['assets'][0]['meter']}**.\n"
                + "\n".join(rows) + "\n**Note:** 'unmetered' is the main meter minus all sub-meters.")
    if intent == "idle":
        i = r["idle_load"]
        rows = [f"- **{x['meter']}**: {x['idle_kwh']:,} kWh, mostly around hours {x['top_hours']}" for x in i["items"][:3]]
        return (f"**Answer:** {i['total_idle_kwh']:,} kWh was used while nothing was being produced.\n" + "\n".join(rows)
                + f"\n**Note:** {i['limitations'][0]}.")
    if intent == "leaks":
        leaks = r["detect_leaks"]
        flagged = [i for i in leaks["items"] if i["flag"]]
        if not flagged:
            return "**Answer:** no meter shows a clear rise in night-time load, so no leak is detected."
        x = flagged[0]
        return (f"**Answer:** likely leak on **{x['meter']}**.\n"
                f"- At night its load is {x['night_vs_day']:.0%} of its working-hours load, up from {x['before']:.0%} "
                f"in the previous period ({leaks['compared_with'][0]} to {leaks['compared_with'][1]}).\n"
                "- A machine that keeps drawing power when the factory is idle usually means a leak or equipment left on.\n"
                f"**What to do:** inspect and repair the {x['meter']}; the fix is a suggestion that needs manager approval.")
    if intent == "savings":
        acts = r["estimate_savings"]["actions"]
        if not acts:
            return "**Answer:** no controllable waste with clear evidence was found, so no action is recommended."
        a = acts[0]
        k, m, c, p = (a["saving_kwh_per_year"], a["saving_money_per_year"], a["saving_co2e_t_per_year"], a["payback_months"])
        return (f"**Answer:** first action: **{a['action']}**.\n"
                f"- **Why:** {a['rationale']}.\n"
                f"- **Savings per year:** {k[0]:,}-{k[1]:,} kWh, {_rs(m[0])}-{m[1]:,}, {c[0]}-{c[1]} t CO2e.\n"
                f"- **Payback:** {p[0]}-{p[1]} months (ease: {a['ease']}, confidence: {a['confidence']}).\n"
                "**What to do:** a suggestion only, it needs manager approval.\n"
                f"**Note:** {r['estimate_savings']['assumptions'][1]}.")
    if intent == "day":
        d = r["daily_detail"]
        return (f"**Answer:** on {d['date']} the factory used {d['actual_kwh']:,} kWh, while the model expected "
                f"{d['expected_kwh']:,} kWh for that day's production and weather.\n"
                f"- Units produced: {d['units_produced']:,}; marked as holiday: {d['is_holiday']}.\n"
                + (f"- **Finding:** {d['note']}." if d["note"] else "- Nothing unusual flagged for this day."))
    return ""

# ---------------------------------------------------------------- graph nodes


class State(TypedDict, total=False):
    run_id: str
    question: str
    period: str | None
    compare_to: str | None
    intent: str
    dates: dict
    stop_reason: str
    results: dict
    summary: str
    checker_passed: bool
    llm_calls: int
    tokens: int
    steps: list
    t0: float
    response: dict


def _step(s: State, node: str, output, tool: str | None = None, inp=None):
    s["steps"].append({"step": len(s["steps"]) + 1, "node": node, "tool": tool, "input": inp, "output": output})


def route(s: State) -> State:
    try:
        lo, hi = tools.data_range()
    except tools.DataNotAvailable as e:
        s["intent"], s["stop_reason"] = "not_in_data", str(e)
        return s
    method = "keyword"
    r = keyword_route(s["question"], hi[:4])
    if not config.MOCK_LLM:
        try:
            r, tokens = llm_route(s["question"], lo, hi)
            s["llm_calls"] += 1
            s["tokens"] += tokens
            method = "llm"
        except (llm.LLMError, ValueError):
            pass                                                    # fall back to keyword routing
    if s.get("period"):
        r["period"] = s["period"]
    if s.get("compare_to"):
        r["compare_to"] = s["compare_to"]
        if r["intent"] in ("bill", "clarification"):
            r["intent"] = "decomposition"
    s["intent"] = r["intent"]

    if s["intent"] in ("not_in_data", "clarification", "refused", "help"):
        s["stop_reason"] = {"not_in_data": "the question needs information that is not in the loaded data",
                            "clarification": "the question does not say which period or asset",
                            "refused": "the request asks for numbers that are not computed from the data",
                            "help": "help"}[s["intent"]]
    else:
        try:   # code, not the LLM, decides the dates and checks they exist
            period = resolve(r.get("period") or "latest_quarter", hi)
            base = resolve(r["compare_to"], hi) if r.get("compare_to") else previous(*period)
            if base[0] > period[0]:
                period, base = base, period                         # always later vs earlier
            tools.check_range(*period)
            if s["intent"] in ("decomposition", "what_if_tariff"):
                tools.check_range(*base)
            s["dates"] = {"period": period, "base": base}
        except ValueError as e:
            s["intent"], s["stop_reason"] = "clarification", str(e)
        except tools.DataNotAvailable as e:
            s["intent"], s["stop_reason"] = "not_in_data", str(e)
    _step(s, "route", {"method": method, "intent": s["intent"], "dates": s.get("dates"),
                       "stop_reason": s.get("stop_reason")}, inp=s["question"])
    return s


PLAN = {  # intent -> (tool, which dates it needs)
    "decomposition": ("decompose_change", "two"), "what_if_tariff": ("what_if_tariff", "two"),
    "bill": ("get_bill", "one"), "assets": ("top_assets", "one"), "idle": ("idle_load", "one"),
    "leaks": ("detect_leaks", "one"), "savings": ("estimate_savings", "one"), "day": ("daily_detail", "day")}


def run_tools(s: State) -> State:
    name, kind = PLAN[s["intent"]]
    (start, end), (bstart, bend) = s["dates"]["period"], s["dates"]["base"]
    args = {"two": {"base_start": bstart, "base_end": bend, "cmp_start": start, "cmp_end": end},
            "one": {"start": start, "end": end}, "day": {"date": start}}[kind]
    try:
        out = json.loads(json.dumps(tools.TOOLS[name](**args), default=float))   # numpy -> plain JSON
        s["results"] = {name: out}
    except (tools.DataNotAvailable, ValueError) as e:
        s["results"], s["intent"], s["stop_reason"] = {}, "not_in_data", str(e)
        out = {"error": str(e)}
    _step(s, "run_tools", out, tool=name, inp=args)
    return s


def write(s: State) -> State:
    if s.get("stop_reason"):
        return s
    text, passed = template(s["intent"], s["results"]), True
    if not config.MOCK_LLM:
        guide = {t: FIELD_GUIDE.get(t, "") for t in s["results"]}
        prompt = (f"QUESTION: {s['question']}\nCALCULATED: {s['intent']} for {s['dates']}\n"
                  f"FIELD GUIDE: {json.dumps(guide)}\nTOOL_RESULTS: {json.dumps(for_llm(s['results']))[:12000]}")
        for _ in range(2):                                   # one retry if the checker finds invented numbers
            try:
                draft, tokens = llm.call(WRITER_PROMPT, prompt)
            except llm.LLMError:
                break
            s["llm_calls"] += 1
            s["tokens"] += tokens
            bad = checker.invented_numbers(draft, s["results"])
            passed = not bad
            _step(s, "check_numbers", {"invented": bad, "used": "llm" if passed else "retry/template"})
            if passed:
                text = draft.strip()
                break
            prompt += (f"\n\nYour last answer used numbers that are not in TOOL_RESULTS: {bad}. "
                       "Rewrite it using only TOOL_RESULTS numbers.")
    s["summary"], s["checker_passed"] = text, passed
    _step(s, "write", text)
    return s


HELP = ("I can answer from the factory's meter, tariff, production and weather data:\n"
        "- **Why did the bill change?** (any two periods, e.g. '2026-Q3 vs 2026-Q2' or 'August vs July')\n"
        "- **What was the bill / energy** in a period\n"
        "- **Which assets use the most energy?**\n"
        "- **How much energy is used when nothing is produced?**\n"
        "- **Is there a leak / night-time load?**\n"
        "- **What should we do first?** (actions with savings, CO2e and payback)\n"
        "- **What happened on <date>?**\n"
        "- **What if the tariff had not changed?**")

STOP_MESSAGES = {
    "not_in_data": "Not in the data: {reason}. I can only answer from the loaded meter, tariff, production and weather data.",
    "clarification_needed": "Please clarify ({reason}).\n\n{help}",
    "refused": "I can't do that: every number I give must come from the factory data and our calculations.",
}


def finish(s: State) -> State:
    answer_type = INTENTS[s["intent"]]
    if s.get("stop_reason"):
        s["summary"] = HELP if s["intent"] == "help" else \
            STOP_MESSAGES[answer_type].format(reason=s["stop_reason"], help=HELP)
    r = s.get("results", {})
    out = next(iter(r.values()), {})
    s["response"] = {
        "run_id": s["run_id"], "answer_type": answer_type, "intent": s["intent"], "summary": s["summary"],
        "period": s.get("dates"),
        "total_pct": out.get("total_pct"), "bridge": out.get("bridge", []), "adds_up": out.get("adds_up"),
        "recommendations": out.get("actions", []),
        "result": {k: v for k, v in out.items() if k not in ("bridge", "actions", "assumptions", "limitations")},
        "assumptions": out.get("assumptions", []) + [f"Plant time zone {config.PLANT_TZ}"],
        "limitations": out.get("limitations", []),
        "data_version": db.data_version()}
    db.execute("""INSERT INTO runs(run_id, created_at, question, answer_type, data_version, llm_calls, tokens,
                  latency_ms, checker_passed, trace, response) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
               (s["run_id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), s["question"], answer_type,
                db.data_version(), s["llm_calls"], s["tokens"], int((time.time() - s["t0"]) * 1000),
                s.get("checker_passed"), db.to_json(s["steps"]), db.to_json(s["response"])))
    return s


graph = StateGraph(State)
for name, fn in [("route", route), ("run_tools", run_tools), ("write", write), ("finish", finish)]:
    graph.add_node(name, fn)
graph.set_entry_point("route")
graph.add_conditional_edges("route", lambda s: "finish" if s.get("stop_reason") else "run_tools")
graph.add_edge("run_tools", "write")
graph.add_edge("write", "finish")
graph.add_edge("finish", END)
AGENT = graph.compile()


def answer(question: str, period: str | None = None, compare_to: str | None = None) -> dict:
    """One question in, one complete answer out. No conversation memory: every call is independent."""
    db.init()
    state = AGENT.invoke({"run_id": "en-" + uuid.uuid4().hex[:6], "question": question, "period": period,
                          "compare_to": compare_to,
                          "llm_calls": 0, "tokens": 0, "steps": [], "t0": time.time()})
    return state["response"]


def get_trace(run_id: str) -> dict | None:
    rows = db.read("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {"run_id": run_id, "question": r["question"], "answer_type": r["answer_type"],
            "data_version": int(r["data_version"]), "llm": llm.status(), "steps": json.loads(r["trace"]),
            "llm_calls": int(r["llm_calls"]), "tokens": int(r["tokens"]), "latency_ms": int(r["latency_ms"]),
            "checker_passed": None if pd.isna(r["checker_passed"]) else bool(r["checker_passed"])}
