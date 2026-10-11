"""The analysis tools. Plain Python, deterministic: every number in an answer comes from here.

Each tool takes local dates 'YYYY-MM-DD' (inclusive) and returns a dict with its numbers plus
evidence_refs, assumptions and limitations.
"""

import numpy as np
import pandas as pd

from app import config, db, model


class DataNotAvailable(Exception):
    pass


def data_range() -> tuple[str, str]:
    if not db.table_exists("daily_energy"):
        raise DataNotAvailable("no data loaded yet")
    r = db.read("SELECT MIN(date) AS lo, MAX(date) AS hi FROM daily_energy WHERE meter_id='main'")
    return r["lo"][0], r["hi"][0]


def check_range(start: str, end: str) -> None:
    lo, hi = data_range()
    if start < lo or end > hi:
        raise DataNotAvailable(f"the data covers {lo} to {hi}, not {start} to {end}")


def _period(start: str, end: str) -> dict:
    """Totals for one period: kWh, bill, days, units per product, degree-days."""
    check_range(start, end)
    d = model.daily(start, end)
    return {"kwh": d["kwh"].sum(), "bill": d["cost"].sum(), "days": len(d),
            "units": {c: d[c].sum() for c in d.columns if c.startswith("u_")},
            "cdd": d["cooling_degree_days"].sum(), "hdd": d["heating_degree_days"].sum()}


def _slots(start: str, end: str) -> pd.DataFrame:
    check_range(start, end)
    s = pd.read_parquet(config.SLOTS_PATH)
    return s[(s["date"] >= start) & (s["date"] <= end) & (s["meter_id"] != "main")]


# ---------------------------------------------------------------- bill and bill bridge (R3, R4)

def get_bill(start: str, end: str) -> dict:
    p = _period(start, end)
    return {"start": start, "end": end, "kwh": round(p["kwh"]), "cost": round(p["bill"]),
            "avg_price": round(p["bill"] / p["kwh"], 3), "days": p["days"],
            "evidence_refs": [f"daily_energy:main:{start}..{end}"], "assumptions": [], "limitations": []}


def _effects(m: dict, B: dict, C: dict, coef=None) -> dict:
    """kWh effects of volume, mix, weather and waste between base B and compare C."""
    scale = sum(C["units"].values()) / max(sum(B["units"].values()), 1)
    units_vol = {k: v * scale for k, v in B["units"].items()}          # base mix at compare volume
    pred_b = model.predict_total(m, B["days"], B["units"], B["cdd"], B["hdd"], coef)
    pred_vol = model.predict_total(m, C["days"], units_vol, B["cdd"], B["hdd"], coef)
    pred_mix = model.predict_total(m, C["days"], C["units"], B["cdd"], B["hdd"], coef)
    pred_c = model.predict_total(m, C["days"], C["units"], C["cdd"], C["hdd"], coef)
    return {"volume": pred_vol - pred_b, "product_mix": pred_mix - pred_vol,
            "weather": pred_c - pred_mix, "waste": (C["kwh"] - pred_c) - (B["kwh"] - pred_b)}


def decompose_change(base_start: str, base_end: str, cmp_start: str, cmp_end: str) -> dict:
    """Split Bill_C - Bill_B into tariff, volume, product_mix, weather, waste (adds up exactly).

    tariff  = Bill_C - kWh_C * base_avg_price         (price effect, no model needed)
    volume / mix / weather = model 'what if' predictions, changing one input at a time
    waste   = change in energy the model cannot explain  (removed last, per Annex A)
    """
    B, C = _period(base_start, base_end), _period(cmp_start, cmp_end)
    m = model.fit(base_start, base_end)
    price_b = B["bill"] / B["kwh"]
    tariff = C["bill"] - C["kwh"] * price_b
    eff = _effects(m, B, C)

    # ranges: re-run with 200 bootstrap models (+ day-to-day noise for waste); keep 5th-95th percentile
    rng = np.random.default_rng(1)
    sims = [_effects(m, B, C, c) for c in m["boot"]]
    for s in sims:
        s["waste"] += rng.normal(0, m["resid_std"] * np.sqrt(C["days"]))

    def pts(kwh_or_money, is_kwh=True):   # -> percentage points of the base bill
        return round(100 * kwh_or_money * (price_b if is_kwh else 1) / B["bill"], 2)

    bridge = [{"cause": "tariff", "pct_points": pts(tariff, False), "range": [pts(tariff, False)] * 2,
               "amount": round(tariff)}]
    for k, v in eff.items():
        lo, hi = np.percentile([s[k] for s in sims], [5, 95])
        bridge.append({"cause": k, "pct_points": pts(v), "range": [pts(lo), pts(hi)], "amount": round(v * price_b)})

    total = C["bill"] - B["bill"]
    limitations = ["Attribution order tariff > volume > mix > weather > waste (Annex A); "
                   "interaction effects go to the later step"]
    if bridge[-1]["range"][0] <= 0 <= bridge[-1]["range"][1]:
        limitations.append("Waste range includes zero: no clear waste at this level")
    return {"base": {"start": base_start, "end": base_end, "bill": round(B["bill"]), "kwh": round(B["kwh"])},
            "compare": {"start": cmp_start, "end": cmp_end, "bill": round(C["bill"]), "kwh": round(C["kwh"])},
            "total_change": round(total), "total_pct": pts(total, False), "bridge": bridge,
            "adds_up": bool(abs(tariff + sum(eff.values()) * price_b - total) < 1),
            "model": model.summary(m),
            "evidence_refs": [f"daily_energy:main:{base_start}..{cmp_end}"],
            "assumptions": ["Energy effects valued at the base-period average price",
                            "Ranges = 5th-95th percentile of 200 bootstrap model refits"],
            "limitations": limitations}


def what_if_tariff(base_start: str, base_end: str, cmp_start: str, cmp_end: str) -> dict:
    """What would the compare period have cost at the base period's average price?"""
    B, C = _period(base_start, base_end), _period(cmp_start, cmp_end)
    at_base = C["kwh"] * B["bill"] / B["kwh"]
    return {"actual_bill": round(C["bill"]), "bill_at_base_price": round(at_base),
            "difference": round(C["bill"] - at_base), "base_avg_price": round(B["bill"] / B["kwh"], 3),
            "compare_avg_price": round(C["bill"] / C["kwh"], 3),
            "evidence_refs": [f"daily_energy:main:{cmp_start}..{cmp_end}"],
            "assumptions": ["Same consumption, priced at the base-period average price"],
            "limitations": ["Ignores any shift between peak and off-peak use"]}


# ---------------------------------------------------------------- assets, idle, leaks (R5, R6, R7)

def top_assets(start: str, end: str) -> dict:
    check_range(start, end)
    d = db.read("SELECT meter_id, SUM(kwh) AS kwh, SUM(cost) AS cost FROM daily_energy "
                "WHERE date BETWEEN ? AND ? GROUP BY meter_id ORDER BY kwh DESC", (start, end))
    main = d.loc[d.meter_id == "main", "kwh"].sum()
    assets = [{"meter": r.meter_id, "kwh": round(r.kwh), "cost": round(r.cost), "share": round(r.kwh / main, 3)}
              for r in d[d.meter_id != "main"].itertuples()]
    return {"start": start, "end": end, "main_kwh": round(main), "assets": assets,
            "evidence_refs": [f"daily_energy:*:{start}..{end}"],
            "assumptions": ["'unmetered' = main meter minus the sum of sub-meters"], "limitations": []}


def idle_load(start: str, end: str) -> dict:
    """Energy used when nothing is produced: non-production days plus night hours."""
    s = _slots(start, end)
    night = s["hour"].between(config.NIGHT_HOURS[0], config.NIGHT_HOURS[1] - 1)
    idle = s[~s["is_production"] | night]
    by_meter = idle.groupby("meter_id")["kwh"].sum().sort_values(ascending=False)
    items = [{"meter": m, "idle_kwh": round(v),
              "top_hours": idle[idle.meter_id == m].groupby("hour")["kwh"].sum().nlargest(3).index.tolist()}
             for m, v in by_meter.items()]
    return {"start": start, "end": end, "total_idle_kwh": round(by_meter.sum()), "items": items,
            "evidence_refs": [f"meter_clean:*:{start}..{end}"],
            "assumptions": [f"Idle = non-production days or {config.NIGHT_HOURS[0]}:00-{config.NIGHT_HOURS[1]}:00 local"],
            "limitations": ["Some idle load (cooling, security lights) is necessary, not waste"]}


def _night_ratio(s: pd.DataFrame) -> pd.Series:
    """Per meter: typical night load / typical production-hours load."""
    night = s["hour"].between(config.NIGHT_HOURS[0], config.NIGHT_HOURS[1] - 1)
    working = s["is_production"] & s["hour"].between(8, 17)
    return (s[night].groupby("meter_id")["kwh"].median() / s[working].groupby("meter_id")["kwh"].median()).dropna()


def _previous(start: str, end: str) -> tuple[str, str] | None:
    """The equally long period just before [start, end], if it is in the data."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    ps, pe = s - (e - s) - pd.Timedelta(days=1), s - pd.Timedelta(days=1)
    lo, _ = data_range()
    return (str(max(ps, pd.Timestamp(lo)).date()), str(pe.date())) if str(pe.date()) >= lo else None


def detect_leaks(start: str, end: str) -> dict:
    """R7: a meter whose night load rose clearly vs the previous period (e.g. compressor air leak)."""
    prev = _previous(start, end)
    if prev is None:
        raise DataNotAvailable("no earlier period to compare night-time load against")
    now, before = _night_ratio(_slots(start, end)), _night_ratio(_slots(*prev))
    items = [{"meter": m, "night_vs_day": round(r, 2), "before": round(before.get(m, np.nan), 2),
              "flag": bool(r > 1.5 * before.get(m, np.inf) and r - before.get(m, 0) > 0.1)}
             for m, r in now.items()]
    items.sort(key=lambda i: (not i["flag"], -i["night_vs_day"]))
    return {"start": start, "end": end, "compared_with": list(prev), "items": items,
            "flagged": [i["meter"] for i in items if i["flag"]],
            "evidence_refs": [f"meter_clean:*:{start}..{end}:night {config.NIGHT_HOURS[0]}-{config.NIGHT_HOURS[1]}h"],
            "assumptions": ["Leak = night/day load ratio more than 1.5x the previous period and +0.1"],
            "limitations": ["Cannot tell a leak from a deliberate new night process"]}


# ---------------------------------------------------------------- savings and ranking (R8, R9)

def estimate_savings(start: str, end: str) -> dict:
    leaks = detect_leaks(start, end)
    prev = leaks["compared_with"]
    now = db.read("SELECT * FROM daily_energy WHERE date BETWEEN ? AND ?", (start, end))
    before = db.read("SELECT * FROM daily_energy WHERE date BETWEEN ? AND ?", tuple(prev))
    main = now[now.meter_id == "main"]
    price = main["cost"].sum() / main["kwh"].sum()
    actions = []
    for m in leaks["flagged"]:
        recent = now[now.meter_id == m].sort_values("date")["night_kwh"].tail(30)   # current state
        extra = (recent - before[before.meter_id == m]["night_kwh"].median()).clip(lower=0)
        lo, hi = np.percentile(extra, [25, 75]) * 365                              # kWh per year
        kwh, money = [round(lo), round(hi)], [round(lo * price), round(hi * price)]
        payback = [round(config.REPAIR_COST[0] / (money[1] / 12), 1), round(config.REPAIR_COST[1] / max(money[0] / 12, 1), 1)]
        ratio = next(i for i in leaks["items"] if i["meter"] == m)
        actions.append({
            "action": f"Inspect and repair {m}: load continues at night (suspected leak)",
            "saving_kwh_per_year": kwh, "saving_money_per_year": money,
            "saving_co2e_t_per_year": [round(k * config.EMISSION_FACTOR / 1000, 1) for k in kwh],
            "payback_months": payback, "ease": "easy",
            "rationale": f"Night load is {ratio['night_vs_day']:.0%} of daytime load, up from {ratio['before']:.0%}",
            "evidence_refs": leaks["evidence_refs"], "confidence": "medium",
            "estimated_impact": {"kwh_per_year": kwh, "money_per_year": money}, "requires_approval": True})
    actions.sort(key=lambda a: a["payback_months"][0])
    return {"start": start, "end": end, "actions": actions, "evidence_refs": leaks["evidence_refs"],
            "assumptions": [f"Emission factor {config.EMISSION_FACTOR} kg CO2e/kWh",
                            f"Repair cost Rs {config.REPAIR_COST[0]}-{config.REPAIR_COST[1]} (assumed, not in data)",
                            "Saving = extra night kWh/day in the last 30 days (25th-75th percentile) x 365"],
            "limitations": ["Night hours only, so savings are conservative if the leak also runs at weekends"]}


# ---------------------------------------------------------------- one day (data trap check)

def daily_detail(date: str) -> dict:
    check_range(date, date)
    lo, hi = data_range()
    d = model.daily(date, date).iloc[0]
    units = {c: d[c] for c in d.index if c.startswith("u_")}
    expected = model.predict_total(model.fit(lo, hi), 1, units, d["cooling_degree_days"], d["heating_degree_days"])
    note = ("A shift was logged but nothing was produced: likely a holiday logged as a normal shift"
            if d["holiday_mismatch"] else None)
    return {"date": date, "actual_kwh": round(d["kwh"]), "expected_kwh": round(expected), "cost": round(d["cost"]),
            "is_holiday": bool(d["is_holiday"]), "units_produced": int(d["units_total"]), "note": note,
            "evidence_refs": [f"daily_energy:main:{date}"],
            "assumptions": ["Expected kWh from the baseline model fitted on all clean days"], "limitations": []}


TOOLS = {f.__name__: f for f in [get_bill, decompose_change, what_if_tariff, top_assets, idle_load,
                                  detect_leaks, estimate_savings, daily_detail]}
