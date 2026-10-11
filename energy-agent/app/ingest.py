"""Ingest pipeline (R1, R4): load -> validate -> clean -> price -> daily tables.

Runs when data arrives (loader script or POST /ingest). Cleaning happens once here, not per question.
Every upload is MERGED with what is stored (see KEYS): new rows are added, rows with an existing key are
replaced. So uploading the same file twice changes nothing, a new batch extends the data, and a corrected
file (the late data drop) overwrites only the rows it contains.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import config, db

TRUE_WORDS, FALSE_WORDS = {"true", "1", "yes", "y"}, {"false", "0", "no", "n", ""}
# A new upload is MERGED with stored data: new keys are added, an existing key is replaced (newest wins).
KEYS = {"meter_15min": ["meter_id", "ts"], "tariff": ["valid_from", "period"],
        "production": ["date", "product"], "weather": ["date"]}


def ingest(path: str, table: str, file_name: str | None = None) -> dict:
    db.init()
    name = file_name or path
    if table not in config.REQUIRED:
        return _log(table, name, 0, 0, "rejected", {"error": f"unknown table '{table}'"})
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip", skipinitialspace=True)   # read as text, parse ourselves
    df.columns = df.columns.str.strip()
    malformed = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1 - len(df)  # lines with wrong field count
    missing = set(config.REQUIRED[table]) - set(df.columns)
    if missing:
        return _log(table, name, 0, 0, "rejected", {"error": f"missing columns {sorted(missing)}"})

    rows_in_file = len(df) + max(malformed, 0)
    df, bad, notes = parse(df, table)
    bad += [{"row": None, "reason": "wrong number of fields"}] * max(malformed, 0)
    report = {"checks": validate(df, table), "fixed_while_parsing": notes, "rejected_examples": bad[:10]}
    expected = config.EXPECTED_ROWS.get(table)
    report["checks"].append({"name": "row_count_matches_data_card", "passed": expected in (None, rows_in_file),
                             "detail": f"{rows_in_file} rows in file, data card says {expected}"})

    if table == "meter_15min":
        slots, report["fixes"] = clean_meters(df)
        if config.SLOTS_PATH.exists():
            old = pd.read_parquet(config.SLOTS_PATH)[["meter_id", "ts", "kwh", "flag"]]
            slots = merge(old, slots, table)
        slots[["meter_id", "ts", "kwh", "flag"]].to_parquet(config.SLOTS_PATH, index=False)
    else:
        if db.table_exists(table):
            old = db.read(f"SELECT * FROM {table}")
            if table == "tariff":                           # stored as text -> back to timestamps to compare keys
                for c in ["valid_from", "valid_to"]:
                    old[c] = pd.to_datetime(old[c], utc=True, format="ISO8601")
            df = merge(old, df, table)
        report["stored_rows"] = len(df)
        db.write_table(table, df)

    unpriced = rebuild()                                    # reprice + daily tables with the latest data
    if unpriced is not None:
        report["checks"].append({"name": "tariff_covers_all_readings", "passed": unpriced["count"] == 0,
                                 "detail": f"{unpriced['count']} meter readings without a tariff price {unpriced['dates']}"})
    return _log(table, name, len(df), len(bad), "ok" if not bad else "partial", report)


def merge(old: pd.DataFrame, new: pd.DataFrame, table: str) -> pd.DataFrame:
    """Add the new rows to the stored ones; where a key already exists the new row replaces the old one."""
    return pd.concat([old, new], ignore_index=True).drop_duplicates(KEYS[table], keep="last") \
        .sort_values(KEYS[table]).reset_index(drop=True)


def _date(s: pd.Series) -> pd.Series:
    """Strict YYYY-MM-DD; impossible dates like 2026-02-30 become NaN (-> rejected)."""
    return pd.to_datetime(s.str.strip(), format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")


def _number(s: pd.Series) -> pd.Series:
    """'1,234.5' -> 1234.5 ; 'ERR', '#VALUE!', 'hot', '' -> NaN."""
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(), errors="coerce")


def parse(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list, dict]:
    """Convert text to proper types and normalise spelling. Unusable rows are rejected with a reason.
    Returns (good rows, rejected rows, counts of things fixed)."""
    df = df[config.REQUIRED[table]].copy()
    notes = {}
    if table == "meter_15min":
        raw_ids = df["meter_id"].fillna("")
        df["meter_id"] = raw_ids.str.strip().str.lower()                     # 'HVAC ', 'Compressors' -> 'hvac'
        notes["meter_names_normalised"] = int((raw_ids != df["meter_id"]).sum())
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="ISO8601")   # 'Z' and '+05:30' -> UTC
        kwh = _number(df["kwh"].fillna(""))
        notes["unreadable_kwh_treated_as_gaps"] = int(kwh.isna().sum())      # repaired in clean_meters
        df["kwh"] = kwh
        df.loc[df["meter_id"] == "", "meter_id"] = np.nan
        must = ["meter_id", "ts"]
    elif table == "tariff":
        df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True, errors="coerce", format="ISO8601")
        df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True, errors="coerce", format="ISO8601")
        df["period"] = df["period"].str.strip().str.lower()
        df["price_per_kwh"] = _number(df["price_per_kwh"])
        df.loc[~df["period"].isin(["peak", "off_peak", "flat"]) | (df["price_per_kwh"] < 0), "price_per_kwh"] = np.nan
        notes["duplicate_rows_dropped"] = int(df.duplicated().sum())
        df = df.drop_duplicates()
        must = config.REQUIRED[table]
    elif table == "production":
        df["date"] = _date(df["date"])
        df["product"] = df["product"].str.strip().str.lower()                # 'BOLTS', ' Gears' -> 'bolts', 'gears'
        df["shift_pattern"] = df["shift_pattern"].fillna("").str.strip().str.lower()
        df["units"] = _number(df["units"].fillna(""))
        df.loc[df["units"] < 0, "units"] = np.nan
        hol = df["is_holiday"].fillna("").str.strip().str.lower()
        notes["unclear_is_holiday_set_false"] = int((~hol.isin(TRUE_WORDS | FALSE_WORDS)).sum())   # e.g. 'maybe'
        df["is_holiday"] = hol.isin(TRUE_WORDS).astype(int)
        must = ["date", "product", "units"]
    else:  # weather
        df["date"] = _date(df["date"])
        for c in ["avg_temp_c", "cooling_degree_days", "heating_degree_days"]:
            df[c] = _number(df[c].fillna(""))
        cdd = df["cooling_degree_days"]
        outlier = cdd > 3 * cdd.median()                                      # e.g. 62.6 when normal is ~12
        notes["degree_day_outliers_removed"] = int(outlier.sum())
        df.loc[outlier, "cooling_degree_days"] = np.nan
        must = ["date"]

    bad_mask = df[must].isna().any(axis=1)
    bad = [{"row": int(i) + 2, "reason": "unusable " + ", ".join(c for c in must if pd.isna(df.at[i, c]))}
           for i in df.index[bad_mask]]
    df = df[~bad_mask]
    if table in ("production", "weather"):                                    # one row per key, newest wins
        key = ["date", "product"] if table == "production" else ["date"]
        notes["duplicates_dropped"] = int(df.duplicated(key).sum())
        df = df.drop_duplicates(key, keep="last").sort_values(key)
    if table == "weather":                                                    # fill missing/outlier degree-days
        for c in ["cooling_degree_days", "heating_degree_days"]:
            notes[f"{c}_filled"] = int(df[c].isna().sum())
            df[c] = df[c].interpolate(limit_direction="both").fillna(0)
    return df.reset_index(drop=True), bad, notes


def validate(df: pd.DataFrame, table: str) -> list[dict]:
    """The validation checks from the guide (4.5). They REPORT problems; cleaning fixes them."""
    checks = []

    def check(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    if table == "meter_15min":
        neg = int((df["kwh"] < 0).sum())
        check("no_negative_energy", neg == 0, f"{neg} negative readings (meter resets, repaired in cleaning)")
        dupes = df.duplicated(["meter_id", "ts"])
        same_key = df[df.duplicated(["meter_id", "ts"], keep=False)]
        conflicts = int((same_key.groupby(["meter_id", "ts"])["kwh"].nunique() > 1).sum())
        check("no_duplicates", not dupes.any(),
              f"{int(dupes.sum())} duplicate (meter_id, ts) rows, {conflicts} with different values (newest kept)")
        steps = df.sort_values("ts").groupby("meter_id")["ts"].diff().dropna()
        gaps = int((steps > pd.Timedelta("15min")).sum())
        check("regular_15min_utc", gaps == 0, f"{gaps} gaps in the 15-min series; all timestamps converted to UTC")
        if "main" in set(df["meter_id"]):
            ok = df[(df["kwh"] >= 0) & (df["kwh"] < df["kwh"].quantile(0.999) * 3)]
            day = ok.assign(d=ok["ts"].dt.tz_convert(config.PLANT_TZ).dt.date)
            piv = day.pivot_table(index="d", columns="meter_id", values="kwh", aggfunc="sum")
            gap = ((piv["main"] - piv.drop(columns="main").sum(axis=1)) / piv["main"]).abs()
            off = int((gap > config.SUBMETER_TOLERANCE).sum())
            check("submeters_match_main", off == 0,
                  f"{off} days outside ±{config.SUBMETER_TOLERANCE:.0%} (median gap {gap.median():.1%}, kept as 'unmetered')")
    elif table == "tariff":
        overlaps = sum(int((g["valid_from"].iloc[1:].values < g["valid_to"].iloc[:-1].values).sum())
                       for _, g in df.sort_values("valid_from").groupby("period"))
        check("tariff_no_overlap", overlaps == 0, f"{overlaps} overlapping periods (the later tariff wins where they overlap)")
    elif table == "production":
        day = df.groupby("date").agg(units=("units", "sum"), shift=("shift_pattern", "max"), holiday=("is_holiday", "max"))
        no_output = day[(day["shift"] != "off") & (day["units"] == 0)].index.tolist()
        check("shift_calendar_matches_production", not no_output,
              f"{len(no_output)} days with a shift logged but nothing produced {no_output[:5]} (likely holidays)")
        hol = day[(day["holiday"] == 1) & (day["units"] > 0)].index.tolist()
        check("holidays_have_no_production", not hol, f"{len(hol)} holidays with production units {hol[:5]}")
    return checks


def clean_meters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Duplicates, gaps, meter resets. Every repaired slot gets a flag so it can be shown."""
    n = len(df)
    df = df.drop_duplicates(["meter_id", "ts"], keep="last")
    wide = df.pivot(index="ts", columns="meter_id", values="kwh").sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="15min", tz="UTC"))
    flag = pd.DataFrame("ok", index=wide.index, columns=wide.columns)

    gaps = wide.isna()
    resets = (wide < 0) | (wide > wide.quantile(0.999) * 3)      # negative or impossible spike
    wide = wide.mask(resets)
    wide = wide.interpolate(limit=config.MAX_GAP_SLOTS, limit_area="inside")   # short gaps only
    flag[gaps] = "interpolated"
    flag[resets] = "reset_fixed"
    flag[wide.isna()] = "missing"                                 # long gaps stay missing

    out = wide.stack(future_stack=True).rename("kwh").reset_index()
    out.columns = ["ts", "meter_id", "kwh"]
    out["flag"] = flag.stack(future_stack=True).values
    fixes = {"duplicates_dropped": n - len(df), "gaps_filled": int((flag == "interpolated").sum().sum()),
             "resets_and_spikes_fixed": int(resets.sum().sum()), "still_missing": int((flag == "missing").sum().sum())}
    return out, fixes


def price_slots(slots: pd.DataFrame, tariff: pd.DataFrame) -> pd.DataFrame:
    """R4: each 15-min slot gets the price valid at that moment (handles the mid-quarter change).
    Rows are applied in start order, so where two tariffs overlap the later one wins."""
    local = slots["ts"].dt.tz_convert(config.PLANT_TZ)
    slots = slots.assign(date=local.dt.strftime("%Y-%m-%d"), hour=local.dt.hour, price=np.nan)
    peak = slots["hour"].between(config.PEAK_HOURS[0], config.PEAK_HOURS[1] - 1)
    period = np.where(peak, "peak", "off_peak")
    for r in tariff.sort_values("valid_from").itertuples():
        in_range = (slots["ts"] >= pd.Timestamp(r.valid_from)) & (slots["ts"] < pd.Timestamp(r.valid_to))
        if r.period != "flat":
            in_range &= period == r.period
        slots.loc[in_range, "price"] = r.price_per_kwh
    slots["cost"] = slots["kwh"] * slots["price"]
    return slots


def rebuild() -> dict | None:
    """Re-price every slot and rebuild the daily tables. Fast at this size, so we always rebuild fully.
    Returns how many readings got no tariff price (the 'tariff covers the whole period' check)."""
    if not (config.SLOTS_PATH.exists() and db.table_exists("tariff")):
        return None
    slots = pd.read_parquet(config.SLOTS_PATH)
    slots["ts"] = pd.to_datetime(slots["ts"], utc=True)
    tariff = db.read("SELECT * FROM tariff")
    tariff["valid_from"] = pd.to_datetime(tariff["valid_from"], utc=True, format="ISO8601")
    tariff["valid_to"] = pd.to_datetime(tariff["valid_to"], utc=True, format="ISO8601")
    slots = price_slots(slots[["meter_id", "ts", "kwh", "flag"]], tariff)
    no_price = slots["price"].isna()
    unpriced = {"count": int(no_price.sum()), "dates": sorted(slots.loc[no_price, "date"].unique())[:5]}

    prod = db.read("SELECT * FROM production") if db.table_exists("production") else \
        pd.DataFrame(columns=["date", "product", "units", "shift_pattern", "is_holiday"])
    weather = db.read("SELECT * FROM weather") if db.table_exists("weather") else pd.DataFrame(columns=["date"])

    per_day = prod.groupby("date").agg(units=("units", "sum"), is_holiday=("is_holiday", "max"),
                                       shift=("shift_pattern", "max"))
    working_days = set(per_day[(per_day["units"] > 0) & (per_day["is_holiday"] == 0)].index)
    slots["is_production"] = slots["date"].isin(working_days)
    night = slots["hour"].between(config.NIGHT_HOURS[0], config.NIGHT_HOURS[1] - 1)
    slots.to_parquet(config.SLOTS_PATH, index=False)

    # daily_energy: one row per day per meter, plus 'unmetered' = main - sum(sub-meters)
    energy = slots.assign(idle=slots["kwh"].where(~slots["is_production"], 0),
                          night=slots["kwh"].where(night, 0),
                          missing=slots["flag"].eq("missing")) \
        .groupby(["date", "meter_id"], as_index=False) \
        .agg(kwh=("kwh", "sum"), cost=("cost", "sum"), idle_kwh=("idle", "sum"),
             night_kwh=("night", "sum"), missing_slots=("missing", "sum"))
    if "main" in set(energy["meter_id"]):
        cols = ["kwh", "cost", "idle_kwh", "night_kwh"]
        main = energy[energy.meter_id == "main"].set_index("date")[cols]
        sub = energy[energy.meter_id != "main"].groupby("date")[cols].sum()
        unmetered = (main - sub).reset_index().assign(meter_id="unmetered", missing_slots=0)
        energy = pd.concat([energy, unmetered], ignore_index=True)
    db.write_table("daily_energy", energy)

    # daily_context: regression inputs + flags
    units = prod.pivot_table(index="date", columns="product", values="units", aggfunc="sum").add_prefix("u_")
    ctx = pd.DataFrame(index=sorted(energy["date"].unique())).join(units).fillna({c: 0 for c in units.columns})
    ctx["units_total"] = ctx[list(units.columns)].sum(axis=1) if len(units.columns) else 0
    ctx["is_holiday"] = per_day["is_holiday"].reindex(ctx.index).fillna(0).astype(int)
    w = weather.set_index("date")
    for c in ["cooling_degree_days", "heating_degree_days"]:
        ctx[c] = w[c].reindex(ctx.index).interpolate(limit_direction="both").fillna(0) if c in w else 0.0
    shift = per_day["shift"].reindex(ctx.index).fillna("off")
    missing = energy[energy.meter_id == "main"].set_index("date")["missing_slots"].reindex(ctx.index).fillna(0)
    # Trap: a holiday logged as a normal shift = shift recorded, but nothing produced (or marked holiday)
    ctx["holiday_mismatch"] = ((shift != "off") & ((ctx["units_total"] == 0) | (ctx["is_holiday"] == 1))).astype(int)
    ctx["exclude_from_training"] = ((ctx["holiday_mismatch"] == 1) | (missing > config.MAX_GAP_SLOTS)).astype(int)
    db.write_table("daily_context", ctx.rename_axis("date").reset_index())
    return unpriced


def _log(table, file_name, loaded, rejected, status, report) -> dict:
    db.execute("""INSERT INTO ingest_log(created_at, table_name, file_name, rows_loaded, rows_rejected, status, report)
                  VALUES (?,?,?,?,?,?,?)""",
               (datetime.now(timezone.utc).isoformat(timespec="seconds"), table, file_name, loaded, rejected,
                status, db.to_json(report)))
    return {"rows_loaded": loaded, "rows_rejected": rejected, "status": status,
            "data_version": db.data_version(), "validation": report}
