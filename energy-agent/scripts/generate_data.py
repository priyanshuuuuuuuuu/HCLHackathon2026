"""Synthetic data pack generator (disclosed in docs/data_card.md).

    python scripts/generate_data.py data            # main pack (4 CSVs, Annex B schema)
    -> also writes data/late_drop/ (corrected tariff + recalibrated line_2) for the 13:45 re-run demo

Fixed seed -> reproducible. Two quarters: Q2 2026 (base) and Q3 2026 (compare). The bill rises ~15%.

What is planted, and why (section 4.3 / 4.4 of the guide):
  causes of the rise : tariff up on 16 Aug · ~5% more output · more energy-heavy forgings (mix)
                       · hot spell 19-31 Aug (HVAC) · compressor air leak from 1 Aug (the real waste)
  data traps         : holiday 15 Aug logged as a '1-shift' day with 0 units · line_2 under-reads ~20% from
                       25 Jul (calibration fault) · main meter ~4% above the sum of sub-meters
                       · tariff periods overlap by one day + a duplicated tariff row
  dirty rows         : missing intervals, meter resets and spikes, duplicate rows (some conflicting),
                       mixed time zones, bad numbers ('ERR', '#VALUE!', '1,234.5'), a row with an extra field,
                       inconsistent names ('HVAC', ' Gears'), invalid dates, unclear holiday flags, a weather outlier
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
(OUT / "late_drop").mkdir(parents=True, exist_ok=True)
TZ = "Asia/Kolkata"
days = pd.date_range("2026-04-01", "2026-09-30", freq="D")
q3 = days >= "2026-07-01"

# ---------------------------------------------------------------- production (daily, per product)
holidays = {"2026-04-14", "2026-05-01"}               # marked correctly
fake_shift = "2026-08-15"                              # TRAP: holiday logged as a normal Saturday shift
rows = []
for d, is_q3 in zip(days, q3):
    ds = str(d.date())
    shift = "off" if d.dayofweek == 6 or ds in holidays else "1-shift" if d.dayofweek == 5 else "2-shift"
    # slightly less per shift in Q3, but Q3 has more working days -> volume still rises
    scale = {"2-shift": 1.0, "1-shift": 0.5, "off": 0.0}[shift] * (0.97 if is_q3 else 1.0)
    if ds == fake_shift:
        scale = 0.0
    forgings_share = 0.036 if is_q3 else 0.026          # mix shift: more energy-heavy forgings in Q3
    for product, share in [("bolts", 0.77), ("gears", 0.20), ("forgings", forgings_share)]:
        units = int(7700 * share * scale * rng.uniform(0.7, 1.3)) if scale else 0   # products vary independently
        rows.append({"date": ds, "product": product, "units": units, "shift_pattern": shift,
                     "is_holiday": ds in holidays})
production = pd.DataFrame(rows)

# ---------------------------------------------------------------- weather (daily)
cdd = 11.5 + rng.normal(0, 1.2, len(days))
cdd[(days >= "2026-08-19") & (days <= "2026-08-31")] += 4                                  # hot spell
weather = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "avg_temp_c": (18 + cdd).round(1),
                        "cooling_degree_days": cdd.round(2), "heating_degree_days": 0.0, "station_id": "SYN-WX-01"})

# ---------------------------------------------------------------- tariff (local time, INR)
tariff = pd.DataFrame([
    ("2026-04-01T00:00:00+05:30", "2026-08-17T00:00:00+05:30", "off_peak", 7.40),
    ("2026-04-01T00:00:00+05:30", "2026-08-17T00:00:00+05:30", "peak", 9.60),
    ("2026-08-16T00:00:00+05:30", "2026-10-01T00:00:00+05:30", "off_peak", 8.35),           # overlaps 1 day
    ("2026-08-16T00:00:00+05:30", "2026-10-01T00:00:00+05:30", "peak", 10.90),
    ("2026-08-16T00:00:00+05:30", "2026-10-01T00:00:00+05:30", "peak", 10.90),              # duplicate row
], columns=["valid_from", "valid_to", "period", "price_per_kwh"]).assign(currency="INR")

# ---------------------------------------------------------------- meters (15-min, built in local time)
slots = pd.date_range("2026-04-01", "2026-10-01", freq="15min", tz=TZ, inclusive="left")
day = slots.strftime("%Y-%m-%d")
hour = slots.hour
per_day = production.pivot_table(index="date", columns="product", values="units", aggfunc="sum")
shift_of = production.groupby("date")["shift_pattern"].first()
bolts, gears, forg = (per_day[p].reindex(day).to_numpy() for p in ["bolts", "gears", "forgings"])
c = pd.Series(cdd, index=weather["date"]).reindex(day).to_numpy()
working_hours = np.where(shift_of.reindex(day).to_numpy() == "2-shift", (hour >= 6) & (hour < 22),
                         np.where(shift_of.reindex(day).to_numpy() == "1-shift", (hour >= 6) & (hour < 14), False))
working_hours = working_hours | ((day == fake_shift) & (hour >= 6) & (hour < 14))   # fake shift: lights and machines on
active = working_hours.astype(float)
hours_worked = pd.Series(active, index=day).groupby(level=0).transform("sum").to_numpy() / 4   # hours per day
per_slot = np.where(hours_worked > 0, 1 / np.maximum(hours_worked * 4, 1), 0)
idle_running = (day == fake_shift) & working_hours                                    # machines on, nothing made

meters = {
    "furnace":     active * (forg * 10.0 + gears * 0.5) * per_slot + 8.0 + idle_running * 25,
    "hvac":        10.0 + c * 0.75 + active * 4.0,
    "line_1":      active * bolts * 0.14 * per_slot + 1.0 + idle_running * 6,
    "line_2":      active * bolts * 0.14 * per_slot + 1.0 + idle_running * 6,
    "line_3":      active * (bolts * 0.18 + gears * 0.7) * per_slot + 1.0 + idle_running * 8,
    "line_4":      active * gears * 0.5 * per_slot + 1.0 + idle_running * 6,
    "lighting":    np.where(active > 0, 6.0, 1.4),
    "compressors": active * (bolts + gears) * 0.07 * per_slot + 0.3,
}
leak_start = pd.Timestamp("2026-08-01", tz=TZ)
growth = np.clip((slots - leak_start).days / 60, 0, 1)                               # leak grows over 2 months
meters["compressors"] += np.where(slots >= leak_start, (1 - active) * (3 + 3 * growth), 0)
noise = pd.Series(rng.normal(1, 0.05, len(days)), index=days.strftime("%Y-%m-%d")).reindex(day).to_numpy()
for m in meters:
    meters[m] = np.clip(meters[m] * noise * rng.normal(1, 0.04, len(slots)), 0, None)
meters["main"] = sum(meters.values()) * 1.04                                          # ~4% unmetered load
true_line_2 = meters["line_2"].copy()
meters["line_2"] = np.where(slots >= pd.Timestamp("2026-07-25", tz=TZ), meters["line_2"] * 0.8, meters["line_2"])

ts = slots.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
order = ["main", "compressors", "hvac", "furnace", "lighting", "line_1", "line_2", "line_3", "line_4"]
meter = pd.DataFrame({"meter_id": np.repeat([order], len(slots), axis=0).ravel(),
                      "ts": np.repeat(ts, len(order)),
                      "kwh": np.column_stack([meters[m] for m in order]).ravel().round(3).astype(object)})

# ---------------------------------------------------------------- dirty rows (what the loader must survive)
idx = rng.choice(meter.index, 220, replace=False)
meter = meter.drop(idx[:180])                                                        # missing intervals
meter.loc[idx[180:210], "kwh"] = ""                                                  # empty readings
meter.loc[idx[210], "kwh"], meter.loc[idx[211], "kwh"] = "ERR", "#VALUE!"
meter.loc[meter.index[(meter.meter_id == "lighting")][-1000], "kwh"] = "1,234.5"     # thousands separator spike
for m, when, val in [("compressors", "2026-05-05T21:30:00Z", -18734.2), ("line_1", "2026-07-28T03:45:00Z", -2210.6),
                     ("lighting", "2026-04-22T20:30:00Z", -0.35), ("main", "2026-09-03T08:30:00Z", 9876.5)]:
    meter.loc[(meter.meter_id == m) & (meter.ts == when), "kwh"] = val              # resets and spikes
dupes = meter.sample(195, random_state=1).copy()
dupes.iloc[:20, 2] = [round(float(v) * 1.1, 3) if str(v).replace(".", "").isdigit() else v for v in dupes.iloc[:20, 2]]
meter = pd.concat([meter, dupes])                                                    # duplicates, 20 conflicting
mixed = meter.ts.str.startswith("2026-05-04T18:30") | meter.ts.str.startswith("2026-05-04T19")
meter.loc[mixed, "ts"] = (pd.to_datetime(meter.loc[mixed, "ts"]).dt.tz_convert(TZ)
                          .dt.strftime("%Y-%m-%dT%H:%M:%S+05:30"))                    # mixed time zones
bad_names = ["HVAC", " HVAC", "Compressors", " Compressors", "LINE_3 ", "LINE_3 "]
for i, name in zip(rng.choice(meter.index, 6, replace=False), bad_names):
    meter.loc[i, "meter_id"] = name                                                 # inconsistent names
meter = pd.concat([meter, pd.DataFrame([{"meter_id": "main", "ts": "2026-13-45T25:00:00Z", "kwh": 123.4},
                                        {"meter_id": "", "ts": "2026-06-01T00:00:00Z", "kwh": 12.0}])])
production.loc[rng.choice(production.index, 4), "product"] = [" Gears", "BOLTS", "Bolts", "forgings "]
production["is_holiday"] = production["is_holiday"].astype(str)
production.loc[rng.choice(production.index, 8), "is_holiday"] = ["0", "no", "FALSE", "false", "N", "maybe", "0", "no"]
production = pd.concat([production, production.sample(3, random_state=3),           # duplicates
                        pd.DataFrame([{"date": "2026-02-30", "product": "bolts", "units": 5000,
                                       "shift_pattern": "2-shift", "is_holiday": "False"}])])
production["units"] = production["units"].astype(object)
production.loc[production.sample(2, random_state=4).index, "units"] = "abc"
weather["avg_temp_c"] = weather["avg_temp_c"].astype(object)
weather.loc[rng.choice(weather.index, 2), "avg_temp_c"] = np.nan
weather.loc[weather.date == "2026-07-21", ["avg_temp_c", "cooling_degree_days"]] = [80.6, 62.6]   # sensor outlier
weather = pd.concat([weather, weather.sample(1, random_state=5).assign(avg_temp_c="hot"),
                     pd.DataFrame([{"date": "2026-09-31", "avg_temp_c": 30.1, "cooling_degree_days": 12.1,
                                    "heating_degree_days": 0.0, "station_id": "SYN-WX-01"}])])

meter.to_csv(OUT / "meter_15min.csv", index=False)
with open(OUT / "meter_15min.csv", "a") as f:
    f.write("compressors,2026-04-18T00:30:00Z,30.1,extra\n")                         # wrong number of fields
production.to_csv(OUT / "production.csv", index=False)
weather.to_csv(OUT / "weather.csv", index=False)
tariff.to_csv(OUT / "tariff.csv", index=False)

# ---------------------------------------------------------------- late data drop (13:45 demo)
corrected = tariff.drop_duplicates().copy()                                          # no overlap, no duplicate
corrected["valid_to"] = corrected["valid_to"].replace("2026-08-17T00:00:00+05:30", "2026-08-16T00:00:00+05:30")
corrected.loc[corrected["price_per_kwh"] == 10.90, "price_per_kwh"] = 10.50          # corrected new peak price
corrected.to_csv(OUT / "late_drop" / "tariff.csv", index=False)
fix = slots >= pd.Timestamp("2026-07-25", tz=TZ)
pd.DataFrame({"meter_id": "line_2", "ts": ts[fix], "kwh": (true_line_2[fix]).round(3)}) \
    .to_csv(OUT / "late_drop" / "meter_15min.csv", index=False)
print(f"written to {OUT}: {len(meter)} meter rows, {len(production)} production rows, late drop in {OUT / 'late_drop'}")
