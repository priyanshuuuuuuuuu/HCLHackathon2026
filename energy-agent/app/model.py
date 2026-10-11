"""Baseline model (R2): expected daily energy from production and weather.

    daily_kWh = baseload + sum(kWh_per_unit[p] * units[p]) + kWh_per_CDD * CDD (+ kWh_per_HDD * HDD)

Plain least squares. Why linear regression: each coefficient has a physical meaning, and the
bill bridge needs exactly those meanings (volume, mix, weather). ~90 daily points -> a complex
model would overfit and could not be split into causes.
"""

import numpy as np
import pandas as pd

from app import db

_cache: dict = {}


class NotEnoughData(Exception):
    pass


def daily(start: str, end: str) -> pd.DataFrame:
    """One row per day: main-meter kWh and cost joined with the day's context."""
    e = db.read("SELECT date, kwh, cost FROM daily_energy WHERE meter_id='main' AND date BETWEEN ? AND ?",
                (start, end))
    c = db.read("SELECT * FROM daily_context WHERE date BETWEEN ? AND ?", (start, end))
    return e.merge(c, on="date")


def features(df: pd.DataFrame) -> list[str]:
    feats = sorted(c for c in df.columns if c.startswith("u_"))
    for w in ["cooling_degree_days", "heating_degree_days"]:
        if df[w].std() > 0:
            feats.append(w)
    return feats


def fit(start: str, end: str) -> dict:
    """Fit on clean days of [start, end]. Cached until the data version changes."""
    key = (start, end, db.data_version())
    if key in _cache:
        return _cache[key]
    df = daily(start, end)
    df = df[df["exclude_from_training"] == 0]          # skip holiday-logged-as-shift and gappy days
    if len(df) < 10:
        raise NotEnoughData(f"only {len(df)} clean days in {start}..{end}")
    feats = features(df)
    X = np.column_stack([np.ones(len(df))] + [df[f] for f in feats])
    y = df["kwh"].to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef

    rng = np.random.default_rng(0)
    test = rng.permutation(len(y))[: len(y) // 5]       # 20% held out to check the fit honestly
    train = np.setdiff1d(np.arange(len(y)), test)
    c80 = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
    mape = float(np.mean(np.abs(y[test] - X[test] @ c80) / y[test]))

    # bootstrap: refit on resampled days -> coefficient spread -> ranges for the bridge
    boot = [np.linalg.lstsq(X[s], y[s], rcond=None)[0] for s in rng.integers(0, len(y), (200, len(y)))]

    model = {"features": ["baseload_per_day"] + feats, "coef": coef, "boot": boot,
             "r2": float(1 - resid.var() / y.var()), "holdout_mape": mape,
             "resid_std": float(resid.std()), "n_days": len(y), "trained_on": f"{start}..{end}"}
    _cache[key] = model
    return model


def predict_total(model: dict, days: int, units: dict, cdd: float, hdd: float, coef=None) -> float:
    """Predicted kWh for a whole period. The model is linear, so summing days = plugging in totals."""
    coef = model["coef"] if coef is None else coef
    x = [days]
    for f in model["features"][1:]:
        x.append(cdd if f == "cooling_degree_days" else hdd if f == "heating_degree_days" else units.get(f, 0))
    return float(np.dot(x, coef))


def summary(model: dict) -> dict:
    return {"r2": round(model["r2"], 3), "holdout_mape": round(model["holdout_mape"], 3),
            "n_days": model["n_days"], "trained_on": model["trained_on"],
            "coefficients": {f: round(float(c), 3) for f, c in zip(model["features"], model["coef"])}}
