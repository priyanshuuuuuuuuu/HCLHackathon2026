"""The bill bridge must always add up, and must not call a price or mix change 'waste'."""

import numpy as np

from app.tools import _effects

MODEL = {"features": ["baseload_per_day", "u_bolts", "u_gears", "cooling_degree_days"]}


def period(days, bolts, gears, cdd, coef, extra_kwh=0.0, price=8.0):
    kwh = np.dot([days, bolts, gears, cdd], coef) + extra_kwh
    return {"days": days, "units": {"u_bolts": bolts, "u_gears": gears}, "cdd": cdd, "hdd": 0.0,
            "kwh": kwh, "bill": kwh * price}


def bridge(B, C, coef):
    m = {**MODEL, "coef": np.array(coef)}
    price_b = B["bill"] / B["kwh"]
    tariff = C["bill"] - C["kwh"] * price_b
    return tariff, {k: v * price_b for k, v in _effects(m, B, C).items()}


def test_adds_up_for_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(100):
        coef = rng.uniform([500, 0.1, 1, 5], [2000, 0.5, 5, 50])
        B = period(91, *rng.uniform([5e4, 1e4, 100], [1e5, 4e4, 900]), coef, price=rng.uniform(6, 9))
        C = period(92, *rng.uniform([5e4, 1e4, 100], [1e5, 4e4, 900]), coef, rng.uniform(-5e3, 5e3), rng.uniform(6, 11))
        tariff, eff = bridge(B, C, coef)
        assert abs(tariff + sum(eff.values()) - (C["bill"] - B["bill"])) < 1e-6


def test_price_rise_is_tariff_not_waste():
    coef = [1000, 0.1, 2.0, 15]
    tariff, eff = bridge(period(91, 9e4, 2e4, 400, coef, price=8), period(91, 9e4, 2e4, 400, coef, price=9), coef)
    assert tariff > 0 and all(abs(v) < 1e-6 for v in eff.values())


def test_mix_shift_is_not_waste():
    coef = [1000, 0.1, 2.0, 15]
    _, eff = bridge(period(91, 9e4, 2e4, 400, coef), period(91, 8e4, 3e4, 400, coef), coef)
    assert eff["product_mix"] > 0 and abs(eff["waste"]) < 1e-6 and abs(eff["volume"]) < 1e-6


def test_unexplained_energy_is_waste():
    coef = [1000, 0.1, 2.0, 15]
    _, eff = bridge(period(91, 9e4, 2e4, 400, coef), period(91, 9e4, 2e4, 400, coef, extra_kwh=5000), coef)
    assert abs(eff["waste"] - 5000 * 8) < 1e-6
