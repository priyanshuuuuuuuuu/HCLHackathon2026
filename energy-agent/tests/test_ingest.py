import pandas as pd

from app.ingest import clean_meters, price_slots


def meter(values):
    ts = pd.date_range("2026-07-01", periods=len(values), freq="15min", tz="UTC")
    return pd.DataFrame({"meter_id": "compressors", "ts": ts, "kwh": values})


def test_duplicates_dropped():
    df = meter([1.0] * 4)
    out, fixes = clean_meters(pd.concat([df, df.iloc[[1]]]))
    assert fixes["duplicates_dropped"] == 1 and len(out) == 4


def test_meter_reset_repaired():
    out, fixes = clean_meters(meter([1.0, 1.0, -500.0, 1.0, 1.0]))
    assert fixes["resets_and_spikes_fixed"] == 1 and out["kwh"][2] == 1.0 and out["flag"][2] == "reset_fixed"


def test_long_gap_not_invented():
    out, fixes = clean_meters(meter([1.0] * 20).drop(index=range(5, 15)))   # 2.5 hour gap
    assert fixes["still_missing"] > 0


def test_tariff_change_mid_period():
    slots = pd.DataFrame({"meter_id": "main", "kwh": [1.0, 1.0], "flag": "ok",
                          "ts": pd.to_datetime(["2026-08-14T10:00Z", "2026-08-16T10:00Z"])})
    tariff = pd.DataFrame({"valid_from": ["2026-01-01T00:00Z", "2026-08-15T00:00Z"],
                           "valid_to": ["2026-08-15T00:00Z", "2027-01-01T00:00Z"],
                           "period": ["flat", "flat"], "price_per_kwh": [7.0, 9.0]})
    assert price_slots(slots, tariff)["price"].tolist() == [7.0, 9.0]


def test_uploads_merge_not_replace(tmp_path, monkeypatch):
    """Two production batches -> both kept; a corrected row replaces the old one; re-upload changes nothing."""
    from app import config, db
    from app.ingest import ingest
    monkeypatch.setattr(config, "STORE", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "SLOTS_PATH", tmp_path / "meter_clean.parquet")
    head = "date,product,units,shift_pattern,is_holiday\n"
    (tmp_path / "a.csv").write_text(head + "2026-07-01,bolts,100,2-shift,False\n2026-07-02,bolts,110,2-shift,False\n")
    (tmp_path / "b.csv").write_text(head + "2026-07-02,bolts,999,2-shift,False\n2026-07-03,bolts,120,2-shift,False\n")
    ingest(str(tmp_path / "a.csv"), "production")
    ingest(str(tmp_path / "b.csv"), "production")
    ingest(str(tmp_path / "b.csv"), "production")
    rows = db.read("SELECT date, units FROM production ORDER BY date")
    assert rows["date"].tolist() == ["2026-07-01", "2026-07-02", "2026-07-03"]
    assert rows["units"].tolist() == [100, 999, 120]


def test_tariff_gap_and_row_count_are_reported(tmp_path, monkeypatch):
    from app import config
    from app.ingest import ingest
    monkeypatch.setattr(config, "STORE", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "SLOTS_PATH", tmp_path / "meter_clean.parquet")
    monkeypatch.setattr(config, "EXPECTED_ROWS", {"tariff": 99})
    (tmp_path / "t.csv").write_text("valid_from,valid_to,period,price_per_kwh\n"
                                    "2026-07-01T00:00:00Z,2026-07-01T12:00:00Z,flat,8\n")   # covers only half a day
    rows = "\n".join(f"main,2026-07-01T{h:02d}:00:00Z,1.0" for h in range(24))
    (tmp_path / "m.csv").write_text("meter_id,ts,kwh\n" + rows + "\n")
    t = ingest(str(tmp_path / "t.csv"), "tariff")
    m = ingest(str(tmp_path / "m.csv"), "meter_15min")
    tariff_checks = {c["name"]: c["passed"] for c in t["validation"]["checks"]}
    meter_checks = {c["name"]: c["passed"] for c in m["validation"]["checks"]}
    assert tariff_checks["row_count_matches_data_card"] is False           # 1 row in file, card says 99
    assert meter_checks["tariff_covers_all_readings"] is False             # afternoon readings have no price
