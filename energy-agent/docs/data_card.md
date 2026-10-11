# Data card: synthetic factory energy pack

| Field | Value |
|---|---|
| Name and version | `generated` v2, 11-Oct-2026 |
| How it was made | `python scripts/generate_data.py data/generated` (numpy + pandas, fixed seed 7). No LLM, no prompts. Every planted effect is a named line in the script. |
| What it contains | `meter_15min.csv`: main + 8 sub-meters (compressors, hvac, furnace, lighting, line_1-4), 15-min, ~158k rows · `production.csv`: bolts, gears, forgings per day with shift pattern and holiday flag · `weather.csv`: daily temperature and degree-days · `tariff.csv`: peak/off-peak INR prices. Period 1 Apr – 30 Sep 2026 (Q2 base, Q3 compare). Plus `late_drop/`: corrected tariff and recalibrated line_2 rows. |
| Planted causes (known truth) | tariff rise on 16 Aug · more working days in Q3 (volume) · more energy-heavy forgings in Q3 (mix) · hot spell 19–31 Aug (weather) · compressor air leak from 1 Aug, growing (waste). Result: bill +~16%, tariff the largest cause, waste the smallest controllable part. |
| Traps (section 4.4) | sub-meters ~4% below main · missing intervals and meter resets · tariff change mid-quarter (with a 1-day overlap and a duplicate row) · mix shift that is not waste · hot period · 15 Aug holiday logged as a "1-shift" day with 0 units · line_2 under-reads 20% from 25 Jul (fixed in the late drop) |
| Dirty rows | empty / `ERR` / `#VALUE!` / `1,234.5` readings · a line with an extra field · mixed `Z` and `+05:30` timestamps · an impossible timestamp · meter names `HVAC`, `Compressors`, `LINE_3 ` and one blank · 195+ duplicates, some conflicting · invalid dates (`2026-02-30`, `2026-09-31`) · product names `BOLTS`, ` Gears` · `is_holiday` as `0/no/FALSE/N/maybe` · `abc` units · temperature `hot` and a 62.6 degree-day outlier |
| Row counts (used by the `row_count_matches_data_card` check) | our CSVs in `data/`: meter_15min 158,022 · production 553 · weather 184 · tariff 5 (`config.EXPECTED_ROWS`) |
| Checks run | loader validation report (`python -m scripts.load data/generated`): all traps detected and repaired or rejected with a reason |
| Section 4.3 minimums | ≥ 6 sub-meters ✓ (8) · ≥ 2 quarters ✓ · mid-period tariff change ✓ · hidden waste pattern ✓ |
| Known limits | smooth daily shapes; three products; one season of weather; the leak is a simple added idle load |
| Privacy | fully synthetic: no real company or personal data |
