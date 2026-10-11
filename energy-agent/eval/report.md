# Evaluation report

Method: exact match on answer type, flagged asset, top cause and additivity; the number checker verifies every number in the summary comes from a tool. Data: synthetic pack (docs/data_card.md).

| metric | mock |
|---|---|
| passed | 22 |
| faithfulness | 22/22 |
| p50_ms | 54 |
| p95_ms | 195 |
| llm_calls_per_request | 0 |
| tokens_per_request | 0 |
| blame_price_mix | 3/3 |
| clarification | 1/1 |
| data_trap | 1/1 |
| decomposition | 2/2 |
| invent_number | 3/3 |
| not_in_data | 3/3 |
| numeric | 3/3 |
| waste | 4/4 |
| what_if | 2/2 |

Choice: (fill in after running both configurations - which one and why, using the numbers above).
