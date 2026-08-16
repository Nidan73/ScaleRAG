# M5 Split Policy

Chronological, expanding-window rolling splits (Phase 2, tasks 4-5). Implemented
in `graphroute_ts.splits`; every split is validated by
`graphroute_ts.leakage` invariants and by `tests/leakage/test_splits.py`.

## Principles

1. **Chronological only** (rule 1). Time is the M5 day index `d_1`..`d_1941`.
   A split trains on days `1..train_end` and forecasts the next `H=28` days
   `[train_end+1, train_end+28]`. No random/shuffled splits, ever.
2. **Public labels only** (rule 2). Every forecast day `<= d_1941`. A split whose
   horizon would exceed `d_1941` is rejected.
3. **No train/horizon overlap** (rule 5). `h_start = train_end + 1`, so the
   training window and the horizon window are disjoint by construction.
4. **Retrieval horizon guard** (rule 3, for later phases). Any retrieved context
   ending at `t_r` must satisfy `t_r + H < target_forecast_origin`. Enforced by
   `leakage.assert_retrieval_horizon` — not used yet (no retrieval in Phase 2).

## Canonical splits (horizon = 28)

| Name | train ≤ | Forecast window |
|------|---------|-----------------|
| `val_m2` | d_1829 | d_1830 – d_1857 |
| `val_m1` | d_1857 | d_1858 – d_1885 |
| `val`    | d_1885 | d_1886 – d_1913 |
| `test`   | d_1913 | d_1914 – d_1941 |

- **Final test horizon:** `d_1914 – d_1941` (last 28 public days).
- **Primary validation horizon:** `d_1886 – d_1913`.
- **Earlier rolling validation origins:** `val_m1`, `val_m2` (≥ 2 required; the
  code refuses `n_earlier_val < 2`). Each is one horizon (28 days) earlier.

Origins are strictly increasing in time; training windows expand. `test` is kept
held out — the smoke baselines are reported on `val` so `test` is not touched
during development.

## Fitting discipline (task 6, rule 5)

Everything learned — feature scalers/encoders, price means, per-series demand
means, the RMSSE/MASE scale denominators, and the WRMSSE dollar weights — is
computed from **training days only** (`day_idx <= train_end`) and then applied to
the horizon. This is enforced in `features.build_features` (train-slice `group_by`)
and guarded by `tests/leakage/test_feature_leakage.py`, which corrupts the horizon
target and asserts no feature or fitted statistic changes.

## Verifying a split

```bash
uv run python scripts/leakage_audit.py --spec configs/split_check_val.json
```
Passes for a valid split; the `--demo` spec shows it failing loudly on an invalid one.
