# LEARNING_PROGRESS — CampusX 100 Days of ML (Days 16–31 packaged)

> Playlist: [CampusX 100 Days of ML](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH).
> Permanent system rules: `AGENT_STYLE/MASTER_ML_LEARNING_SYSTEM.md`.
> CampusX clone (secondary reference, never edit): `vendor/100-days-of-machine-learning/`.
> Day folder contract: learner notebook + `*_simplified.ipynb` + `reference_notebook.ipynb` + `datasets/` + `topics.md`.

## Per-day status (verified 2026-11-02)

| Day | topics.md | reference_notebook | datasets/ | simplified | Learner notebooks |
|-----|-----------|--------------------|-----------|------------|-------------------|
| Day_16 | Yes | Yes | Yes (`train.json`) | 2/2 | JSON deep-dive, JSON-SQL-API |
| Day_17 | Yes | Yes | Yes (`movies.csv`) | 1/1 | API fetch |
| Day_18 | Yes | Yes | — (live scrape) | 1/1 | Web scraping |
| Day_19 | Yes | Yes | Yes (`train.csv`) | 1/1 | Understanding your data |
| Day_20 | Yes | Yes | Yes (`train.csv`) | 2/2 | Univariate + cheat sheet |
| Day_21 | Yes | Yes | Yes (`train.csv`) | 1/1 | Bivariate/multivariate |
| Day_22 | Yes | Yes | Yes (`train.csv`) | 3/3 | ydata, Sweetviz, placeholder |
| Day_23 | Yes | REFERENCE_STATUS.md (no CampusX day23) | — | 1/1 | FE concepts |
| Day_24 | Yes | Yes | Yes (`Social_Network_Ads.csv`) | 3/3 | Standardization ×3 |
| Day_25 | Yes | Yes | Yes (`wine_data.csv`) | 2/2 | MinMax + stub |
| Day_26 | Yes | Yes | Yes (`customer.csv`) | 1/1 | Ordinal/Label |
| Day_27 | Yes | Yes | Yes (`cars.csv`) | 1/1 | One-hot |
| Day_28 | Yes | Yes | Yes (`covid_toy.csv`) | 1/1 | ColumnTransformer |
| Day_29 | Yes | Yes (+ `_2`…`_4`) | Yes (`train.csv`) | 5/5 | Pipelines ×5 |
| Day_30 | Yes | Yes | Yes (`train.csv`) | 2/2 | FunctionTransformer ×2 |
| DAY_31 | Yes | Yes | Yes (`concrete_data.csv`) | 1/1 | POWER-TRANSFORM |

**Totals:** 16 `topics.md` · 18 `reference_notebook*.ipynb` · 16 `datasets/` · 28 `*_simplified.ipynb`.

## How to study a day

1. Open **my notebook** (original) and **`_simplified.ipynb`** — same code/outputs, easier markdown.
2. Open **`reference_notebook.ipynb`** side by side (CampusX original, untouched).
3. Read **`topics.md`** for independent revision (What / Why / How / Syntax / Execution / Mistakes).
4. Use **`datasets/`** to re-run without hunting files.

## Older notes still available

- `topic_notes.md` per day (deeper notes; Day 30 has RS5 “from zero” explanation; Day 31 formal notes in `DAY_31/topic_notes.md`).
- Teaching style: `AGENT_STYLE/MASTER_TEACHING_STYLE.md` (Rule Set 5 = ML Concept-from-Zero).
- Day 31 chapter/transcript material: `ML_Notes_Videos_31_40/`.

## Gaps (labeled, not invented)

- Days 1–15: not packaged (background / no day folders yet).
- Day 18: no local dataset (page is the source).
- Day 23: no official CampusX day23 notebook.
- Stub notebooks (Day 24/25): history only — revise from full notebooks.

## Next priorities

1. Study Day 16→31 using the four-file folder pattern.
2. Re-run cells with `null` / unexecuted outputs when convenient.
3. Day 32+: apply the same packaging automatically when new notebooks arrive.
4. Keep leakage rules: fit on train only; Pipeline inside CV (Days 24–31).
