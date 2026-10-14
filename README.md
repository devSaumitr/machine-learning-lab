# Machine Learning — Day-by-Day Learning Repository (CampusX 100 Days)

> Reference: [CampusX 100 Days of Machine Learning](https://github.com/campusx-official/100-days-of-machine-learning/tree/main) (repo) and the [YouTube playlist](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&si=OKqpjE7JagrQ9QyL) (videos through Day 29 watched).
> Rule of this repo: **one day, one folder** — each `Day_XX/` holds that day's notebooks, detailed study notes, and a short README.

## Purpose

Learn ML the way the course teaches it (one day at a time), keep my own experiments, and be able to revise any day without reopening videos. Notebooks teach through code; `topic_notes.md` files teach the theory for independent revision.

## Study days (what actually exists — no empty future folders)

| Day | Topic | Status |
|---|---|---|
| [Day_16](Day_16/) | Working with JSON and SQL | Complete (SQL re-run needs live MySQL) |
| [Day_17](Day_17/) | Fetching Data from an API | Complete (no pagination/error handling yet) |
| [Day_18](Day_18/) | Web Scraping into a DataFrame | Incomplete (method proven, export missing) |
| [Day_19](Day_19/) | Understanding Your Data | Complete |
| [Day_20](Day_20/) | Univariate Analysis (+ plot cheat sheet) | Complete (`distplot` deprecated) |
| [Day_21](Day_21/) | Bivariate and Multivariate Analysis | Complete (3 small documented bugs) |
| [Day_22](Day_22/) | Pandas Profiling | Complete (reports may be truncated; filename conflict noted) |
| [Day_23](Day_23/) | What is Feature Engineering | Partial by design (concepts only; code lives in Days 24–29) |
| [Day_24](Day_24/) | Standardization | Complete (accuracy re-run suggested) |
| [Day_25](Day_25/) | Normalization / MinMax Scaling | Complete (MaxAbs/Robust hands-on missing) |
| [Day_26](Day_26/) | Ordinal and Label Encoding | Complete |
| [Day_27](Day_27/) | One-Hot Encoding | Complete (2 documented bugs to fix on re-run) |
| [Day_28](Day_28/) | ColumnTransformer | Complete (test-leak bug documented) |
| [Day_29](Day_29/) | sklearn Pipelines A–Z | Complete (broken draft + crashed inference kept as lessons) |
| [Day_30](Day_30/) | FunctionTransformer (log1p studied) | Studied with code + outputs (notes pending) |

Days 1–15 have no folders: no local work exists for them (background/intro/CSV material). Nothing beyond Day 30 exists.

## How to study a day

1. Open `Day_XX/README.md` — what was studied, which files, which order, honest status.
2. Work the notebook(s) — intro → mental model → step-by-step code → results → mistakes → **experiment** → takeaways.
3. Revise from `Day_XX/topic_notes.md` — theory, formulas, interpretation, Q&A (except Day_30, which has no notes yet).

Follow the days in numeric order — the course sequence is the prerequisite chain (collection → EDA → engineering → pipelines).

## Progress, honestly

- Hands-on complete: Days 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29 (with the gaps listed above).
- Incomplete/partial: Day 18 (export), Day 23 (code lives in Days 24–29), Day 30 (studied with code, notes pending).
- Missing hands-on: binning/construction/extraction, MaxAbs/Robust practice, API/SQL error handling, Day 13–15 background topics.
- Deliberately kept broken: Day-29 draft (`ValueError`) and manual inference (crash) — read their explanations, never copy the code.

## Practical notes (VS Code / Jupyter)

- Kernels: `.venv` (most notebooks), `.venv312` (ydata-profiling notebook only).
- Datasets load from absolute `E:/MACHINE_LEARNING/...` paths, so day folders are freely movable.
- Day_29 holds frozen `pipe.pkl` + `models/` (originals also in `BASICS/` leftovers: `students.json`, `models/`, `pipe.pkl` — do not delete without review).
- Shared artifacts at root: `DATA_SETS/`, `train.csv`, `eda_utils.py`, `eda_plots/`, `*_report.html`. `LEARNING_PROGRESS.md` tracks video↔notebook↔notes; `WORK_LOG.md` logs operations.
- Do not touch `.venv*/`, `.git/`, or datasets.
