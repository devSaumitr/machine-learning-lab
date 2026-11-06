# Day_16

CampusX 100 Days of ML · folder `Day_16/`

## Day number

**16**

## Main topics

Working with JSON, APIs, and SQL; JSON deep dive.

## Learning objectives

1. Read JSON/API responses into Pandas and query data with SQL.
2. Explain how the JSON structure maps to rows and columns.
3. Interpret outputs and avoid common loading mistakes (see `notes.md`).
4. Connect data acquisition to the later cleaning and feature-engineering stages.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook (official Day 16 flow) | 13 |
| `ML_DAY16_JSON-deep-dive.ipynb` | Separate personal deep-dive (kept distinct) | 84 |
| `reference_notebooks/day16.ipynb` | CampusX original — **do not edit** | 11 |

## Dataset

| File | Source | Verified shape |
|------|--------|----------------|
| `datasets/train.json` | CampusX course dataset | 39,774 recipes × 3 columns |
| `datasets/world.sql` | MySQL `world` database dump linked by the official course README; copied from the project root without modification | 239 countries, 984 country-language rows |

The SQL notebook expects a local MySQL server with a database named `world`. Import `datasets/world.sql` first; the JSON portion uses `datasets/train.json`. The dump creates the `world` schema and its tables.

## Dependencies

```text
python >= 3.9
pandas, numpy, jupyter
mysql-connector-python   (SQL notebook; the course installs the older `mysql.connector` package)
sqlalchemy               (optional; pandas' supported SQL integration)
```

```bash
pip install pandas numpy jupyter mysql-connector-python sqlalchemy
```

## How to run

1. Open `Day_16/` as the working directory.
2. Read `notes.md` Part A for the complete learning chapter; Part B is optional advanced reference.
3. Run `my_notebook.ipynb` first, then `ML_DAY16_JSON-deep-dive.ipynb`.
4. Import `datasets/world.sql` into MySQL before running the SQL cells.
5. Use `datasets/train.json`; prefer the relative path over the notebook's absolute path.

## Study order

1. `notes.md` Part A — JSON foundations, `read_json`, `json_normalize`, SQL, notebook results, essential practice
2. `my_notebook.ipynb` — official Day 16 flow plus GitHub API and `world.country`
3. `ML_DAY16_JSON-deep-dive.ipynb` — orientations, JSON Lines, nesting, dates, Unicode, mini project
4. `notes.md` Part B — formal references, API/SQL engineering, source audit, extended practice
5. Revise from the revision summary at the end of Part A

## Daily folder structure

```text
Day_16/
├── my_notebook.ipynb
├── ML_DAY16_JSON-deep-dive.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
└── datasets/
    ├── train.json
    └── world.sql
```

## Status

Existing personal notebooks and the official reference copy are preserved. The SQL data dump was added from the project root; it creates the `world` schema and contains 239 countries, 984 country-language rows, and 17 countries with missing life expectancy.

- 2026-11-04: `notes.md` rewritten in place to the approved Day 39 two-layer standard; the JSON, API, SQL, notebook-audit, and interview content was preserved and expanded.
- Validation: 18 math items rendered by KaTeX with 0 errors; 0 raw LaTeX delimiters; 168 balanced fences; 0 Mermaid diagrams.
- The SQL cells require a live MySQL server; the JSON and deep-dive cells run without it.
