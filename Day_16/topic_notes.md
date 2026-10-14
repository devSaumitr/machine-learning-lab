# Working with JSON and SQL — Study Notes (Day 16)

> Reference: `day16 - working-with-json-and-sql` in the CampusX repo. Your work: `ML_DAY16_JSON-SQL-API.ipynb` (train.json 39774×3, GitHub issues API 30×36, MySQL world.country 167×15) + `ML_DAY16_JSON-deep-dive.ipynb` (JSON mastery + clean_orders mini-project).

## 1. Basic Idea

JSON files, API responses, and SQL tables all become the same thing in your hands: a Pandas DataFrame. Day 16 teaches the three loaders: `pd.read_json` for files (and simple URLs), `pd.json_normalize` for nested structures, and `pd.read_sql_query` for database tables.

## 2. Intuition and Mental Model

Think of every source as "rows and columns wearing a different wrapper". CSV is already tabular. JSON is tabular-looking text that sometimes hides lists inside cells. An API is JSON delivered over the network. SQL is a table living on a server — you mail it a question (the query) and it mails back rows.

## 3. Why Do We Need It?

Real ML data rarely arrives as a clean CSV. Recipe data came as `train.json` (39774 recipes with `id/cuisine/ingredients`), issue trackers serve JSON over HTTP, and company data lives in MySQL. If collection is wrong, every later step (EDA → cleaning → model) is wrong.

## 4. Detailed Explanation

- **Tabular JSON → `pd.read_json(path)`**. Works when the JSON already looks like rows: a list of objects (`orient="records"`), or column dicts. The `orient` argument must match the file's layout (`records`, `columns`, `index`, `split`, `table`, `values`).
- **JSON Lines → `lines=True`**. One object per line (e.g. `students.jsonl`, `data.jsonl`). Forgetting `lines=True` (or adding it for normal JSON) fails.
- **JSON string → `StringIO`**. `pd.read_json` treats a plain string as a *file path* (hence the famous `FileNotFoundError` demo in your notebook). Wrap the string: `pd.read_json(StringIO(text))`.
- **Nested JSON → `pd.json_normalize`**. Lists/dicts inside cells (like `ingredients`, or `address{state,pincode}`) need flattening: `record_path` turns a nested list into rows, `meta` repeats parent fields, `sep` joins key names, `max_level` caps depth. Rule of thumb: tabular → `read_json`, nested → `normalize`.
- **List column → `explode`**. One row per list element when you need ingredient-level rows.
- **SQL → connect + `pd.read_sql_query(sql, conn)`**. `mysql.connector.connect(host, user, database)` opens the session; the query (`SELECT * FROM country WHERE LifeExpectancy>60`) returns 167×15. Always `conn.close()`; production code prefers SQLAlchemy (your notebook shows the `UserWarning`).

## 5. Examples

- `students.json` (Amit/Rahul/Priya with `skills` + `address`): `read_json` gives 3 rows with `address` still a dict per cell → `json_normalize` splits it into `address_state`, `address_pincode`.
- Courses: `pd.json_normalize(data, record_path="courses", meta=["id", "name"])` makes one row per course while repeating the student.
- Writing: `df.to_json("students.jsonl", orient="records", lines=True)` round-trips with `read_json(..., lines=True)`.

## 6. Important Formulas

No formulas this day — but two signatures to memorize:
- `pd.read_json(path_or_buf, orient=..., lines=...)`
- `pd.json_normalize(data, record_path=..., meta=..., sep="_", max_level=...)`

## 7. Code Connection

```python
df = pd.read_json("students.json")                       # tabular JSON file
df = pd.read_json("students.jsonl", lines=True)          # JSON Lines
from io import StringIO
df = pd.read_json(StringIO(records_json), orient="records")  # JSON string
flat = pd.json_normalize(nested_data, sep="_")           # nested dicts
courses = pd.json_normalize(data, record_path="courses", meta=["id", "name"])
import mysql.connector as cc
conn = cc.connect(host="localhost", user="root", database="world")
df = pd.read_sql_query("SELECT * FROM country WHERE LifeExpectancy>60", conn)
conn.close()
```

## 8. Interpretation

- `train.json` 39774×3: rows are *recipes*, and `ingredients` cells are lists — row count is not ingredient count.
- GitHub issues 30×36: 30 rows is *one API page*, not all issues; most of the 36 columns are payload noise.
- SQL result 167×15: countries passing the filter in that snapshot, nothing more.

## 9. Common Mistakes

1. Passing a JSON string without `StringIO` → `FileNotFoundError`.
2. Wrong/missing `lines=True` for JSONL.
3. Using `read_json` on deeply nested payloads instead of `normalize`.
4. Assuming API keys (`data`, `results`) without printing them first.
5. Leaving SQL connections open; `pip install` inside notebooks instead of the environment.

## 10. Machine Learning Connection

Collection → inspection (Day 19). Normalized JSON (`to_datetime` dates, exploded lists, `isna` checks — see the `clean_orders.json` mini-project) feeds directly into EDA and feature engineering.

## 11. Interview and Conceptual Questions

1. When do you use `read_json` vs `json_normalize`? — Tabular-shaped JSON vs nested dicts/lists.
2. Why does `pd.read_json(json_string)` fail? — It is treated as a file path; wrap in `StringIO`.
3. What do `record_path` and `meta` do? — Explode a nested list into rows while repeating parent fields.
4. What does `max_level=0` vs `1` change? — Flattening depth.
5. Why close the SQL connection, and what is the SQLAlchemy warning about? — Release the session; `mysql.connector` direct use is legacy.

## 12. Quick Revision

Tabular JSON → `read_json` (+`orient`/`lines`/`StringIO`); nested → `normalize` (+`record_path`/`meta`/`sep`); SQL → connect → `read_sql_query` → close. Inspect keys and shapes before assuming anything.
