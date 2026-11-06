# Day 16 - Working with JSON, APIs, and SQL

> CampusX 100 Days of Machine Learning, Day 16.
> One unified chapter covering the official notebook, `my_notebook.ipynb`, and the separate `ML_DAY16_JSON-deep-dive.ipynb`.

### How to read this chapter

**Part A** teaches the problem, the JSON and relational foundations, the three loaders, the notebook results, the practical workflow, and essential interview material. It is sufficient to understand and implement the day without reading Part B.

**Part B** is optional advanced reading: formal orientation references, API and SQL engineering, a cell-level source-fidelity audit, extended comparisons, an extended interview bank, and extra examination practice.

The day has no mathematical model and no train/test split. The quantitative work is about **data shapes, cardinality, selectivity, serialization, and memory**, which are the mathematical foundations of data loading.

---

# Part A - Main Learning Chapter

## 1. The problem: data is not born as a CSV

A machine-learning project begins before `train_test_split`. It begins when data is still trapped in a file, an HTTP response, or a database.

Day 16 deals with three wrappers around the same tabular idea:

| Source | Typical form | What Pandas receives |
|---|---|---|
| JSON file | text containing objects, arrays, and nested values | a DataFrame, sometimes with lists or dictionaries still inside cells |
| API | JSON delivered over HTTP | a Python list/dict, then a DataFrame |
| SQL database | tables on a database server | rows and columns returned by a query |

The destination is always a DataFrame:

```text
JSON file ──────────────┐
API response ───────────┼──> Python objects ──> pandas DataFrame ──> inspection / cleaning
SQL query result ───────┘
```

If this step is wrong, every later step is wrong. A column that looks numeric may have arrived as text. A nested list may be hiding one row per ingredient. A query may have silently excluded rows because of `NULL`. A live API may return a different page than the one saved in the notebook.

### 1.1 What the course actually demonstrates

The official `reference_notebooks/day16.ipynb` is short. It shows:

1. `pd.read_json('train.json')` on a recipe file.
2. `pd.read_json(url)` on a currency API.
3. `mysql.connector` plus `pd.read_sql_query` on the `world` database.

The local notebooks extend this:

- `my_notebook.ipynb` uses the recipe file, a GitHub issues API, and a filtered `world.country` query.
- `ML_DAY16_JSON-deep-dive.ipynb` is a separate 84-cell JSON laboratory: orientations, JSON Lines, `StringIO`, `json_normalize`, `record_path`, `meta`, `explode`, dates, Unicode, and a small cleaning project.

The three notebooks are not one notebook and must not be merged. This chapter teaches all three.

### 1.2 The central decision

Every loading task starts with one question:

> What is the shape of the source, and what shape do I need?

- If the JSON already looks like a table, use `pd.read_json`.
- If the JSON contains nested lists or dictionaries, use `pd.json_normalize`.
- If the data lives in a database, send a query and let the database do the filtering before the rows enter memory.

The rest of Part A explains how to make that decision reliably.

---

## 2. Foundations: JSON, tables, and the DataFrame

### 2.1 JSON is a small data model

JSON has six kinds of values:

| JSON value | Example | Python equivalent |
|---|---|---|
| object | `{"name": "Amit", "age": 22}` | `dict` |
| array | `["Python", "SQL"]` | `list` |
| string | `"Noida"` | `str` |
| number | `22`, `3.14` | `int` or `float` |
| boolean | `true`, `false` | `bool` |
| null | `null` | `None` |

An object maps keys to values. An array is an ordered list. These two containers can nest arbitrarily.

A JSON **document** is text. It is not a Python dictionary until a parser converts it:

```python
import json

text = '{"name": "Amit", "age": 22}'
obj = json.loads(text)      # str -> dict
print(type(obj))            # <class 'dict'>
```

This distinction matters because `pd.read_json` expects a file path, URL, or file-like object, not an arbitrary Python dictionary. A raw JSON string should be wrapped in `StringIO`.

### 2.2 Tabular JSON

The simplest JSON for Pandas is a list of objects:

```json
[
  {"id": 1, "name": "Amit", "city": "Noida"},
  {"id": 2, "name": "Rahul", "city": "Delhi"}
]
```

Each object becomes a row; each key becomes a column. This is the `records` orientation.

A column-oriented JSON object can represent the same table:

```json
{
  "id":   {"0": 1, "1": 2},
  "name": {"0": "Amit", "1": "Rahul"}
}
```

This is the `columns` orientation. The outer keys are columns; the inner keys are row labels.

A split representation stores index, columns, and data separately:

```json
{
  "index": [0, 1],
  "columns": ["id", "name"],
  "data": [[1, "Amit"], [2, "Rahul"]]
}
```

The same table can therefore be serialized in several ways. `orient` is the instruction that tells Pandas which representation it is looking at.

### 2.3 Nested JSON

Real APIs rarely return a clean list of flat objects. They return objects containing lists and objects:

```json
{
  "student": "Amit",
  "courses": [
    {"name": "Python", "score": 90},
    {"name": "ML", "score": 85}
  ]
}
```

If this were loaded with `pd.read_json`, the `courses` value would remain a list inside one cell. That is not necessarily wrong, but it is not a table of courses. `json_normalize` can turn the repeated `courses` list into rows while repeating the parent student.

### 2.4 JSON Lines

JSON Lines, also called NDJSON, stores one independent JSON object per line:

```text
{"id": 1, "name": "Amit"}
{"id": 2, "name": "Rahul"}
```

There is no enclosing array. It is useful for logs, streaming data, and large exports because a reader can process one line at a time. Pandas needs `lines=True` on both read and write.

### 2.5 The DataFrame mental model

A DataFrame is a two-dimensional table with an index and named columns. A cell can contain any Python object, including a list or a dictionary. That flexibility is useful, but it also hides structure:

```text
DataFrame with a list column

id   cuisine   ingredients
1    greek     [romaine lettuce, black olives, ...]
2    indian    [water, vegetable oil, wheat, salt]
```

The table has 39,774 rows, but it does not have 39,774 ingredients. It has 39,774 **recipes**, each with a list of ingredients. The row count is not the ingredient count. This is the first cardinality trap of the day.

### 2.6 The recipe dataset in numbers

The local `datasets/train.json` is 13,081,987 bytes. Loading it gives:

$$
39{,}774 \text{ rows} \times 3 \text{ columns}.
$$

The columns are `id`, `cuisine`, and `ingredients`. There are 20 cuisines. Every row has a non-missing ingredient list. The list lengths range from 1 to 65, with a mean of approximately 10.7677 ingredients per recipe.

The average file size per row is

$$
\frac{13{,}081{,}987}{39{,}774}\approx 328.91 \text{ bytes per row}.
$$

That number is file size, not DataFrame memory: the JSON text includes keys, brackets, commas, and repeated ingredient strings. A parsed DataFrame can use more memory than the file because Python objects and list containers have overhead.

### 2.7 The relational model

A relational database stores data in tables. A table has a fixed schema: named columns with types. Rows are records. A query returns a new relation:

```text
database world
├── country          239 rows, 15 columns
├── city             cities and populations
└── countrylanguage  984 rows, 4 columns
```

The `country` table has columns such as `Code`, `Name`, `Continent`, `Region`, `SurfaceArea`, `Population`, and `LifeExpectancy`. The `countrylanguage` table has `CountryCode`, `Language`, `IsOfficial`, and `Percentage`.

The SQL query is not a Pandas operation. The database evaluates it, and Pandas receives only the result. That is why filtering in SQL can be more efficient than loading an entire table and filtering in Pandas.

---

## 3. Reading tabular JSON with `pd.read_json`

### 3.1 The basic call

```python
import pandas as pd

df = pd.read_json("datasets/train.json")
```

`pd.read_json` parses the file, infers the orientation when `orient` is not supplied, and returns a DataFrame. The recipe file is a list of objects, so inference succeeds and produces 39,774 rows and 3 columns.

The function signature is roughly:

```python
pd.read_json(
    path_or_buf,
    orient=None,
    typ="frame",
    dtype=None,
    convert_dates=True,
    convert_axes=None,
    keep_default_dates=True,
    precise_float=False,
    date_unit=None,
    encoding=None,
    lines=False,
    chunksize=None,
    compression="infer",
    nrows=None,
)
```

The exact parameter set changes across pandas versions. Learn the common ones and check the documentation for the version you use.

### 3.2 File, URL, and string

`path_or_buf` can be:

- a local file path;
- a URL;
- a file-like object;
- in some versions, a JSON string, but this is the least reliable form.

Use an explicit `StringIO` for a string:

```python
from io import StringIO

json_text = '[{"name":"Amit","age":22},{"name":"Rahul","age":23}]'
df = pd.read_json(StringIO(json_text), orient="records")
```

Why not pass `json_text` directly? Because `read_json` is designed around a path or buffer. A literal string can be interpreted as a path or can trigger a confusing parser error. `StringIO` removes the ambiguity.

### 3.3 `orient`: the main knob

| `orient` | JSON shape | Result |
|---|---|---|
| `records` | list of objects | one row per object |
| `columns` | `{column: {index: value}}` | columns from outer keys |
| `index` | `{index: {column: value}}` | rows from outer keys |
| `split` | `{index, columns, data}` | exact table reconstruction |
| `table` | schema plus data | schema-aware table |
| `values` | list of lists | values only; names lost |

`records` is the most common orientation for APIs. `columns` is the default orientation used by `DataFrame.to_json` in current pandas. `split` is useful when you need an exact round-trip. `values` is compact but discards column names and index labels.

A wrong orientation does not always raise an error. It can produce an empty DataFrame, transposed data, or columns made from the wrong keys. Always inspect the shape and columns after loading.

### 3.4 Worked example: students

The deep-dive creates this Python list:

```python
data = [
    {
        "id": 1,
        "name": "Amit",
        "age": 22,
        "city": "Noida",
        "skills": ["Python", "SQL"],
        "address": {"state": "UP", "pincode": 201301},
    },
    {
        "id": 2,
        "name": "Rahul",
        "age": 23,
        "city": "Delhi",
        "skills": ["Python", "ML"],
        "address": {"state": "Delhi", "pincode": 110001},
    },
    {
        "id": 3,
        "name": "Priya",
        "age": 21,
        "city": "Lucknow",
        "skills": ["SQL", "Tableau"],
        "address": {"state": "UP", "pincode": 226001},
    },
]
```

After writing it to `students.json` and reading it:

```python
df = pd.read_json("students.json")
```

the result has 3 rows and 6 columns. The `skills` column still contains lists. The `address` column still contains dictionaries. `read_json` has not flattened them.

That is a valid intermediate state. It is not yet a table of skills or states. Section 6 shows how to flatten it.

### 3.5 JSON Lines

Read:

```python
df = pd.read_json("students.jsonl", lines=True)
```

Write:

```python
df.to_json("students.jsonl", orient="records", lines=True)
```

The `lines=True` flag is not optional for JSON Lines. Without it, Pandas expects one enclosing JSON document. With it, Pandas expects one object per line.

`nrows` can sample a large JSON Lines file:

```python
sample = pd.read_json("huge.jsonl", lines=True, nrows=100)
```

This reads 100 lines without parsing the whole file. It is a good habit before loading a multi-gigabyte log.

### 3.6 Reading from a URL

The official notebook does:

```python
pd.read_json("https://api.exchangerate-api.com/v4/latest/INR")
```

The saved output is a 160-row by 7-column DataFrame. The API returns a JSON object whose `rates` field is a dictionary of currencies. Pandas inferred a table in which the currencies became the index, the rate values became a `rates` column, and metadata such as `base`, `date`, and `api_url` were repeated across rows.

This is a useful demonstration, but it is not a general API client:

- no authentication headers;
- no pagination;
- no retry policy;
- no control over timeouts;
- no explicit schema.

For production, use `requests` (or another HTTP client), inspect `response.json()`, and then call `pd.json_normalize`.

The local `my_notebook.ipynb` uses:

```python
url = "https://api.github.com/repositories/19438/issues"
df = pd.read_json(url)
```

The saved output is 30 rows by 36 columns. The GitHub endpoint returns a list of issue objects, so `records` inference works. But 30 rows is one page, not the whole issue history. Pagination and rate limits are real API concerns, not details to ignore.

### 3.7 What `read_json` does internally

At a high level:

1. Open the path, URL, or buffer.
2. Read and decode the text.
3. Parse JSON into Python objects.
4. Apply the orientation rule to decide rows, columns, and index.
5. Infer dtypes.
6. Build the DataFrame.

The expensive parts are parsing and object construction. For a 13 MB file with 39,774 lists, the parser creates a Python list for every `ingredients` cell. That is why a JSON file can consume more memory than its byte size suggests.

### 3.8 Dtypes after loading

On pandas 3.0.5, the recipe frame has:

```text
id             int64
cuisine          str
ingredients    object
```

Older pandas versions may show `object` for `cuisine`. The `ingredients` column is `object` because each cell is a Python list, not a scalar. Do not expect a numeric dtype for a column that contains containers.

Always inspect:

```python
df.shape
df.columns
df.dtypes
df.isna().sum()
df.head()
```

---

## 4. Writing JSON with `DataFrame.to_json`

### 4.1 The basic call

```python
df.to_json("output.json", orient="records", indent=2)
```

If `path_or_buf` is a path, Pandas writes a file. If it is `None`, Pandas returns a JSON string.

Common parameters:

| Parameter | Meaning |
|---|---|
| `path_or_buf` | output path or `None` for a string |
| `orient` | structure of the JSON |
| `indent` | pretty-print indentation |
| `lines` | one object per line |
| `force_ascii` | escape non-ASCII when `True` |
| `date_format` | `"epoch"` or `"iso"` |
| `date_unit` | `"s"`, `"ms"`, `"us"`, `"ns"` for epoch values |

### 4.2 Orientation examples

For a small frame:

```python
df_small = pd.DataFrame({
    "name": ["Amit", "Rahul"],
    "age": [22, 23],
    "city": ["Noida", "Delhi"],
})
```

`orient="records"` gives:

```json
[
  {"name": "Amit", "age": 22, "city": "Noida"},
  {"name": "Rahul", "age": 23, "city": "Delhi"}
]
```

`orient="columns"` gives:

```json
{
  "name": {"0": "Amit", "1": "Rahul"},
  "age":  {"0": 22, "1": 23},
  "city": {"0": "Noida", "1": "Delhi"}
}
```

`orient="index"` uses the index as the outer keys. `orient="split"` stores index, columns, and data separately. `orient="table"` adds schema information. `orient="values"` stores only the value matrix.

### 4.3 JSON Lines for large data

```python
df.to_json("data.jsonl", orient="records", lines=True)
```

Each row becomes one JSON object on one line. This is the preferred format for logs and streaming exports. Do not add pretty-print indentation to a JSON Lines file: the format depends on one object per line.

### 4.4 Dates and epochs

JSON has no native date type. A datetime column can be written as:

- an ISO string, which is readable and portable;
- an epoch number, which is compact but needs a unit.

```python
dates_df = pd.DataFrame({
    "name": ["Amit", "Rahul"],
    "date": pd.to_datetime(["2026-09-23", "2026-09-24"]),
})

dates_df.to_json("dates.json", orient="records", date_format="iso", indent=2)
```

For `2026-09-23`, the UTC epoch seconds are

$$
1{,}790{,}121{,}600,
$$

the epoch milliseconds are

$$
1{,}790{,}121{,}600{,}000,
$$

and the epoch nanoseconds are

$$
1{,}790{,}121{,}600{,}000{,}000{,}000.
$$

The unit must travel with the data; a bare epoch number is ambiguous.

### 4.5 Unicode

By default, `to_json` escapes non-ASCII characters. Use:

```python
df.to_json("output.json", orient="records", force_ascii=False)
```

to preserve readable Unicode. The deep-dive demonstrates this with Hindi text. The saved notebook has some mojibake in its markdown and string literals, which is an encoding problem in the notebook file, not a failure of `force_ascii=False`.

### 4.6 Round-trip warning

A round-trip is not automatically lossless:

- `orient="values"` loses column names and index;
- `orient="table"` may add pandas-specific schema metadata;
- a list column becomes a list again only if the parser preserves it;
- dates become strings unless converted after loading;
- integer columns with missing values may become floats.

Always verify the round-trip:

```python
assert df.shape == df_roundtrip.shape
assert df.columns.tolist() == df_roundtrip.columns.tolist()
```

---

## 5. The Python `json` module versus Pandas

### 5.1 Four functions

| Function | Direction | Input/output |
|---|---|---|
| `json.load()` | JSON file to Python | file object -> dict/list |
| `json.loads()` | JSON string to Python | string -> dict/list |
| `json.dump()` | Python to JSON file | object + file -> file |
| `json.dumps()` | Python to JSON string | object -> string |

Example:

```python
import json

obj = json.loads('{"name": "Amit", "age": 22}')
text = json.dumps(obj, indent=2)
```

### 5.2 When to use which

Use the `json` module when:

- you need to inspect the raw structure before deciding how to tabulate it;
- you need to parse a JSON string stored inside a DataFrame column;
- you need to write a Python object that is not a DataFrame;
- you need fine control over serialization.

Use Pandas when:

- the JSON already maps to rows and columns;
- you want dtype inference and a DataFrame immediately;
- you need `json_normalize` for nested records.

### 5.3 JSON inside a DataFrame column

A common warehouse pattern is a table with a text column that contains JSON:

```text
id | details
1  | '{"city":"Noida","age":22}'
2  | '{"city":"Delhi","age":23}'
```

The `details` column contains strings, not dictionaries. Parse first:

```python
df["details"] = df["details"].apply(json.loads)
```

Then normalize and concatenate:

```python
details_flat = pd.json_normalize(df["details"])

result = pd.concat(
    [
        df.drop(columns="details").reset_index(drop=True),
        details_flat.reset_index(drop=True),
    ],
    axis=1,
)
```

The `reset_index(drop=True)` calls align the two frames by position. Without them, pandas aligns on the index, which can silently introduce missing values if the indexes differ.

### 5.4 JSON null and missing values

JSON `null` becomes Python `None`, and Pandas usually represents it as `NaN` in numeric columns or as a nullable value in extension dtypes.

```python
missing_json = '[{"name":"Amit","age":22},{"name":"Rahul","age":null}]'
missing_df = pd.read_json(StringIO(missing_json), orient="records")
```

The `age` column becomes floating-point because `NaN` is a float. If exact integer semantics matter, use a nullable integer dtype after loading:

```python
missing_df["age"] = missing_df["age"].astype("Int64")
```

Do not assume that a missing value in JSON is the same as an empty string. `null`, `""`, and a missing key are three different states.

### 5.5 Dates after loading

JSON dates arrive as strings. Convert them explicitly:

```python
df["date"] = pd.to_datetime(df["date"])
```

Use `errors="coerce"` when bad dates should become missing rather than stopping the pipeline:

```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
```

Then inspect how many values became `NaT`.

---

## 6. Nested JSON with `pd.json_normalize`

### 6.1 Why normalization is needed

A nested object loaded by `read_json` stays nested:

```text
id | name  | address                        | education
1  | Amit  | {'city': 'Noida', 'state': 'UP'} | {'degree': 'BTech', 'year': 2027}
```

Many models and CSV exports cannot use a dictionary cell directly. `json_normalize` flattens nested dictionaries into dotted column names.

### 6.2 The basic call

```python
flat = pd.json_normalize(nested_data)
```

For the two-student example, the result has 2 rows and 6 columns:

```text
id, name, address.city, address.state, education.degree, education.year
```

The nested keys become part of the column name, joined by `"."` by default.

### 6.3 `sep`

Change the separator:

```python
pd.json_normalize(nested_data, sep="_")
```

Now the columns are `address_city`, `address_state`, `education_degree`, and `education_year`.

The separator is a naming convention, not a semantic change. Choose one and use it consistently across the project.

### 6.4 `max_level`

`max_level` controls how deeply dictionaries are flattened:

| `max_level` | Result on the two-student example |
|---:|---|
| `0` | 4 columns: `id`, `name`, `address`, `education` (nested dicts preserved) |
| `1` | 6 columns: nested dictionaries flattened one level |
| `None` | all levels flattened |

This is useful when an API response is too deeply nested and complete flattening would create hundreds of columns.

### 6.5 `record_path`: one row per child

Consider:

```python
course_data = [
    {
        "student": "Amit",
        "college": "GL Bajaj",
        "courses": [
            {"name": "Python", "score": 90},
            {"name": "ML", "score": 85},
        ],
    },
    {
        "student": "Rahul",
        "college": "GL Bajaj",
        "courses": [
            {"name": "Python", "score": 88},
            {"name": "SQL", "score": 92},
        ],
    },
]
```

`pd.json_normalize(course_data, record_path="courses")` returns 4 rows and 2 columns:

```text
name    score
Python  90
ML      85
Python  88
SQL     92
```

The parent information is gone. The row count follows the cardinality rule

$$
\text{rows}=\sum_{i=1}^{n} n_i,
$$

where $n_i$ is the number of child records for parent $i$. Here $2+2=4$.

### 6.6 `meta`: keep parent fields

```python
pd.json_normalize(
    course_data,
    record_path="courses",
    meta=["student", "college"],
)
```

returns 4 rows and 4 columns:

```text
name    score  student  college
Python  90     Amit     GL Bajaj
ML      85     Amit     GL Bajaj
Python  88     Rahul    GL Bajaj
SQL     92     Rahul    GL Bajaj
```

The parent fields are repeated on every child row. This is the correct denormalized form for a course-level table: each row is one course, with enough parent context to identify it.

### 6.7 `meta_prefix`

If a parent field has the same name as a child field, prefix the parent:

```python
pd.json_normalize(
    course_data,
    record_path="courses",
    meta=["student", "college"],
    meta_prefix="parent_",
)
```

The columns become `parent_student` and `parent_college`. The child columns remain `name` and `score`.

`record_prefix` does the same for child columns when those collide with metadata.

### 6.8 Nested `record_path`

`record_path` can be a list when the repeated records are several levels deep:

```python
deep_data = [
    {
        "company": "ABC",
        "department": {
            "name": "AI",
            "employees": [
                {"name": "Amit", "role": "Intern"},
                {"name": "Rahul", "role": "Engineer"},
            ],
        },
    }
]

pd.json_normalize(
    deep_data,
    record_path=["department", "employees"],
    meta=["company", ["department", "name"]],
)
```

The result has 2 rows and 4 columns: `name`, `role`, `company`, and `department.name`. The nested `meta` path `["department", "name"]` is flattened to `department.name`.

### 6.9 `errors="ignore"`

If some records are missing a metadata key, the default `errors="raise"` stops with an error. `errors="ignore"` keeps going and leaves the missing field as `NaN`:

```python
incomplete = [
    {"id": 1, "profile": {"city": "Noida"}},
    {"id": 2},
]

pd.json_normalize(incomplete, meta=["profile"], errors="ignore")
```

The output has `id` and `profile.city`, with `NaN` for the second row. Use this only when missing metadata is genuinely acceptable; otherwise fix the source schema.

### 6.10 `explode`: one row per list element

`json_normalize` handles nesting during parsing. `explode` handles a list that is already a column:

```python
skills_df = pd.DataFrame({
    "id": [1, 2],
    "skills": [["Python", "SQL"], ["ML", "Python"]],
})

skills_df.explode("skills")
```

The result has 4 rows. The index is repeated; use `reset_index(drop=True)` if a unique index is required.

The cardinality rule is the same:

$$
\text{rows}=\sum_{i=1}^{n} \lvert \text{list}_i \rvert.
$$

### 6.11 Choosing between `record_path` and `explode`

| Situation | Use |
|---|---|
| Nested list inside a raw JSON object | `record_path` |
| List already stored in a DataFrame column | `explode` |
| Need parent metadata repeated | `record_path` with `meta` |
| Need to keep the original nested structure | `read_json` or `json_normalize` without flattening |

### 6.12 Mini project: clean orders

The deep-dive ends with a small project:

```python
project_data = [
    {
        "user_id": 101,
        "profile": {"name": "Amit", "city": "Noida"},
        "joined": "2026-09-01",
        "orders": [
            {"product": "Laptop", "amount": 60000},
            {"product": "Mouse", "amount": 1000},
        ],
    },
    {
        "user_id": 102,
        "profile": {"name": "Rahul", "city": "Delhi"},
        "joined": "2026-09-05",
        "orders": [
            {"product": "Keyboard", "amount": 2000},
        ],
    },
]
```

The normalization is:

```python
orders = pd.json_normalize(
    project_data,
    record_path="orders",
    meta=["user_id", ["profile", "name"], ["profile", "city"], "joined"],
    sep="_",
)

orders["joined"] = pd.to_datetime(orders["joined"])
```

The result has 3 rows: two orders for user 101 and one for user 102. The columns include `product`, `amount`, `user_id`, `profile_name`, `profile_city`, and `joined`.

Then the notebook:

1. lowercases column names;
2. checks `isna().sum()`;
3. exports to `clean_orders.json` with ISO dates.

This is the correct order: normalize, type-convert, clean names, validate missingness, export. Do not export before checking the shape and missing values.

---

## 7. SQL: from a database to a DataFrame

### 7.1 The relational model in one page

A database contains schemas; a schema contains tables; a table contains rows and typed columns. The `world` database used by the course contains:

| Table | Rows | Columns | Purpose |
|---|---:|---:|---|
| `country` | 239 | 15 | one row per country |
| `countrylanguage` | 984 | 4 | languages spoken per country |
| `city` | many | 5 | cities and populations |

The `countrylanguage` table links to `country` through `CountryCode -> Code`.

### 7.2 Connecting with `mysql.connector`

The official notebook uses:

```python
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="world",
)
```

This requires a running MySQL server with the `world` database imported from `datasets/world.sql`.

A typical import command in a shell that supports input redirection is:

```bash
mysql -u root -p < datasets/world.sql
```

In Windows PowerShell, where `<` is not a native input redirection operator, use one of these instead:

```powershell
Get-Content -Raw datasets/world.sql | mysql -u root -p
```

or import the file through MySQL Workbench. The dump contains `CREATE SCHEMA world; USE world;`, so it creates and selects the database.

### 7.3 Reading a query

```python
import pandas as pd

df = pd.read_sql_query("SELECT * FROM countrylanguage", conn)
```

The saved official output is 984 rows by 4 columns. The local notebook uses:

```python
df = pd.read_sql_query(
    "SELECT * FROM country WHERE LifeExpectancy > 60",
    conn,
)
```

The saved local output is 167 rows by 15 columns.

### 7.4 Why 167 and not 239?

The `country` table has 239 rows. Seventeen countries have `NULL` life expectancy. SQL `NULL` means "unknown", and a comparison with `NULL` is not true:

```sql
NULL > 60  -- unknown, not true
```

Therefore those 17 rows are excluded. The filter returns 167 rows. The selectivity is

$$
\frac{167}{239}\approx 0.6987,
$$

or about 69.87% of the table.

This is not a Pandas quirk. It is SQL three-valued logic: true, false, and unknown. If you wanted to keep the missing values, you would write:

```sql
WHERE LifeExpectancy > 60 OR LifeExpectancy IS NULL
```

and then decide explicitly how to treat them.

### 7.5 Closing the connection

The notebooks do not close the connection. Production code should:

```python
conn = mysql.connector.connect(...)
try:
    df = pd.read_sql_query("SELECT * FROM country", conn)
finally:
    conn.close()
```

or use a context manager whose semantics you have verified for the driver. Leaving connections open consumes server resources and can exhaust the connection limit.

### 7.6 The SQLAlchemy warning

In pandas 3.0.5, `pd.read_sql_query` with a `mysql.connector` connection emits:

```text
UserWarning: pandas only supports SQLAlchemy connectable (engine/connection)
or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects
are not tested. Please consider using SQLAlchemy.
```

The query still works, but the warning is a compatibility signal. The supported production pattern is SQLAlchemy:

```python
from sqlalchemy import create_engine

engine = create_engine("mysql+mysqlconnector://root:@localhost/world")
df = pd.read_sql_query("SELECT * FROM country", engine)
```

This requires `sqlalchemy` and a suitable MySQL driver. The exact URL depends on the driver and credentials.

### 7.7 SQL injection

Never build SQL by concatenating untrusted input:

```python
# Dangerous
name = input("Country: ")
sql = "SELECT * FROM country WHERE Name = '" + name + "'"
```

A malicious value can change the query. Use parameters:

```python
sql = "SELECT * FROM country WHERE Continent = %s"
df = pd.read_sql_query(sql, conn, params=("Asia",))
```

The placeholder style depends on the driver (`%s` for MySQL Connector/Python; `?` for sqlite3; named parameters for SQLAlchemy). The principle is the same: data must not be parsed as SQL syntax.

### 7.8 Credentials

Do not hardcode real passwords in a notebook or commit them to git. Use environment variables or a secrets manager:

```python
import os

conn = mysql.connector.connect(
    host=os.environ["DB_HOST"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database="world",
)
```

The course uses an empty local password for convenience. That is a classroom setup, not a production pattern.

### 7.9 SQL versus Pandas filtering

Both of these can produce the same rows:

```python
# SQL filtering
df = pd.read_sql_query(
    "SELECT * FROM country WHERE LifeExpectancy > 60",
    conn,
)

# Pandas filtering
df_all = pd.read_sql_query("SELECT * FROM country", conn)
df = df_all[df_all["LifeExpectancy"] > 60]
```

The SQL version transfers fewer rows and uses the database's indexes and optimizer. The Pandas version transfers the whole table and filters in memory. For large tables, filter in SQL. For small tables, either is fine, but be consistent and document the query.

### 7.10 A useful join

The two tables can be joined:

```sql
SELECT c.Name, cl.Language, cl.Percentage
FROM country AS c
JOIN countrylanguage AS cl
  ON c.Code = cl.CountryCode
WHERE cl.IsOfficial = 'T'
ORDER BY cl.Percentage DESC
LIMIT 5;
```

This returns the five most widely spoken official languages in the snapshot. The join is the relational equivalent of merging two DataFrames on a key.

### 7.11 `read_sql_query`, `read_sql_table`, and `read_sql`

- `read_sql_query(sql, con)` runs a SQL query.
- `read_sql_table(table_name, con)` reads a table; it requires SQLAlchemy.
- `read_sql(sql, con)` is a convenience wrapper that dispatches to one of the above.

Use `read_sql_query` when you want an explicit query and portable DB-API behavior.

---

## 8. The notebooks and the actual results

### 8.1 Official notebook: 11 cells

`reference_notebooks/day16.ipynb` is the CampusX original. Its sequence is:

1. import pandas;
2. `pd.read_json('train.json')` -> 39,774 rows x 3 columns;
3. `pd.read_json('https://api.exchangerate-api.com/v4/latest/INR')` -> saved 160 rows x 7 columns;
4. `!pip install mysql.connector`;
5. connect to `world`;
6. `pd.read_sql_query("SELECT * FROM countrylanguage", conn)` -> 984 rows x 4 columns.

The saved API output is dated 2021-03-31. A live call today will not return the same rates. The notebook does not close the MySQL connection.

### 8.2 `my_notebook.ipynb`: 13 cells

The local notebook follows the same three-source idea:

1. import pandas;
2. read `E:/MACHINE_LEARNING/train.json` by absolute path -> 39,774 x 3;
3. read `https://api.github.com/repositories/19438/issues` -> saved 30 x 36;
4. install `mysql.connector`;
5. connect to `world`;
6. query `country WHERE LifeExpectancy > 60` -> 167 x 15.

The GitHub output is one page of issues. The 36 columns include URLs, IDs, titles, bodies, labels, reactions, and other API metadata. Most of them are not useful for a simple analysis; this is a good example of "API payload is not the same as analysis table."

### 8.3 `ML_DAY16_JSON-deep-dive.ipynb`: 84 cells

The deep-dive is a separate personal notebook. It covers:

- JSON basics;
- `read_json` and orientations;
- `to_json`;
- JSON Lines;
- `StringIO`;
- the `json` module;
- `json_normalize`, `sep`, `max_level`, `record_path`, `meta`, `meta_prefix`, nested paths, and `errors`;
- JSON strings inside DataFrame columns;
- `explode`;
- null handling;
- dates, `date_format`, and `date_unit`;
- Unicode;
- API response envelopes;
- a reusable `inspect_json` function;
- the `clean_orders` mini project.

It creates files such as `students.json`, `students.jsonl`, `dates.json`, `data.jsonl`, and `clean_orders.json` in the working directory. Run it from `Day_16/` so those files do not scatter across the project.

### 8.4 Source-fidelity notes

These are observations about the actual saved notebooks, not requirements to change them:

- The official notebook uses a relative `train.json`; the local `my_notebook` uses an absolute `E:/MACHINE_LEARNING/train.json`. Prefer `datasets/train.json` for portability.
- The official and local notebooks both use `!pip install mysql.connector` inside the notebook. Environment setup belongs outside the notebook.
- Neither notebook closes the MySQL connection.
- The official API output is from 2021; the GitHub output is from a later live call. Both are snapshots, not stable data.
- The local notebook markdown contains mojibake such as `�?"`. The Python code is unaffected; this is an encoding artifact.
- The deep-dive markdown also contains mojibake and corrupted Hindi strings. The JSON concepts are intact, but the saved text should not be treated as a clean Unicode example.
- The deep-dive variable `sample_100` actually reads 3 rows (`nrows=3`). The name is misleading, not the code.
- The deep-dive's `clean_orders` export writes a file in the current directory. That is expected for a practice notebook.
- Some older notes referred to `database.sqlite` for the SQL side. That file is not part of this folder. The actual SQL asset is `datasets/world.sql`, used through a MySQL server.

---

## 9. Practical workflow and decisions

### 9.1 Inspect before loading

For an unfamiliar JSON file:

```python
import json

with open("file.json", encoding="utf-8") as f:
    data = json.load(f)

print(type(data))
print(len(data) if isinstance(data, list) else list(data.keys()))
print(data[:2] if isinstance(data, list) else data)
```

The deep-dive wraps this in a reusable function:

```python
def inspect_json(data):
    print("Python type:", type(data))
    if isinstance(data, list):
        print("Number of records:", len(data))
        if data:
            print("First record:", data[0])
    elif isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
```

Do this before choosing `read_json` or `json_normalize`.

### 9.2 A decision table

| Observation | Action |
|---|---|
| list of flat objects | `pd.read_json(..., orient="records")` or `pd.DataFrame(data)` |
| one object per line | `pd.read_json(..., lines=True)` |
| nested dictionaries | `pd.json_normalize(data)` |
| nested list of records | `pd.json_normalize(data, record_path=...)` with `meta` |
| list already in a DataFrame column | `df.explode(column)` |
| JSON string inside a column | `json.loads` then `json_normalize` |
| database table | `pd.read_sql_query` |
| huge JSON Lines | `lines=True` plus `nrows` or chunked reading |

### 9.3 Validate after loading

Always check:

```python
df.shape
df.columns.tolist()
df.dtypes
df.isna().sum()
df.head()
df.sample(5, random_state=0)
```

For a known dataset, compare against expected counts. For the recipe file, 39,774 rows and 3 columns. For `countrylanguage`, 984 rows and 4 columns. For the filtered `country` query, 167 rows and 15 columns.

### 9.4 Reproducibility

- Save the raw API response to disk with a timestamp.
- Record the URL, query parameters, and API version.
- Record the SQL query and the database snapshot.
- Do not write tests that depend on a live API returning the same page.
- Pin the pandas version when JSON behavior matters.

### 9.5 Memory

`pd.read_json` builds Python objects. A 13 MB file with nested lists can occupy much more than 13 MB in memory. For large data:

- use JSON Lines;
- read a sample with `nrows`;
- select only needed columns after loading;
- consider a columnar format such as Parquet for repeated analysis;
- avoid keeping multiple full copies of the DataFrame.

### 9.6 Security

- Never hardcode production credentials.
- Use parameterized SQL.
- Treat downloaded JSON as untrusted input.
- Validate file paths and sizes before parsing.
- Be careful with deeply nested JSON that can cause excessive memory use.

### 9.7 Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Wrong `orient` | empty or transposed frame | inspect the JSON shape; pass the correct orient |
| Missing `lines=True` | parser error or one giant row | use `lines=True` for JSON Lines |
| Passing a JSON string directly | path error or parser error | wrap in `StringIO` |
| Using `read_json` on nested API data | dict/list cells | use `json_normalize` |
| Forgetting `meta` | parent fields lost after `record_path` | add `meta` |
| Forgetting `reset_index` after `explode` | duplicate index | `reset_index(drop=True)` |
| Not closing SQL connection | resource leak | `try/finally` or context manager |
| Concatenating SQL strings | injection risk | parameterized query |
| Assuming API output is complete | only first page | pagination and rate limits |
| Treating `null` as an empty string | wrong missing-value semantics | inspect and convert explicitly |

---

## 10. Essential interview and examination preparation

### 10.1 Level 1 - definitions

**Q1. What is JSON?**

JSON is a text format for structured data built from objects, arrays, strings, numbers, booleans, and null. It is language-independent and commonly used by APIs and configuration files.

**Q2. What is a DataFrame?**

A two-dimensional labeled data structure with rows, columns, and an index. It can hold heterogeneous column types and, in a cell, Python objects such as lists or dictionaries.

**Q3. What is JSON Lines?**

A format with one independent JSON object per line, used for logs and streaming data. Pandas reads it with `lines=True`.

**Q4. What is a relational database?**

A database organized into tables with fixed schemas, where rows are records and columns are attributes. SQL queries return relations.

### 10.2 Level 2 - mechanics

**Q5. `read_json` versus `json_normalize`?**

`read_json` reads JSON that already maps reasonably to a table. `json_normalize` flattens nested dictionaries and lists, especially API payloads.

**Q6. What does `orient` do?**

It tells Pandas how the JSON maps to rows, columns, and index. Common values are `records`, `columns`, `index`, `split`, `table`, and `values`.

**Q7. Why use `StringIO`?**

Because `read_json` expects a path or file-like object. `StringIO` wraps a JSON string as a buffer and avoids path/parser ambiguity.

**Q8. What do `record_path` and `meta` do?**

`record_path` selects the nested list that should become rows. `meta` copies parent fields onto each generated row.

**Q9. What does `max_level` control?**

How many nested dictionary levels are flattened. `0` keeps top-level dictionaries, `1` flattens one level, and `None` flattens all levels.

**Q10. What is the difference between `json.load` and `json.loads`?**

`json.load` reads a file object; `json.loads` reads a string.

### 10.3 Level 3 - applied and numerical

**Q11. Two students each have two courses. How many rows does `record_path="courses"` produce?**

$$
2 \times 2 = 4.
$$

With `meta`, the parent fields are repeated on all four rows.

**Q12. A list column has lengths 2 and 3. How many rows after `explode`?**

$$
2+3=5.
$$

**Q13. The recipe file has 39,774 rows and 3 columns. Does it contain 39,774 ingredients?**

No. It contains 39,774 recipes. The ingredient lists have a mean length of about 10.7677, so the total number of ingredient entries is much larger than the row count.

**Q14. The `country` table has 239 rows, 17 with `NULL` life expectancy, and 167 with life expectancy greater than 60. What fraction passes the filter?**

$$
\frac{167}{239}\approx 0.6987.
$$

The 17 `NULL` rows are excluded because `NULL > 60` is unknown, not true.

**Q15. A 13,081,987-byte file has 39,774 rows. What is the average file size per row?**

$$
\frac{13{,}081{,}987}{39{,}774}\approx 328.91 \text{ bytes}.
$$

This is file size per row, not memory per row.

### 10.4 Level 4 - debugging and scenarios

**Q16. `pd.read_json("data.jsonl")` fails. What is the first thing to check?**

Whether the file is JSON Lines. If each line is an independent object, pass `lines=True`.

**Q17. `json_normalize` returns 4 rows but the parent fields are missing. What did you forget?**

`meta`. `record_path` creates child rows but does not automatically copy parent fields.

**Q18. A column contains `'{"city":"Noida"}'` strings. Can `json_normalize` flatten it directly?**

Not reliably. Parse each string with `json.loads` first, then normalize the resulting dictionaries.

**Q19. `pd.read_sql_query` works but emits a warning about SQLAlchemy. What does it mean?**

Pandas supports SQLAlchemy connectables, database URIs, and sqlite3 DB-API connections. Other DB-API objects such as `mysql.connector` are not tested. The query may work, but SQLAlchemy is the supported path.

**Q20. You need the first 100 rows of a 50 GB JSON Lines file. What do you do?**

Use `pd.read_json(..., lines=True, nrows=100)`, or stream the file line by line. Do not load the whole file.

### 10.5 Worked code-reading questions

**Q21. What is the shape of the result?**

```python
data = [
    {"id": 1, "items": [{"x": 10}, {"x": 20}]},
    {"id": 2, "items": [{"x": 30}]},
]
pd.json_normalize(data, record_path="items", meta="id")
```

There are 3 child records, so the shape is `(3, 2)`: columns `x` and `id`.

**Q22. What is the difference between these two calls?**

```python
pd.json_normalize(data, max_level=0)
pd.json_normalize(data, max_level=1)
```

`max_level=0` keeps top-level nested dictionaries as cells. `max_level=1` flattens one level into dotted column names.

**Q23. Why does this fail?**

```python
pd.read_json('[{"a": 1}]')
```

Because the literal string is not a reliable file-like input. Wrap it:

```python
pd.read_json(StringIO('[{"a": 1}]'), orient="records")
```

### 10.6 Exam-style theory with answers

**Question 1.** Explain why `read_json` can return a DataFrame with a list in a cell. Is that an error?

It is not an error. `read_json` maps the top-level JSON structure to rows and columns. A nested list remains a Python object in its cell until it is flattened. The user must decide whether the list is a valid cell value or a repeated record that needs normalization.

**Question 2.** A colleague says "JSON is just a text version of a table." Correct the statement.

JSON can represent tabular data, but it is a general data model with objects, arrays, and nesting. A table is one possible shape. Nested JSON often needs normalization before it becomes tabular.

**Question 3.** Why is `WHERE LifeExpectancy > 60` not equivalent to "all rows except those with life expectancy at most 60"?

Because SQL uses three-valued logic. Rows with `NULL` life expectancy are unknown, not false. They are excluded by the filter. To include them, use `OR LifeExpectancy IS NULL` and handle the missing values explicitly.

**Question 4.** Why is a parameterized SQL query safer than string concatenation?

Parameterization separates SQL syntax from data. The driver sends the query structure and the values separately, so a value cannot change the query's meaning. String concatenation allows SQL injection.

**Question 5.** Why can a 13 MB JSON file consume much more than 13 MB in memory?

The file is compact text. Parsing creates Python objects: dictionaries, lists, strings, and numeric objects. Each object has overhead. A DataFrame with list-valued cells also stores those containers. File size and in-memory size are different measurements.

---

## 11. Revision summary

### 11.1 The three loaders

| Goal | Tool | Key arguments |
|---|---|---|
| tabular JSON file | `pd.read_json` | `orient`, `lines`, `nrows` |
| JSON string | `pd.read_json(StringIO(...))` | `orient` |
| nested JSON | `pd.json_normalize` | `record_path`, `meta`, `sep`, `max_level` |
| list column | `df.explode` | column name |
| JSON text in a column | `json.loads` then `json_normalize` | parse first |
| SQL table | `pd.read_sql_query` | SQL string, connection, `params` |

### 11.2 The three shapes

```text
Tabular JSON  -> read_json  -> DataFrame
Nested JSON   -> json_normalize -> DataFrame
SQL relation  -> read_sql_query -> DataFrame
```

### 11.3 The five checks

1. What is the top-level JSON type: list or dict?
2. Is the file JSON Lines?
3. Is the data nested?
4. Does the query return the expected row count?
5. Did the connection close, and were credentials kept out of the notebook?

### 11.4 The three cardinality traps

- A recipe row is not an ingredient.
- A parent row is not a child row after `record_path`.
- A `NULL` comparison is not `False`; it is unknown.

### 11.5 The one-sentence summary

Inspect the source structure, choose `read_json` for tabular JSON, `json_normalize` for nested JSON, and `read_sql_query` for relational data; then validate shape, dtypes, missingness, and provenance before the data enters the ML workflow.

---

# Part B - Advanced Reading and Reference

## 12. Formal JSON and orientation reference

### 12.1 JSON grammar

A JSON value is one of:

```text
object   := { "key": value, ... }
array    := [ value, ... ]
string   := "..."
number   := integer or decimal
boolean  := true | false
null     := null
```

An object's keys are strings. Arrays preserve order. Objects are unordered in the JSON specification, although Python dictionaries preserve insertion order.

### 12.2 Type mapping

| JSON | Python | Pandas tendency |
|---|---|---|
| object | `dict` | `object` cell or flattened columns |
| array | `list` | `object` cell or exploded rows |
| string | `str` | `object` or `str` dtype |
| integer | `int` | `int64` or `float64` if missing |
| decimal | `float` | `float64` |
| boolean | `bool` | `bool` or `object` if missing |
| null | `None` | `NaN`, `NaT`, or nullable value |

### 12.3 Orientation reference

| Orientation | Read shape | Write shape | Round-trip property |
|---|---|---|---|
| `records` | list of objects | one object per row | names and values preserved |
| `columns` | column dicts | column dicts | names and index preserved |
| `index` | index dicts | index dicts | index labels preserved |
| `split` | index, columns, data | same | exact structural round-trip |
| `table` | schema + data | schema + data | schema-aware, more metadata |
| `values` | list of lists | list of lists | names and index lost |

### 12.4 JSON Lines specification

A JSON Lines file:

- uses UTF-8;
- has one JSON value per line;
- does not have a top-level array;
- may end with a newline;
- is not pretty-printed across lines.

The reader must process each line independently. This is why `lines=True` is a separate mode.

---

## 13. Parameter reference

### 13.1 `pd.read_json`

| Parameter | Default | Meaning |
|---|---|---|
| `path_or_buf` | required | path, URL, or file-like object |
| `orient` | `None` | JSON orientation; inferred when possible |
| `typ` | `"frame"` | `"frame"` or `"series"` |
| `dtype` | `None` | dtype specification |
| `convert_dates` | `True` | convert date-like columns |
| `convert_axes` | `None` | convert axes to dtypes |
| `keep_default_dates` | `True` | keep default date parsing |
| `precise_float` | `False` | use higher-precision float parsing |
| `date_unit` | `None` | timestamp unit |
| `encoding` | `None` | file encoding |
| `lines` | `False` | JSON Lines mode |
| `chunksize` | `None` | chunked reading when supported |
| `compression` | `"infer"` | compression detection |
| `nrows` | `None` | read only this many lines (JSON Lines) |

### 13.2 `DataFrame.to_json`

| Parameter | Default | Meaning |
|---|---|---|
| `path_or_buf` | required | path or `None` |
| `orient` | `None` | JSON structure; resolves to columns in current pandas |
| `date_format` | `None` | epoch behavior by default; use `"iso"` for readable dates |
| `double_precision` | `10` | digits for floats |
| `force_ascii` | `True` | escape non-ASCII |
| `date_unit` | `"ms"` | epoch unit |
| `default_handler` | `None` | handler for unsupported objects |
| `lines` | `False` | JSON Lines mode |
| `compression` | `"infer"` | compression |
| `index` | version-dependent | include index |

### 13.3 `pd.json_normalize`

| Parameter | Default | Meaning |
|---|---|---|
| `data` | required | dict or list of dicts |
| `record_path` | `None` | path to repeated records |
| `meta` | `None` | parent fields to repeat |
| `meta_prefix` | `None` | prefix for metadata columns |
| `record_prefix` | `None` | prefix for record columns |
| `errors` | `"raise"` | `"raise"` or `"ignore"` for missing metadata |
| `sep` | `"."` | separator for nested names |
| `max_level` | `None` | maximum flattening depth |

### 13.4 `json` module

| Function | Input | Output |
|---|---|---|
| `json.load` | file object | Python object |
| `json.loads` | string | Python object |
| `json.dump` | object + file | file |
| `json.dumps` | object | string |

---

## 14. API engineering

### 14.1 HTTP basics

An API request has:

- a method (`GET`, `POST`, ...);
- a URL;
- headers (authentication, content type, user agent);
- optional query parameters;
- an optional body.

A response has:

- a status code (`200`, `401`, `403`, `404`, `429`, `500`, ...);
- headers;
- a body, often JSON.

`pd.read_json(url)` performs a simple GET. It cannot set custom headers or handle pagination. For real APIs, use `requests`:

```python
import requests

response = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
response.raise_for_status()
payload = response.json()
```

Then inspect `payload` before normalizing.

### 14.2 Response envelopes

APIs often wrap records in metadata:

```json
{
  "page": 1,
  "total": 100,
  "results": [
    {"id": 1, "name": "Amit"},
    {"id": 2, "name": "Rahul"}
  ]
}
```

Extract the list:

```python
df = pd.json_normalize(payload["results"])
metadata = {"page": payload["page"], "total": payload["total"]}
```

Do not pass the whole envelope to `json_normalize` and hope the result is the table you wanted. Inspect the keys.

### 14.3 Pagination

The GitHub issues endpoint returns one page of 30 issues by default. The saved `my_notebook` output is 30 rows. To collect more:

- send `per_page` and `page` parameters;
- follow the `Link` header;
- stop when a page is empty or the rate limit is reached;
- store the page number with each row.

A loop must have a termination condition. An API that always returns the same page will otherwise loop forever.

### 14.4 Authentication and rate limits

- Use an access token in an `Authorization` header, not in a notebook cell.
- Read tokens from environment variables or a secrets manager.
- Handle `401` (unauthorized), `403` (forbidden), and `429` (too many requests).
- Respect `Retry-After` headers.
- Cache responses during development.

### 14.5 Reproducibility

A live API is not a reproducible data source. Save the raw response:

```python
with open("raw/github_issues_2026-11-04.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
```

Then build the DataFrame from the saved file. This separates data collection from data analysis.

### 14.6 API versus `read_json`

| Requirement | `pd.read_json(url)` | `requests` + `json_normalize` |
|---|---|---|
| simple public JSON | good | good |
| custom headers | no | yes |
| pagination | no | yes |
| retries/timeouts | limited | full control |
| nested envelope | awkward | explicit |
| production logging | no | yes |

---

## 15. SQL engineering

### 15.1 DB-API and SQLAlchemy

Python database access has two layers:

- **DB-API 2.0**: the low-level driver interface (`connect`, `cursor`, `execute`, `fetchall`, `close`).
- **SQLAlchemy**: a higher-level engine and SQL toolkit that manages connections and dialects.

`mysql.connector` is a DB-API driver. Pandas can use it, but pandas 3.0.5 warns that it is not a tested DB-API2 type. SQLAlchemy is the supported integration.

### 15.2 Connection lifecycle

```text
connect -> execute query -> fetch rows -> close
```

A connection is a resource. Close it even if the query fails:

```python
conn = mysql.connector.connect(...)
try:
    df = pd.read_sql_query("SELECT ...", conn)
finally:
    conn.close()
```

### 15.3 Parameterized queries

Never interpolate values into SQL text. Use placeholders:

```python
sql = "SELECT * FROM country WHERE Continent = %s"
df = pd.read_sql_query(sql, conn, params=("Asia",))
```

For SQLAlchemy:

```python
from sqlalchemy import text

sql = text("SELECT * FROM country WHERE Continent = :continent")
df = pd.read_sql_query(sql, engine, params={"continent": "Asia"})
```

### 15.4 Transactions

`SELECT` queries do not modify data, so no commit is required. If you use `INSERT`, `UPDATE`, or `DELETE`, you must understand the driver's autocommit and transaction behavior. A read-only analysis should use a read-only account when possible.

### 15.5 Schema and keys

The `world` schema uses:

- `country.Code` as the primary key;
- `countrylanguage.CountryCode` as a foreign key to `country.Code`;
- `city.CountryCode` as a foreign key to `country.Code`.

A join is valid only when the key types and meanings match. The `CountryCode` is a three-letter code, not a country name.

### 15.6 Indexes and selectivity

An index can make a filtered query faster. Selectivity is the fraction of rows that pass a predicate:

$$
\text{selectivity}=\frac{\text{rows returned}}{\text{rows examined}}.
$$

For `LifeExpectancy > 60`, the selectivity is $167/239\approx 0.6987$. A filter that returns most of the table may not benefit much from an index; a highly selective filter can.

### 15.7 SQL versus pandas operations

| Task | SQL | Pandas |
|---|---|---|
| filter rows | `WHERE` | boolean mask |
| select columns | `SELECT` | `df[cols]` |
| aggregate | `GROUP BY` | `groupby` |
| join | `JOIN` | `merge` |
| sort | `ORDER BY` | `sort_values` |
| limit | `LIMIT` | `head` |

Use SQL for reducing data before transfer. Use Pandas for iterative analysis and modeling.

### 15.8 SQL injection example

Dangerous:

```python
name = "Aruba' OR '1'='1"
sql = "SELECT * FROM country WHERE Name = '" + name + "'"
```

The value changes the query's logic. Parameterized:

```python
sql = "SELECT * FROM country WHERE Name = %s"
df = pd.read_sql_query(sql, conn, params=(name,))
```

The driver treats the value as data, not syntax.

---

## 16. Source-fidelity audit

### 16.1 Official notebook: 11 cells

| Cells | Content | Saved output |
|---|---|---|
| 0 | import pandas | - |
| 1 | markdown: working with JSON | - |
| 2 | `pd.read_json('train.json')` | 39,774 x 3 |
| 3 | `pd.read_json(exchangerate URL)` | 160 x 7, dated 2021-03-31 |
| 4 | markdown: working with SQL | - |
| 5 | `!pip install mysql.connector` | install log |
| 6 | `import mysql.connector` | - |
| 7 | connect to `world` | - |
| 8 | `SELECT * FROM countrylanguage` | 984 x 4 |
| 9 | display `df` | 984 x 4 |
| 10 | empty | - |

The official notebook does not close the connection, does not use a context manager, and does not handle API or database errors.

### 16.2 `my_notebook.ipynb`: 13 cells

| Cells | Content | Saved output |
|---|---|---|
| 0 | documentation banner | - |
| 1-4 | read `train.json` | 39,774 x 3 |
| 5-6 | read GitHub issues URL | 30 x 36 |
| 7-8 | SQL documentation | - |
| 9 | install `mysql.connector` | install log |
| 10 | connect and query `country` | 167 x 15, UserWarning |
| 11 | experiment markdown | - |
| 12 | key takeaways | - |

The local notebook uses an absolute path, does not close the connection, and relies on a live API for the GitHub cell.

### 16.3 `ML_DAY16_JSON-deep-dive.ipynb`: 84 cells

The notebook is organized into 38 numbered sections plus banners and experiments. Its major blocks are:

| Block | Cells | Topic |
|---|---|---|
| JSON basics and sample data | 0-7 | JSON structures, `students.json` |
| `read_json` and orientations | 8-19 | records, columns, index, split, table, values |
| `to_json` and JSON Lines | 20-25 | writing, NDJSON, `StringIO` |
| `json` module | 26-27 | `loads`, `dumps` |
| `json_normalize` | 28-41 | nesting, `sep`, `max_level`, `record_path`, `meta`, nested paths |
| JSON in columns and `explode` | 42-45 | parsing strings, list columns |
| Missing values and dates | 46-57 | nulls, `to_datetime`, `date_format`, `date_unit`, Unicode |
| Large files and APIs | 58-70 | JSON Lines, `nrows`, envelopes, `errors` |
| Workflow and mini project | 71-83 | inspection, mistakes, cheat sheets, `clean_orders` |

The deep-dive writes files in the current working directory. It is a practice notebook, not a production pipeline.

### 16.4 Discrepancies and cautions

- The official and local SQL cells use `mysql.connector` directly; pandas warns in 3.0.5.
- The official notebook's API output is a 2021 snapshot.
- The local GitHub output is a later snapshot; the API is live.
- The local notebook's markdown has mojibake.
- The deep-dive's Hindi strings are corrupted in the saved file.
- The deep-dive's `sample_100` variable reads 3 rows, not 100.
- Older notes mentioned `database.sqlite`; the actual SQL asset is `datasets/world.sql`.
- The notebook installs packages with `!pip`; environment setup should be external.
- The official `countrylanguage` query returns 984 rows; the local `country` query returns 167 rows. They are different tables and must not be compared as if they were the same experiment.

---

## 17. Extended comparisons

### 17.1 JSON versus CSV versus SQL versus Parquet

| Format | Structure | Types | Nested data | Best use |
|---|---|---|---|---|
| CSV | flat rows | inferred | no | simple interchange |
| JSON | objects/arrays | inferred | yes | APIs, configuration, semi-structured data |
| SQL | relational tables | schema-defined | no (normalized) | transactional data, filtering, joins |
| Parquet | columnar | schema-defined | limited | analytics, compression, repeated reads |

### 17.2 `read_json` versus `json_normalize`

| Question | `read_json` | `json_normalize` |
|---|---|---|
| Is the JSON already tabular? | yes | not required |
| Does it flatten nested dicts? | no | yes |
| Does it explode nested lists? | no | with `record_path` |
| Does it preserve parent fields? | no | with `meta` |
| Does it handle JSON Lines? | yes, with `lines=True` | not directly |

### 17.3 `record_path` versus `explode`

| Question | `record_path` | `explode` |
|---|---|---|
| Input | raw nested JSON | DataFrame column |
| Parent metadata | `meta` | already present |
| Cardinality | sum of child counts | sum of list lengths |
| Index | new RangeIndex | repeated original index |

### 17.4 SQL versus Pandas filtering

| Question | SQL | Pandas |
|---|---|---|
| Where does filtering happen? | database | memory |
| Network transfer | only matching rows | all selected rows |
| Index use | possible | not applicable |
| Reproducibility | query text | code plus data snapshot |
| Flexibility | SQL dialect | Python ecosystem |

### 17.5 `read_json` versus `requests` plus `json_normalize`

| Question | `read_json(url)` | `requests` |
|---|---|---|
| Headers/auth | no | yes |
| Pagination | no | yes |
| Error handling | limited | full |
| Nested envelope | implicit | explicit |
| Production readiness | low | high |

---

## 18. Extended interview bank

### Level 1 - Foundations

1. What is JSON? Name its six value types.
2. What is a DataFrame?
3. What is JSON Lines?
4. What is a relational table?
5. What is an API?

### Level 2 - Mechanics

6. What does `orient="records"` mean?
7. What does `orient="columns"` mean?
8. What does `lines=True` change?
9. Why does `StringIO` help?
10. What is the difference between `json.load` and `json.loads`?
11. What does `json_normalize` do?
12. What does `record_path` do?
13. What does `meta` do?
14. What does `max_level` do?
15. What does `explode` do?

### Level 3 - Mathematics and applied reasoning

16. Derive the row count after `record_path` for parent records with $n_i$ children.
17. Derive the row count after `explode` for a list column.
18. Explain why `NULL > 60` is not true.
19. Compute the selectivity of a filter that returns 167 of 239 rows.
20. Explain why file size per row is not memory per row.

### Level 4 - Debugging and leakage

21. A JSON file loads as one column. What went wrong?
22. A JSON Lines file fails without `lines=True`. Why?
23. A nested API response produces dict cells. What do you do?
24. `record_path` output lost the parent ID. What did you forget?
25. A SQL query returns fewer rows than expected. How do you check `NULL`?
26. A pandas SQL call warns about SQLAlchemy. What does it mean?
27. A notebook works locally but fails in CI because the API changed. How do you fix the process?
28. A column contains JSON strings and `json_normalize` raises. Why?

### Level 5 - Internals and performance

29. Why can a 13 MB JSON file use more than 13 MB of memory?
30. How does `read_json` infer orientation?
31. What is the cost of `json_normalize` on a large list?
32. When is `nrows` useful?
33. When is Parquet a better choice than JSON?
34. Why is filtering in SQL often cheaper than filtering in Pandas?
35. What does a database index do?

### Level 6 - Design and scenarios

36. Design a pipeline for a paginated JSON API with authentication.
37. Design a reproducible ingestion job for a daily CSV export.
38. Design a schema for recipes and ingredients in a relational database.
39. When would you store JSON in a database column instead of normalizing it?
40. How would you validate a JSON ingestion pipeline?

### Level 7 - Advanced

41. Compare `orient="split"` and `orient="table"` for round-trip fidelity.
42. Explain the trade-offs of denormalizing parent fields with `meta`.
43. How would you handle a JSON Lines file larger than memory?
44. How would you detect and handle schema drift in API responses?
45. How would you secure database credentials in a notebook-based workflow?
46. How would you test a SQL query without a production database?
47. Why is SQL three-valued logic a source of silent data loss?
48. How would you reconcile a JSON API snapshot with a SQL snapshot of the same entities?

---

## 19. Extended examination questions

### 19.1 Theory

1. Explain the difference between a JSON object and a JSON array.
2. Explain why JSON is self-describing but not schema-enforcing.
3. Explain the role of `orient` in `read_json` and `to_json`.
4. Explain why JSON Lines is suitable for streaming.
5. Explain the relational model and the meaning of a primary key.
6. Explain SQL `NULL` semantics with an example.
7. Explain why parameterized queries prevent SQL injection.
8. Explain the difference between `read_sql_query` and `read_sql_table`.

### 19.2 Numerical

9. A parent list has 4 parents with child counts 2, 3, 1, and 5. How many rows after `record_path`?
10. A list column has lengths 4, 0, and 2. How many rows after `explode`?
11. A table has 500 rows and a filter returns 125. What is the selectivity?
12. A 20 MB JSON file has 50,000 rows. What is the average file size per row?
13. A JSON date is `2026-09-23`. What is its UTC epoch in seconds? (Use the value from Section 4.4.)
14. A nested object has 3 top-level scalar fields and two nested objects with 2 fields each. How many columns after full normalization?
15. A nested object has the same structure but `max_level=0`. How many columns?

### 19.3 Analytical

16. A colleague claims that loading JSON with `read_json` is always sufficient. Refute the claim with a nested API example.
17. A colleague claims that `WHERE LifeExpectancy > 60` returns all non-missing rows. Correct the claim.
18. A colleague claims that a 30-row API response is the full dataset. What questions do you ask?
19. A colleague stores JSON strings in a CSV column and expects `read_csv` to flatten them. What is wrong with the expectation?
20. A colleague uses string concatenation for a SQL query because "the input is internal." Explain why that is still risky.

---

## 20. References and glossary

### 20.1 Course references

- CampusX Day 16 official notebook: `vendor/100-days-of-machine-learning/day16 - working-with-json-and-sql/day16.ipynb`.
- Local reference copy: `reference_notebooks/day16.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- Local JSON deep-dive: `ML_DAY16_JSON-deep-dive.ipynb`.
- Dataset: `datasets/train.json`, `datasets/world.sql`.

### 20.2 Technical references

- pandas documentation: `read_json`, `to_json`, `json_normalize`, `read_sql_query`.
- Python documentation: `json` module, `io.StringIO`.
- MySQL Connector/Python documentation.
- SQLAlchemy documentation.
- JSON Lines specification.
- GitHub REST API documentation.

### 20.3 Glossary

| Term | Meaning |
|---|---|
| API | application programming interface; here, an HTTP service returning JSON |
| DB-API | Python's standard low-level database interface |
| DataFrame | two-dimensional labeled table |
| JSON | JavaScript Object Notation; text format for structured data |
| JSON Lines | one JSON value per line |
| `meta` | parent fields repeated on child rows in `json_normalize` |
| `orient` | JSON-to-table mapping |
| `record_path` | path to repeated child records |
| selectivity | fraction of rows returned by a filter |
| SQL injection | attack that changes a query through unescaped input |
| `StringIO` | in-memory text buffer used as a file-like object |

### 20.4 Final checklist

- [ ] I can explain why JSON, APIs, and SQL all end as DataFrames.
- [ ] I can choose `read_json` versus `json_normalize`.
- [ ] I can explain `orient`, `lines`, and `StringIO`.
- [ ] I can flatten nested records with `record_path` and `meta`.
- [ ] I can compute row counts after `record_path` and `explode`.
- [ ] I can connect to MySQL, run a query, and close the connection.
- [ ] I can explain why `NULL` changes a filter's row count.
- [ ] I can explain why SQLAlchemy is preferred over raw DB-API connections in pandas.
- [ ] I can describe pagination, authentication, and rate limits.
- [ ] I can validate shape, dtypes, missingness, and provenance before modeling.
