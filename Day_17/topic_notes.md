# Fetching Data from an API — Study Notes (Day 17)

> Reference: `day17-api-to-dataframe` in the CampusX repo. Your work: `DAY17_Fetching-Data-From-API.ipynb` (jsonplaceholder users + jsonfakery movies → 20 rows of title/popularity).

## 1. Basic Idea

A REST API is a URL that returns data (usually JSON) instead of a web page. Day 17's pattern is three steps: `requests.get(url)` → `response.json()` → `pd.DataFrame(...)`.

## 2. Intuition and Mental Model

Ordering food by phone: you call (GET request), they read back your order (JSON response), you write it into your notebook (DataFrame). The menu (available fields) is discovered by looking, not by guessing — print the keys before selecting columns.

## 3. Why Do We Need It?

Live, complete datasets often exist only behind APIs (movies, GitHub, weather, finance). Fetching programmatically beats downloading CSVs by hand and keeps the data fresh.

## 4. Detailed Explanation

- **`requests.get(url)`** performs the HTTP call and returns a response object. Check it before parsing: `resp.raise_for_status()` turns HTTP errors into loud exceptions instead of confusing parse failures.
- **`.json()`** parses the body text into Python (list or dict). No local file is involved — the data lives on the server.
- **Find the row list.** Payloads wrap rows under a key (`data`, `results`, ...). Your movies API nests rows under `result["data"]`; the users API returns the list directly. Inspect first, index second.
- **`pd.DataFrame(rows)[cols]`** keeps only needed columns (`original_title`, `popularity`). Values change between calls — live data is not reproducible, so record snapshots for experiments.

## 5. Examples

```python
import requests, pandas as pd
resp = requests.get("https://jsonfakery.com/movies/simple-paginate")
resp.raise_for_status()
data = resp.json()                       # dict with key "data"
movies = pd.DataFrame(data["data"])[["original_title", "popularity"]]
```

One raw record carries many fields (`movie_id`, `overview`, `popularity` 21.04, ...); the notebook keeps two.

## 6. Important Formulas

None — but memorize the three-step shape: GET → `.json()` → `DataFrame`.

## 7. Code Connection

`requests` handles the network; everything after `.json()` is plain Python/Pandas (dict keys, `DataFrame`, column selection) — the same tools as Day 16, minus the file.

## 8. Interpretation

20 rows is *one page* of movies, not the catalog. Popularity numbers (e.g. 368.52) are snapshot values for comparison, not stable facts.

## 9. Common Mistakes

1. Indexing a guessed key (`data`) without printing keys → `KeyError`.
2. No status check — parsing an error page as JSON.
3. Treating live results as reproducible (no seed, no snapshot).
4. Ignoring pagination/auth/rate limits on real APIs.
5. Your notebook's first fetch (users API) is never displayed — always inspect what you fetched.

## 10. Machine Learning Connection

API tables enter the same pipeline as files: inspect (Day 19) → EDA (Days 20–22) → clean → split. Live fetching also powers data updates in deployed systems.

## 11. Interview and Conceptual Questions

1. What three steps turn an API into a DataFrame? — GET, `.json()`, `DataFrame` + column selection.
2. Why call `raise_for_status()` first? — Fail loudly on HTTP errors instead of mis-parsing.
3. The payload key is unknown — what do you do? — Print `data.keys()` (or `type`) and find the row list.
4. Why snapshot API data for experiments? — Live values change; reproducibility needs a fixed copy.

## 12. Quick Revision

GET → status check → `.json()` → find the row list → `DataFrame` → keep needed columns. Inspect keys first; snapshot what matters.
