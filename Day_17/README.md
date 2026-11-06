# Day 17 - Fetching Data from an API

CampusX 100 Days of Machine Learning, Day 17.

## Main topics

- HTTP `GET` requests with `requests`
- JSON responses and status codes
- Converting JSON records into a pandas DataFrame
- API pagination and the TMDb top-rated movies example
- Live-API reproducibility, snapshots, and API-key security

## Learning objectives

1. Explain the request -> JSON -> DataFrame pipeline.
2. Use `requests.get` with parameters, headers, and timeouts.
3. Check HTTP status before parsing JSON.
4. Paginate an API and avoid the removed `DataFrame.append`.
5. Save and reload a reproducible data snapshot without an accidental index column.
6. Identify the security and reproducibility limits of live APIs.

## Files

| File | Role |
|---|---|
| `my_notebook.ipynb` | Personal practical notebook: JSONPlaceholder and JSONFakery examples (9 cells) |
| `reference_notebooks/day17.ipynb` | CampusX original TMDb notebook (11 cells); read-only reference |
| `notes.md` | Complete Day 17 chapter: Part A main learning, Part B advanced reference |
| `datasets/movies.csv` | Saved TMDb snapshot from the official notebook: 8,551 rows x 8 columns |

`datasets/movies.csv` includes an `Unnamed: 0` index column because the original `to_csv` call wrote the index. Read it with `pd.read_csv("datasets/movies.csv", index_col=0)` or save future snapshots with `index=False`.

## Dataset source

- TMDb top-rated movies API, collected by the official Day 17 notebook.
- The CSV is a fixed snapshot; the live API can return different values.
- The local notebook also calls JSONPlaceholder and JSONFakery live endpoints; those results are not reproducible without a saved snapshot.

## Dependencies

```text
python >= 3.10
pandas
requests
jupyter
```

```bash
pip install pandas requests jupyter
```

Tested with Python 3.14.2 and pandas 3.0.5.

## How to run

1. Open `Day_17/` as the working directory.
2. Start Jupyter and open `my_notebook.ipynb`.
3. Run the cells in order. The notebook uses live public APIs, so results can change.
4. For the official pagination experiment, open `reference_notebooks/day17.ipynb`.
5. The tracked reference copy has the original TMDb key redacted to `YOUR_TMDB_API_KEY`; the vendor copy retains the original. Supply your own key through an environment variable and rotate any exposed key.
6. Read `notes.md` for the full explanation and the source-fidelity audit.

## Study order

1. `notes.md` Part A: HTTP, JSON, `requests`, pagination, results, and mistakes.
2. `my_notebook.ipynb`: run the small local API examples.
3. `reference_notebooks/day17.ipynb`: compare the official TMDb implementation and its saved outputs.
4. `notes.md` Part B: advanced reference, source audit, and extended interview/exam practice.

## Status

Rewritten and validated against the Day 39 standard on **2026-11-04**.

- Notes: 1,456 lines; Part A and Part B present; 11 math items; 0 KaTeX errors; 0 raw delimiters; fences balanced.
- The redundant legacy text dump was removed; the local notebook remains the preserved implementation.
