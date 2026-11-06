# Day 18 - Fetching Data with Web Scraping

CampusX 100 Days of Machine Learning, Day 18.

## Main topics

- Fetching HTML with `requests`
- Parsing HTML with BeautifulSoup and `lxml`
- Selecting page, card, and field with `find` / `find_all` / `get_text`
- Scraping AmbitionBox company cards
- Pagination, selector drift, missing fields, and verification pages
- Cleaning scraped strings into numeric and categorical features

## Learning objectives

1. Explain the request -> HTML -> parse tree -> DataFrame pipeline.
2. Use `find`, `find_all`, `select`, and `get_text` correctly.
3. Guard `find` results against `None`.
4. Extract fields card by card so rows stay aligned.
5. Paginate, check status/markers, and avoid the removed `DataFrame.append`.
6. Convert ratings, review counts, ages, employee counts, and headquarters.
7. Explain the ethics, legality, and fragility of scraping.

## Files

| File | Role |
|---|---|
| `my_notebook.ipynb` | Personal practical notebook: AmbitionBox selector demo (22 cells) |
| `reference_notebooks/day18.ipynb` | CampusX original AmbitionBox notebook (33 cells); read-only reference |
| `notes.md` | Complete Day 18 chapter: Part A main learning, Part B advanced reference |

## Dataset

No local dataset is bundled. The official notebook scrapes live pages and never writes a CSV; `datasets/` is empty. The official saved result is `final.shape == (30000, 7)`, but it cannot be reproduced from disk.

- Prefer an API or official export when one exists.
- Save scraped data immediately with `to_csv(..., index=False)`.
- A live AmbitionBox check on 2026-11-04 returned a `Verify - AmbitionBox` page, so the saved selectors are not assumed to work today.

## Dependencies

```text
python >= 3.10
pandas
requests
beautifulsoup4
lxml
jupyter
```

```bash
pip install pandas requests beautifulsoup4 lxml jupyter
```

Tested with Python 3.14.2 and pandas 3.0.5.

## How to run

1. Open `Day_18/` as the working directory.
2. Start Jupyter and open `my_notebook.ipynb`.
3. Install `lxml` in the environment; the notebook's `pip install lxml` cell is not valid Python.
4. Run the cells in order. The notebook targets a live site and has no saved outputs; it may be blocked by a verification page.
5. For the official 1,000-page experiment, open `reference_notebooks/day18.ipynb` and compare its saved outputs.
6. Read `notes.md` for the full explanation, source-fidelity audit, and ethics discussion.

## Study order

1. `notes.md` Part A: HTML, BeautifulSoup, the official and local notebooks, cleaning, and mistakes.
2. `my_notebook.ipynb`: read and run the selector demo; expect possible bot verification.
3. `reference_notebooks/day18.ipynb`: compare the official 30-row page and 30,000-row loop.
4. `notes.md` Part B: advanced reference, source audit, and extended interview/exam practice.

## Status

Rewritten and validated against the Day 39 standard on **2026-11-04**.

- Notes: 2,127 lines; Part A and Part B present; 12 math items; 0 KaTeX errors; 0 raw delimiters; fences balanced.
- The redundant legacy text dump was removed; the local notebook remains the preserved implementation.
