# Day 15 — Working with CSV Files

CampusX 100 Days of ML · folder `Day_15/`

---

## Day number

**15**

---

## Main topics

- CSV input with pandas: `read_csv` and common loading options
- Separators, headers, names, index columns, and selected columns
- Missing-value markers, data types, dates, and chunked reading
- Saving a cleaned table with `to_csv` (supplementary)
- Practical file-loading errors and how to diagnose them

---

## Learning objectives

1. Load a CSV into a DataFrame and explain what each important argument changes.
2. Inspect shape, columns, dtypes, index, and missing values after loading.
3. Use `usecols`, `dtype`, `na_values`, `parse_dates`, `nrows`, and `chunksize` deliberately.
4. Recognise encoding, separator, header, and decimal-format problems.
5. Connect correct data loading to later cleaning and feature engineering.

---

## Reference notebooks (official, read-only)

| File in `vendor/100-days-of-machine-learning/` | Role | Cells |
|---|---|---:|
| `day15 - working with csv files/working-with-csv.ipynb` | Official CSV loading and saving lesson | 36 |

> The official notebooks are **not copied** into this folder. They stay in the read-only master repository and are mapped here by path.

---

## Datasets

| Local file | Official source |
|---|---|
| `datasets/aug_train.csv` | `day15 - working with csv files/aug_train.csv` |
| `datasets/IPL Matches 2008-2020.csv` | `day15 - working with csv files/IPL Matches 2008-2020.csv` |
| `datasets/movie_titles_metadata.tsv` | `day15 - working with csv files/movie_titles_metadata.tsv` |
| `datasets/test.csv` | `day15 - working with csv files/test.csv` |
| `datasets/zomato.csv` | `day15 - working with csv files/zomato.csv` |

**Data note.** `BX-Books.csv` is referenced by the notebook but is not included in the repository; the course README links the Book-Crossing dataset at `http://www2.informatik.uni-freiburg.de/~cziegler/BX/`.

---

## Dependencies

```text
pandas, numpy, jupyter
```

---

## How to run

1. Open the official notebook from the master repository; it is read-only.
2. Use this folder's `datasets/` copy when a notebook expects a local file.
3. Read `notes.md` for the deep theory, worked examples, and interview material.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

Video: https://www.youtube.com/watch?v=a_XrmKlaGTs

Books dataset: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

---

## Daily folder structure

```text
Day_15/
├── notes.md
├── README.md
└── datasets/         # only when data files are required
```

---

## Status

Deep `notes.md` complete: all 36 official cells covered, math checked, and key outputs verified against reruns. No personal notebook is created during the preparation phase.
