# Day 34 — Working with Dates and Time

CampusX 100 Days of ML · folder `DAY_34/`

---

## Day number

**Day 34**

---

## Main topics

- Why CSV date columns arrive as `object` strings
- `pd.to_datetime`, datetime dtypes, and parsing policies
- The `.dt` accessor: year, month, day, weekday, week, quarter, semester
- Weekend and semester flags built with `np.where`
- Durations, `Timedelta`, elapsed days, and approximate months
- Timestamp vs `Period` vs `Timedelta` vs `DateOffset`
- Time-zone localisation/conversion and DST pitfalls
- Leakage-safe date feature engineering and reproducible `as_of` references

---

## Learning objectives

1. Convert date and timestamp columns with an explicit format and validate the result.
2. Extract and interpret calendar components with `.dt`.
3. Build weekend, quarter, and semester features and explain their conventions.
4. Compute elapsed time in days, seconds, minutes, hours, and approximate months.
5. Distinguish a point in time from a duration and a calendar period.
6. Handle naive/aware timestamps and DST gaps or ambiguities.
7. Place date features in a leakage-safe train/test workflow.

---

## Notebooks

| Local file | Role | Cells |
|---|---|---:|
| `reference_notebooks/working-with-dates-and-time.ipynb` | Copied official reference, read-only | 42 |
| `DATE_TIME.ipynb` | Existing personal notebook, preserved | 28 |

**Official source:** `vendor/100-days-of-machine-learning/day34-handling-date-and-time/working-with-dates-and-time.ipynb`. The reference copy is byte-identical to the vendor file (SHA-256 `51ccd35a40915d7af5d7db90c0143b0b381de8fca7b37493663ae372470b4063`). The personal notebook was not modified.

---

## Datasets

| Local file | Official source | Rows |
|---|---|---:|
| `Datasets/orders.csv` | `day34-handling-date-and-time/orders.csv` | 1,000 |
| `Datasets/messages.csv` | `day34-handling-date-and-time/messages.csv` | 1,000 |

The folder name is **`Datasets`** (capital D). Local copies use LF line endings; vendor copies use CRLF. Both parse to equal DataFrames, and the local files were not altered.

The personal notebook reads the two files through absolute paths under `E:\MACHINE_LEARNING\DAY_34\Datasets\`.

---

## Dependencies

```text
pandas, numpy, jupyter
```

---

## How to run and open the references

1. Open `DATE_TIME.ipynb` from this folder and run the cells in order. It reads the datasets through the existing absolute paths.
2. Read `notes.md` for the unified theory, worked examples, interview/exam preparation, and the 42-cell coverage matrix.
3. Open `reference_notebooks/working-with-dates-and-time.ipynb` read-only for the official sequence. Its cells load `orders.csv` and `messages.csv` by relative name, so to execute them, start Jupyter with `Datasets/` as the working directory or use a scratch copy with adjusted paths. Do not edit the reference notebook.

---

## Folder structure

```text
DAY_34/
├── DATE_TIME.ipynb
├── Datasets/
│   ├── messages.csv
│   └── orders.csv
├── reference_notebooks/
│   └── working-with-dates-and-time.ipynb
├── notes.md
└── README.md
```

---

## Status

**NOTES COMPLETE.** The notes cover all 42 cells of the official reference notebook, including every saved output. There are no plots in the official notebook; this is stated explicitly in the notes. Official 2021 outputs and current-environment reruns are labelled separately. Fences and LaTeX math were validated, and the personal notebook, datasets, `reference_notebooks/`, and `vendor/` were left unchanged.
