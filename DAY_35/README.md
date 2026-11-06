# Day 35 — Complete Case Analysis

CampusX 100 Days of Machine Learning · `DAY_35/`

## Main topics

- Complete-case analysis (listwise deletion) and its estimand
- Missingness percentages, column filtering, and sample loss
- MCAR, MAR, and MNAR mechanisms
- Included-versus-excluded diagnostics and bias assessment
- Original versus retained distributions
- When complete-case analysis is acceptable, and its alternatives

## Notebooks

| Local file | Role | Cells |
|---|---|---:|
| `reference_notebooks/day35.ipynb` | Copied official reference notebook; read-only | 20 |
| `missing_data.ipynb` | Personal working notebook; preserved unchanged | 20 |

The copied reference is byte-identical to `vendor/100-days-of-machine-learning/day35-complete-case-analysis/day35.ipynb`. Open either notebook read-only in Jupyter; do not edit the reference or vendor copy.

## Datasets

| Local file | Provenance | Notes |
|---|---|---|
| `data_science_job.csv` | Existing working copy | 1,909,540 bytes; CRLF line endings; read by the personal notebook via its absolute path |
| `datasets/data_science_job.csv` | `day35-complete-case-analysis/data_science_job.csv` | 1,928,699 bytes; LF line endings; official copy for portable reruns |

Both copies contain 19,158 rows × 13 columns and parse to identical values. Their byte hashes differ only because of line endings. The target column is `target`; the five low-missing analysis columns are `city_development_index`, `enrolled_university`, `education_level`, `experience`, and `training_hours`.

## Dependencies

Tested with Python 3.14.2 and:

```text
pandas 3.0.5
numpy 2.5.3
matplotlib 3.11.2
scipy 1.18.1
scikit-learn 1.9.1
jupyter
```

The official notebook itself requires only pandas, NumPy, matplotlib, and Jupyter. SciPy and scikit-learn are used by the supplementary diagnostics in `notes.md`.

## How to run

1. Open `DAY_35/` as the working directory.
2. Run `missing_data.ipynb` in order. It reads the root `data_science_job.csv` by absolute path.
3. For a portable rerun, point the read path to `datasets/data_science_job.csv`; the parsed data are identical.
4. Open `reference_notebooks/day35.ipynb` read-only to compare the official 20-cell sequence.
5. Read `notes.md` for the theory, diagnostics, worked examples, and interview preparation.

## Folder structure

```text
DAY_35/
├── data_science_job.csv
├── missing_data.ipynb
├── notes.md
├── README.md
├── datasets/
│   └── data_science_job.csv
└── reference_notebooks/
    └── day35.ipynb
```

## Status

**NOTES COMPLETE**

`notes.md` covers all 20 official cells, the missing-data mechanisms, complete-case assumptions and mathematics, retention and variance loss, included-versus-excluded diagnostics, pandas execution details, graph interpretation, worked examples, alternatives, leakage risks, and interview/exam preparation. The personal notebook, both dataset copies, the reference notebook, and `vendor/` are preserved; no extra notes or notebook copies were created.
