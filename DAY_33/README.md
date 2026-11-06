# Day 33 — Handling Mixed Variables

CampusX 100 Days of ML · folder `DAY_33/`

---

## Day number

**Day 33**

---

## Main topics

- Variables that combine numbers and categories in one column
- Separating numerical and categorical parts
- Extracting numbers from text with regular expressions
- Extracting the initial letter of a cabin identifier
- Splitting ticket identifiers into numeric and categorical components
- Distinguishing meaningful features from identifiers and high-cardinality noise
- Missingness, validation, and train/test leakage concerns around extraction

---

## Learning objectives

1. Recognise a mixed variable and explain why its raw dtype is misleading.
2. Split a column into numeric and categorical components without losing rows.
3. Use `pandas.to_numeric(..., errors='coerce')` and a missing mask correctly.
4. Extract patterns with `str.extract` and string indexing.
5. Explain how mixed-variable handling differs from ordinary imputation.
6. Decide whether an extracted component is a meaningful feature or an identifier artefact.
7. Validate extraction with row counts, masks, and dtypes instead of relying on `head()`.
8. Explain high-cardinality and group-leakage risks when tickets or cabins create related records.

---

## Notebooks

| Local file | Local cells | Official source | Official cells | Role |
|---|---:|---|---:|---|
| `mixedata.ipynb` | 11 | `day33-handling-mixed-variables/Untitled.ipynb` | 14 | Personal mixed-variable notebook, preserved |

Official notebook stays read-only in `vendor/100-days-of-machine-learning/`; it is not copied here. The personal notebook is not merged into the official notebook.

---

## Datasets

| Local file | Official source | Note |
|---|---|---|
| `DAtasets/titanic.csv` | `day33-handling-mixed-variables/titanic.csv` | Existing local dataset folder; the notebook uses the absolute path `E:\MACHINE_LEARNING\DAY_33\DAtasets\titanic.csv` |

The actual folder name is `DAtasets` (capital `D`, lowercase `a` in `Atasets`), matching the notebook's absolute path. Earlier README wording used `Datasets`; do not rename the folder without updating the notebook and verifying the rerun.

---

## Dependencies

```text
pandas, numpy, matplotlib, jupyter
```

---

## How to run

1. Open this folder as the working directory.
2. Run `mixedata.ipynb` in its existing order.
3. Read `notes.md` for the full treatment of mixed variables and identifier extraction.

---

## Daily folder structure

```text
DAY_33/
├── mixedata.ipynb
├── notes.md          # complete
├── README.md
└── DAtasets/
    └── titanic.csv
```

---

## Status

Official notebook is mapped to the existing personal notebook. Dataset is present, `notes.md` is complete, and the README path has been corrected to `DAtasets`. No notebook, dataset, or vendor file was modified. The notebook's non-raw regex and absolute path are documented as known issues, not silently changed.
