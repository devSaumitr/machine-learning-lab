# Day 55 - Ridge Regression

CampusX 100 Days of Machine Learning - `Day_55/`

## Day number

**55**

## Main topics

- Multicollinearity and unstable OLS coefficients
- The ridge objective, L2 penalty, and shrinkage
- Closed-form, SVD, and gradient-descent views
- Coefficients paths and alpha selection
- From-scratch and scikit-learn implementations
- Bias-variance interpretation of regularization

## Learning objectives

1. Explain why OLS becomes unstable with correlated or numerous features.
2. Derive the ridge closed form and interpret the penalty.
3. Compare OLS and ridge coefficients as alpha changes.
4. Implement ridge with scikit-learn and from scratch.
5. Select alpha with validation data instead of the test set.
6. Interpret ridge coefficients without treating them as feature importance.

## Notebooks

The five official notebooks are kept in `reference_notebooks/` as read-only reference copies.

| File | Role | Cells |
|---|---|---:|
| `Ridge Regularization.ipynb` | scikit-learn ridge, extreme alpha, polynomial features | 18 |
| `ridge-regression-from-scratch.ipynb` | Closed-form ridge with NumPy | 12 |
| `ridge-regression-from-scratch-m-and-b.ipynb` | Univariate slope/intercept derivation and class | 14 |
| `ridge-regression-gradient-descent.ipynb` | SGD, sparse_cg, and batch gradient descent | 13 |
| `ridge-regression-key-understandings.ipynb` | Coefficient paths, bias-variance, loss surface | 30 |

These copies are byte-identical to the originals in:

`vendor/100-days-of-machine-learning/day55-regularized-linear-models/`

Do not edit the copies. The vendor repository is the master reference and is also read-only.

## Datasets

No local dataset file is required.

- The diabetes dataset is loaded through `sklearn.datasets.load_diabetes()`.
- The polynomial examples use `sklearn.datasets.make_regression()` and synthetic data generated in the notebooks.
- Notebook 3 uses the diabetes feature matrix directly.
- Notebook 5 contains an external geometry image embedded in the notebook; no separate image file is required.

## Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Optional:

```text
mlxtend
```

`mlxtend` is needed only for the bias-variance decomposition cell in
`ridge-regression-key-understandings.ipynb`. The rest of the notebook runs without it.

## How to run

1. Start Jupyter from `E:\MACHINE_LEARNING` or from `Day_55/`.
2. Open the notebooks in `reference_notebooks/` in this order for a guided study sequence:
   - `Ridge Regularization.ipynb`
   - `ridge-regression-from-scratch.ipynb`
   - `ridge-regression-from-scratch-m-and-b.ipynb`
   - `ridge-regression-gradient-descent.ipynb`
   - `ridge-regression-key-understandings.ipynb`
3. Run the cells from top to bottom. The notebooks are independent; the order above is pedagogical, not a dependency requirement.
4. Read `notes.md` alongside the notebooks for derivations, worked examples, graph interpretation, and interview practice.

## Folder structure

```text
Day_55/
    README.md
    notes.md
    reference_notebooks/
        Ridge Regularization.ipynb
        ridge-regression-from-scratch.ipynb
        ridge-regression-from-scratch-m-and-b.ipynb
        ridge-regression-gradient-descent.ipynb
        ridge-regression-key-understandings.ipynb
```

## Status

**NOTES COMPLETE**

`notes.md` covers all five notebooks: theory, mathematics, implementations, alpha
selection, standardization, comparisons with lasso and elastic net, worked numerical
examples, graph interpretation, common mistakes, interview Levels 1-7, university exam
practice, revision material, and a coverage matrix.

Validation: 5,227 lines; 1,172 math expressions rendered with 0 KaTeX errors; balanced
code fences; 3 Mermaid workflow diagrams. Notebooks and vendor files were not modified.
