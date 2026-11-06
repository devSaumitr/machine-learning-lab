# Day 47 — Principal Component Analysis: Step by Step

CampusX 100 Days of Machine Learning · folder `Day_47/`

## Main topics

- Curse of dimensionality, correlated features, and redundancy
- Centring, standardisation, covariance, and correlation matrices
- Eigenvalues, eigenvectors, principal components, and scores
- Explained variance, projection, reconstruction error, and SVD
- Choosing components, sign/rotation ambiguity, and PCA limitations
- Leakage-safe pipelines and comparisons with feature selection, LDA, t-SNE, and UMAP

## Learning objectives

1. Explain why PCA is useful and when it is not appropriate.
2. Derive principal components by variance maximisation and reconstruction error.
3. Reproduce the notebook's covariance, eigenvalues, eigenvectors, and projections.
4. Interpret explained variance and choose the number of components.
5. Implement PCA correctly with `StandardScaler`, `PCA`, and `Pipeline`.
6. Identify the notebook's row-versus-column projection bug and its numerical effect.

## Reference notebook (copied, read-only)

| File | Source | Cells |
|---|---|---:|
| `reference_notebooks/pca_step_by_step (1).ipynb` | `vendor/100-days-of-machine-learning/day47-pca/pca_step_by_step (1).ipynb` | 13 |

The local copy is byte-identical to the vendor notebook (SHA-256 match). Treat it as read-only. It contains saved outputs, including the covariance matrix, eigenvalues, eigenvectors, and projected rows. There is no personal notebook for this day.

## Dataset

No dataset file is required. The notebook generates a synthetic 40-row, three-feature, two-class sample with NumPy. `datasets/` is intentionally empty.

## Dependencies

```text
numpy, pandas, matplotlib, scikit-learn, plotly, jupyter
```

`plotly` is needed for the two interactive scatter plots. The notebook uses `DataFrame.append`, which was removed in pandas 2.0; to execute it in a modern environment, use `pd.concat([df, df1], ignore_index=True)` or run with pandas < 2. The saved outputs can be read without executing the notebook.

## How to open

1. Start Jupyter or VS Code in this folder.
2. Open `reference_notebooks/pca_step_by_step (1).ipynb`.
3. Read `notes.md` for the full theory, cell-by-cell explanation, corrected PCA, worked examples, interview questions, and exam practice.
4. If rerunning, expect version differences in `df.append`, `%pylab`, the `Arrow3D` renderer, and NumPy's `eig` dtype.

## Official sources

- Video: https://youtu.be/tXXnxjj2wM4
- Kaggle notebook: https://www.kaggle.com/nitsin/pca-demo-1
- Course repository: https://github.com/campusx-official/100-days-of-machine-learning

## Folder structure

```text
Day_47/
├── notes.md
├── README.md
├── datasets/                    # empty; no external data
└── reference_notebooks/
    └── pca_step_by_step (1).ipynb
```

## Status

**NOTES COMPLETE.** The unified `notes.md` covers all 13 notebook cells, the underlying theory and derivations, exact notebook values, the projection bug and its correction, comparisons, common mistakes, interview Levels 1–7, and examination-style practice.
