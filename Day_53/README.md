# Day 53 - Polynomial Regression and the Bias-Variance Trade-off

CampusX 100 Days of ML - Day 53 - Folder `Day_53/`

---

## Day number

**Day 53**

---

## Main topics

- Fitting curved relationships with a linear-in-parameters model
- `PolynomialFeatures`, powers, interactions, and feature explosion
- Underfitting, overfitting, and degree selection
- Bias-variance trade-off and regularization preview
- Extrapolation danger, Runge's phenomenon, and numerical conditioning
- Scikit-learn pipelines and leakage-safe validation
- Correcting the official notebook's malformed 3-D design matrices

The official notebook's saved results: linear test $R^2 \approx 0.1613$; degree-2 test $R^2 \approx 0.8332$. `notes.md` explains the notebook, diagnoses its shape and plotting bugs, and documents corrected supplementary experiments separately.

---

## Learning objectives

1. Explain why polynomial regression is nonlinear in the input but linear in its coefficients.
2. Expand one or more features and interpret `powers_` and the design matrix.
3. Diagnose underfitting and overfitting from training and validation evidence.
4. Select degree and regularization strength with leakage-safe cross-validation.
5. Derive and interpret least squares, ridge, and the bias-variance decomposition.
6. Recognise ill-conditioning, overflow, Runge oscillations, and unsafe extrapolation.
7. Build a production-ready pipeline and answer interview questions at Levels 1-7.

---

## Reference notebook

| File | Role | Cells |
|---|---|---:|
| `reference_notebooks/polynomial-regression.ipynb` | Read-only copy of the CampusX notebook, with saved outputs | 37 |

Source: `vendor/100-days-of-machine-learning/day53-polynomial-regression/polynomial-regression.ipynb`

Do not edit either notebook. The local reference copy is byte-identical to the vendor source and is kept separately so the day can be studied without touching the master repository.

---

## Dataset

**No external dataset is required.** The notebook generates its data with NumPy:

- 1-D case: `X = 6 * np.random.rand(200, 1) - 3`
- 1-D target: `y = 0.8*X**2 + 0.9*X + 2 + noise`
- 3-D case: two random inputs and a quadratic surface with noise

The notebook does not set a random seed, so its saved output is not reproducible from the code alone. No local `datasets/` folder is used for this day.

---

## Dependencies

```text
python >= 3.9
numpy
matplotlib
scikit-learn
plotly
jupyter
```

Install example:

```bash
pip install numpy matplotlib scikit-learn plotly jupyter
```

---

## How to run

1. Open the folder `Day_53/`.
2. Open `reference_notebooks/polynomial-regression.ipynb` in Jupyter and run the cells in order.
3. Read `notes.md` for the full theory, worked examples, cell-by-cell walkthrough, exam questions, and interview preparation.
4. Treat the reference notebook as read-only. The notebook's 3-D section contains known bugs; `notes.md` explains them and gives the corrected experiment.

---

## Folder structure

```text
Day_53/
├── notes.md                              # complete study chapter
├── README.md
└── reference_notebooks/
    └── polynomial-regression.ipynb       # CampusX original - DO NOT EDIT
```

---

## Validation

- `notes.md`: 3,875 lines, 1,099 math items.
- Raw `\(` / `\[` delimiters: **0**.
- KaTeX rendering errors: **0**.
- Fenced code blocks: balanced.
- Mermaid diagrams: 3.
- Official notebook left unchanged; vendor repository untouched.

---

## Official source

Video: https://youtu.be/BNWLf3cKdbQ

---

## Status

**NOTES COMPLETE.** The day is documentation-only: the reference notebook is preserved as the practical source, and `notes.md` is the unified study document. There is no personal notebook for this preparation phase.
