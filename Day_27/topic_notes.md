# One-Hot Encoding — Study Notes (Day 27)

> Reference: `day27-one-hot-encoding` in the CampusX repo. Your work: `DAY27_OneHot-Encoding.ipynb` on cars.csv (~8128 rows: brand/km/fuel/owner → price; train 5689).

## 1. Basic Idea

No order? No numbers-with-meaning. One-hot encoding gives each category level its own binary (0/1) column: Diesel becomes its own yes/no question instead of a misleading rank number.

## 2. Intuition and Mental Model

A multiple-choice checkbox vs a 1–5 rating: colors are checkboxes (tick Diesel *or* Petrol), satisfaction is a rating (ordered). One-hot builds checkboxes; ordinal builds ratings. Never rate colors.

## 3. Why Do We Need It?

Numbering unordered levels (Diesel=0, Petrol=1, CNG=2) tells linear/distance models that CNG is "twice Diesel" — pure fiction that warps training. One-hot removes the lie at the cost of width.

## 4. Detailed Explanation

- **`pd.get_dummies` (quick) vs `OneHotEncoder` (pipeline-safe).** `get_dummies` is fine for exploration; the estimator remembers train categories and, with `handle_unknown="ignore"`, maps unseen test levels to all-zero rows instead of crashing or misaligning columns.
- **Dummy trap → keep n−1.** With all n dummies plus an intercept, one column is predictable from the rest (multicollinearity). `drop="first"` (or `drop_first=True`) keeps n−1; the dropped level becomes the all-zero baseline that coefficients compare against.
- **High cardinality → group first.** Raw brands would explode width: levels under 100 rows collapse into `other` via `replace`, then encode. Your run: `value_counts(owner)` → dummies → split (70/30) → OHE block (5689, 7) → `hstack` with numerics (5689, 9).
- **Preserved bugs (learn, don't imitate).** `y=df.iloc[:, :-1]` overlaps X (select the target column instead); `fit_transform` on `X_test` leaks (use `transform`). Both are explained in the notebook's takeaways.

## 5. Examples

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
Xtr = ohe.fit_transform(X_train)   # train only
Xte = ohe.transform(X_test)        # never fit_transform here
ohe.get_feature_names_out()        # read the generated columns
```

Fuel {Diesel, Petrol, CNG, LPG} → 3 columns; Diesel-as-baseline = all-zero row.

## 6. Important Formulas

Width rule: n levels → n−1 columns (with intercept). Multicollinearity target: zero dependence between input columns.

## 7. Code Connection

`drop`, `sparse_output=False` (dense for inspection), `handle_unknown="ignore"` (production safety), `get_feature_names_out` (readability). `get_dummies(drop_first=True)` mirrors the behavior for quick work.

## 8. Interpretation

- A `1` means membership, nothing more; all-zero rows mean the baseline level.
- (5689, 7) → (5689, 9): encoded block width plus numeric columns after stacking — read shapes to verify every glue step.

## 9. Common Mistakes

1. Numbering unordered levels (the original sin this day removes).
2. Keeping all n dummies with an intercept (dummy trap).
3. `get_dummies` separately on train/test → mismatched columns in production.
4. One-hotting raw IDs/high-cardinality text (Name/Ticket/Cabin) — group or drop first.
5. The notebook's two preserved bugs above.

## 10. Machine Learning Connection

One-hot columns feed linear/distance models honestly and become the categorical arm of [Day-28 ColumnTransformer](../Day_28/topic_notes.md) and [Day-29 pipelines](../Day_29/topic_notes.md) (`trf2`).

## 11. Interview and Conceptual Questions

1. What is the dummy trap and how does `drop="first"` fix it? — n dummies + intercept are collinear; n−1 removes the redundancy.
2. `get_dummies` vs `OneHotEncoder` in production? — The estimator remembers categories and tolerates unseen levels.
3. 30 levels, 20 rare — what do you do? — Group rare into `other`, then encode.
4. All-zero encoded row — meaning? — The dropped baseline level.

## 12. Quick Revision

No order → OneHot, n−1 columns, group the rare. Fit-train/transform-test; `handle_unknown="ignore"` for the real world. Ordered inputs stay with [Day 26](../Day_26/topic_notes.md).
