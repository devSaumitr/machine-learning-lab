# Standardization — Study Notes (Day 24)

> Reference: `day24-standardization` in the CampusX repo. Your work: `DAY24_Standardization-with-Outliers.ipynb` (Social_Network_Ads + synthetic outliers; LogReg gains, trees flat) + `DAY24_Feature-Scaling-stub.ipynb` (2-line stub — do not revise from it).

## 1. Basic Idea

Standardization re-expresses every feature as "how many standard deviations from the mean": `z = (x − μ) / σ`. Result: mean 0, spread 1 — Age (20–90) and Salary (10k–350k) stop fighting on different scales.

## 2. Intuition and Mental Model

Two runners measured in meters vs kilometers look incomparable until both use the same unit. Standardization converts every feature to the shared unit of "standard deviations away from its own average".

## 3. Why Do We Need It?

Distance and gradient math is dominated by the biggest axis: unscaled, Salary differences drown Age differences, so LogisticRegression (and KNN/SVM/neural nets) learn badly. Your notebook proves it: LogReg accuracy rises scaled, while the DecisionTree barely moves (threshold splits don't care about scale).

## 4. Detailed Explanation

- **Formula.** `z = (x − μ) / σ`, learned per column. Unbounded: outliers stay outliers, just recentered.
- **Fit on train, apply everywhere.** `fit` learns μ/σ from `X_train` only; `transform` applies to train and test. `fit_transform` on test re-learns from test data — leakage plus an inconsistent mapping.
- **Seeing it work.** Before: Salary axis dwarfs Age in scatter; KDEs sit on different scales. After: cloud circularizes, both KDEs center at 0, `describe` shows ~0/1, `sc.mean_` matches train means.
- **Not outlier-proof.** Injecting Age [5, 90, 95] and salaries up to 350k re-stretches everything — means and deviations move with extremes. Outlier resistance needs RobustScaler/capping, not StandardScaler.

## 5. Examples

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtr = sc.fit_transform(X_train)   # learn μ/σ on TRAIN
Xte = sc.transform(X_test)        # apply to test — never fit here
```

Social_Network_Ads via `iloc[:, 2:]` (Age/EstimatedSalary/Purchased), split 80/20 (`random_state=42`).

## 6. Important Formulas

`z = (x − μ) / σ`. Inverse (for interpretation): `x = z·σ + μ`.

## 7. Code Connection

`StandardScaler` wraps the formula with `fit/transform` discipline; `train_test_split` must come first so statistics never see test data. The notebook's scatter + `kdeplot` triplets (both columns, Age-only, Salary-only) are the before/after evidence.

## 8. Interpretation

- Post-scale `describe` ≈ mean 0, std 1 → correct application.
- LogReg scaled > unscaled → scaling helped optimization; Tree scaled ≈ unscaled → expected, scale-invariant splits.
- Re-stretched axes after outlier injection → Standardization centers but never removes extremes.

## 9. Common Mistakes

1. `fit_transform(X_test)` — the classic leak (also present in your Days 27–28 notebooks).
2. Fitting on full data before splitting.
3. Expecting outliers to vanish (use RobustScaler/cap/log instead).
4. Scaling the target column by accident.
5. The stub's line "Standardization also called Normalization" — wrong; different formulas, different purposes (see Day_25).

## 10. Machine Learning Connection

Scale for distance/gradient models; skip (harmlessly) for trees. Standardization reappears as preprocessing inside Day-28/29 pipelines under the same fit-on-train rule.

## 11. Interview and Conceptual Questions

1. Write the formula and the fit/transform rule. — `(x−μ)/σ`; fit-train, transform-test.
2. Why does LogReg gain but trees don't? — Distances/gradients distort with scale; thresholds don't.
3. New test point far above train max — problem? — No error; large z-score, model extrapolates (or fails honestly).
4. Outliers dominate — which scaler? — RobustScaler (median/IQR), not Standard.

## 12. Quick Revision

z-scores: unbounded, zero-mean, unit-spread. Fit-train/transform-test always. Helps gradient/distance models, not trees; never fixes outliers.
