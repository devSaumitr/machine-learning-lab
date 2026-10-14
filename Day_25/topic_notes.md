# Normalization and MinMax Scaling — Study Notes (Day 25)

> Reference: `day25-normalization` in the CampusX repo. Your work: `DAY25_Normalization-MinMaxScaler.ipynb` (hand formula → scaler → leak-free split → toy Pipeline; wine data Alcohol/Malic Acid) + `DAY25_Normalization-stub.ipynb` (1-line stub — do not revise from it).

## 1. Basic Idea

Normalization (MinMax) squeezes each feature into a fixed interval, usually [0, 1]: `x' = (x − min) / (max − min)`. The smallest train value becomes 0, the largest becomes 1, everything else lands proportionally between.

## 2. Intuition and Mental Model

Grading on a curve where the worst paper scores 0 and the best scores 100: relative order preserved, absolute gaps rescaled. The catch — one absurd paper (outlier) stretches the whole curve, squeezing everyone else together.

## 3. Why Do We Need It?

Bounded features suit bounded needs: pixel values, coordinates, and algorithms assuming [0, 1] inputs. It is Standardization's sibling with opposite tradeoffs — bounded output, but hostage to min/max.

## 4. Detailed Explanation

- **Formula.** `x' = (x − min) / (max − min)`; general range `[a, b]`: `a + (x−min)(b−a)/(max−min)`. Your notebook starts by hand-computing it per column, then switches to the API.
- **`MinMaxScaler`.** `fit_transform(train)` learns min/max from train; `transform(test)` applies. `feature_range=(-1, 1)` moves the interval — same shape, different bounds.
- **Unseen values.** A test value above the train max maps *above* 1 (below min → below 0). Expected, not a bug — and exactly why ranges must come from train alone.
- **Choosing.** MinMax when bounded output matters and outliers are absent; Standard for Gaussian-ish/zero-mean needs; Robust (median/IQR) when outliers dominate; MaxAbs for sparse data (keeps zeros); row-wise `Normalizer` (unit norm) is a different thing entirely — per-sample, not per-feature.
- **Pipeline habit.** Split → fit scaler on train → transform test; the notebook's toy `LogisticRegression` + `Pipeline` is a correctness check of that discipline, not a tuned model.

## 5. Examples

```python
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range=(-1, 1))
Xtr = mm.fit_transform(X_train)
Xte = mm.transform(X_test)   # unseen extremes map outside [-1, 1]
```

Wine data columns [0, 1, 2] (`Class` label, `Alcohol`, `Malic Acid`): KDE + scatter before/after show the squeeze; `describe` confirms the bounds.

## 6. Important Formulas

`x' = (x − min)/(max − min)`; ranged: `a + (x−min)(b−a)/(max−min)`. Robust: `(x − median)/IQR`. MaxAbs: `x/max|x|`.

## 7. Code Connection

Hand formula first (understanding), `MinMaxScaler` second (practice), `Pipeline` third (habit). `feature_range` is the only knob most beginners ever need.

## 8. Interpretation

- Post-scale `describe`: min 0 / max 1 (or your custom range) → correct.
- Out-of-range transformed values → unseen extremes, investigate rather than clip blindly.
- Trailing notebook cells sketch MaxAbs/Robust/Normalizer — theory only, no hands-on yet.

## 9. Common Mistakes

1. `fit` on test or full data — leakage (same rule as Day 24).
2. MinMax with wild outliers — min/max collapse the rest; prefer Robust.
3. Confusing MinMax (column-wise, bounded) with `Normalizer` (row-wise, unit norm).
4. Clipping out-of-range values without asking *why* they exceed train bounds.

## 10. Machine Learning Connection

MinMax feeds bounded-input models and coordinate/pixel features; inside pipelines it is one interchangeable preprocessing step (your Day-29 `trf3` is exactly this role).

## 11. Interview and Conceptual Questions

1. Formula + what happens to a new maximum? — `(x−min)/range`; maps above 1.
2. MinMax vs Standard vs Robust — when each? — Bounded clean data / Gaussian zero-mean needs / outlier-heavy data.
3. What does `feature_range=(-1,1)` change? — Output interval only.
4. `Normalizer` vs `MinMaxScaler`? — Row unit-norm vs column bounding.

## 12. Quick Revision

MinMax = bounded [0,1], min/max-sensitive. Fit-train/transform-test; unseen extremes overflow honestly. Pair with [Day-24 standardization](../Day_24/topic_notes.md): different tools, same leak discipline.
