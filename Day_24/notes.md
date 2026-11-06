# Day_24 — Standardization (z-score Scaling) — Topics Guide

> Deep notes for **Day_24**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# Standardization (z-score Scaling) — Topics Guide

**Day 24.** Local notebooks, in study sequence:

1. `DAY24_Standardization-with-Outliers.ipynb` — **MAIN** notebook (run this one).
2. `DAY24_Standardization_documented.ipynb` — same code, cell-by-cell explanations.
3. `DAY24_Feature-Scaling-stub.ipynb` — 2 markdown lines, **no code** (never revise from it).

Easy copies (same code + same outputs, simpler markdown):
`DAY24_Standardization-with-Outliers_simplified.ipynb`,
`DAY24_Standardization_documented_simplified.ipynb`,
`DAY24_Feature-Scaling-stub_simplified.ipynb`

Secondary (only for gaps): CampusX Day 24 —
https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day24-standardization
(video: https://youtu.be/1Yw9sC0PNwY)

---

## 1 What

**Feature scaling** = putting every numeric feature on a comparable range so no column
dominates just because its numbers are big.

**Standardization (z-score)** = rewrite each value as *"how many standard deviations
away from this column's mean am I?"*

- Formula: `z = (x - mean) / std`
- After scaling: column **mean ≈ 0**, column **std ≈ 1**
- Values are **unbounded** (no fixed range — an outlier just becomes a big z like `+5.1`)

In these notebooks the two features are `Age` (roughly 18–60) and
`EstimatedSalary` (roughly 15,000–150,000) from `Social_Network_Ads.csv`.

The stub notebook says *"Standardization also called Normalization"* — **that is wrong**.
Standardization (Day 24) and Min-Max Normalization (Day 25) are different formulas.

## 2 Why

- Salary numbers are ~1000x bigger than Age numbers. Distance/gradient math is
  dominated by the biggest axis, so **LogisticRegression, KNN, SVM and neural nets**
  learn better when everything is on one scale.
- **Trees (DecisionTree, RandomForest) do not care** — they only ask "is value > threshold?",
  and rescaling just relabels the thresholds. The notebooks prove both claims with accuracy.
- The statistics (mean, std) must come from **training data only**. If you scale before
  splitting, test-set information leaks into training → `fit` on full data is forbidden.

## 3 How

### 3a. z-score (this day's formula)

```
z = (x - mean) / std
```

Small example — `Age = [22, 35, 48]`:

1. `mean = (22 + 35 + 48) / 3 = 35`
2. deviations = `[-13, 0, +13]`
3. `std = sqrt((169 + 0 + 169) / 3) = sqrt(112.67) = 10.61`  (population std, ddof=0 — same as sklearn)
4. `z = [-13/10.61, 0, 10.61/10.61] = [-1.225, 0.000, +1.225]`

Mean of z is `0`, std of z is `1`. That is the whole idea — the notebook repeats it on
thousands of rows.

### 3b. min-max (Day 25 preview, for contrast)

```
x' = (x - min) / (max - min)        -> always inside [0, 1]
```

Example — `Salary = [25000, 120000]`, value `60000`:
`(60000 - 25000) / (120000 - 25000) = 35000 / 95000 = 0.368`

| | Standardization (Day 24) | Normalization / Min-Max (Day 25) |
|---|---|---|
| Formula | `(x - mean) / std` | `(x - min) / (max - min)` |
| Output range | unbounded z-scores | fixed `[0, 1]` |
| Result on fit data | mean 0, std 1 | min 0, max 1 |
| Outliers | push mean/std around | push min/max around |

## 4 Syntax breakdown

### `train_test_split` — runs FIRST, always

```python
from sklearn.model_selection import train_test_split as tts
x = df.drop(columns=['Purchased'])   # features
y = df['Purchased']                   # target
X_train, X_test, Y_train, Y_test = tts(x, y, test_size=.2, random_state=42)
```

- `test_size=.2` → 20% test, 80% train.
- `random_state=42` → same shuffle every run (reproducible).
- Rule printed in the notebook: **FOR FEATURE SCALING ALWAYS DO TRAIN TEST SPLIT**.

### `StandardScaler()` — constructor arguments

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   # with_mean=True, with_std=True, copy=True (all defaults)
```

- `with_mean=True` → subtract the mean (center at 0).
- `with_std=True` → divide by standard deviation.
- `copy=True` → original data is not overwritten.
- No argument here means "use the defaults" — that is what the notebook does.

### The three methods — when does `fit` run?

```python
scaler = scaler.fit(X_train)        # LEARNS: stores mean_ , var_ , scale_  (from TRAIN only)
X_train_scaled = scaler.transform(X_train)   # APPLIES stored stats, learns nothing
X_test_scaled  = scaler.transform(X_test)    # APPLIES same stats — never fit here
```

| Method | What it does | Where it may run |
|---|---|---|
| `fit(X)` | computes `mean_`, `var_`, `scale_`, stores them, returns `self` | **only on X_train**, once |
| `transform(X)` | `z = (x - mean)/std` using stored stats | train and test |
| `fit_transform(X)` | shorthand for `fit(X)` then `transform(X)` | **only on X_train** |

- `scaler = scaler.fit(X_train)` works because `fit` returns the scaler itself.
- Trailing underscore = learned attribute: `scaler.mean_`, `scaler.scale_`.
- `fit_transform(X_test)` would learn a *second* set of statistics from test data →
  **data leakage** and an inconsistent mapping. Never do it.

## 5 Implementation

### Notebook 1 — MAIN (`DAY24_Standardization-with-Outliers.ipynb`), in order

```python
# 1. imports + dark plot theme
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np

# 2. load + keep only modeling columns
df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\Social_Network_Ads.csv")
df = df.iloc[:, 2:]            # keeps Age, EstimatedSalary, Purchased

# 3. split BEFORE any scaling
X_train, X_test, Y_train, Y_test = tts(x, y, test_size=.2, random_state=42)

# 4. scale the leak-free way
scaler = StandardScaler()
scaler = scaler.fit(X_train)                    # learn mean/std from TRAIN
X_train_scaled = scaler.transform(X_train)      # apply to train
X_test_scaled  = scaler.transform(X_test)       # apply to test (same stats)

# 5. bring column names back (transform returns a bare numpy array)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# 6. check the math
scaler.mean_
np.round(X_train_scaled.describe(), 1)          # expect mean 0.0, std 1.0
```

Then: before/after `scatter`, three `kdeplot` panels (both columns, Age only, Salary only),
two models scored with `accuracy_score`, then the outlier round:

```python
new_data = pd.DataFrame({'Age': [5, 90, 95],
                         'EstimatedSalary': [1000, 250000, 350000],
                         'Purchased': [0, 1, 1]})
df = pd.concat([df, new_data], ignore_index=True)   # +3 extreme rows
# ... re-split, re-fit, re-transform, re-plot, re-score
```

### Notebook 2 — `DAY24_Standardization_documented.ipynb`

Identical code, extra markdown after each cell explaining *why*. Read it after the MAIN
notebook when a step feels unclear. (Its saved outputs come from a later run — see §7.)

### Notebook 3 — `DAY24_Feature-Scaling-stub.ipynb`

Two markdown lines, zero code cells. Its line *"Standardization also called
Normalization"* is **false** — ignore it and use this guide instead.

## 6 Execution walkthrough (what you actually see)

Run the MAIN notebook top to bottom:

1. **Cells 3–5 — setup.** Imports, dark theme, `df.sample(5)` shows random rows with
   `Age`, `EstimatedSalary`, `Purchased`.
2. **Cell 8 — split.** 80/20 split, `random_state=42`. Clean run: 320 train / 80 test
   rows (322 / 81 once outliers are in).
3. **Cells 11–12 — the core.** `fit(X_train)` runs once and learns 2 means + 2 stds;
   `transform` runs twice (train, test). Nothing is learned from test data.
4. **Cell 13 — `scaler.mean_`.** Prints the two learned means (Age, Salary).
   With outliers aboard the recorded value is `array([3.82080745e+01, 7.21521739e+04])`
   → mean Age **38.2**, mean Salary **72,152.2**.
5. **Cell 14 — `X_train_scaled`.** A bare numpy array of z-scores, mostly between
   −3 and +3.
6. **Cell 16 — `describe()`.** The numeric proof:

   ```
          Age  EstimatedSalary
   mean   0.0              0.0
   std    1.0              1.0
   min   -3.0             -1.8
   max    5.1              7.1      <- big z = outliers still there
   ```

7. **Cell 18 — scatter before/after.** Left: Salary axis dwarfs Age (elongated cloud).
   Right (red): cloud becomes round and centered on (0,0). Saved as PNG.
8. **Cells 20/22/23 — KDE panels.** Before: two humps on very different x-ranges.
   After: both densities centered at 0. (Watch the titles: the code sets
   `ax1.set_title("BEFORE SCALLING")` twice, so the second panel's title is overwritten —
   a harmless bug kept in the original.)
9. **Cells 26–32 — LogisticRegression.** Two fresh models, one fit on raw data, one on
   scaled data, each predicting its *matching* test slice:
   ```
   ACT     -> 0.8641975308641975   (70/81 correct)
    SCALED -> 0.8518518518518519   (69/81 correct)
   ```
   One test row apart → **noise, not proof** that scaling helped here.
10. **Cells 33–35 — DecisionTreeClassifier.**
    MAIN notebook records `ACT -> 0.8375` and `SCALED -> 0.8375` — an **exact tie**
    (documented notebook records `0.8888… / 0.8888…`, also an exact tie). Trees are
    scale-invariant; rescaling only relabels thresholds.
11. **Cells 37–39 — inject outliers.** Age `[5, 90, 95]`, Salary `[1000, 250000, 350000]`
    appended with `pd.concat`. The raw scatter now shows 3 extremes crushing the bulk
    into a corner. (In the MAIN notebook the right panel's `ax2.scatter` line is
    commented out, so that panel renders **blank** — the real comparison is cell 18.)
12. **Cells 41–45 — round two.** Same split, same scaler code, new data:
    `scaler.mean_ = [38.2, 72152.2]` — the 250k/350k salaries dragged the mean.
13. **Cells 47–48 — plots with outliers.** Right panel is standardized *but* the extreme
    points still fly far from the crowd; the Salary KDE is still skewed.
14. **Cells 49–50 — conclusion.** *"STANDARIZATION DOES NOT REDUCE THE EFFECT OF THE
    OUTLIERS — YOU HAVE TO HANDLE IT MANUALLY"* + the `when-to-apply` image.
15. **Cells 51–52 — experiment + takeaways.** Change `test_size` to 0.3; drop the 3
    outlier rows and re-score LogReg.

## 7 Observations

- **Outliers survive.** After injection, z-scores reach `+5.1` (Age) and `+7.1`
  (Salary) — standardization *centers* data, it never removes extremes. Fix: cap/remove
  outliers or use `RobustScaler` (median/IQR, Day 25 notes).
- **Leakage is avoided the right way** in these notebooks: split → `fit(X_train)` →
  `transform` both. The anti-pattern (`fit` on all data, or `fit_transform(X_test)`)
  is discussed in the markdown but never executed — know why it is wrong: test
  statistics would enter training.
- **Saved outputs are from mixed runs** — read them critically:
  - documented notebook's *first* `describe()` already shows `max 5.1 / 7.1` and
    `count 322`, i.e. that cell was executed *after* outliers were added;
  - documented notebook's `scaler.mean_` first cell also shows `[38.2, 72152.2]`;
  - tree scores differ between notebooks (0.8375 vs 0.8889) but both show a tie.
- **Small bugs kept on purpose** (do not "fix" the originals): duplicated
  `lr = LogisticRegression()` cell, `ax1.set_title` called twice in the KDE cells,
  commented-out `ax2.scatter` (blank panel), typos `SCALLING / standorize / techinique`.
- **Stub error:** *"Standardization also called Normalization"* — false. Different
  formula, different output range (Day 25).

## 8 Connection to prior concepts

- **Days 19–21 (EDA):** `scatter`, `kdeplot`, `describe()` — the same tools now used as
  *evidence* that scaling worked.
- **Day 23 (Feature Engineering):** scaling is a preprocessing step like encoding —
  it happens after splitting, before the model.
- **Train/test split:** introduced earlier; here it becomes mandatory because scalers
  learn statistics (leakage risk).
- **Day 25 (Normalization / MinMaxScaler):** the sibling technique — bounded `[0,1]`
  output, different formula, plus `RobustScaler` / `MaxAbsScaler` for outliers and
  sparse data.
- **Days 28–29 (ColumnTransformer / Pipeline):** `StandardScaler` reappears inside
  pipelines under the same rule: `fit` on train folds only.
- **CampusX Day 24 video** for a second explanation of the same z-score idea.

### From `topic_notes.md` / deep notes

# Standardization — Study Notes (Day 24)

> Reference: `day24-standardization` in the CampusX repo. Your work: `DAY24_Standardization-with-Outliers.ipynb` (Social_Network_Ads + synthetic outliers; honest scoreboard: LogReg 0.8642 vs 0.8519, Tree tied 0.8375) + `DAY24_Standardization_documented.ipynb` (explained copy) + `DAY24_Feature-Scaling-stub.ipynb` (2-line stub — do not revise from it).

## 1. Basic Idea

Standardization re-expresses every feature as "how many standard deviations from the mean": `z = (x − μ) / σ`. Result: mean 0, spread 1 — Age (20–90) and Salary (10k–350k) stop fighting on different scales.

## 2. Intuition and Mental Model

Two runners measured in meters vs kilometers look incomparable until both use the same unit. Standardization converts every feature to the shared unit of "standard deviations away from its own average".

## 3. Why Do We Need It?

Distance and gradient math is dominated by the biggest axis: unscaled, Salary differences drown Age differences, so LogisticRegression (and KNN/SVM/neural nets) *should* learn worse unscaled. Your run is the honest caveat: 0.8642 unscaled vs 0.8519 scaled — 70/81 vs 69/81, a single test row, i.e. noise rather than a verdict — while the DecisionTree tied exactly at 0.8375 (threshold splits don't care about scale).

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
- LogReg 0.8642 vs 0.8519 (one row apart — inconclusive here, not proof either way); Tree 0.8375 vs 0.8375 (exactly invariant, as theory predicts).
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
2. Your run shows LogReg 0.8642 unscaled vs 0.8519 scaled — did scaling fail, and why don't trees care? — 70/81 vs 69/81 is one test row: noise, not failure or success; scaling's value shows where optimization is genuinely harder. Trees tie because thresholds only relabel under rescaling.
3. New test point far above train max — problem? — No error; large z-score, model extrapolates (or fails honestly).
4. Outliers dominate — which scaler? — RobustScaler (median/IQR), not Standard.

## 12. Quick Revision

z-scores: unbounded, zero-mean, unit-spread. Fit-train/transform-test always. Helps gradient/distance models, not trees; never fixes outliers.

---

## Section map (Part 9 checklist)

Content above covers the day’s topics. When expanding further, place new material under these headings without deleting preserved sources:

- **1. Introduction** — Course intro / what this day covers (from topics + notebook headings).
- **2. Problem Being Solved** — Why this technique exists; what breaks if ignored.
- **3. Required Foundations** — Prerequisites from earlier days.
- **4. Conceptual Understanding** — Core idea in plain English with technical accuracy.
- **5. Mathematical Foundation** — Formulas, variables, numerical examples (when relevant).
- **6. Internal Working** — fit/transform internals, what is stored, execution flow.
- **7. Python and Scikit-learn Implementation** — Complete meaningful implementation patterns.
- **8. Syntax Breakdown** — Arguments, defaults, object vs call, return values.
- **9. Practical Examples** — Before/after tables, runnable snippets from the day.
- **10. Visual Interpretation** — Graphs: axes, pattern, conclusion, limits.
- **11. Comparison with Related Techniques** — vs alternatives; when to use / avoid.
- **12. Assumptions and Limitations** — Hard limits and edge cases.
- **13. Common Mistakes** — Beginner + workflow mistakes with fixes.
- **14. Real-World Applications** — Project scenarios and decision process.
- **15. Connection to the ML Workflow** — Where it sits in the pipeline; course links.
- **16. Interview Questions and Answers** — Levels 1–6 with reasoned answers.
- **17. Practice Questions** — Conceptual / code / debug / decision; answers separate.
- **18. Topic Summary and Connections** — Cheat sheet + link to next topic.

---

## Interview quick set

Use preserved Q&A in the body above. Add Level 1–6 follow-ups here as you study:

1. What problem does this day solve?
2. How does the method work internally?
3. What is the common leakage / domain mistake?
4. How do you compare alternatives with CV?
5. Where does this sit in a Pipeline?

## Practice

Re-run `my_notebook.ipynb`, predict one output before running, then check. Update mastery checkboxes as you go.

## Mastery checklist

- [ ] I can explain the problem in plain words
- [ ] I can write the core implementation
- [ ] I can explain important syntax / defaults
- [ ] I can name common mistakes and fixes
- [ ] I can connect this day to the ML workflow
- [ ] I can answer interview follow-ups
