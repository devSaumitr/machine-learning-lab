# Day_30 — FunctionTransformer and Log Transform — Topics Guide (Day 30)

> Deep notes for **Day_30**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# FunctionTransformer and Log Transform — Topics Guide (Day 30)

> Local main: `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` (executed, outputs kept).
> Simplified read-along: `DAY30_Function-Transformer-preview-OUT-OF-SCOPE_simplified.ipynb` (same code + outputs, easy markdown).
> Deep notes + scores: `topic_notes.md` (do not invent new numbers — all accuracies below are quoted from it / the notebook outputs).
> Secondary (gaps only): CampusX `day30-function-transformer`.

---

## 1. What

**Data transformation** changes the *shape* of feature values — not the rows, not the order.

**`FunctionTransformer(func)`** takes any ordinary Python function and wraps it into a scikit-learn transformer (something that has `.fit()` / `.transform()`), so it can sit inside `ColumnTransformer` and `Pipeline` like `StandardScaler` does.

**Log transform** is the specific function used in this notebook:

```text
log1p(x) = log(1 + x)
```

It rewrites every value in a column. Big values shrink hard. Small values barely move. A right-skewed column becomes more symmetric.

**Q-Q plot** (`stats.probplot(x, dist="norm", plot=plt)`) draws sample values against normal values. If the points hug the 45° line → "looks normal".

---

## 2. Why

Titanic `Fare` looks like this:

```text
Fare ≈ [0, 7, 8, 8, 10, 15, 20, 25, 30, 500, 800, 1000]
```

Most rows are tiny. A few are giants → **right skew**.

- **Linear models (LogisticRegression):** feel those giants in the squared loss.
- **Distance models:** one big fare dominates the metric.
- **Trees:** split on order, so they care less — but the tail still wastes clean split points.

Goal: compress the tail **without deleting rows**.

Your notebook's scoreboard (from `topic_notes.md`):

| Model | Raw features | After log1p |
|---|---|---|
| LogisticRegression | 0.7207 | 0.7374 |
| DecisionTree | 0.6816 | 0.7095 |

Both went up. That is the whole reason for the transform.

---

## 3. How

### 3a. `log1p` (the Day-30 tool)

```text
log1p(x) = log(1 + x)
```

Small numeric example:

```text
x     = 0,    10,    100,   1000
log1p = 0,    2.40,  4.62,  6.91
```

Ratio 1000 : 10 = **100×** in raw data becomes 6.91 : 2.40 ≈ **2.9×** after log.
Compression grows with size — that is tail-taming.

Why `log1p` and not `log`? Fares contain **0**, and `log(0) = -inf` poisons the column. `log1p(0) = 0` keeps every row finite.

### 3b. `FunctionTransformer` (the wrapper)

```python
trf = FunctionTransformer(func=np.log1p)   # stores the function, runs nothing
X_train_trans = trf.fit_transform(X_train) # applies log1p now
X_test_trans  = trf.transform(X_test)      # applies the same log1p
```

- The wrapper is **stateless**: `log1p` has no mean, no categories, no counts to learn. `fit` learns nothing.
- To log only `Fare` and pass `Age` untouched:

```python
trf2 = ColumnTransformer(
    [('log', FunctionTransformer(np.log1p), ['Fare'])],
    remainder='passthrough'
)
```

Output is a plain numpy array, **transformed block first, passthrough last** → column 0 = logged Fare, column 1 = raw Age, column names gone.

### 3c. Box-Cox / Yeo-Johnson (outlook — Day 31)

Log fixes λ = 0 by hand. `PowerTransformer` **searches for the best λ** from data (see `DAY_31/topics.md`).

---

## 4. Syntax breakdown

### `func=` stores a reference — it does NOT compute

```python
trf = FunctionTransformer(func=np.log1p)   # ✅ stores the function object (a reference)
trf = FunctionTransformer(func=np.log1p(X))# ❌ calls it NOW, on X, and passes the RESULT array
```

Difference in one line:

```text
np.log1p     → a function (you can call it later, on any X)
np.log1p(X)  → an array (already computed, on one specific X)
```

`func=` wants the **recipe**, not the cooked dish. sklearn calls the recipe itself later inside `transform`.

### `fit_transform` vs `transform`

```python
X_train_trans = trf.fit_transform(X_train)  # train: fit + apply
X_test_trans  = trf.transform(X_test)       # test:  apply ONLY
```

> **⚠ Your notebook's cell uses `trf.fit_transform(X_test)`.** That is a refit on test data. It is harmless *here only because log1p is stateless* (refitting learns nothing). With an imputer/scaler/encoder the same line would leak test statistics.

### `ColumnTransformer` tuple shape

```python
[(name, transformer_object, column_selector), ...]
remainder='passthrough'   # keep unlisted columns; default 'drop' would delete Age
```

### Domain rules

| Function | Allowed input |
|---|---|
| `np.log1p` | x ≥ 0 (never negatives) |
| `np.log` | x > 0 (breaks at 0) |
| Box-Cox (Day 31) | **x > 0 strictly** |
| Yeo-Johnson (Day 31) | any real number |

---

## 5. Implementation

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# 1. data
df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\train.csv")
df = df[['Survived', 'Age', 'Fare']]
df['Age'] = df['Age'].fillna(df['Age'].mean())
X = df.iloc[:, 1:3]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=34)

# 2. baseline models on raw features
clf, clf2 = LogisticRegression(), DecisionTreeClassifier()
clf.fit(X_train, y_train); clf2.fit(X_train, y_train)

# 3. wrap log1p and transform
trf = FunctionTransformer(func=np.log1p)
X_train_trans = trf.fit_transform(X_train)
X_test_trans  = trf.transform(X_test)          # correct line (notebook has fit_transform)

# 4. fair rematch on logged features
clf.fit(X_train_trans, y_train); clf2.fit(X_train_trans, y_train)

# 5. selective routing: log Fare only, Age passes through
trf2 = ColumnTransformer([('log', FunctionTransformer(np.log1p), ['Fare'])],
                         remainder='passthrough')
X_train_trans2 = trf2.fit_transform(X_train)
X_test_trans2  = trf2.transform(X_test)

# 6. reusable helper — takes any function OBJECT
def apply_transform(transform):
    X = df[['Age', 'Fare']]; y = df['Survived']
    trf = ColumnTransformer([('log', FunctionTransformer(transform), ['Fare'])],
                            remainder='passthrough')
    X_trans = trf.fit_transform(X)
    clf = LogisticRegression()
    print("Accuracy", np.mean(cross_val_score(clf, X_trans, y, scoring='accuracy', cv=15)))
    # ... Q-Q before/after plot ...

apply_transform(np.log1p)
apply_transform(np.sin)     # deliberate bad idea — see Observations
```

---

## 6. Execution walkthrough (your notebook, in order)

1. **Imports** — nothing runs yet.
2. **Load + slice + impute + split** — 891 rows → `X` (891, 2), `y` (891,); 80/20 split, `random_state=34`.
3. **Distplot + Q-Q for Age and Fare** — `distplot` prints a deprecation warning (dead in seaborn ≥ 0.14; use `histplot`). Fare's Q-Q bends hard away from the diagonal → skewed. Age is closer to the line.
4. **Baseline fit** → `Accuracy LR 0.720670...`, `Accuracy DT 0.681564...`
5. **`trf = FunctionTransformer(func=np.log1p)`** — stores the reference. No output, no computation.
6. **Transform train, (re)fit test** — `fit_transform(X_test)` on the test line (bug, see Observations).
7. **Rematch** → `Accuracy LR 0.737430...`, `Accuracy DT 0.709497...` — both up. Log helped.
8. **Transform-all-then-CV** → `LR 0.677005...`, `DT 0.662109...` (15 folds). Lower than the holdout — different protocol, not a contradiction; also preprocessing ran *before* CV (leaky shape — fix with Pipeline).
9. **Q-Q before/after for Age and Fare** — judge by shrinking distance to the diagonal; Fare moves most.
10. **Selective `ColumnTransformer`** → `Accuracy LR 0.715083...`, `Accuracy DT 0.703910...` — only Fare logged. Tree still gains, linear dips below baseline (Age stayed raw).
11. **Second CV cell** → `LR 0.667947...`, `DT 0.667947...` — **identical to 16 decimals = copy-paste bug** (the DT line evaluates `clf`, not `clf2`).
12. **`apply_transform(np.sin)`** → `Accuracy 0.618455...` — far under baseline ~0.72. Sine is periodic: it folds different fares onto the same value and destroys the signal. A negative result with a positive lesson.

---

## 7. Observations

### Refit-on-test

```python
X_test_trans = trf.fit_transform(X_test)   # notebook line
X_test_trans = trf.transform(X_test)       # correct line
```

- **Stateless** (`FunctionTransformer(np.log1p)`): refit learns nothing → numbers are still valid, but the *habit* is wrong.
- **Stateful** (imputer, `StandardScaler`, `PowerTransformer`, encoders): the same line fits on test rows → **data leakage**, inflated scores. Day 31's `PowerTransformer` is exactly this trap.

### Stateless vs stateful

| | `FunctionTransformer(np.log1p)` | `StandardScaler` / `PowerTransformer` |
|---|---|---|
| Learns from data? | No — fixed recipe | Yes — mean/σ or λ |
| `fit` does | nothing (checks only) | estimates parameters |
| Fit on test? | harmless but bad habit | **leakage** |
| sklearn term | stateless | stateful |

### Other traps seen in this notebook

1. Transforming all of `X` before `cross_val_score` → preprocessing must live *inside* each fold (Pipeline).
2. Reading transformed columns by old names → after routing, output is a nameless numpy array; column 0 = logged Fare by construction (`X_trans[:, 0]`).
3. Identical 16-decimal CV scores → you scored the same estimator twice.
4. Choosing a transform by vibes → compare holdout **and** CV against baseline; keep it only if both improve.

---

## 8. Connection with prior

- **Day 24–25 (scaling):** `StandardScaler` recenters/rescales but *keeps* skew (`z = (x − μ)/σ`). Log *changes the shape*. They are complementary, not rivals — and PowerTransformer (Day 31) does both in one object.
- **Day 29 (pipelines):** same "define once, reuse everywhere" pattern — the transform belongs inside `ColumnTransformer` → `Pipeline`, so split, CV, and predict all repeat training exactly.
- **Day 31 (next):** log is one fixed choice of a knob (λ = 0 style). `PowerTransformer` **learns** the knob per column: Box-Cox (strictly positive, `X > 0`) or Yeo-Johnson (any sign, sklearn default). See `DAY_31/topics.md`.
- **Interview one-liner:** *"`func=` stores a function object for later use; `np.log1p(X)` calls it immediately — that is why the first composes into a pipeline and the second does not."*

### From `topic_notes.md` / deep notes

# FunctionTransformer and the Log Transform — Study Notes (Day 30)

> Reference: `day30-function-transformer` in the CampusX repo. Your work: `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` (29 cells, executed) and its explained copy `DAY30_FunctionTransformer_documented.ipynb`. All numbers below are your actual outputs.
> Explanation section follows Rule Set 5 (same “from zero, no bragging” ladder as the Box-Cox walkthrough you liked).

## Explanation — Log transform from zero

### 1. Why we need it

Look at Titanic-like fares:

```text
Fare ≈ [0, 7, 8, 8, 10, 15, 20, 25, 30, 500, 800, 1000]
```

Most rows are small. A few are huge.

- Mass on the left, rare giants on the right → **right skew**.
- Linear models feel those giants in squared error.
- Distance models let one big fare dominate the metric.
- Trees split on order, so they care less — but the tail still wastes clean split points.

**Goal:** compress the tail so the column is more symmetric — without dropping rows.

---

### 2. What is the log transform?

Rewrite each positive value with a logarithm. Big numbers shrink hard; small numbers barely move.

```text
x_new = log(1 + x)     # log1p — the version you use in code
```

Why `log1p` and not `log`? Fares include **0**, and `log(0) = −inf` poisons the column. `log1p(0) = 0` keeps every row finite.

---

### 3. Understanding the “knob” (and why Day 31 exists)

Log is one fixed choice. The bigger family is a power dial (λ):

| λ | Effect |
|---|--------|
| 1 | almost no change |
| 0.5 | square-root-like |
| 0 | **log** (your Day-30 tool) |
| −1 | reciprocal-like, harder compression |

Worked check:

```text
Fare = 1000,  log1p(1000) ≈ 6.91
Fare = 10,    log1p(10)    ≈ 2.40
```

Ratio 1000:10 = 100× becomes ≈ 6.91:2.40 ≈ 2.9× — compression grows with size. That is tail-taming.

**Day 31 preview:** Box-Cox / `PowerTransformer` **searches for the best λ** instead of fixing λ = 0.

---

### 4. How do we choose the function in practice?

You do not invent a new transform each morning. You pick a candidate (`log1p`, sqrt, later Box-Cox), wire it into a transformer, and let **holdout + cross-validation** tell you if it helped.

```python
from sklearn.preprocessing import FunctionTransformer

trf = FunctionTransformer(func=np.log1p)   # stores the function; does not run it yet
```

---

### 5. Important conditions

- Only for values where the function is defined: `log1p` wants **x ≥ 0** (never negatives).
- It **reduces** skew; it does **not** prove normality (use histogram + Q-Q, not vibes).
- Fit/transform discipline: on test data call `transform`, never refit a stateful preprocessor.

---

### 6. Implementation (sklearn)

```python
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

# Day 30 — fixed function
FunctionTransformer(np.log1p)

# Day 31 — learned power (outlook)
PowerTransformer(method="yeo-johnson")   # default; handles zeros/negatives
PowerTransformer(method="box-cox")       # strictly positive columns only
```

---

### 7. Log vs Box-Cox (bridge to Day 31)

| | Log / `log1p` (Day 30) | Box-Cox (Day 31) |
|--|------------------------|------------------|
| Knob | fixed (λ = 0 style) | optimal λ by MLE |
| Domain | ≥ 0 for log1p | **X > 0 only** |
| Flexible | simple | more flexible |
| sklearn | `FunctionTransformer` | `PowerTransformer` |

**Log is the special case of Box-Cox when λ = 0.**

---

### 8. Check understanding

```text
salary = [20000, 25000, 30000, 35000, 500000]
```

**Question:** Why would a log (or Box-Cox) transform help here?

**Think:** 500000 is ~15× the next value. After `log1p`, that gap shrinks so one salary no longer owns the whole scale. Skew drops; scale-sensitive models stop overreacting to a single row.

---

### 9. Where it lives in your workflow

Same reusable pattern as pipelines day:

```python
def create_transformer(transform):
    return ColumnTransformer(
        [("trans", FunctionTransformer(transform), ["Fare"])],
        remainder="passthrough",
    )

pipe = Pipeline([
    ("prep", create_transformer(np.log1p)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

| Train-test split | Cross-validation |
|-------------------|------------------|
| One split, one score | Many folds, mean score |
| Faster | Slower, steadier |
| Final look at unseen data | Fair **comparison** of transforms/models |

**Use both:** compare transforms with CV → pick the winner → one final holdout score. Define preprocessing once inside the Pipeline; holdout, CV, and prediction all reuse it.

**Lock-in:** wrap the transform once in `FunctionTransformer` + `Pipeline` — less repetition, fewer mistakes, same object for split, CV, and predict.

---

## 1. Concept and definition

- **Data transformation** reshapes feature *values* (here: compressing a long tail). It differs from cleaning (repairing entries), encoding (text → numbers), and scaling (matching magnitudes): transformation changes the distribution's shape.
- **`FunctionTransformer(func)`** adapts any ordinary element-wise function into a scikit-learn transformer with `fit`/`transform`, so it composes inside `ColumnTransformer` and `Pipeline`.
- **Log transform** (`log1p`) maps `x → log(1+x)`: huge values shrink dramatically, small values barely move — a right-skewed column becomes more symmetric.
- **Q-Q plot** (`scipy.stats.probplot(x, dist="norm", plot=plt)`) plots sample quantiles against theoretical normal quantiles; points on the 45° reference line mean "looks normal".

## 2. The problem it solves

Titanic Fare piles near 0 with rare huge values. That tail dominates distances (hurting LogisticRegression) and wastes tree splits. The log compresses the tail so models see signal instead of scale. Your scoreboard: LogReg 0.7207 → 0.7374, Tree 0.6816 → 0.7095.

## 3. How it works internally

- `FunctionTransformer(func=np.log1p)` **stores the function reference** — nothing computes at construction, and `fit` learns nothing (stateless: `log1p` has no means, categories, or counts). `transform` simply applies `log1p` element-wise and returns the array.
- `ColumnTransformer([("log", FunctionTransformer(np.log1p), ["Fare"])], remainder="passthrough")` applies the logger to `Fare` only and passes `Age` through untouched. Output is a plain numpy array ordered **transformed block first, passthrough last** — column 0 is logged-Fare, column 1 is raw Age, names gone. Your helper relies on this (`X_trans[:, 0]`).
- `cross_val_score(est, X, y, scoring="accuracy", cv=15)` rotates 15 train/test folds and returns 15 scores; the notebook reports their mean.
- `stats.probplot` sorts the sample, pairs each value with the matching normal quantile, and draws both plus the quartile reference line.

## 4. Mathematical foundation

- Standardization (contrast): `z = (x − μ) / σ`, with `x` original value, `μ` fitted feature mean, `σ` fitted feature standard deviation, `z` standardized value. It recenters but keeps skew — hence a different tool from the log.
- Log: `log1p(x) = log(1 + x)`. Small example: fares 0, 10, 100, 1000 → `log1p` ≈ 0, 2.40, 4.62, 6.91. The 1000:10 ratio (100×) becomes 6.91:2.40 (≈2.9×): compression grows with size, which is exactly tail-taming.
- Why `log1p` over `log`: `log(0)` is negative infinity and fares contain 0; `log1p(0) = 0` keeps every row finite. Never use either on negative values.

## 5. Syntax and API reference

```python
FunctionTransformer(func=None, inverse_func=None, validate=False, ...)
ColumnTransformer([(name, transformer, columns)], remainder="passthrough")
stats.probplot(x, dist="norm", plot=plt)
cross_val_score(estimator, X, y, scoring="accuracy", cv=15)
```
- `func`: the wrapped function (a reference like `np.log1p`, not a call like `np.log1p(X)`).
- Transformer tuples are always `(string name, transformer object, column selector)`.
- `remainder="passthrough"` keeps unlisted columns; default `"drop"` would silently remove Age.
- `cv=15`: fifteen folds; higher than the usual 5, so slower but steadier means.

## 6. Code walkthrough (your cells, in order)

- **Cells 7–8** (Style-A lines in the documented copy): raw-string CSV path; double-bracket 3-column slice; whole-column mean imputation (simplification — mean sees future test rows; strict habit is train-only); positional `iloc` split into `X` (891, 2) and `y` (891,); 80/20 split with fixed shuffle 34.
- **Cells 9–10**: per-column histogram + Q-Q panels; the `distplot` deprecation warning is real (dead in seaborn ≥ 0.14 — corrected cell uses `histplot`).
- **Cells 11–12**: fresh estimators → fit on train → predict on test → baselines 0.7207 / 0.6816.
- **Cells 13–14**: wrap `log1p`; train line correct, test line refits (documented bug — harmless here *only* because the transformer is stateless; corrected cell uses `transform`).
- **Cells 15–16**: fair rematch on logged features → 0.7374 / 0.7095, both up.
- **Cell 17**: transform-all-then-validate → 15-fold means 0.6770 / 0.6621 (lower than holdout = different protocol, not contradiction) plus the same fit-before-split caveat; Pipeline cell shows the fold-honest shape.
- **Cells 18–19**: Q-Q before/after — judge by shrinking distance to the diagonal; Fare should move most.
- **Cell 20**: selective routing — logged-Fare col 0, raw Age col 1; correct `transform`-on-test; 0.7151 / 0.7039 (selective effects: tree still gains, linear dips under baseline with Age raw).
- **Cell 21**: the `clf`-twice bug — identical 0.6679/0.6679 outputs confess it; one-token corrected cell (value unknown until run).
- **Cells 25–26**: `apply_transform(transform)` takes a *function object*; `np.sin` scores 0.6185, far under baseline — periodic folding destroys monotonic signal. Negative result, positive lesson.

## 7. Input and output

- `FunctionTransformer.transform`: array-like in (any non-negative values for `log1p`), same-shape array out.
- Selective `ColumnTransformer` out: numpy array, **reordered** (transformed block, then passthrough), dtypes unified to float — downstream code must use positions (`X_trans[:, 0]`), never old names.
- `accuracy_score`: fraction correct in [0, 1]; `cross_val_score`: one score per fold — report the mean *and* remember the spread.

## 8. Differences between related concepts

- `fit()` (learn parameters) vs `transform()` (apply them) vs `fit_transform()` (both at once — train only).
- `StandardScaler` (recenter/rescale, keeps skew, needs μ/σ → stateful) vs log transform (reshapes skew, stateless) — complementary, not rivals.
- `np.log` (breaks at 0) vs `np.log1p` (safe at 0) vs `np.sin` (periodic — wrong tool for skew).
- `FunctionTransformer` (wraps a fixed function, learns nothing) vs `PowerTransformer` (learns the exponent from data — Day-31 outlook).
- Histogram (shape at a glance) vs Q-Q plot (normality test against a reference line).
- Holdout score (one split, 0.7374) vs CV mean (fifteen splits, 0.6770) — different protocols, never directly compared.

## 9. Common mistakes and debugging

1. `fit_transform` on test (cell 14) — always `transform` with the train-fitted object.
2. Evaluating the wrong estimator (cell 21) — identical scores to 16 decimals mean one model ran twice.
3. `distplot` on modern seaborn — `ImportError`/failure; use `histplot`.
4. `log` on zeros/negatives — `-inf`/NaN silently poisons the column; prefer `log1p` for ≥0 data.
5. Transforming all of `X` before `cross_val_score` — preprocessing must live inside each fold (Pipeline).
6. Reading transformed columns by old names/positions — after routing, column 0 is Fare-by-construction; verify shapes.
7. Treating a Q-Q plot as proof of normality — it is visual evidence with limits, not a statistical guarantee.

## 10. Practical ML application

Log-family transforms belong wherever right-skewed magnitudes feed scale-sensitive models (fares, prices, counts, incomes): try log, compare holdout + CV against baseline, keep it only with improvement on both. Trees often gain less but can still gain (yours: +0.028). In production the transform lives inside the `ColumnTransformer`/`Pipeline` so inference repeats training exactly — never as a loose pre-step.

## 11. Interview and conceptual questions

1. Why `log1p` and not `log` for Fare?
2. What does `func=` store, and how does that differ from calling `np.log1p(X)`?
3. Your test line refits the transformer — why are the scores still valid here, and when would the same line corrupt results?
4. After selective routing, which output column is Fare and why? What type are the outputs?
5. Two models print identical 16-decimal CV scores. Diagnose it.
6. Sine scores 0.6185 vs baseline ~0.72. What does that teach about choosing transforms?
7. Holdout says 0.7374, CV says 0.6770. Contradiction? Explain.
8. How would you make cell 17's validation fold-honest?

<details><summary>Answers</summary>

1. Fares include 0; `log(0) = -inf` would poison the column, while `log1p(0) = 0` keeps all rows finite.
2. `func=` stores a function reference for later application inside `fit`/`transform`; `np.log1p(X)` computes immediately outside any transformer, so it cannot compose into pipelines.
3. `log1p` is stateless — refitting learns nothing, so numbers equal a correct `transform`. With a stateful transformer (imputer/scaler/encoder) the same line would leak test statistics.
4. Column 0, because transformed blocks precede passthrough columns; outputs are a nameless numpy array, so positions (not names) must be used.
5. Copy-paste bug: the DT line evaluates `clf` instead of `clf2` — two different algorithms never tie to 16 decimals by chance.
6. `sin` is periodic and folds distinct fares onto identical values, destroying monotonic signal. Transforms must match data shape; negative results are evidence.
7. No contradiction: different protocols — one fixed split vs mean over 15 folds with different compositions.
8. Wrap unfitted preprocessing + estimator in a `Pipeline` and pass it (with raw `X`) to `cross_val_score`, as the corrected cell shows.
</details>

## 12. Revision sheet

- Normality: histogram shape, Q-Q diagonal closeness, `skew()` ≈ 0.
- `log1p(x) = log(1+x)` for right-skewed ≥0 data; never `log` on zeros, never logs on negatives.
- `FunctionTransformer(func)` wraps without running; stateless → `fit` learns nothing.
- `ColumnTransformer([(name, trf, cols)], remainder="passthrough")`; outputs reordered, nameless.
- Discipline: `transform` (never refit) on test; preprocessing inside CV folds (Pipeline).
- Your numbers: baseline 0.7207/0.6816 → full-log 0.7374/0.7095 → selective 0.7151/0.7039 → sine 0.6185; CV means 0.6770/0.6621 (different protocol).
- Bugs you can now spot: refit-on-test, wrong-estimator scoring, deprecated `distplot`, pre-fold fitting.

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
