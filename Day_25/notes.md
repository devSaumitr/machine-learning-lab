# Day_25 — Normalization / MinMaxScaler — Topics Guide (Day 25)

> Deep notes for **Day_25**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# Normalization / MinMaxScaler — Topics Guide (Day 25)

> Primary: `DAY25_Normalization-MinMaxScaler.ipynb` (main); stub: `DAY25_Normalization-stub.ipynb`.
> Secondary: CampusX `day25-normalization` (`reference_notebook.ipynb`).
> Dataset: `datasets/wine_data.csv`.

## 1. What is it?

**Min-max normalization** squeezes each numeric column into a fixed range, usually **[0, 1]**. It is one kind of **feature scaling** (Day 24 was standardization/z-scores; this day is bounded min-max and related scalers).

## 2. Why do we need it?

Distance-based and gradient-based models care about units. Salary in thousands vs Age in years: without scaling, salary dominates. Min-max puts every feature on the same “ruler” when you know the bounds.

## 3. How does it work?

For one feature with minimum \(x_{min}\) and maximum \(x_{max}\):

\[
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
\]

So \(x_{min} \to 0\) and \(x_{max} \to 1\).

**Hand example** — feature `[10, 20, 30, 40, 50]`, value `30`:

```text
(30 - 10) / (50 - 10) = 20/40 = 0.5
whole column → [0, 0.25, 0.5, 0.75, 1]
```

Scale **per column**, never one min/max for the whole DataFrame.

**Related scalers** (same day’s overview):

| Scaler | Idea | Formula (sketch) |
|--------|------|------------------|
| MinMax | map to [0,1] (or custom) | `(x-min)/(max-min)` |
| Mean normalization | centre by mean, scale by range | `(x-mean)/(max-min)` — rare in sklearn |
| MaxAbs | divide by max \|x\| | good for **sparse** data |
| Robust | centre/scale by median & IQR | resistant to **outliers** |

## 4. Syntax breakdown

### Manual vs library

```python
# manual
(x - df.min()) / (df.max() - df.min())

# sklearn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_sc = scaler.fit_transform(X_train)  # learns min/max from train
X_test_sc  = scaler.transform(X_test)       # reuses train min/max
```

| Piece | Meaning |
|-------|---------|
| `MinMaxScaler()` | create object — **no data yet** |
| `fit_transform(X)` | `fit` then `transform` on the **same** X (train only) |
| `transform(X)` | apply learned `data_min_`, `data_max_` |
| `feature_range=(a,b)` | output interval, default (0,1) |
| `scaler.data_min_` / `data_max_` | learned bounds (attributes after fit) |

**fit vs transform:** `fit` **learns** parameters; `transform` **applies** them. Test data must only be `transform`ed.

### Pipeline (leak-free habit)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", MinMaxScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

Tuples: `(name, object)` — name for humans, object with `fit`/`transform`/`predict`.

## 5. Practical implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/wine_data.csv")
# columns depend on the file: often Class + chemical features
X = df.drop(columns=[df.columns[0]])  # or explicit feature list
y = df[df.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_sc, y_train)
print(accuracy_score(y_test, model.predict(X_test_sc)))
```

## 6. Execution walkthrough

1. Load wine CSV; separate X and y.  
2. **Split first** — test set must stay unseen.  
3. `fit_transform(X_train)` → store train min/max inside `scaler`.  
4. `transform(X_test)` → same formula; test values can fall **outside** [0,1] if beyond train range (expected).  
5. Fit model on scaled train; score on scaled test.  
6. Notebook may also show hand calculations and mean-norm / MaxAbs / Robust formulas in markdown — same pattern: formula → small numbers → sklearn class.

## 7. Important observations

- **Split before scale.** Fitting on all data leaks test min/max into training (same rule as Day 24).  
- **Never `fit_transform(X_test)`** as the default — different bounds than train.  
- **Outliers move min/max** a lot → whole feature compresses into a tiny band (why RobustScaler exists).  
- **Test values outside [0,1]** are normal if test has more extreme values than train.  
- **Identical values** in a feature → range 0 → division by zero risk; sklearn handles degenerate cases carefully, but a constant feature is useless anyway.  
- **Scale X, not y** by default — transforming the target changes the loss (separate, intentional choice).  
- **MinMax vs StandardScaler:** bounded vs unbounded; both need train-only fit.  
- **Row-wise `Normalizer`** (unit length per row) is a **different** API — do not confuse with column min-max.  
- Stub notebook is incomplete by design — revise from the main notebook.

## 8. Connection with previous concepts

- **Day 24:** z-score standardization — sibling tool, same leakage discipline.  
- **Day 19–21:** ranges and skew you saw in EDA explain **why** a column needs scaling.  
- **Day 28–29:** MinMaxScaler becomes a stage inside `ColumnTransformer` / `Pipeline`.  
- **Day 30–31:** power transforms change **shape**; scaling changes **range** — different jobs, can compose.

## Common mistakes

1. Scale before `train_test_split`.  
2. `fit_transform` on train and separately `fit_transform` on test.  
3. Using MinMax when extreme outliers wreck the range — prefer Robust.  
4. Forgetting `max_iter` on LogisticRegression after scaling (convergence warnings).  
5. Confusing **column** min-max with **row** L2 normalization (`sklearn.preprocessing.Normalizer`).

## Mini cheat sheet

```text
CONCEPT     → put features on a shared bounded scale
FORMULA     → (x - min) / (max - min)  → usually [0,1]
FIT RULE    → fit on train only; transform train and test
API         → MinMaxScaler().fit_transform(train); .transform(test)
PAIR WITH   → Pipeline so CV never leaks
TRAP        → outliers move min/max; test can exit [0,1]
```

### From `topic_notes.md` / deep notes

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
