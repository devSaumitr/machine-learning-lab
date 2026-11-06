# Day_28 — ColumnTransformer — Topics Guide (Day 28)

> Deep notes for **Day_28**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# ColumnTransformer — Topics Guide (Day 28)

> Primary: `DAY28_Column-Transformer.ipynb`.
> Secondary: CampusX `day28-column-transformer` (`reference_notebook.ipynb`).
> Dataset: `datasets/covid_toy.csv` (age, gender, fever, cough, city → has_covid).

## 1. What is it?

`ColumnTransformer` applies **different preprocessing to different columns**, then glues the results into one array. It is the object that says: “impute fever here, ordinal-encode cough there, one-hot gender/city, pass age through.”

## 2. Why do we need it?

Numeric columns need imputation/scaling. Text columns need encoding. Doing that by hand means tracking every block and `np.concatenate` order — easy to break, hard to reuse. One `ColumnTransformer` declares the plan once.

## 3. How does it work?

You write a list of **triples**:

```python
ColumnTransformer(
    transformers=[
        ("name1", transformer_object1, ["col_a"]),
        ("name2", transformer_object2, ["col_b", "col_c"]),
    ],
    remainder="passthrough"   # or "drop"
)
```

```text
raw DataFrame
    │
    ├─ tnf1 on selected cols ──► block A
    ├─ tnf2 on selected cols ──► block B
    └─ remainder ──────────────► block C (optional)
    │
    ▼
np.concatenate([A, B, C])  → one numpy array (order = declaration order)
```

`fit` runs `fit` on each sub-transformer (learns means, categories, …).  
`transform` runs each `transform` and concatenates.  
`fit_transform` = fit then transform in one call (**train only**).

## 4. Syntax breakdown

### Triple `(name, transformer, columns)`

| Part | Type | Example |
|------|------|---------|
| name | str | `"imputer"`, `"onehot"` |
| transformer | object with fit/transform | `SimpleImputer()`, `OneHotEncoder()` |
| columns | list of names or indices | `["fever"]`, `[1, 6]` |

Names must be unique — they identify steps in `named_transformers_`.

### `remainder`

| Value | Meaning |
|-------|---------|
| `"drop"` | ignore columns not listed (sklearn default) |
| `"passthrough"` | append unlisted columns as-is |

Default `"drop"` **silently removes** columns — a common surprise.

### fit vs transform on test

```python
X_train_p = ct.fit_transform(X_train)  # learns
X_test_p  = ct.transform(X_test)       # applies only
# X_test_p = ct.fit_transform(X_test)  # WRONG — leaks test statistics/categories
```

## 5. Practical implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("datasets/covid_toy.csv")
X = df.drop(columns=["has_covid"])
y = df["has_covid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Manual-style blocks (as in the notebook) vs one CT:
ct = ColumnTransformer(
    transformers=[
        ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
        ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"),
         ["gender", "city"]),
    ],
    remainder="passthrough",
)

X_train_ct = ct.fit_transform(X_train)
X_test_ct  = ct.transform(X_test)
print(X_train_ct.shape, X_test_ct.shape)
```

**Manual path (what the notebook shows first):** impute fever → ordinal cough → one-hot → `np.concatenate` in a fixed order. Same math, more bookkeeping.

## 6. Execution walkthrough

1. Load `covid_toy.csv`; split train/test.  
2. **Without CT:** each transformer runs on a slice; you `hstack`/`concatenate` yourself; column order is your responsibility.  
3. **With CT:** declare triples; one `fit_transform(X_train)` learns all sub-steps and returns one array.  
4. `transform(X_test)` reuses learned imputer values and encoder categories.  
5. Compare shapes/contents of manual vs CT outputs (notebook may show a deliberate column-choice mismatch between paths — compare carefully, do not assume identical features).  
6. Preserved bug reminder: calling `fit_transform` on **test** twice → leakage + possibly different categories; correct is `transform`.

## 7. Important observations

- **Output is a numpy array** — column **names disappear**; downstream steps use **positions**.  
- **Transformer blocks run in list order** — that defines output column order.  
- **`remainder="drop"` default** drops Age-like columns if you forget them.  
- **Index vs name selectors:** names work on DataFrames; after the first transform, later stages often see arrays and need indices (critical in Day 29 pipelines).  
- **No model inside CT** — CT is preprocessing only; estimator goes in a `Pipeline` (Day 29).  
- **Split before `fit_transform`** — same leakage rule as scalers.  
- Notebook quirks to notice: `train_test_split` without `random_state` (non-reproducible); possible double imports; test-side `fit_transform` (bug preserved for learning).

## 8. Connection with previous concepts

- **Days 24–27:** StandardScaler, MinMax, Ordinal/Label/OneHot are exactly the **transformer objects** you plug into triples.  
- **Day 19–21:** EDA told you which columns need which treatment.  
- **Day 29:** the same `ColumnTransformer` becomes stage `trf1` inside `Pipeline`, next to impute/scale/select/model.  
- **Day 30+:** `FunctionTransformer` / `PowerTransformer` also sit inside CT/Pipeline.

## Common mistakes

1. Forgetting `remainder` and losing columns.  
2. Duplicate triple names.  
3. `fit_transform` on test.  
4. Assuming CT output keeps DataFrame column names (it usually does not).  
5. Mixing **manual concatenate order** with CT order and then comparing the wrong columns.  
6. Putting the estimator inside `ColumnTransformer` — wrong tool; use `Pipeline`.

## Mini cheat sheet

```text
CONCEPT     → per-column preprocessing, one object
SYNTAX      → [("name", trf, cols), ...], remainder=...
FIT RULE    → fit_transform(train); transform(test)
OUTPUT      → numpy array, declaration order, names gone
NEXT        → wrap CT + model in Pipeline (Day 29)
TRAP        → default remainder="drop"; fit on test leaks
```

### From `topic_notes.md` / deep notes

# ColumnTransformer (Day 28)

> Playlist: Day 28 Column Transformer / How to use ColumnTransformer in Sklearn.
> Notebook: `DAY28_Column-Transformer.ipynb` (covid_toy.csv: age/gender/fever/cough/city→has_covid; WITHOUT-CT (80,7) vs WITH-CT (20,23)).

## 1. Learning Objectives

- Reproduce the manual path (impute → ordinal → one-hot → concatenate) and the `ColumnTransformer` path.
- Write `(name, transformer, columns)` triples with `remainder="passthrough"`.
- Explain why `transform` (not `fit_transform`) belongs on test data.
- Map this notebook to Day-29 `trf1`.

## 2. Why This Topic Matters

Different columns need different math (fever → median, cough → ordinal, gender/city → one-hot, age → pass). Doing it by hand means tracking every block and concatenation order — the exact pain `ColumnTransformer` removes, and the exact object inside Day-29 pipelines.

## 3. Prerequisites

- Days 26–27 encoders; `SimpleImputer`; `train_test_split`; `np.concatenate`/`hstack`.

## 4. First-Principles Explanation

List, per column group, "which transformer touches it". `ColumnTransformer` runs all groups in parallel and glues the outputs into one array. `remainder="passthrough"` keeps unlisted columns (age) as-is.

## 5. Formal Definition

- **`ColumnTransformer(transformers=[(name, transformer, columns)], remainder=...)`**: applies each transformer to its columns, concatenates outputs. `fit` learns all sub-transformers on train; `transform` applies them.

## 6. How It Works Internally

- WITHOUT-CT: `SimpleImputer(fever)` → `OrdinalEncoder(["Mild","Strong"])` on cough → `OneHotEncoder(drop="first")` on gender/city → age passthrough → `np.concatenate` → (80,7). You track order manually.
- WITH-CT: `tnf1=(imputer, [fever])`, `tnf2=(ordinal, [cough])`, `tnf3=(one-hot, [age, gender])`, `remainder="passthrough"` → `fit_transform(train)` / `fit_transform(test)` → (20,23). Preserved quirks: CT one-hots age+gender while manual one-hots gender+city (compare, don't assume identical); test uses `fit_transform` twice (bug — see §12); split has no `random_state` (non-reproducible); only tail rows show execution (71,78,79,81).

## 7. Step-by-Step Examples

covid_toy row → tnf1 fills fever → tnf2 maps Mild/Strong→0/1 → tnf3 expands gender(+age per this notebook) → remainder passes the rest → one output row. Same row through manual path needs three separate transforms + manual `concatenate` in the same order.

## 8. Code Examples

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ct = ColumnTransformer([
    ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
    ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
    ("ohe_cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["gender", "city"]),
], remainder="passthrough")

Xtr = ct.fit_transform(X_train)  # learn on TRAIN
Xte = ct.transform(X_test)       # apply to test
```

## 9. Output Interpretation

- (80,7) manual vs (20,23) CT reflect different column choices/splits in this demo, not a universal ratio — read the transformer's column lists to explain any shape.
- Executed tail indices (71,78,79,81) only; `null` above means cells never ran — re-run top-to-bottom.

## 10. Important Parameters and Variations

| Arg | Meaning |
|---|---|
| `(name, transformer, columns)` | name for debugging, estimator, column selector (names or indices) |
| `remainder="passthrough"/"drop"` | keep or drop unlisted columns |
| `sparse_threshold` | dense vs sparse output |
| `verbose_feature_names_out` | readable output names via `get_feature_names_out()` |

Day-29 uses index-based columns on numpy arrays (`trf1` imputes Age[2]/Embarked[6]) — same object, different selector style.

## 11. Practical Use Cases

- Any mixed-type table: one `ct` replaces pages of manual block-tracking; embed as pipeline step `trf1`.

## 12. Common Mistakes and Misconceptions

1. `fit_transform` on test (twice here) → leakage + mismatched categories. Fix: `transform`.
2. Column mismatch manual-vs-CT (gender+city vs age+gender) — compare before concluding equivalence.
3. No `random_state` in split → irreproducible shapes/order.
4. Forgetting `remainder` drops columns silently (default `"drop"`).
5. Index-vs-name selectors break after `train_test_split` → array conversion (Day-29 uses indices deliberately).

## 13. Relationship With Other Topics

Days 24–27 (the three transformers) → Day 28 (combine) → Day 29 (automate: `ct` becomes pipeline `trf1`, followed by `trf2` OHE / `trf3` scale / `trf4` select / `trf5` model).

## 14. Revision Summary

- One object, per-column transformers, single output array. Fit-train/transform-test; declare columns + remainder; reproducible split.

## 15. Practice Questions

1. Write the `ColumnTransformer` for: impute fever / ordinal cough / one-hot gender+city / passthrough age.
2. What does `remainder="passthrough"` do, and what is the default?
3. Why is `fit_transform(X_test)` wrong here?
4. Manual gives (80,7), CT gives (20,23) — what two things explain the difference?
5. How does this notebook's `ct` relate to Day-29 `trf1`?
6. Split has no `random_state` — what breaks for revision?

<details><summary>Answers</summary>

1. See §8 code block.
2. Keeps unlisted columns; default `"drop"` removes them.
3. Learns test statistics/categories — leakage + train/test mapping mismatch.
4. Different column selections (gender+city vs age+gender) and different effective splits/rows executed.
5. Same pattern: imputation stage inside the larger pipeline; Day-29 uses index selectors.
6. Reproducibility — shapes/rows/order change per run.
</details>

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
