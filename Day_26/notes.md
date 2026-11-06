# Day_26 — Ordinal Encoding & Label Encoding — Topics Guide (Day 26)

> Deep notes for **Day_26**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# Ordinal Encoding & Label Encoding — Topics Guide (Day 26)

> Primary: `DAY26_Ordinal-Label-Encoding.ipynb` (this folder). Secondary: CampusX [day26-ordinal-encoding](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day26-ordinal-encoding) (`day26.ipynb`, video `w2GglmYHfmM`) — fills the gap that CampusX skips the train/test split and encodes the full frame; your local notebook adds the leak-free split.

## 1. What

Encoding turns text categories into numbers a model can read. Day 26 covers the two sklearn encoders:

- **`OrdinalEncoder`** — for **input** columns whose levels have a real rank (`Poor < Average < Good`, `School < UG < PG`). Each level gets an integer: you choose which level is 0, 1, 2.
- **`LabelEncoder`** — for the **target** column `y` only (`No/Yes → 0/1`). Same idea, but designed for a single 1-D label vector, not a feature matrix.

Data: `customer.csv` (`age/gender/review/education/purchased`, 50 rows). Slice `df.iloc[:, 2:]` → `review/education/purchased`. Example row: `97,Male,Poor,PG,Yes`.

## 2. Why

- Models (LogReg, trees, SVM…) do arithmetic — strings crash them.
- Nominal categories (gender) should not pretend to have a rank → Day 27 one-hot. Ordered categories *do* have a natural numeric story, so one integer per level carries the rank for free (0 < 1 < 2 mirrors Poor < Average < Good).
- The target must be numeric for loss computation; binary labels become 0/1 via `LabelEncoder`.
- The **order you declare is the order the model sees**. If you let sklearn sort alphabetically, "Good" can end up as 1 and "Poor" as 2 — the model then learns the opposite of reality.

## 3. How

1. Load `customer.csv`, keep last 3 columns (`iloc[:, 2:]`).
2. `train_test_split(X=review+education, y=purchased, test_size=0.3, random_state=34)` → `X_train.shape == (35, 2)` (fit **only** on train).
3. `OrdinalEncoder(categories=[["Poor","Average","Good"],["School","UG","PG"]])` → `fit(X_train)` → `transform(X_train)` and `transform(X_test)`.
4. `LabelEncoder()` → `fit(y_train)` → `transform(y_train)` / `transform(y_test)`; check `le.classes_`.
5. Rule of thumb: **ordinal for ordered inputs, label for the target, one-hot (Day 27) for unordered inputs.**

## 4. Syntax breakdown

### `OrdinalEncoder` init args

```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(
    categories=[["Poor", "Average", "Good"],   # column 1 order = codes 0,1,2
                ["School", "UG", "PG"]],       # column 2 order = codes 0,1,2
    dtype=np.float64,   # output dtype (default float64 → you see 0. 1. 2.)
    handle_unknown="error"   # unseen test value → raise; "use_encoded_value" → fill with a number
)
```

- `categories`: "auto" (learn from data, **alphabetical for strings**) or a **list of lists — one inner list per column, in column order**. This is the whole lesson of the day: declare it yourself.
- `dtype`, `handle_unknown`: the other two knobs; the notebook leaves them default.
- No `drop` / no `sparse` here — those belong to `OneHotEncoder` (Day 27).

### `categories` / fitted attributes

- After `fit`, `oe.categories_` returns the arrays actually used (your lists, or learned ones).
- Codes are positional: index in the list = the number written out.

### `fit_transform` vs `transform`

| call | meaning | when |
|---|---|---|
| `oe.fit(X_train)` | learn category lists | once, on **train** |
| `oe.transform(X_train)` / `X_test` | map strings → numbers with the learned lists | after fit |
| `oe.fit_transform(X_train)` | shortcut = fit + transform | train only, never test |

`fit_transform(X_test)` would re-learn categories from test — leakage plus a possibly different mapping (same rule as Day 24/25 scalers).

### `inverse_transform`

```python
oe.inverse_transform([[0., 1.]])   # → [['Poor', 'UG']]
le.inverse_transform([0, 1])       # → ['No', 'Yes']
```

Both encoders can map numbers back to strings — useful when you must print predictions as labels. The notebook does not call it; remember it exists.

### `LabelEncoder` (target only)

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)          # le.classes_ → ['No', 'Yes']  (alphabetical, automatic)
y_train = le.transform(y_train)   # ['No','Yes',...] → [0,1,...]
y_test  = le.transform(y_test)
```

- No `categories=` argument — classes are always learned (alphabetically for strings). Check `le.classes_` before trusting 0/1.
- 1-D input only (a Series/array of labels), not a DataFrame of features.
- No `drop`, no `sparse` — those do not apply.

## 5. Implementation

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\customer.csv")
df = df.iloc[:, 2:]                      # review, education, purchased

X_train, X_test, y_train, y_test = tts(
    df.iloc[:, 0:2], df.iloc[:, 2], test_size=0.3, random_state=34
)                                        # X_train.shape → (35, 2)

oe = OrdinalEncoder(categories=[
    ["Poor", "Average", "Good"],
    ["School", "UG", "PG"]
])
oe.fit(X_train)
X_train = oe.transform(X_train)          # ndarray of 0./1./2.
X_test  = oe.transform(X_test)

le = LabelEncoder()
le.fit(y_train)                          # classes_ = ['No','Yes']
y_train = le.transform(y_train)          # 0/1 array
y_test  = le.transform(y_test)
```

## 6. Execution walkthrough

1. `df.sample()` → row 7: `60, Female, Poor, School, Yes` (age/gender are dropped next — gender is nominal and unused today).
2. `df = df.iloc[:, 2:]` → 50 rows × 3 cols (`review`, `education`, `purchased`).
3. Split with `test_size=0.3, random_state=34` → `X_train.shape == (35, 2)` (15 rows held out).
4. `oe = OrdinalEncoder(categories=[["Poor","Average","Good"],["School","UG","PG"]])` → sklearn repr shows both lists as user-set parameters.
5. `oe.fit(X_train)` → stores the declared lists (train only saw them; test must contain nothing new).
6. `X_train = oe.transform(X_train)`; `X_test = oe.transform(X_test)` → float arrays, e.g. `Average→1.0`, `School→0.0`, `PG→2.0`. Column names are lost (now a bare ndarray — wrap with `pd.DataFrame(..., columns=...)` if you need them).
7. `oe.categories_` → `[array(['Poor','Average','Good']), array(['School','UG','PG'])]` — confirms your order won.
8. `le.fit(y_train)` → `le.classes_ == ['No','Yes']` → `No=0`, `Yes=1`.
9. `y_train = le.transform(y_train)`; `y_test = le.transform(y_test)` → 1-D int arrays (35 and 15 values).
10. Trailing cells are empty — no model is trained on Day 26.

## 7. Observations

- **Ordinal vs label vs one-hot.** Same number-mapping idea; the difference is *where* it is applied: ordered **inputs** → `OrdinalEncoder` (multi-column, you set order); **target** → `LabelEncoder` (1-D, auto order); unordered **inputs** → one-hot (Day 27).
- **Category order pitfall.** Default `categories="auto"` sorts strings alphabetically: `Average=0, Good=1, Poor=2` — the exact reverse of the real rank. Always pass `categories=` for ordinal data (CampusX and your notebook both do).
- **Leakage.** Fit encoders on `X_train` / `y_train` only, then `transform` test — the same fit-on-train rule as Day 24 (StandardScaler) and Day 25 (MinMaxScaler). CampusX's own day26 notebook encodes the *whole* frame before any split — a gap your local notebook fixes.
- **Type gotchas.** Ordinal output is `float64` by default (`0., 1., 2.`), not int. `LabelEncoder` output is int. Reassigning `X_train = oe.transform(X_train)` discards the DataFrame (index + columns).
- **LabelEncoder misuse.** Never feed it input features — it hides rank info for nominal data and is built for y only. For binary targets it is fine; for multi-class targets it imposes an alphabetical code order the model may treat as ordinal (acceptable for classification losses that use class indices).
- **Unseen categories.** Default `handle_unknown="error"` — a test row with a level never seen in train crashes on purpose. That is a signal to re-check your split/groups, not to silently guess.

## 8. Connection with prior + comparison tables

Builds directly on the leak discipline of [Day 24 standardization](../Day_24/topic_notes.md) and [Day 25 normalization](../Day_25/topic_notes.md): fit on train, transform test — now applied to *categories* instead of *numbers*. Day 27 replaces ordered mapping with one-hot for nominal columns; Days 28–29 put these encoders inside `ColumnTransformer`/`Pipeline`.

### Encoder choice

| Situation | Encoder | Order | Output |
|---|---|---|---|
| Input, ranked levels (Poor/Avg/Good) | `OrdinalEncoder` | you declare it | one float column per feature |
| Target labels (No/Yes) | `LabelEncoder` | learned (alphabetical) | 1-D int array |
| Input, no rank (gender, fuel) | OneHot (Day 27) | irrelevant | n or n−1 columns |

### `OrdinalEncoder` vs `LabelEncoder`

| | OrdinalEncoder | LabelEncoder |
|---|---|---|
| Fits on | 2-D X (multiple columns) | 1-D y only |
| `categories=` arg | yes (declare order) | no (auto alphabetical) |
| Typical use | ordered features | target |
| `inverse_transform` | yes | yes |

### CampusX vs local

| | CampusX day26 | Local DAY26 notebook |
|---|---|---|
| Encoder usage | OrdinalEncoder + LabelEncoder, same category lists | same |
| Train/test split | **absent** (encodes full data) | present (`rs=34`, 70/30) — leak-free |
| Extra teaching | — | comments (gender nominal / review ordered), experiment cell (drop `categories=` and watch it break) |

### From `topic_notes.md` / deep notes

# Ordinal and Label Encoding — Study Notes (Day 26)

> Reference: `day26-ordinal-encoding` in the CampusX repo. Your work: `DAY26_Ordinal-Label-Encoding.ipynb` on customer.csv (`review/education → purchased`; 35-row train split).

## 1. Basic Idea

Models need numbers, but numbers must not invent order. Day 26's rule: ordered *inputs* → `OrdinalEncoder` (Poor < Average < Good becomes 0/1/2), the *target* → `LabelEncoder` (No/Yes becomes 0/1). Unordered inputs wait for Day 27.

## 2. Intuition and Mental Model

Medals have a real order (gold > silver > bronze) — numbering them 3/2/1 tells the truth. Shirt colors have none — numbering them would lie. Ordinal encoding is for medals; the target label just needs two distinct jerseys (0/1).

## 3. Why Do We Need It?

Strings crash sklearn, and naive numbering (`{"Poor": 2, ...}` by accident) teaches false rankings to linear and distance models. Declared-order encoding preserves true rank information in a single column.

## 4. Detailed Explanation

- **Nominal vs ordinal.** Nominal = equal status, no rank (gender, city, fuel). Ordinal = meaningful rank (Poor < Average < Good; School < UG < PG). Classify first, encode second.
- **`OrdinalEncoder(categories=[...])`.** The `categories` argument is mandatory in spirit: without it the encoder sorts alphabetically (Average/Good/Poor) and silently scrambles rank. Fit on train, transform both splits.
- **`LabelEncoder` on the target only.** `fit_transform(y)` maps No→0/Yes→1. Never use it on input features (it invents order) and never OneHot the target.
- **Your run.** `iloc[:, 2:]` isolates review/education/purchased; split 70/30 (`random_state=34`) gives a (35, 2) train block; ordinal + label transforms complete the encoding. No OneHot appears here by design; no model is trained (encoding-only scope).

## 5. Examples

```python
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
ord_enc = OrdinalEncoder(categories=[["Poor", "Average", "Good"],
                                     ["School", "UG", "PG"]])
Xtr = ord_enc.fit_transform(X_train)   # train only
Xte = ord_enc.transform(X_test)
ytr = LabelEncoder().fit_transform(y_train)   # target only
```

## 6. Important Formulas

None — mappings, not math. Document the order alongside the code or the numbers are meaningless.

## 7. Code Connection

Two estimators, two roles: `OrdinalEncoder` (inputs, declared order) and `LabelEncoder` (target). Both follow fit-train/transform-test; the overwritten `X_train` array losing column names is preserved as-is (wrap in a DataFrame in new code).

## 8. Interpretation

Encoded values 0/1/2 mean rank positions, not quantities — a "2" is not twice a "1". Target 0/1 are class ids for the classifier's output.

## 9. Common Mistakes

1. Omitting `categories=` → alphabetical scrambling.
2. `LabelEncoder` on inputs; OneHot on the target.
3. Fitting encoders before the split (leakage of test category distributions).
4. Encoding ID-like/high-cardinality columns raw (width explosion — group first, see Day 27).

## 10. Machine Learning Connection

Ordinal inputs + label target are the minimal numeric diet most classifiers accept; unordered columns join via [Day-27 OneHot](../Day_27/topic_notes.md), and all of it is automated in [Day-28/29](../Day_28/topic_notes.md) preprocessing.

## 11. Interview and Conceptual Questions

1. Ordered input vs unordered input vs target — which encoder? — Ordinal / OneHot / Label.
2. What breaks without `categories=`? — Alphabetical order replaces true rank.
3. Why not `LabelEncoder` on features? — Imposes a false ranking the model will exploit.
4. Should Titanic's Pclass be ordinal or one-hot? — Ordinal respects 1<2<3 in one column; OneHot allows non-monotonic effects in two — model-dependent.

## 12. Quick Revision

Rank-real inputs → Ordinal with declared order; target → Label; everything unordered → [Day 27](../Day_27/topic_notes.md). Fit-train/transform-test, always.

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
