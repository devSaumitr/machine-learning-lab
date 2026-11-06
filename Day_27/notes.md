# Day_27 — One-Hot Encoding — Topics Guide (Day 27)

> Deep notes for **Day_27**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# One-Hot Encoding — Topics Guide (Day 27)

> Primary: `DAY27_OneHot-Encoding.ipynb` (this folder). Secondary: CampusX [day27-one-hot-encoding](https://github.com/campusx-official/100-days-of-machine-learning/tree/main/day27-one-hot-encoding) (`day27.ipynb`, video `U5oCv3JKWKA`) — same four blocks (pandas dummies → k−1 → sklearn `OneHotEncoder` → top categories); your local notebook adds explicit dummy-trap markdown and keeps documented bugs for you to fix.

## 1. What

One-hot encoding gives **every level its own 0/1 column**, used when categories have **no order** (fuel: CNG/Diesel/LPG/Petrol; owner: First/Second/…; brand: Maruti/Honda/…). A row with `fuel=Diesel` is `fuel_Diesel=1`, all other fuel columns `0`.

Two tools, same idea:

- **`pd.get_dummies(df, columns=[...])`** — one-shot pandas, great for EDA.
- **`OneHotEncoder`** (sklearn) — fits on train, remembers categories, transforms test safely inside pipelines.

Plus the **dummy trap** rule: for n levels keep **n−1** columns (`drop_first=True` / `drop='first'`), and the **high-cardinality fix**: rare levels (brand count ≤ 100) collapse into one bucket before encoding.

Data: `cars.csv` — `brand/km_driven/fuel/owner/selling_price`, 8128 rows (train 5689 at `test_size=0.3, rs=34`).

## 2. Why

- Unordered categories must not be squeezed into one integer: ordinal codes would invent a rank (CNG < Diesel < Petrol) that does not exist — the model would learn nonsense. Day 26's lesson: only encode order when order is real.
- **Dummy trap / multicollinearity:** with all n columns present, `fuel_CNG + fuel_Diesel + fuel_LPG + fuel_Petrol = 1` always — one column is a perfect linear combination of the others. Input columns must not predict each other (multicollinearity must be zero). n−1 columns still fully describe n levels: the dropped level is the all-zero baseline.
- **High cardinality:** 32 brands → 32 extra columns, most of them almost never 1 (Isuzu=5 rows). Grouping rare brands into `other` keeps the matrix narrow without dropping the signal column.
- **Pipeline safety:** `get_dummies` on test alone can produce different columns than train (or error on unseen levels). `OneHotEncoder` fit on train always emits the same columns; `handle_unknown="ignore"` maps unseen test levels to all-zeros.

## 3. How

1. Load `cars.csv`; inspect `df['owner'].value_counts()` (First 5289 … Test Drive 5).
2. **Pandas path:** `pd.get_dummies(df, columns=['fuel','owner'])` → full n columns; then `drop_first=True` → n−1 (dummy trap fixed).
3. **Sklearn path:** split 70/30 (`rs=34`); `OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)`; `fit_transform(X_train[['fuel','owner']])` → `(5689, 7)`; `transform(X_test[...])`; `np.hstack` with `brand`/`km_driven` → `(5689, 9)`.
4. **Top categories:** `counts = df['brand'].value_counts()`; `threshold = 100`; `repl = counts[counts <= threshold].index`; `df['brand'].replace(repl, 'uncommon')` → `pd.get_dummies(...)` → 12 frequent brands + 1 bucket.

## 4. Syntax breakdown

### `pd.get_dummies`

```python
pd.get_dummies(
    df,
    columns=["fuel", "owner"],   # which columns to expand; others pass through
    drop_first=False,            # True → keep n-1 columns (dummy trap fix)
    dtype=np.int8                # output type (bool by default in modern pandas)
)
```

- Prefix auto-named `fuel_CNG`, `owner_First Owner`, … (use `prefix=` / `sep=` to change).
- **No fit state** — pure function. Calling it separately on train and test is the classic column-mismatch bug.

### `OneHotEncoder` init args

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(
    drop="first",            # None (default, n cols) | "first" | "if_binary" | list per column
    sparse_output=False,     # False → dense array (sklearn ≥1.2; older name: sparse=False)
    dtype=np.int32,          # 0/1 ints instead of bool/float
    handle_unknown="error"   # "error" | "ignore" (unseen → all zeros) | "infrequent_if_exist"
    # also: min_frequency / max_categories → auto-group rare levels ("infrequent")
)
```

- `drop='first'` = k−1 encoding, the sklearn twin of `drop_first=True`.
- `sparse_output=True` (default) returns a scipy sparse matrix — memory-friendly for huge one-hots; the notebook forces dense for easy `hstack`/display.
- `handle_unknown='ignore'` is the pipeline-safe setting the notebook's markdown recommends (not set in the code here).

### `categories` (implicit)

- You do not pass `categories=` in this notebook; `fit` learns them from train **in sorted order** (alphabetical). `ohe.categories_` / `ohe.drop_idx_` show what was learned and which level was dropped. Declare lists yourself only if the sorted order is wrong for you.

### `fit_transform` vs `transform`

| call | meaning | when |
|---|---|---|
| `ohe.fit_transform(X_train[['fuel','owner']])` | learn levels + encode | train only |
| `ohe.transform(X_test[['fuel','owner']])` | encode with train's levels | test, never fit here |

`fit_transform` on test = leakage **and** possibly different columns (documented bug in this notebook — see §7).

### `inverse_transform`

```python
ohe.inverse_transform(X_train_new)   # 0/1 rows → original ['Diesel', 'First Owner'] strings
```

Works on the encoded block; pandas `get_dummies` has no inverse (bools are not reversible without the original frame).

### drop / sparse at a glance

| | pandas `get_dummies` | sklearn `OneHotEncoder` |
|---|---|---|
| drop one level | `drop_first=True` | `drop='first'` |
| sparse output | no (dense DataFrame) | `sparse_output=True` default |
| remembers categories | no | yes (`categories_`) |
| unseen test level | column mismatch risk | `handle_unknown='ignore'` |

## 5. Implementation

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

cars = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\cars.csv")
df = cars

# pandas EDA path
pd.get_dummies(df, columns=['fuel', 'owner'])
pd.get_dummies(df, columns=['fuel', 'owner'], drop_first=True)   # k-1

# sklearn pipeline path
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4], df.iloc[:, -1],   # FIX (see §7): y must be selling_price only
    test_size=0.3, random_state=34
)
ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)
X_train_new = ohe.fit_transform(X_train[['fuel', 'owner']])   # (5689, 7)
X_test_new  = ohe.transform(X_test[['fuel', 'owner']])
np.hstack((X_train[['brand', 'km_driven']].values, X_train_new)).shape  # (5689, 9)

# top categories
counts = df['brand'].value_counts()
threshold = 100
repl = counts[counts <= threshold].index
pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5)
```

## 6. Execution walkthrough

1. `cars` → 8128 rows × 5 cols: `brand, km_driven, fuel, owner, selling_price` (row 0: Maruti, 145500, Diesel, First Owner, 450000).
2. `df['owner'].value_counts()` → First 5289, Second 2105, Third 555, Fourth & Above 174, Test Drive 5 (5 levels → 5 or 4 dummy columns).
3. `pd.get_dummies(df, columns=['fuel','owner'])` → 8128 × **12** (brand + km + price pass through; fuel 4 + owner 5 = 9 new columns replace 2 old: −2+9 = +7).
4. `drop_first=True` → 8128 × **10**: `fuel_CNG` dropped (baseline = CNG), `owner_First Owner` dropped (baseline = First Owner); 3 + 4 = 7 dummies.
5. Split `df.iloc[:,0:4]` vs `df.iloc[:,:-1]`, `test_size=0.3, rs=34` → train 5689 / test 2439. (Bug: `y` is not the target — see §7.)
6. `ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)`.
7. `X_train_new = ohe.fit_transform(X_train[['fuel','owner']])` → dense int array **(5689, 7)** = fuel 3 + owner 4.
8. `X_test_new = ohe.transform(X_test[['fuel','owner']])` — same 7 columns, no refit.
9. `np.hstack((X_train[['brand','km_driven']].values, X_train_new)).shape` → **(5689, 9)** = 2 passthrough + 7 encoded (brand itself is still a string column — not encoded here).
10. Top-category block: `counts` shows Maruti 2448 … Peugeot 1; `threshold = 100`; `repl = counts[counts <= 100].index` → 15 rare brands (Fiat…Peugeot); `replace(repl, 'unommon')` then `get_dummies(...).sample(5)` → **13 columns** (12 brands with count > 100 + the bucket), booleans True/False.

## 7. Observations

- **Ordinal vs label vs one-hot (Day 26 recap + Day 27).** Ordinal/label = one number per level (rank must be real, or it is the target). One-hot = one column per level (rank must *not* exist). Putting `OrdinalEncoder` on `fuel` would invent Diesel > Petrol; putting `LabelEncoder` on inputs has the same flaw.
- **Dummy trap.** Full dummies are collinear: columns sum to 1. Keep n−1 (`drop_first` / `drop='first'`) whenever multicollinearity matters (linear models especially). Trees tolerate full dummies, but k−1 is the safe default CampusX teaches.
- **Category order.** One-hot does not care about level order — `get_dummies` and `OneHotEncoder` both sort alphabetically, and that is fine here. Only ordinal encoding needs a declared order (Day 26).
- **Leakage — documented bug in this notebook:** `y_train, y_test = train_test_split(..., df.iloc[:, :-1], ...)` makes `y` **identical to X** (`brand, km_driven, fuel, owner`) instead of `selling_price`. Fix: `df.iloc[:, -1]`. Also, never `fit_transform` the encoder on `X_test` (the notebook correctly uses `transform` here — keep it that way; the takeaways cell still warns about the pattern).
- **Small bugs preserved on purpose:** typo `'unommon'` (CampusX spells it `'uncommon'`); `OneHotEncoder` imported twice; a code cell with only `nunique()`/`threshold` and an empty cell — clean these before a full re-run.
- **Width explosion.** fuel+owner is cheap (7 columns); brand would add 31. That is why top-k grouping exists: threshold 100 → 13 columns instead of 32. Move the threshold to 500 and the bucket grows, width shrinks (experiment cell).
- **pandas vs sklearn.** `get_dummies` forgets everything between calls — fine for one-off EDA, wrong for train/test pipelines. `OneHotEncoder` is the production choice; combine with `handle_unknown='ignore'` for unseen test levels.

## 8. Connection with prior + comparison tables

Extends the fit-on-train rule from [Day 24](../Day_24/topic_notes.md) / [Day 25](../Day_25/topic_notes.md) and the encoder-for-columns idea from [Day 26 topics](../Day_26/topics.md): scaling handles numeric features, Day 26 handles ordered categories + target, Day 27 handles unordered categories. Days 28–29 assemble `OrdinalEncoder` + `OneHotEncoder` + scalers into one `ColumnTransformer` pipeline — the `drop`/`sparse_output`/`handle_unknown` args you meet here are exactly what you will set there.

### One-hot vs ordinal vs label

| | OneHot (Day 27) | Ordinal (Day 26) | Label (Day 26) |
|---|---|---|---|
| Level order | none | you declare it | learned (alphabetical) |
| Output | n or n−1 columns | 1 column, codes 0..k−1 | 1-D codes |
| Applied to | unordered inputs | ordered inputs | target only |
| Dummy trap | yes (n−1 fix) | n/a | n/a |

### `get_dummies` vs `OneHotEncoder`

| | `pd.get_dummies` | `OneHotEncoder` |
|---|---|---|
| Style | function, no state | estimator, fit/transform |
| Best for | quick EDA | pipelines / production |
| Unseen test level | risk of wrong columns | `handle_unknown='ignore'` |
| Drop level | `drop_first=True` | `drop='first'` |
| Sparse | no | `sparse_output=True` default |
| Inverse | no | `inverse_transform` |

### CampusX vs local

| | CampusX day27 | Local DAY27 notebook |
|---|---|---|
| Four blocks | pandas → k−1 → sklearn → top categories | same sequence |
| Dummy-trap theory | light headings | dedicated markdown (n−1 / multicollinearity) |
| Split | `test_size=0.2, rs=2`, y correct (`iloc[:,-1]`) | `test_size=0.3, rs=34`, **y bug kept** as a fix-it exercise |
| Known quirks | clean `uncommon` | typo `unommon`, duplicate import, empty cells — all documented in takeaways |

### From `topic_notes.md` / deep notes

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
