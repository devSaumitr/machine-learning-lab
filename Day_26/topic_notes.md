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
