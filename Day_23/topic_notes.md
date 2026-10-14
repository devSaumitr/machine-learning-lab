# What is Feature Engineering (Day 23)

> Playlist: Day 23 What is Feature Engineering.
> Notebook: `DAY23_What-is-Feature-Engineering.ipynb` (outline only: 7 markdown cells, 1 empty code cell, zero implementation). Implementations live in Days 24–29.

## 1. Learning Objectives

- Define the five jobs: transform, construct, select, extract, and handle missing/outliers/scale.
- Map each line of the outline to its implementation notebook (table below).
- Explain why sklearn rejects NaN and why strings must become numbers.

## 2. Why This Topic Matters

Models eat numbers, not raw spreadsheets. Feature engineering is the conversion: fill gaps, convert types, tame outliers, match scales, create missing signals, drop noise. Day 23 names the jobs; Days 24–29 do them.

## 3. Prerequisites

- Days 19–22 findings (which columns are missing/skewed/categorical).

## 4. First-Principles Explanation

Raw table → model-ready matrix. Each operation fixes one blocker: NaN (model crashes) → fill; strings (no math) → encode; wild ranges (distance distortion) → scale; absent signal (e.g. family size from SibSp+Parch) → construct; too many columns (noise/slowness) → select; buried structure → extract.

## 5. Formal Definition

- **Feature transformation**: change representation (scaling, encoding, binning, log).
- **Construction**: create new features from existing ones (manual/groups).
- **Selection**: keep only informative features. **Extraction**: project to new axes (e.g. PCA, later).
- **Binning**: numeric → categorical bands. **Outlier handling**: cap/remove/robust-scale extreme points.

## 6. How It Works Internally (map to your repo)

| Outline line | Meaning | Implemented in |
|---|---|---|
| sklearn rejects NaN; fill it | imputers learn fill values on train | Day 28 (`SimpleImputer`), Day 29 pipelines |
| categorical→one-hot, string→numeric | unordered → OneHot; ordered → Ordinal; target → Label | Days 26–27 |
| binning num→string | discretize ranges | concept only (no notebook yet) |
| outliers impact algos | distance/gradient models swing; trees resist | Day 24 outlier demo |
| Age/Salary scaling | match magnitudes | Days 24–25 |
| create groups manually | new features from domain | concept only |
| use only important features | SelectKBest etc. | Day 29 `SelectKBest(chi2, k=8)` |

## 7. Step-by-Step Examples

Titanic preview: Age (177 missing → impute) → Sex/Embarked (strings → OneHot) → Fare (skew → scale) → FamilySize = SibSp+Parch+1 (construct) → Cabin (687 missing → drop/engineer deck) → SelectKBest keeps top-8 (select). Each arrow is one Day 24–29 notebook.

## 8. Code Examples

None in the original (empty code cell). Representative pattern from later notebooks:

```python
# fill -> encode -> scale (the Day-23 checklist in 3 lines of spirit)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

## 9. Output Interpretation

No outputs in this notebook by design — interpret it as a checklist: every bullet should point to a later implementation; bullets without implementations (binning, construction, extraction) are flagged as gaps in `LEARNING_PROGRESS.md`.

## 10. Important Parameters and Variations

- Selection vs extraction are different (outline lists "selection" twice — read once as selection, once as extraction).
- Binning trades resolution for robustness; construction trades simplicity for signal; selection trades completeness for generalization.

## 11. Practical Use Cases

- Planning cleaning sprints: turn each EDA finding into exactly one engineering step.

## 12. Common Mistakes and Misconceptions

1. Treating this outline as "done" — it is vocabulary, not skill; skill is Days 24–29.
2. Imputing/encoding before splitting (leakage) — fit on train only (Days 24–25 drills).
3. Encoding the target with OneHot; one-hotting high-cardinality IDs (Name/Ticket/Cabin).
4. Scaling for trees out of habit (harmless but pointless).

## 13. Relationship With Other Topics

Day 23 (what) → Days 24–25 (scale) → 26–27 (encode) → 28 (combine) → 29 (automate + select). EDA findings are the input; pipelines are the output.

## 14. Revision Summary

- Fill → encode → scale → create → select. Your notebook defines the words; the next four folders define the how.

## 15. Practice Questions

1. Why does sklearn crash on NaN, and where should fill values be learned?
2. Ordered input vs unordered input vs target — which encoder for each?
3. When does scaling matter vs not matter?
4. Give one construction example for Titanic.
5. Selection vs extraction — difference?

<details><summary>Answers</summary>

1. No NaN math defined; learn fill on train, apply to test (no leakage).
2. Ordinal / OneHot / Label.
3. Matters for distance/gradient models (LogReg/KNN/SVM/nets); barely for trees.
4. FamilySize = SibSp + Parch + 1; Title from Name; Deck from Cabin.
5. Selection keeps original columns; extraction builds new axes.
</details>
