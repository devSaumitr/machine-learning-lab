# Bivariate and Multivariate Analysis (Day 21)

> Playlist: Day 21 EDA using Bivariate and Multivariate Analysis.
> Notebook: `DAY21_Bivariate-Multivariate-Analysis.ipynb` (tips + iris + Titanic `E:/MACHINE_LEARNING/train.csv`).

## 1. Learning Objectives

- Choose plots by data-type pair: num–num (scatter/relplot), cat–num (bar/box/catplot), cat–cat (crosstab/heatmap/clustermap), many-cols (pairplot/pivot-heatmap).
- Use hue/style/size/row facets to add a 3rd variable to one plot.
- Read `crosstab`/`pivot_table` numbers behind heatmaps.

## 2. Why This Topic Matters

Single-column EDA cannot show relationships — and ML features are chosen by relationships (Pclass/Sex/SibSp → Survived). This video teaches the plot-per-pair system your notebook practices end to end.

## 3. Prerequisites

- Day 20 univariate plots; `crosstab`/`pivot_table` names; hue concept.

## 4. First-Principles Explanation

Bivariate = "how do TWO columns move together?" Add hue/style/size = third variable on the same canvas. Multivariate = grids of many pairs (`pairplot`) or matrices (`heatmap`, `clustermap`, `pivot_table`).

## 5. Formal Definition

- **Bivariate**: joint analysis of two variables (scatter, bar-by-group, box-by-group, contingency heatmap).
- **Multivariate**: 3+ variables jointly (facets, pairplot grids, correlation/clustermaps).
- **`crosstab(a, b)`**: frequency table of two categoricals. **`pivot_table(values, index, columns)`**: aggregated matrix.
- **Clustermap**: heatmap + hierarchical row/column reordering.

## 6. How It Works Internally

- `relplot(tip vs total_bill, hue/row=sex, style=smoker, size=size)`: scatter points positioned by (total_bill, tip); color/shape/size channels encode the extra columns; `row=sex` splits into facet subplots.
- `catplot(Pclass vs Age, kind="bar", hue=Sex)`: mean Age per (Pclass, Sex) with confidence bars. `boxplot(Pclass vs Age)`: full quartile boxes per class.
- `crosstab(Pclass, Survived)` → integer matrix → `heatmap` colors cells; `clustermap` additionally clusters similar rows/columns.
- `pairplot(iris, hue="species")`: every numeric pair scattered, diagonal KDEs, color = species.
- `pivot_table(values="Pclass", index="Age", columns="SibSp")` → mean-Pclass matrix → viridis heatmap.

## 7. Step-by-Step Examples (your notebook order)

1. Load `iris`, `tips`, `titanic.head()`.
2. `relplot`: does `tip` rise with `total_bill`? Do `sex`/`smoker` shift clouds?
3. `catplot` bar vs `boxplot`: mean-age-per-class vs full age distribution per class.
4. `distplot(Age|Survived)` ×2 — deprecated; concept = survived vs died age curves.
5. `crosstab(Pclass, Survived)` + `heatmap`: which class concentrates survival.
6. `clustermap(crosstab(SibSp, Survived))` ×2 (2nd has `Sibsp` typo → KeyError; drop duplicate).
7. `pairplot(iris, hue=species)`: species separate in petal dimensions.
8. `pivot_table` heatmap (mislabeled `LinePlot`): mean Pclass over (Age, SibSp) grid.

## 8. Code Examples

```python
sns.relplot(data=tips, x="total_bill", y="tip", hue="sex", row="sex", style="smoker", size="size")
sns.catplot(data=titanic, x="Pclass", y="Age", hue="Sex", kind="bar")
sns.boxplot(data=titanic, x="Pclass", y="Age")
pd.crosstab(titanic["Pclass"], titanic["Survived"])
sns.heatmap(pd.crosstab(titanic["Pclass"], titanic["Survived"]), annot=True)
sns.clustermap(pd.crosstab(titanic["SibSp"], titanic["Survived"]))
sns.pairplot(iris, hue="species")
sns.heatmap(titanic.pivot_table(values="Pclass", index="Age", columns="SibSp"), cmap="viridis")
# modern fix for the deprecated cells:
sns.histplot(data=titanic, x="Age", hue="Survived", kde=True)
```

## 9. Output Interpretation

- Upward `tip` cloud = positive association; separated hue clouds = group effect.
- Bar = averages (CIs); box = medians/spread/outliers — report both, not one.
- Heatmap numbers = counts/means; clustermap order = similarity, not raw category order.
- Pairplot: non-overlapping species clouds = separable features (good for modeling).

## 10. Important Parameters and Variations

| Plot | Encodes | When |
|---|---|---|
| `relplot` | x/y/hue/style/size/row/col | num–num + categories |
| `catplot(kind=bar/box/count/violin)` | x/y/hue | cat–num |
| `boxplot` | distributions per group | spread/outliers |
| `heatmap(annot, cmap)` | matrix values | crosstab/corr/pivot |
| `clustermap` | + dendrogram order | similarity pattern |
| `pairplot(hue)` | all pairs | quick separability scan |

## 11. Practical Use Cases

- Feature screening: which columns separate the target (keep Pclass/Sex/SibSp-style signals for Days 26–29).

## 12. Common Mistakes and Misconceptions

1. `distplot` deprecated → use `histplot`/`kdeplot`.
2. `Sibsp` typo → `KeyError`; duplicate clustermap call; 2 empty code cells.
3. Final cell labeled `LinePlot` but is a heatmap — read code, not header.
4. Bar-only reading (means) without boxes (spread); color-only heatmap reading without annotations.
5. `relplot` is figure-level (facets) vs `scatterplot` axes-level — customization differs.

## 13. Relationship With Other Topics

Day 20 (one col) → Day 21 (pairs) → Day 22 (auto-reports) → encoding/scaling/pipelines: e.g. Pclass/Sex/SibSp patterns justify Days 26–27 encoders and Day 28–29 preprocessing.

## 14. Revision Summary

- Pair → plot: num–num scatter/rel; cat–num bar+box; cat–cat crosstab+heatmap; many → pairplot/clustermap/pivot-heatmap. Facets add variables without new plots.

## 15. Practice Questions

1. Which plot for: tip-vs-bill by sex / mean age per class / age spread per class / Pclass-vs-Survived counts?
2. What do hue/style/size/row each add to a scatter?
3. Bar vs box on the same cat–num pair — what does each hide/show?
4. What does `crosstab` compute, and why annotate the heatmap?
5. What does clustermap reordering mean?
6. Fix the three bugs in the notebook's dist/cluster/LinePlot cells.
7. Pairplot species clouds overlap heavily — what does that imply for modeling?

<details><summary>Answers</summary>

1. relplot / catplot-bar / boxplot / crosstab+heatmap.
2. Extra variables on one canvas: color / marker / point-size / subplot-split.
3. Bar shows means+CIs, hides spread; box shows quartiles/outliers, hides precise means.
4. Joint frequency table; annotations give exact counts behind colors.
5. Rows/cols sorted by similarity (dendrogram), revealing groups.
6. `distplot`→`histplot/kdeplot`; `Sibsp`→`SibSp` + drop duplicate; rename header to heatmap.
7. Those features separate classes poorly — need better features/transforms.
</details>
