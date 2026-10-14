# Automated EDA: Pandas Profiling, Sweetviz and Manual Reports (Day 22)

> Playlist: Day 22 Pandas Profiling.
> Notebooks: `DAY22_Pandas-Profiling-ydata.ipynb` (ydata-profiling 4.18.4, `.venv312`), `DAY22_Pandas-Profiling-Sweetviz.ipynb` (Sweetviz 2.3.3, `.venv`), `DAY22_Pandas-Profiling-empty-placeholder.ipynb` (0 cells). Shared artifacts at root: `titanic_report.html`, `eda_report.html`, `eda_plots/`, `eda_utils.py`.

## 1. Learning Objectives

- Generate a one-line EDA HTML report with ydata-profiling AND with Sweetviz.
- Read the report: missingness, distributions, correlations, duplicates.
- Explain why both notebooks writing `titanic_report.html` is a conflict.
- Use `eda_utils.py` as a dependency-light manual alternative.

## 2. Why This Topic Matters

Hand EDA (Days 19–21) teaches; automated reports scale. One call produces the full first-pass report you would otherwise hand-plot in 30 cells — then you verify key findings manually.

## 3. Prerequisites

- Days 19–21 outputs (what missing/corr/distribution mean).

## 4. First-Principles Explanation

A profiler scans every column once (types, uniques, missing, stats, correlations, duplicates, interactions) and renders one HTML page. Sweetviz and ydata-profiling differ in layout/engine, not in goal.

## 5. Formal Definition

- **ydata-profiling** (`ProfileReport`): generates descriptive stats, missing matrix, distributions, correlations, interactions, duplicates into self-contained HTML.
- **Sweetviz** (`sv.analyze` + `show_html`): similar one-call EDA with target-comparison views.

## 6. How It Works Internally

- ydata: `ProfileReport(df, title=..., explorative=True)` → describe phase (per-column + `Describe Embarked`-style progress) → summarize (47/47 steps) → render HTML → `to_file` writes it.
- Sweetviz: `sv.analyze(df)` builds the report object (0% → done) → `show_html(path)` writes it.
- `eda_utils.py` (root, pandas/numpy/matplotlib/seaborn only): `overview` → `data_quality` (`missing_report`, duplicates, uniques, constant/ID-like cols) → `numerical_summary` (describe + IQR outlier counts) → `categorical_summary` (top-10 + proportions + high-cardinality flag) → `analyze_relationships` (group means, crosstab rates) → `plot_eda` (histograms, boxplots, corr heatmap, count plots, missing bars, target splits into `eda_plots/*.png`) → `generate_html_report` (lightweight HTML, NOT equivalent to ydata).

## 7. Step-by-Step Examples (your Titanic run)

1. `pd.read_csv("E:/MACHINE_LEARNING/train.csv")` → 891×12 (head shows PassengerId…Embarked).
2. ydata: `ProfileReport(df, title="Titanic EDA Report", explorative=True)` → `profile.to_file("titanic_report.html")`.
3. Sweetviz: `sv.analyze(titanic)` → `report.show_html("titanic_report.html")` — SAME filename, overwrites #2.
4. Manual: `import eda_utils; eda_utils.run_full_eda(titanic, target="Survived")` → console tables + `eda_plots/*.png` + `eda_report.html`.

## 8. Code Examples

```python
# ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Titanic EDA Report", explorative=True)
profile.to_file("titanic_report_ydata.html")  # renamed to avoid clobbering

# Sweetviz
import sweetviz as sv
report = sv.analyze(titanic)
report.show_html("titanic_report_sweetviz.html")  # renamed

# Manual, no extra deps
import eda_utils
eda_utils.run_full_eda(titanic, target="Survived")
```

## 9. Output Interpretation

- Missing section → Cabin (drop/engineer), Age (impute), Embarked×2 (mode).
- Distribution panels → Fare right-skew, Age humps; correlation panels → weak numeric links (match Day-19 numbers).
- 100% progress + kernel crash (both notebooks) = HTML may be truncated — regenerate and check file size/date before citing.

## 10. Important Parameters and Variations

| Call | Key args | Notes |
|---|---|---|
| `ProfileReport` | `title`, `explorative` | explorative = deeper interactions, slower |
| `to_file` | path | self-contained HTML |
| `sv.analyze` | df, optional target | comparison views |
| `show_html` | path | writes file |
| `run_full_eda` | `target`, `output_html`, `plots_dir` | reproducible, lightweight |

## 11. Practical Use Cases

- First-pass report for stakeholders; regression check after cleaning; quick comparison of train vs test splits.

## 12. Common Mistakes and Misconceptions

1. Both notebooks write `titanic_report.html` — they overwrite each other. Rename (as above).
2. `!pip install` / `%pip install` inside notebooks; different kernels (`.venv312` vs `.venv`) per library.
3. Code pasted as markdown (Sweetviz notebook); 2 empty markdowns (ydata); redundant re-`read_csv`.
4. Treating the auto-report as proof — verify key plots manually (Days 19–21).
5. `findfont: Failed...Roboto` warnings are cosmetic; kernel crashes are not — re-run.

## 13. Relationship With Other Topics

Days 19–21 (manual) ↔ Day 22 (auto) → Days 23–29 (act on findings: impute Age, drop/engineer Cabin, encode categoricals, scale numerics).

## 14. Revision Summary

- One call → full HTML: ydata (`ProfileReport`/`to_file`), Sweetviz (`analyze`/`show_html`). Rename outputs; expect crashes on first run; verify manually. `eda_utils` = light manual path.

## 15. Practice Questions

1. What three things must you read in any profiling report before modeling?
2. Why do the two notebooks conflict, and what rename fixes it?
3. Progress hit 100% but the kernel crashed — is the HTML trustworthy?
4. When would you prefer `eda_utils` over ydata-profiling?
5. The Sweetviz notebook is titled "pandas profiling" — why is the title misleading?

<details><summary>Answers</summary>

1. Missingness, distributions/skew, correlations/duplicates.
2. Same output filename `titanic_report.html`; write `*_ydata.html` / `*_sweetviz.html`.
3. No — may be truncated; regenerate and verify size/completeness.
4. Restricted env / no extra installs / need scripted PNGs + small HTML.
5. It implements Sweetviz, not pandas/ydata-profiling — title describes Day 22, not the library used.
</details>
