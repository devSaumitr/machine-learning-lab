# LEARNING_PROGRESS — Playlist ↔ Notebooks ↔ Notes (through Day 29)

> Playlist: [CampusX 100 Days of ML](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&si=OKqpjE7JagrQ9QyL). Titles for Days 13–29 verified via playlist/curriculum indexes (transcripts unavailable — mapping is title-level, labeled accordingly). Days 1–12 treated as background (no hands-on notebooks found); do not assume mastery.
> Statuses: `Notebook exists` → `Notebook documented` (banner+sections+takeaways added, code untouched) → `Notes drafted/reviewed` → `Complete` (only when verified, not merely present).

| Video | Video Title (title-level evidence) | Topic | Notebook | Notebook Documented | Notes File | Notes Status |
|---|---|---|---|---|---|---|
| Day 13 | End to End Toy Project | Toy project | — | — | — | Not started (no notebook found) |
| Day 14 | How to Frame a ML Problem (title-level) | Framing | — | — | — | Not started (no notebook found) |
| Day 15 | Working with CSV files | CSV | — (background; `read_csv` used everywhere) | — | covered in [01 note](Day_16/topic_notes.md) §4–8 | Notes drafted |
| Day 16 | Working with JSON/SQL | JSON/SQL/API intro | [ML_DAY16_JSON-deep-dive](Day_16/ML_DAY16_JSON-deep-dive.ipynb), [ML_DAY16_JSON-SQL-API](Day_16/ML_DAY16_JSON-SQL-API.ipynb) | Yes (both) | [01 note](Day_16/topic_notes.md) | Notes reviewed |
| Day 17 | Fetching Data From an API | API fetch | [DAY17](Day_17/DAY17_Fetching-Data-From-API.ipynb) | Yes | [01 note](Day_17/topic_notes.md) | Notes reviewed (gaps: no pagination/error handling) |
| Day 18 | Fetching data using Web Scraping | Scraping | [DAY18](Day_18/DAY18_Web-Scraping-Demo.ipynb) | Yes | [01 note](Day_18/topic_notes.md) | Notes reviewed (gap: no DataFrame/CSV export) |
| Day 19 | Understanding Your Data | First look | [DAY19](Day_19/DAY19_Understanding-Your-Data.ipynb) | Yes | [02 note](Day_19/topic_notes.md) | Notes reviewed |
| Day 20 | EDA using Univariate Analysis | Univariate | [DAY20](Day_20/DAY20_Univariate-Analysis.ipynb) (+ [Cheat sheet](Day_20/Matplotlib-Seaborn-Cheat-Sheet.ipynb) ref) | Yes (both) | [02 note](Day_20/topic_notes.md) | Notes reviewed (gap: `distplot` modernization) |
| Day 21 | EDA using Bivariate and Multivariate Analysis | Bivariate+ | [DAY21](Day_21/DAY21_Bivariate-Multivariate-Analysis.ipynb) | Yes | [03 note](Day_21/topic_notes.md) | Notes reviewed (3 preserved bugs documented) |
| Day 22 | Pandas Profiling | Auto EDA | [ydata](Day_22/DAY22_Pandas-Profiling-ydata.ipynb), [Sweetviz](Day_22/DAY22_Pandas-Profiling-Sweetviz.ipynb), [empty placeholder](Day_22/DAY22_Pandas-Profiling-empty-placeholder.ipynb) | Yes (all 3) | [04 note](Day_22/topic_notes.md) | Notes reviewed (gap: shared-filename conflict) |
| Day 23 | What is Feature Engineering | FE concepts | [DAY23](Day_23/DAY23_What-is-Feature-Engineering.ipynb) (outline, 0 code) | Yes | [05 note](Day_23/topic_notes.md) | Notes reviewed |
| Day 24 | Feature Scaling – Standardization | Standardization | [DAY24 full](Day_24/DAY24_Standardization-with-Outliers.ipynb), [stub](Day_24/DAY24_Feature-Scaling-stub.ipynb) | Yes (both) | [06 note](Day_24/topic_notes.md) | Notes reviewed (stub: do not use) |
| Day 25 | Feature Scaling – Normalization/MinMax/MaxAbs/Robust | Normalization | [DAY25 full](Day_25/DAY25_Normalization-MinMaxScaler.ipynb), [stub](Day_25/DAY25_Normalization-stub.ipynb) | Yes (both) | [06 note](Day_25/topic_notes.md) | Notes reviewed (gap: MaxAbs/Robust hands-on missing) |
| Day 26 | Encoding Categorical Data / Ordinal / Label | Ordinal+Label | [DAY26](Day_26/DAY26_Ordinal-Label-Encoding.ipynb) | Yes | [07 note](Day_26/topic_notes.md) | Notes reviewed |
| Day 27 | One Hot Encoding / Handling Categorical Data | OneHot | [DAY27](Day_27/DAY27_OneHot-Encoding.ipynb) | Yes | [07 note](Day_27/topic_notes.md) | Notes reviewed (2 preserved bugs documented) |
| Day 28 | Column Transformer / How to use ColumnTransformer | ColumnTransformer | [DAY28](Day_28/DAY28_Column-Transformer.ipynb) | Yes | [08 note](Day_28/topic_notes.md) | Notes reviewed (`fit_transform`-on-test bug documented) |
| Day 29 | Machine Learning Pipelines A–Z | Pipelines | [manual](Day_29/DAY29_Workflow-Without-Pipelines.ipynb), [broken](Day_29/DAY29_Sklearn-Pipelines-broken.ipynb), [pipeline](Day_29/DAY29_Titanic-Using-Pipeline.ipynb), [manual-infer](Day_29/DAY29_Predict-Without-Pipeline.ipynb), [pipe-infer](Day_29/DAY29_Pipeline-Export-Predict.ipynb) | Yes (all 5) | [09 note](Day_29/topic_notes.md) | Notes reviewed |
| Day 30+ | Function Transformer, Power Transformer, Binning, … (Day 30+ titles known) | Studied with code + executed outputs (notes pending) | [DAY30 preview (Day_30)](Day_30/DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb) (29 cells: normality checks, log1p, LogReg vs Tree, ColumnTransformer; executed) | Yes (status updated after study) | Not written yet | In progress |

## Uncertain mappings (labeled, not assumed)
- `FEAture_eng.py`, `nomialencoding.py` (root): companion scripts to Day 23/27 notebooks — unmapped line-by-line, reference only.
- Days 1–12 content (ML intro, Python/anaconda setup, job roles per curriculum index): no notebooks found; treated as background, NOT marked complete.
- No transcript access was available; titles come from playlist/curriculum indexes, not invented transcripts.

## Next Study Priorities
1. Poorly-explained-in-notebook but important: re-run accuracy cells (`null` executions in Days 24–25) to print LogReg-vs-Tree numbers; regenerate both Day-22 HTML reports under non-clobbering names and verify completeness.
2. Playlist-covered but missing hands-on: binning/binarization, feature construction (FamilySize/Title/Deck), extraction (PCA preview), MaxAbs/Robust practice, FunctionTransformer (Day 30 — only as context, not new scope).
3. Incomplete notes: none structural — all 9 drafted/reviewed; next pass = add binning/construction supplements when implemented.
4. Prerequisites for later topics: leakage discipline (`transform`-on-test), `handle_unknown="ignore"`, CV vs GridSearch distinction — drill via Day 27/28/29 practice questions before any Day-30+ material.
