# PROGRESS.md — Live work status

> **Rules live in `AGENTS.md` (permanent, includes CRITICAL DEPTH OVERRIDE + NEVER-MERGE RULE). Status lives here only.**
> Recovery: read `AGENTS.md` → read this file → inspect files → continue next unfinished task.
> Last verified: **2026-11-06**

---

## Current objective

**Course expansion is paused except for the user-requested Day 41 theory class. The current priority is applying the approved Day 39 two-layer pedagogical standard to every existing `notes.md`: keep the technical depth, teach it in a gradual, explanation-first narrative, and split each chapter into Part A (main learning chapter, sufficient on its own) and Part B (optional advanced reference). There is no mandatory word count. `Day_39/notes.md` is the approved standard; `Day_16`, `Day_17`, `Day_18`, `Day_19`, `Day_20`, `Day_21`, `Day_22`, `DAY_31`, and `DAY_32` have been rewritten and validated. `Day_41/notes.md` is a validated theory-only supplement for the user-supplied YouTube outlier class; it is not an official vendor day. Do not create other new day folders, notes, reference copies, or datasets until the existing notes have been rewritten.**

Next: rewrite the remaining existing notes in sequence, starting with Days 23–30; then Days 33–38 and 40; then Days 42–45 and 47–49; then Days 50–53; then Days 55–56.

### Folder policy (course-wide preparation)

```text
Day_XX/                      # or existing uppercase DAY_XX
├── notes.md                 # ONE unified textbook-depth document per day
├── README.md                # source map, datasets, dependencies, run notes
├── datasets/                # only when required
├── <existing personal notebooks>     # preserved, never generated anew
└── reference_notebooks/     # existing copies preserved; never delete
```

- New official numbered days are documentation-only folders: no personal notebook, no copied vendor notebook.
- Existing Days 16–36 keep their personal notebooks, `reference_notebooks/`, and extra experiments under their original names.
- Multiple official notebooks are never merged: one README map and one unified `notes.md` cover all notebooks for that day.
- The vendor clone is read-only; new READMEs map official notebooks by relative path instead of copying them.
- Existing Days 32–35 already have personal notebooks; `DAY_36` also has an existing `univariate_imputation.ipynb` stub. Preserve these files.

---

## Depth policy (from AGENTS.md)

| File | Experience |
|------|------------|
| `<personal>.ipynb` | Open section → understand what/why → comments → run → understand output |
| `notes.md` | Deep theory, math, worked examples, interview/exam prep |

- Consolidation merges **full unique content** into `notes.md` — removes duplicate **files**, not knowledge.
- No arbitrary length cap on notes.
- Interview levels 1–7 with explanatory answers when notes are expanded.

---

## Course-wide coverage table (inventory verified 2026-11-02; Days 16–22, 31, and 32 rewritten 2026-11-04–06)

Status meanings: **NOTES PRESENT** = an older notes file exists but has not been audited against the new depth standard; **NOTES COMPLETE** = deep notes written and checked (math plus source/content spot-checks); **REWRITTEN (DAY 39 STANDARD)** = two-layer Part A/Part B rewrite validated for math, structure, and content; **THEORY NOTES** = validated theory-only supplement with no official notebook or dataset; **TYPESET** = completed and math-validated on 2026-11-02; **SCAFFOLDED** = README/source map/datasets prepared, deep notes pending; **EXTRA** = no official numbered source; **UNNUMBERED** = official module without a course-day number.

| Day | Official source | Refs | Transcript | Datasets | Folder | Notes | QA |
|---|---|---:|---|---|---|---|---|
| 15 | `day15 - working with csv files` | 1 | No | 5 CSVs/TSV; BX-Books missing | `Day_15` | NOTES COMPLETE | PASS (MATH; CONTENT) |
| 16 | `day16 - working-with-json-and-sql` | 1 | No | `train.json`; MySQL `world` external | `Day_16` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 17 | `day17-api-to-dataframe` | 1 | No | `movies.csv` (API output) | `Day_17` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 18 | `day18-pandas-dataframe-using-web-scraping` | 1 | No | None | `Day_18` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 19 | `day19-understanding-your-data-descriptive-stats` | 1 | No | `train.csv` | `Day_19` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 20 | `day20-univariate-analysis` | 1 | No | `train.csv` | `Day_20` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 21 | `day21-bivariate-analysis` | 1 | No | `train.csv` | `Day_21` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 22 | `day22-pandas-profiling` | 1 | No | `train.csv` | `Day_22` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 24 | `day24-standardization` | 1 | No | `Social_Network_Ads.csv` | `Day_24` | NOTES PRESENT | PENDING AUDIT |
| 25 | `day25-normalization` | 1 | No | `wine_data.csv` | `Day_25` | NOTES PRESENT | PENDING AUDIT |
| 26 | `day26-ordinal-encoding` | 1 | No | `customer.csv` | `Day_26` | NOTES PRESENT | PENDING AUDIT |
| 27 | `day27-one-hot-encoding` | 1 | No | `cars.csv` | `Day_27` | NOTES PRESENT | PENDING AUDIT |
| 28 | `day28-column-transformer` | 1 | No | `covid_toy.csv` | `Day_28` | NOTES PRESENT | PENDING AUDIT |
| 29 | `day29-sklearn-pipelines` | 4 | No | `train.csv`; generated `.pkl` models | `Day_29` | NOTES PRESENT | PENDING AUDIT |
| 30 | `day30-function-transformer` | 1 | No | `train.csv` | `Day_30` | NOTES PRESENT | PENDING AUDIT |
| 31 | `day31-power-transformer` | 1 | Yes | `concrete_data.csv` | `DAY_31` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 32 | `day32-binning-and-binarization` | 2 | Yes | `train.csv` | `DAY_32` | **REWRITTEN (DAY 39 STANDARD)** | PASS (MATH; PEDAGOGY; CONTENT) |
| 33 | `day33-handling-mixed-variables` | 1 | Yes | `titanic.csv` | `DAY_33` | NOTES COMPLETE | PASS (MATH; CONTENT) |
| 34 | `day34-handling-date-and-time` | 1 | Yes | `messages.csv`, `orders.csv` | `DAY_34` | NOTES COMPLETE | PASS (MATH; CONTENT) |
| 35 | `day35-complete-case-analysis` | 1 | Yes | `data_science_job.csv` | `DAY_35` | NOTES COMPLETE | PASS (MATH; CONTENT) |
| 36 | `day36-imputing-numerical-data` | 2 | Yes | `titanic_toy.csv` | `DAY_36` | NOTES COMPLETE | PASS (MATH; CONTENT) |
| 37 | `day37-handling-missing-categorical-data` | 2 | Yes | `train.csv` | `Day_37` | SCAFFOLDED | PENDING |
| 38 | `day38-missing-indicator` | 3 | Yes | `train.csv`, `house-train.csv` | `Day_38` | SCAFFOLDED | PENDING |
| 39 | `day39-knn-imputer` | 1 | Yes | `train.csv` | `Day_39` | **APPROVED STANDARD** | PASS (MATH; PEDAGOGY; CONTENT) |
| 40 | `day40-iterative-imputer` | 1 | Yes | `50_Startups.csv` | `Day_40` | SCAFFOLDED | PENDING |
| 41 | YouTube theory class: What are Outliers (user-supplied) | 0 | No | None | `Day_41` | **THEORY NOTES** | PASS (MATH; STRUCTURE) |
| 42 | `day42-outlier-removal-using-zscore` | 1 | No | `placement.csv` | `Day_42` | SCAFFOLDED | PENDING |
| 43 | `day43-outlier-removal-using-iqr-method` | 1 | No | `placement.csv` | `Day_43` | SCAFFOLDED | PENDING |
| 44 | `day44-outlier-detection-using-percentiles` | 1 | No | `weight-height.csv` | `Day_44` | SCAFFOLDED | PENDING |
| 45 | `day45-feature-construction-and-feature-splitting` | 1 | No | `train.csv` | `Day_45` | SCAFFOLDED | PENDING |
| 47 | `day47-pca` | 1 | No | Synthetic | `Day_47` | SCAFFOLDED | PENDING |
| 48 | `day48-simple-linear-regression` | 2 | No | `placement.csv` | `Day_48` | SCAFFOLDED | PENDING |
| 49 | `day49-regression-metrics` | 1 | No | `placement.csv` | `Day_49` | SCAFFOLDED | PENDING |
| 50 | `day50-multiple-linear-regression` | 2 | No | Synthetic; built-in diabetes | `Day_50` | SCAFFOLDED | PENDING |
| 51 | `day51-gradient-descent` | 5 | No | Synthetic; built-in diabetes | `Day_51` | SCAFFOLDED | PENDING |
| 52 | `day52-types-of-gradient-descent` | 4 | No | Synthetic; built-in diabetes | `Day_52` | SCAFFOLDED | PENDING |
| 53 | `day53-polynomial-regression` | 1 | No | Synthetic | `Day_53` | SCAFFOLDED | PENDING |
| 55 | `day55-regularized-linear-models` | 5 | No | Built-in diabetes | `Day_55` | SCAFFOLDED | PENDING |
| 56 | `day56-lasso-regression` | 2 | No | Synthetic; built-in diabetes | `Day_56` | SCAFFOLDED | PENDING |
| 57 | `day57-elasticnet-regression` | 1 | No | Built-in diabetes | `Day_57` | SCAFFOLDED | PENDING |
| 58 | `day58-logistic-regression` | 3 | No | Synthetic | `Day_58` | SCAFFOLDED | PENDING |
| 59 | `day59-classification-metrics` | 3 | No | Kaggle heart/Iris/MNIST external | `Day_59` | SCAFFOLDED | PENDING |
| 60 | `day60-logistic-regression-contd` | 2 | No | `ushape.csv` missing; softmax synthetic | `Day_60` | SCAFFOLDED | PENDING |
| 65 | `day65-random-forest` | 7 | No | `heart.csv`; digit `train.csv` missing | `Day_65` | SCAFFOLDED | PENDING |
| 66 | `day66-adaboost` | 2 | No | Synthetic; hard-coded demo | `Day_66` | SCAFFOLDED | PENDING |
| 68 | `day68-stacking-and-blending` | 1 | No | `heart.csv` | `Day_68` | SCAFFOLDED | PENDING |
| Extra | No official Day 23 folder | 0 | No | None | `Day_23` | NOTES PRESENT (EXTRA) | UNVERIFIED |
| Extra | `gradient-boosting` (unnumbered) | 1 | No | Synthetic | `EXTRA_gradient_boosting` | SCAFFOLDED | PENDING |
| Extra | `kmeans` (unnumbered) | 1 | No | `student_clustering.csv` | `EXTRA_kmeans` | SCAFFOLDED | PENDING |

### Coverage facts and source anomalies

- **45 numbered vendor course days** exist: `15–22, 24–40, 42–45, 47–53, 55–60, 65–66, 68`.
- Vendor gaps: `23, 41, 46, 54, 61–64, 67`; the vendor repo starts at Day 15 and has no Day 69+ folders.
- `Day_41` is not a vendor folder. It is a user-requested theory-only supplement created from the YouTube class `https://youtu.be/Lln1PKgGr_M` (“What are Outliers | Outliers in Machine Learning”, CampusX, 17:07). It has no official notebook or dataset; do not treat it as a vendor source.
- **82 official notebooks** are present recursively, including multiple notebooks on Days 29, 32, 36–38, 48, 50–52, 55–56, 58–60, 65–66.
- Vendor transcripts exist only for Days **31–40**, so all other vendor rows are marked `No`; Day 41 uses a separate user-supplied YouTube transcript, and no vendor transcript use may be claimed.
- All copied dataset files were SHA-256 checked against the vendor source, normalising CRLF/LF line endings. `DAY_35/data_science_job.csv` differs only by line endings; parsed content is identical.
- `day68-stacking` is a 20-byte text file containing `nothing to display`; the real directory is `day68-stacking-and-blending`. Root `adaboost_demo.ipynb` is byte-identical to `day66-adaboost/adaboost_demo.ipynb`; it is a duplicate, not a separate lesson.
- `BX-Books.csv` (Day 15), MySQL `world` (Day 16), Kaggle inputs (Day 59), `ushape.csv` (Day 60), and digit `train.csv` (Day 65) are not bundled. Their READMEs state this explicitly; no substitutes were invented.

---

## Restoration result (verified 2026-11-02)

Cell-level mapping was produced from git history before any change; user approved keeping `DAY29_*` names.

| Day | Restored / kept | Cells |
|-----|-----------------|-------|
| 16 | `my_notebook.ipynb` (base) + `ML_DAY16_JSON-deep-dive.ipynb` | 13 + 84 |
| 20 | `my_notebook.ipynb` (base) + `Matplotlib-Seaborn-Cheat-Sheet.ipynb` | 22 + 74 |
| 22 | `my_notebook.ipynb` (base) + Sweetviz + placeholder | 8 + 10 + 3 |
| 24 | `my_notebook.ipynb` (documented base) + with-outliers + stub | 106 + 53 + 5 |
| 25 | `my_notebook.ipynb` (MinMaxScaler) + stub | 61 + 4 |
| 29 | 5 separate `DAY29_*` notebooks (4 official + broken counterexample) | 46 + 9 + 15 + 17 + 25 |
| 30 | `my_notebook.ipynb` (documented) + preview | 77 + 29 |
| 31 | `my_notebook.ipynb` | 42 |

- Day 29 improved 46-cell `my_notebook` was renamed to `DAY29_Titanic-Using-Pipeline.ipynb`; the 44-cell git copy was a subset and was not duplicated.
- All other restored notebooks were written byte-identically from `git show HEAD:<path>`.
- READMEs for affected days rewritten with accurate notebook inventories.
- Removed Day_16 raw text dumps (`*.ipynb.txt`, `campusx_dump.txt`) after confirming their content is preserved in the notebooks/git.
- `reference_notebooks/` untouched; `vendor/` untouched; `notes.md` remains one file per day.

---

## Master-notes pilot — Day 31 (2026-11-02)

`DAY_31/notes.md` was initially written to the user's master-notes standard:

- Initial version: 1,495 lines / ~80 KB; 18 chapters (objectives → prerequisites → problem → skewness → power family →
  Box–Cox derivation → Yeo–Johnson → sklearn internals → concrete experiment → 6 worked examples →
  plots → comparison/limitations/mistakes → interview L1–7 (Q1–Q40 + 4 worked problems) → exam
  (5 theory + 5 numerical + 3 analytical) → connections/summary/checklist → coverage matrix → references).
- 4 Mermaid diagrams; 104 table rows; coverage matrix maps all 22 official notebook cells.
- Official vs personal numbers reported separately; official CV leakage in the course notebook documented
  and corrected with a Pipeline (pipeline-based 5-fold YJ = 0.6737 vs official 0.6835; CV still uses full `X`).
- Transcript limitations and "additions beyond the course" disclosed.

QA performed before evaluation:

- Re-verified zeros/minima (BFS 471, Fly Ash 566, Superplasticizer 379), raw YJ Cement mean/std
  (9.464559 / 1.024432), skewness demo (1.1384 → 0.0275, λ = −0.120722), Box–Cox hand values,
  and all reported scores/lambdas against the dataset and saved official outputs.
- Fixed one rounding error (Yeo–Johnson λ=0.5 at x=10: 4.6333 → 4.6332) and one imprecise branch label
  in the Box–Cox vs Yeo–Johnson table.
- Prior 60,155-byte notes backed up outside the workspace (temp `DAY31_notes_previous.md`); no unique
  content intentionally discarded.

Self-assessment against the master-prompt dimensions: **93/100** (source fidelity 20/20, theory 19/20,
mathematics 14/15, examples 14/15, interview/exam 14/15, implementation 9/10, structure 3/5). This score
was for the initial version and is historical: the refinement pass later found the cell-20 mislabel,
among other corrections.

---

## Day 31 refinement pass (2026-11-02)

Refined the existing `DAY_31/notes.md` in place. No rewrite from scratch, no new notes file, no notebook
or dataset changes; worked examples, interview/exam questions, diagrams, and source distinctions preserved.

- Current size: **1,700 lines / 106,214 bytes / ~15,045 words**; 18 sections, 40 interview questions,
  12 follow-ups, 4 Mermaid diagrams.
- Corrected OLS conceptual conflation: separated squared-residual loss from feature leverage; rewrote
  §4.2 and added §4.4 (hat matrix, \(h_{ii}\), Cook's distance, worked \(x=(1,2,3,4,10)\), \(h_{10}=0.92\)).
- Fixed workflow order in §2 and §16.2 (split before fitting preprocessing); made §16.2 clear that the
  course compares methods on the holdout, so those scores are model-selection evidence.
- Rewrote §9.1 to distinguish public fitted attributes (`lambdas_`, `n_features_in_`,
  `feature_names_in_`) from private `_scaler`; clarified `standardize=False` and NaN behaviour.
- Added the cell-20 source-fidelity trap: cell 17 reassigns `pt` to a full-data Yeo–Johnson fit, so cell 20's `box_cox_lambdas` column is actually full-data Yeo–Johnson (not Box–Cox); the second column is the training-split Yeo–Johnson fit from cell 16. Reported the actual training fits in §10.3 and verified the visible differences (Cement 0.1695 vs 0.1770/0.1743; Age 0.0018 vs 0.0666/0.0199).
- Labelled every reported score by split, random state, CV strategy, preprocessing, and full-data CV;
  added the distinction between preprocessing leakage and holdout reuse; corrected the earlier
  "leakage-safe" overclaim for the personal notebook (it uses a Pipeline but still CVs on full `X`).
- Expanded §10.4 into functional-form, leverage, residual, and cross-method mechanisms, plus explicit
  limits: no normality guarantee, shift confound, no significance claim, no independent-CV claim.
- Corrected §11 Example 2 (Jacobian interpretation), Example 5 (constant-feature behaviour; executable
  code added), and Example 6 (official split labels); added the Box–Cox/Yeo–Johnson anchor distinction.
- Expanded 12 interview answers (Q4, Q5, Q9, Q11, Q16, Q18, Q22, Q24, Q27, Q30, Q33, Q40) and added
  transitions between sections.
- Re-verified official outputs, personal scores, lambdas, skewness, raw YJ mean/std, small-sample
  \(\lambda=-0.120722\), profile-likelihood grid, leverage example, and constant-feature behaviour
  (YJ \(\lambda=1\), all-zero standardised output; Box–Cox `ValueError: Data must not be constant`).
- Corrected stale `DAY_31/README.md` claims: holdout results now 0.6276 → 0.8048/0.8162; removed
  nonexistent "Appendix A–I"/"cheat sheet" references; clarified CV/holdout wording.

Verification (pre-formatting counts): display math 47/47, inline math 458/458, 54 fences, 4 Mermaid blocks,
40 questions, 12 follow-ups, no mojibake, no TODO/FIXME. `notes.md` is the only notes file;
`reference_notebooks/`, `my_notebook.ipynb`, `datasets/`, and `vendor/` were not modified.

---

## Day 31 math formatting pass (2026-11-02)

Converted every mathematical expression in `DAY_31/notes.md` to Markdown-compatible LaTeX outside fenced code:

- `\(...\)` → `$...$`; `\[...\]` → `$$...$$`; no raw `\(...\)` or `\[...\]` remains outside code.
- **478 inline expressions** and **47 display blocks**; **8 `aligned` derivations** and **3 `cases` blocks**.
- Converted headings and prose symbols: λ → `$\lambda$`, R² → `$R^2$`, and numeric arrow chains to `$...\to...$`.
- Preserved `\\[8pt]`/`\\[6pt]` row spacing, code fences, tables, and Mermaid diagrams.
- Current size: **1,723 lines / 105,803 bytes / 15,105 words**; 54 fences, 4 Mermaid, 40 questions, 12 follow-ups.
- Validation: all **525 math items rendered by KaTeX 0.16.22 with 0 errors**; balanced `aligned`/`cases`;
  0 blank-line issues around display blocks; 0 raw LaTeX commands in prose; 0 Unicode math symbols in prose;
  balanced dollar delimiters. KaTeX is the VS Code math renderer; the constructs used are supported by
  GitHub MathJax, but a final visual GitHub preview remains a user-facing check.
- Backup: `%TEMP%\opencode\DAY31_notes_before_math_format.md`. No notebook, dataset, README, `vendor/`,
  or `reference_notebooks/` changes.

---

## Day 32 deep-notes pass (2026-11-03)

`DAY_32/notes.md` was written as one unified chapter covering both official notebooks and both preserved personal notebooks. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **2,665 lines / 134,705 bytes / 21,027 words**; 114 fences, 2 Mermaid diagrams, 122 math items (53 display, 69 inline), all rendered by the KaTeX checker with 0 errors; no raw LaTeX delimiters.
- Coverage: official `day32.ipynb` 27/27 cells and `binarization.ipynb` 21/21 cells; personal `BINNING_BINERIZATION.ipynb` 23 cells and `BINNERZATION.ipynb` 20 cells mapped in the coverage matrix. The two personal notebooks remain separate.
- Verified source-fidelity issues: official binning CV passes raw `X` instead of `X_trf`; `discretize` repeats the bug; official trees are unseeded; the official holdout differences are exactly one test prediction (143-row test set); `pd.cut` boundary labels disagree with `KBinsDiscretizer` codes; `Binarizer(copy=False)` mutates `X_train`; local `clf_trans` and `'mdoel'` naming errors are noted without changing code.
- Verified data facts: 891 raw rows; 714 complete rows after selecting `Age`, `Fare`, `Survived`; 571/143 split; Age edges and 15-bin counts; family distribution 404 zeros and 310 positives; official/local metrics and pipeline comparisons separated by split, CV folds, and model seed.
- Supplementary leakage-safe checks: seeded binning pipeline CV 0.6303 baseline vs 0.6753 quantile (repeated 0.6255 vs 0.6597); seeded binarization CV 0.6541 raw family vs 0.6262 binary (repeated 0.6399 vs 0.6205). These are explicitly labelled supplementary, not official outputs.
- Validation: `check_notes.py` reports balanced fences, no raw LaTeX, 114 fences, 2 Mermaid blocks; `extract_math.py` + KaTeX render all 122 math items with 0 errors. `DAY_32/README.md` updated from pending to complete.

---

## Day 33 deep-notes pass (2026-11-03)

`DAY_33/notes.md` was written as one unified chapter covering the official mixed-variables notebook and the preserved personal notebook. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **2,076 lines / 100,024 bytes / 15,336 words**; 86 fences, 23 math items (16 display, 7 inline), all rendered by the KaTeX checker with 0 errors; no raw LaTeX delimiters, no Mermaid diagrams.
- Coverage: official `Untitled.ipynb` 14/14 cells and personal `mixedata.ipynb` 11/11 cells mapped in the coverage matrix. The personal notebook remains separate and unchanged.
- Verified data facts: 891×4 raw data; `Cabin` missing 687 (204 observed); 147 non-missing distinct cabins plus `NaN`; `cabin_num` 200 non-missing, 691 missing, 92 distinct; four observed letter-only cabins (`D`, `D`, `D`, `T`); deck counts C 59, B 47, D 33, E 32, A 15, F 13, G 4, T 1; `number` 752 numeric and 139 `A`; `Ticket` 887 numeric endings and 4 `LINE`; 681 distinct tickets with 210 duplicate rows; `ticket_cat` 230 non-missing, 661 missing, 43 non-missing prefixes plus `NaN`; target 549/342.
- Source-fidelity findings: official notebook has 14 cells, no train/test split, no model, no ticket-prefix normalisation, no multi-cabin handling, no missing indicator, and leaves `cabin_num` as text; personal notebook has 11 cells, uses the absolute `DAtasets` path, combines cabin inspection and extraction in one cell, emits a non-raw-regex `SyntaxWarning`, does not inspect `ticket_cat`, and plots all 678 ticket-number levels.
- README corrected from `Datasets` to the actual `DAtasets` path; the dataset folder was not renamed and the notebook absolute path was not edited.
- Validation: `check_notes.py` reports balanced fences, no raw LaTeX, 86 fences; `extract_math.py` + KaTeX render all 23 math items with 0 errors. `DAY_33/README.md` updated from pending to complete.

---

## Day 34 deep-notes pass (2026-11-03)

`DAY_34/notes.md` was written as one unified chapter covering the official Day 34 notebook and the preserved personal notebook. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **3,670 lines / 163,370 bytes / 25,188 words**; 206 fences, 4 Mermaid diagrams, 133 math items (67 display, 66 inline), all rendered by the KaTeX checker with 0 errors; no raw LaTeX delimiters, no mojibake.
- Coverage: official `working-with-dates-and-time.ipynb` **42/42 cells** mapped in the coverage matrix. The personal `DATE_TIME.ipynb` (28 cells) remains separate and unchanged.
- Data facts verified: both CSVs have 1,000 rows and no missing values; orders range 2018-07-10 to 2019-12-16 (394 distinct dates, 818 products, 31 cities); messages range 2012-01-01 02:58 to 2017-01-18 00:11 (999 distinct timestamps, all seconds zero); month/quarter/semester/weekday/weekend counts checked.
- Source-fidelity findings: the official saved outputs use Python 3.8.3-era pandas (`datetime64[ns]`, `dt.week`, `np.timedelta64(1,'M')`, `today` = 2021-04-21). In the current environment, `pd.to_datetime` returns `datetime64[us]`, `dt.week` raises `AttributeError`, and month-unit division raises `ValueError`; these are documented with equivalent reruns. The official `dt.week` values match `dt.isocalendar().week` on the checked rows.
- Notebook outputs are labelled **Official** (saved 2021 output) or **Rerun** (pandas 3.0.5 / NumPy 2.5.3). The official notebook contains no plots; Section 16 interprets the tabular distributions and states what cannot be concluded.
- Validation: `check_notes.py` reports balanced fences, 4 Mermaid blocks, no raw `\(`/`\[`; `extract_math.py` + KaTeX render all 133 math items with 0 errors. `DAY_34/README.md` updated to list the 42-cell reference copy, `Datasets/` (capital D), sources, dependencies, reference-opening notes, and `NOTES COMPLETE`.

---

## Day 35 deep-notes pass (2026-11-03)

`DAY_35/notes.md` was written as one unified chapter covering the official Day 35 notebook and the preserved personal notebook. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **3,320 lines / 161,537 bytes / 24,261 words**; 114 fences, 4 Mermaid diagrams, 461 math items (123 display, 338 inline), all rendered by the KaTeX checker with 0 errors; no raw LaTeX delimiters.
- Coverage: official `day35.ipynb` **20/20 cells** mapped in the coverage matrix. The personal `missing_data.ipynb` (20 cells) remains separate and unchanged.
- Source fidelity: the copied reference notebook is byte-identical to the vendor source. Both dataset copies parse identically (19,158 rows × 13 columns); the byte hashes differ only by CRLF/LF line endings. The personal notebook reads the root copy by absolute path.
- Data facts verified by rerun: five selected columns; 17,182 retained (89.6858%); 1,976 dropped (10.3142%); missing counts and percentages; row-pattern counts; target counts and prevalence; included/excluded means, SDs, SMDs, Welch/KS tests; missingness-by-target rates; category proportions.
- Source-fidelity findings: the official cell 10 histogram grid has four axes because three numeric columns fill a 2×2 layout with one unused axes; the string columns are skipped. Cell 19 is empty. No train/test split, model, or imputation is present in the official notebook.
- Notebook outputs are labelled **Official** or **Rerun**; supplementary diagnostics are labelled separately and are not attributed to the course.
- Validation: `check_notes.py` reports balanced fences, 4 Mermaid blocks, no raw `\(`/`\[`; `extract_math.py` + KaTeX render all 461 math items with 0 errors. `DAY_35/README.md` updated to list the 20-cell reference copy, both dataset copies, sources, dependencies, reference-opening notes, and `NOTES COMPLETE`.

## Day 39 pedagogical rewrite pilot (2026-11-03)

`Day_39/notes.md` was restructured in place from **2,422 lines / 17,494 words** to **1,023 lines / 9,227 words**. The rewrite preserves the technical content while separating it into two layers:

- **Part A — Main Learning Chapter:** **6,022 words**, 4 Mermaid diagrams, narrative explanation-first teaching, one running Titanic example, terminology introduced before use, formulas derived and interpreted, implementation connected to theory, interview essentials, exam practice, one-page revision.
- **Part B — Advanced Reading and Reference:** **3,203 words**, formal notation and algorithm, full distance-metric table, NaN-Euclidean and weighting examples, curse-of-dimensionality derivation, API internals, missingness/uncertainty, expanded experiment tables, extended interview Levels 1–7, extended exam questions, coverage/source fidelity, references.
- **Preserved:** official KNN accuracy 0.7151 vs mean 0.6927, rerun KNN 0.7039, 712/179 split, 148/29 missing ages, confusion matrices, paired comparison, $k$/weights and CV tables, imputed distribution, correlations, `add_indicator` result, actual row 334 calculation, all important formulas and limitations.
- **Validation:** balanced fences; 22 fences; 4 Mermaid blocks; 194 math items (25 display, 169 inline) rendered by KaTeX with **0 errors**; **0 raw LaTeX delimiters**. `README.md` remains accurate; no notebook, dataset, `vendor/`, or reference file changed.
- **Quality result:** main chapter can be read without Part B; advanced material remains available; interview and examination coverage preserved. No duplicate notes file was created.

## Day 31 Day 39-standard rewrite (2026-11-04)

`DAY_31/notes.md` was rewritten in place from the older 1,723-line chapter to a two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **1,476 lines / 78,031 bytes / 11,145 checker words** (12,025 regex words); Part A **8,468 words**, Part B **3,557 words**.
- Preserved: Box–Cox/Yeo–Johnson derivations, `lambdas_`, `standardize`, the concrete experiment, official vs personal outputs, leakage analysis, source-fidelity audit, worked examples, interview/exam material, and limitations.
- Validation: 433 math items rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 36 fences; 1 Mermaid diagram; balanced fences.
- `DAY_31/README.md` updated with the Part A/Part B study order and rewrite status.
- Backup: `%TEMP%\opencode\Day_31_notes_before_rewrite.md`.

## Day 32 Day 39-standard rewrite (2026-11-04)

`DAY_32/notes.md` was rewritten in place to the two-layer standard. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **2,499 lines / 127,181 bytes / 19,759 checker words** (19,660 regex words); Part A **14,925 words**, Part B **4,733 words**.
- Structure: Part A covers the problem, three binning strategies, binarization, API internals, both official notebooks, both local notebooks, result audit, workflow, mistakes, essential interview/exam, and revision. Part B holds formal edge algorithms, implementation internals, full cell-level source audit, extended comparisons, extended interview/exam, references, and glossary.
- Preserved: official/local saved metrics, one-row holdout difference, `pd.cut` boundary mismatch, `Binarizer(copy=False)` mutation, unseeded-tree variability, official CV bugs, source-fidelity audit, and all supplementary leakage-safe comparisons.
- Validation: 154 math items rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 108 fences; 3 Mermaid diagrams; balanced fences.
- `DAY_32/README.md` updated with the Part A/Part B study order and rewrite status.
- Backup: `%TEMP%\opencode\Day_32_notes_before_rewrite.md`.

## Day 16 Day 39-standard rewrite (2026-11-04)

`Day_16/notes.md` was rewritten in place from a 297-line concatenation of older topic guides into a unified two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **2,055 lines / 68,084 bytes / 10,470 checker words** (9,898 regex words); Part A **6,911 words**, Part B **2,985 words**.
- Coverage: JSON foundations and orientations, `read_json`/`to_json`, JSON Lines, `StringIO`, the `json` module, `json_normalize` (`record_path`, `meta`, `sep`, `max_level`, `meta_prefix`, `errors`), `explode`, dates, Unicode, APIs, SQL, and all three notebooks.
- Verified data facts: `train.json` 39,774 × 3, 20 cuisines, ingredient lists 1-65 (mean 10.7677), 13,081,987 bytes; `world.sql` 239 countries, 17 missing life expectancy, 167 with life expectancy > 60, 984 country-language rows; official API output 160 × 7; local GitHub output 30 × 36; local SQL output 167 × 15.
- Source-fidelity audit: official notebook 11 cells, `my_notebook` 13, deep-dive 84; absolute paths, live API snapshots, mojibake, `!pip install`, missing `conn.close()`, and the stale `database.sqlite` reference are documented without changing notebook code.
- Validation: 18 math items rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 168 balanced fences; 0 Mermaid diagrams.
- `Day_16/README.md` updated with the Part A/Part B study order, verified dataset shapes, dependencies, and rewrite status.
- Backup: `%TEMP%\opencode\Day_16_notes_before_rewrite.md`.

## Day 17 Day 39-standard rewrite (2026-11-04)

`Day_17/notes.md` was rewritten in place from an 11,545-byte concatenation of older topic guides into a unified two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made. The redundant `DAY17_Fetching-Data-From-API.ipynb.txt` dump was removed after confirming all nine local-notebook cell sources were preserved in `my_notebook.ipynb`.

- Current size: **1,456 lines / 43,073 bytes / 6,833 checker words** (6,440 regex words); Part A **4,360 words**, Part B **1,937 words**.
- Coverage: HTTP/REST/JSON foundations, `requests` and `Response`, status codes, timeouts, JSON envelopes, DataFrame construction, pagination, TMDb top-rated example, JSONPlaceholder/JSONFakery examples, snapshots, API-key security, source-fidelity audit, and interview/exam material.
- Verified source facts: official `day17.ipynb` 11 cells; local `my_notebook.ipynb` 9 cells; TMDb saved frame 8,551 x 7; `movies.csv` 8,551 x 8 with `Unnamed: 0`; pagination `427 x 20 + 11 = 8,551`; local JSONFakery result 20 x 2.
- Source-fidelity issues documented without changing notebook code: hard-coded TMDb key, `df.head()` before `df` is defined with out-of-order execution counts, `df.append` removed in pandas 2.0, `to_csv` index artifact, missing status/timeout/error handling, live APIs without snapshots, stale README references, and the redundant text dump.
- Validation: 11 math items rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 104 balanced fences; 0 Mermaid diagrams.
- `Day_17/README.md` updated with actual files, dataset source, dependencies, live-API caveats, Part A/Part B study order, and rewrite status.
- Backup: `%TEMP%\opencode\Day_17_notes_before_rewrite.md`.

## Day 18 Day 39-standard rewrite (2026-11-04)

`Day_18/notes.md` was rewritten in place from an 11,966-byte concatenation of older topic guides into a unified two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made. The redundant `DAY18_Web-Scraping-Demo.ipynb.txt` dump was removed after confirming all 22 local-notebook cell sources were preserved in `my_notebook.ipynb`.

- Current size: **2,127 lines / 60,103 bytes / 8,993 checker words** (8,681 regex words); Part A **6,083 words**, Part B **2,452 words**.
- Coverage: HTML/tree/CSS foundations, `requests`, BeautifulSoup `find`/`find_all`/`select`/`get_text`, the official 1,000-page AmbitionBox loop, local selector demo, cleaning ratings/reviews/ages/employees/HQ, pagination, ethics, source-fidelity audit, and interview/exam material.
- Verified source facts: official `day18.ipynb` 33 cells; local `my_notebook.ipynb` 22 cells; official first page 30 cards, 30 ratings, 30 review counts, `df.shape (30, 7)`; 1,000 pages at 30 cards/page gives 30,000 rows and `final.shape (30000, 7)`; displayed company count 477,887, so 30,000 rows is about 6.28%.
- Source-fidelity issues documented without changing notebook code: no status/timeout/delay, no saved CSV, `final.append` removed in pandas 2.0, bare `except`, positional `infoEntity` indexing, `df` overwritten to the last page, unseeded `df.sample`, selector drift, malformed local page URL, invalid `pip install lxml` cell, no `None` guards, no local outputs, and a 2026-11-04 verification-page check.
- Validation: 12 math items rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 172 balanced fences; 0 Mermaid diagrams.
- `Day_18/README.md` updated with actual files, no-dataset caveat, dependencies, live-site caveats, Part A/Part B study order, and rewrite status.
- Backup: `%TEMP%\opencode\Day_18_notes_before_rewrite.md`.

## Day 19 Day 39-standard rewrite (2026-11-04)

`Day_19/notes.md` was rewritten in place from a 12,168-byte older topic guide into a unified two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **1,877 lines / 56,018 bytes / 8,906 checker words**; Part A **6,144 words**, Part B **2,762 words**.
- Coverage: the seven first-look checks, Titanic data facts, missing-value counts and percentages, `describe()` interpretation, mean/median/skew, quartiles and IQR, duplicates, Pearson correlation and the heatmap, source-fidelity audit, extended comparisons, interview/exam material, references, and glossary.
- Verified data facts: 891 x 12; Age missing 177 (19.8653%), Cabin 687 (77.1044%), Embarked 2 (0.2245%); total missing 866/10,692 = 8.0995%; duplicates 0; Survived 342 (38.3838%), died 549 (61.6162%); correlations `Pclass -0.338481`, `Fare 0.257307`, `Parch 0.081629`, `SibSp -0.035322`, `Age -0.077221`, `PassengerId -0.005007`; `Pclass`-`Fare` correlation -0.5495; Fare upper fence 65.6344 with 116 above; Age upper fence 64.8125 with 11 above; Sex, Pclass, and Embarked group survival rates; Ticket/Cabin/Embarked `describe(include="all")` values.
- Source-fidelity issues documented without changing notebook code: official `day19.ipynb` has 17 cells, local `my_notebook.ipynb` has 23 cells (9 code, 14 markdown); official `df.corr()` saved output works in the older pandas version, but pandas 3.0.5 raises `ValueError` on string columns; official relative `train.csv` path is not present under `reference_notebooks/`; local absolute root path is byte-identical to `datasets/train.csv`; the local heatmap cell has no execution count and its markdown reports a kernel crash after rendering; neither notebook sets `random_state`.
- Validation: 57 math items (40 display, 17 inline) rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 82 balanced fences; 0 Mermaid diagrams.
- `Day_19/README.md` updated with actual files, dataset details, pandas 3.0.5 caveats, study order, and rewrite status.
- Backup: `%TEMP%\opencode\Day_19_notes_before_rewrite.md`.

## Day 20 Day 39-standard rewrite (2026-11-04)

`Day_20/notes.md` was rewritten in place from a 267-line / 11,751-byte older guide into a unified two-layer chapter. No notebook, dataset, cheat-sheet, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **2,659 lines / 84,993 bytes / 13,457 checker words**; Part A **8,988 words**, Part B **4,461 words**.
- Coverage: univariate categorical plots (countplot, bar chart, pie), univariate numeric plots (histogram, bin choice, KDE/distplot, boxplot, IQR fences), Titanic Age/Fare/Embarked/Sex/Pclass/Survived facts, Matplotlib figure/axes internals, pandas/Seaborn APIs, deprecations, source-fidelity audit, cheat-sheet audit, extended comparisons, and extended interview/exam material.
- Verified source facts: official `day20.ipynb` 21 cells (13 code, 8 markdown); local `my_notebook.ipynb` 22 cells (9 code, 13 markdown); cheat sheet 74 cells (34 code, 40 markdown), all code cells unexecuted and no saved outputs. Official uses Embarked countplot, Sex pie, Age `bins=5`, Age distplot/boxplot, and Age min/max/mean/skew. Local uses Pclass countplot, Survived pie, bare `plt.hist`, Age `bins=300`, Age distplot, Fare boxplot, and `Age.min()` saved kernel-crash error; no saved plot outputs. Local reads the absolute root `train.csv`; root and `datasets/train.csv` are byte-identical (SHA-256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`).
- Verified data facts: 891 x 12; Embarked S 644 / C 168 / Q 77 / NaN 2; Sex 577/314; Survived 549/342; Pclass 216/184/491; Age mean 29.699118, median 28, std 14.526497, Q1 20.125, Q3 38, min 0.42, max 80, skew 0.3891; Fare mean 32.204208, median 14.4542, std 49.693429, Q1 7.9104, Q3 31, min 0, max 512.3292, skew 4.7873; Fare 116 above upper fence 65.6344; Age 11 above 64.8125; histogram bins 5 counts `[100, 346, 188, 69, 11]`; Freedman-Diaconis about 20 bins.
- Source-fidelity and version notes: `sns.distplot` exists in seaborn 0.13.2 but is deprecated and slated for removal in v0.14.0; legacy positional `sns.countplot(df['Embarked'])` still works; old Matplotlib axes repr differs; pandas 3.0 uses `str` text dtype; the pre-rewrite README pointed to nonexistent `reference_notebooks/reference_notebook.ipynb` instead of `reference_notebooks/day20.ipynb`.
- Validation: 135 math items (47 display, 88 inline) rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 106 balanced fences; 0 Mermaid diagrams. `check_notes.py` reports 13 headings because code comments begin with `#`; structural H1 headings are title, Part A, and Part B.
- `Day_20/README.md` updated with actual files, dataset details, dependencies, version caveats, study order, and rewrite status.
- Backups: `%TEMP%\opencode\Day_20_notes_before_rewrite.md` (11,751 bytes); `%TEMP%\opencode\Day_20_README_before_rewrite.md` (1,574 bytes).

## Day 21 Day 39-standard rewrite (2026-11-04)

`Day_21/notes.md` was rewritten in place from a 322-line / 14,624-byte older guide into a unified two-layer chapter. No notebook, dataset, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **3,303 lines / 106,709 bytes / 16,805 checker words**; Part A **11,932 words**, Part B **4,873 words**.
- Coverage: bivariate/multivariate framing and plot-choice map; tips scatterplot/relplot; Titanic barplot/boxplot/catplot; Age-by-survival distplot/KDE; Pclass crosstab heatmaps; Parch/SibSp clustermaps; iris pairplot; flights lineplot and month-by-year clustermap; sparse pivot-table heatmap; Matplotlib/Seaborn internals; source-fidelity audit; extended comparisons; interview Levels 1–7; exam material.
- Verified source facts: official `reference_notebooks/day21.ipynb` 33 cells (25 code, 8 markdown); local `my_notebook.ipynb` 26 cells (13 code, 13 markdown). Official cell 9 uses undefined `df` and old positional `scatterplot`; official barplot/boxplot/lineplot cells use old positional APIs; official `groupby("Embarked").mean()` and `flights.groupby("year").sum()` fail in pandas 3.0; `distplot` is deprecated; saved flights month names are full names, current are abbreviations. Local cell 17 has `Sibsp` typo plus duplicate clustermap; local pivot table is 88 × 6 with 185/528 non-null cells; local pairplot saved text reports 20 axes while current creates 16; local notebook has no saved outputs for relplot, catplot, boxplot, distplot, or first heatmap.
- Verified data facts: Titanic 891 × 12; Pclass × Survived `[[80,136],[97,87],[372,119]]`; survival 62.96%/47.28%/24.24%; Sex female 233/314 (74.20%), male 109/577 (18.89%), OR 12.35, RR 3.93; Embarked C 93/168 (55.36%), Q 30/77 (38.96%), S 217/644 (33.70%); Age missing 177 (125 died, 52 survived); Age by survival died mean 30.626/median 28, survived mean 28.344/median 28; tips r 0.6757, R² 0.4566, slope 0.1050, intercept 0.9203; iris 150 rows/50 per species; flights 144 rows, total 40,363, 1949 1,520, 1960 5,714, growth 3.759; chi-square/Cramér's V values verified for Pclass, Sex, Embarked, SibSp, Parch.
- Source-fidelity and version notes: `sns.distplot` works in seaborn 0.13.2 but is deprecated; positional `scatterplot`/`barplot`/`boxplot`/`lineplot` raise `TypeError`; pandas 3.0 rejects `groupby(...).mean()` on strings and `groupby(...).sum()` on categorical month; old Matplotlib axes repr differs; pandas 3.0 uses `str` text dtype. The pre-rewrite README pointed to nonexistent `DAY21_*` notebooks and `reference_notebooks/reference_notebook.ipynb`.
- Validation: 105 math items (32 display, 73 inline) rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 146 balanced fences; 0 Mermaid diagrams. `check_notes.py` reports 3 structural H1 headings.
- `Day_21/README.md` rewritten to the actual files, 26/33-cell notebook differences, dataset facts, dependencies/version caveats, run notes, study order, and rewrite status.
- Backups: `%TEMP%\opencode\Day_21_notes_before_rewrite.md` (14,624 bytes); `%TEMP%\opencode\Day_21_README_before_rewrite.md` (2,445 bytes).

## Day 22 Day 39-standard rewrite (2026-11-06)

`Day_22/notes.md` was rewritten in place from the 17,311-byte older guide into a unified two-layer chapter. No notebook, dataset, report artifact, `vendor/`, or `reference_notebooks/` changes were made.

- Current size: **1,784 lines / 70,682 bytes / 10,819 checker words**; Part A **7,418 words**, Part B **3,405 words**.
- Coverage: all four distinct notebooks (ydata-profiling, Sweetviz, empty placeholder, official legacy), the current ydata report, the older root ydata duplicate, and the official pandas-profiling report. The notebooks remain separate.
- Verified data facts: 891 x 12; 866 missing cells (8.1%); 0 duplicate rows; Sex survival female 233/314 (74.20%) vs male 109/577 (18.89%), OR about 12.35; Pclass survival 62.96% / 47.28% / 24.24%; Age missing by class 30 / 11 / 136; Fare 15 zeros, skew 4.7873; Age skew 0.3891; correlations `Survived`-`Pclass` -0.3385, `Survived`-`Fare` 0.2573, `Pclass`-`Fare` -0.5495.
- Source-fidelity findings: no Sweetviz HTML artifact exists; both local HTML files are ydata-profiling v4.18.4; the Sweetviz notebook claims generation but the artifact is absent; both local notebooks show kernel crashes after report generation; the saved ydata report predates the notebook timestamp, so the exact run is uncertain; the old notes referenced a nonexistent ydata notebook and the old README referenced a nonexistent reference notebook.
- Validation: 57 math items (18 display, 39 inline) rendered by KaTeX with **0 errors**; 0 raw LaTeX delimiters; 30 balanced fences; 0 Mermaid diagrams; 3 structural H1 headings.
- `Day_22/README.md` rewritten with actual notebook/report inventory, environment differences, report caveats, study order, and privacy note.
- Backups: `%TEMP%\opencode\Day_22_notes_before_rewrite.md` (17,311 bytes); `%TEMP%\opencode\Day_22_README_before_rewrite.md` (1,817 bytes).

## Day 41 theory-only notes (2026-11-06)

`Day_41/notes.md` was created at the user's explicit request from the YouTube class `https://youtu.be/Lln1PKgGr_M` (“What are Outliers | Outliers in Machine Learning”, CampusX, 17:07). This is an exception to the no-new-day-folder policy: the vendor repository has no Day 41, and this day has no official notebook or dataset.

- Current size: **1,647 lines / 10,757 words**; 127 math items (38 display, 89 inline) rendered by KaTeX with **0 errors**; 12 balanced fences; 0 Mermaid diagrams; 0 raw LaTeX delimiters; 3 structural H1 headings.
- Coverage: outlier definition and examples; effects on statistics, linear/logistic regression, boosting, deep learning, and trees; trimming, capping, missing-value treatment, and discretization; previews of z-score, IQR, percentile, and Winsorization methods; Part B robust statistics, influence diagnostics, multivariate detection, code patterns, and extended interview/exam material.
- Source discipline: the auto-generated transcript (382 snippets) is noisy; examples are reconstructed rather than quoted. No official notebook results are claimed. Supplementary mathematics and practice are clearly separated from the video content.
- `Day_41/README.md` added with the source, theory-only status, study order, and validation note.
- No notebook or dataset was created.

---

## Completed (do not repeat)

1. [x] Permanent rules + depth override + gold-standard/never-merge section → `AGENTS.md`
2. [x] CampusX `vendor/` clone (untouched)
3. [x] Day_31 pilot + cleanup
4. [x] Day_16–30 cleanup (refs kept)
5. [x] Inspection: official notebook counts vs personal notebooks (Days 16–31)
6. [x] Cell-level restoration mapping + user verification
7. [x] Notebook separation restored (Days 16, 20, 22, 24, 25, 29, 30)
8. [x] READMEs updated for restored days
9. [x] Day 31 notes refinement + math-typesetting pass (accuracy, source fidelity, CV/holdout distinction, interview depth, Markdown LaTeX)
10. [x] Course-wide inventory, 26 target folders, README/source maps, dataset copies, and progress table
11. [x] `Day_15/notes.md` deep pass: 36/36 cells covered, 41 math items render, key claims checked against reruns, README updated
12. [x] `DAY_32/notes.md` deep pass: 48/48 official cells mapped, 122 math items render, source-fidelity bugs and reruns documented, README updated
13. [x] `DAY_33/notes.md` deep pass: 25/25 notebook cells mapped (14 official + 11 personal), 23 math items render, extraction facts and source-fidelity issues verified, README corrected and updated
14. [x] `DAY_34/notes.md` deep pass: 42/42 official cells mapped, 133 math items render, date/time facts and version-dependent outputs verified, README updated
15. [x] `DAY_35/notes.md` deep pass: 20/20 official cells mapped, 457 math items render, retention/diagnostics/target facts verified, README updated
16. [x] `DAY_36/notes.md` deep pass: 53/53 official cells mapped (30 mean/median + 23 arbitrary), 630 math items render, variance/covariance formulas and notebook outputs verified, README updated
17. [x] `DAY_31/notes.md` Day 39-standard rewrite: 1,476 lines, 433 math items render, source fidelity and experiment results preserved, README updated
18. [x] `DAY_32/notes.md` Day 39-standard rewrite: 2,499 lines, 154 math items render, 48 official cells and both local notebooks covered, README updated
19. [x] `Day_16/notes.md` Day 39-standard rewrite: 2,055 lines, 18 math items render, JSON/API/SQL and all three notebooks covered, README updated
20. [x] `Day_17/notes.md` Day 39-standard rewrite: 1,456 lines, 11 math items render, API/JSON/pagination and both notebooks covered, README updated
21. [x] `Day_18/notes.md` Day 39-standard rewrite: 2,127 lines, 12 math items render, web scraping/BeautifulSoup and both notebooks covered, README updated
22. [x] `Day_19/notes.md` Day 39-standard rewrite: 1,877 lines, 57 math items render, descriptive statistics/Titanic EDA and both notebooks covered, README updated
23. [x] `Day_20/notes.md` Day 39-standard rewrite: 2,659 lines, 135 math items render, univariate Titanic EDA and all three notebooks covered, README updated
24. [x] `Day_21/notes.md` Day 39-standard rewrite: 3,303 lines, 105 math items render, bivariate/multivariate EDA and both notebooks covered, README updated
25. [x] `Day_22/notes.md` Day 39-standard rewrite: 1,784 lines, 57 math items render, all four notebooks and both report generations covered, README updated
26. [x] `Day_41/notes.md` theory-only supplement: 1,647 lines, 127 math items render, YouTube transcript used, no notebook/dataset, README added
27. [ ] Day 37+ deep notes and remaining scaffolded days

---

## Remaining (next unfinished first)

1. **Verification pass** — confirm every restored notebook opens, matches its source (or documented improvement), and that README inventory matches disk.
2. **Pedagogical rewrite of existing notes** — `Day_39` is the approved standard; `Day_16`, `Day_17`, `Day_18`, `Day_19`, `Day_20`, `Day_21`, `Day_22`, `DAY_31`, and `DAY_32` are rewritten and validated. `Day_41` is a validated theory-only supplement. Next rewrite Days 23–30, then Days 33–38/40, 42–45/47–49, 50–53, and 55–56 using the two-layer, explanation-first standard. Course expansion remains paused except for explicit user requests.
3. **Notebook polish** — meaningful headings + short what/why + useful code comments, preserving implementations. Day 32 has local typos and duplicated draft markdown to address later; do not rewrite the working code.
4. **Future day notes** — after the existing-note rewrite is complete, resume Day 37+ only under the approved standard; inspect each personal notebook against its official source before editing.
5. **Persist the master-notes standard** — the approved pedagogical pilot and Day 31/32 depth exemplars become the writing standard; add it to `AGENT_STYLE/` once the user approves it (the verbatim prompt is not stored in the repo).

---

## Exact next action

> Rewrite Days 23–30 next using the approved Day 39 two-layer standard: Part A must be sufficient on its own, Part B optional; preserve all technical depth, formulas, notebook results, source-fidelity notes, and interview/exam coverage; rewrite in place; validate fences, tables, Mermaid, and KaTeX; update each README and this progress file. Do not resume course expansion except for explicit user requests.

---

## Errors / notes

- Inline Python scripts in `%TEMP%\opencode\` (PowerShell has no heredoc).
- PowerShell expands `$` and backticks inside double-quoted inline Python; use standalone `.py` files for math/code-fence scripts.
- `PYTHONIOENCODING=utf-8` for Unicode.
- `Day_31` ≡ `DAY_31`.
- User: **do not delete `reference_notebooks/`**.
- User: **do not reduce theoretical depth**.
- User: **never merge distinct notebooks**; keep `DAY29_*` names.
