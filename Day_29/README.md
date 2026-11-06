# Day 29 — Machine Learning Pipelines A–Z

CampusX 100 Days of ML · folder `Day_29/`

## Main topics

- Manual Titanic preprocessing without a pipeline.
- `Pipeline` and `ColumnTransformer` automation.
- A deliberately broken workflow for debugging practice.
- Cross-validation, `GridSearchCV`, pipeline export, and inference.

## Learning objectives

1. Run the manual impute → encode → concatenate → train workflow.
2. Explain `fit`, `transform`, and `predict` for a pipeline.
3. Diagnose unassigned drops, swapped train/test blocks, and column-index bugs.
4. Validate with `cross_val_score` and tune with `GridSearchCV`.
5. Export a fitted pipeline and predict from raw rows.

## Notebooks

| File | Role |
|------|------|
| `DAY29_Workflow-Without-Pipelines.ipynb` | Correct manual workflow |
| `DAY29_Titanic-Using-Pipeline.ipynb` | Five-stage pipeline; preserves the notebook's `trf2` index bug |
| `DAY29_Sklearn-Pipelines-broken.ipynb` | Deliberately broken counterexample — do not copy |
| `DAY29_Predict-Without-Pipeline.ipynb` | Manual inference; saved kernel crash |
| `DAY29_Pipeline-Export-Predict.ipynb` | Load `pipe.pkl` and predict a raw row |
| `reference_notebooks/` | Four CampusX originals — **do not edit** |

## Dataset and artifacts

| File | Source |
|------|--------|
| `datasets/train.csv` | CampusX Titanic dataset |
| `models/*.pkl` | Manual-workflow encoders and classifier |
| `pipe.pkl` | Exported fitted pipeline |

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, jupyter
```

```bash
pip install pandas numpy scikit-learn jupyter
```

## How to run

1. Open `Day_29/` as the working directory.
2. Run the notebooks in this order: manual workflow → pipeline → broken → manual inference → export/predict.
3. Read `notes.md` for theory, the preserved bugs, and interview material.

## Notes

The pipeline notebook's `trf2` selects `[1, 6]` after `trf1`, which are `Embarked` and `Fare` — not `Sex` and `Embarked`. This preserved bug is explained in `notes.md`; do not silently "fix" the notebook.
