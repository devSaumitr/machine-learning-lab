# Day_29

CampusX 100 Days of ML · folder `Day_29/`

## Day number

**29**

## Main topics

Machine learning pipelines A–Z: the manual workflow, the pipeline workflow, the broken counterexample, pipeline export, and one-line inference.

## Learning objectives

1. Run the full manual preprocessing + training workflow without a pipeline.
2. Explain what `Pipeline` and `ColumnTransformer` automate and in what order.
3. Diagnose the classic broken pipeline (leakage, wrong order) and fix it.
4. Export a fitted pipeline with pickle and reuse it for unseen rows.
5. Compare manual inference with pipeline inference.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `DAY29_Workflow-Without-Pipelines.ipynb` | Manual preprocessing + training workflow | 25 |
| `DAY29_Titanic-Using-Pipeline.ipynb` | Correct end-to-end pipeline (improved personal copy) | 46 |
| `DAY29_Sklearn-Pipelines-broken.ipynb` | Deliberately broken counterexample — do not copy | 17 |
| `DAY29_Predict-Without-Pipeline.ipynb` | Manual inference on new rows | 15 |
| `DAY29_Pipeline-Export-Predict.ipynb` | Export fitted pipeline and predict in one line | 9 |
| `reference_notebooks/reference_notebook*.ipynb` | CampusX originals (4) — **do not edit** | — |

## Dataset and artifacts

| File | Source |
|------|--------|
| `datasets/train.csv` | CampusX course dataset (Titanic) |
| `models/*.pkl` | Artifacts saved by the export notebook |
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
2. Run the notebooks in this order: workflow without pipelines → using pipeline → broken → predict without pipeline → export/predict.
3. Read `notes.md` for the unified theory and interview material.

## Daily folder structure

```text
Day_29/
├── DAY29_Workflow-Without-Pipelines.ipynb
├── DAY29_Titanic-Using-Pipeline.ipynb
├── DAY29_Sklearn-Pipelines-broken.ipynb
├── DAY29_Predict-Without-Pipeline.ipynb
├── DAY29_Pipeline-Export-Predict.ipynb
├── notes.md
├── README.md
├── reference_notebooks/  (4 originals)
├── datasets/
├── models/
└── pipe.pkl
```

## Status

All five personal notebooks restored under their original `DAY29_*` names, matching the four official notebooks plus one personal counterexample. `notes.md` covers all of them.
