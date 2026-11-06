# Machine Learning Lab — Python, DSA, OOP, and ML

My personal learning repository for **Python, OOP, DSA, and Machine Learning** — built step by step, topic by topic.

**Repo:** [devSaumitr/machine-learning-lab](https://github.com/devSaumitr/machine-learning-lab)

> ML course reference: [CampusX 100 Days of Machine Learning](https://github.com/campusx-official/100-days-of-machine-learning/tree/main) and the [YouTube playlist](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH).
> Rule for the ML track: **one day, one folder** — each `Day_XX/` holds that day's notebooks, detailed study notes, and a short README.

---

## What I'm Learning

```text
CORE_PYTHON/
├── topics/Day2/    → args/kwargs, exceptions, mutability, scope, truthy/falsy, None
├── topics/Day3/    → lambda, map/filter/reduce, recursion, iterators, generators, modules
└── EXCEPTION_HANDELING/ → custom exceptions, practice

OOPS/
├── topics/         → classes, encapsulation, inheritance, polymorphism, abstraction, magic methods, file handling
├── codes/          → practice files (ATM, bank, fraction, customer, etc.)
└── codes/notes/    → deep-dive notes on name mangling, MRO, composition, ABC, gotchas

PYTHON_LIBRARIES/
├── numpy/          → array creation, indexing, slicing, broadcasting, stacking, vector ops, advanced topics
└── others/         → math tricks, loops, mask operations

DSA/
└── red_black_tree  → full guide with BST, rotations, insertion, deletion, C++ implementation

LECTURES/
└── PYTHON/         → CampusX Data Science Mentorship Program videos

Day_15 … Day_68/    → CampusX 100 Days of ML: notebooks, notes.md, README, datasets
EXTRA_*/            → independent ML practice folders
ML_Notes_Videos_31_40/ → generated chapter PDFs, figures, transcripts, and build scripts
AGENT_STYLE/        → permanent teaching/format rules for this workspace
```

### Topic Roadmap

```text
Day 1: Variables, data types, I/O, operators, conditions, loops, strings
  ↓
Day 2: Functions, *args/**kwargs, exceptions, mutability, scope/LEGB, truthy/falsy, None
  ↓
Day 3: Lambda, map/filter/reduce, recursion, iterators, generators, modules, __name__/__main__
  ↓
Day 4: OOP — classes, objects, encapsulation, inheritance, polymorphism, abstraction, magic methods
  ↓
NumPy → Pandas → Matplotlib → Statistics → Scikit-learn → ML models
  ↓
CampusX 100 Days of ML → feature engineering → preprocessing → model building
```

---

## Machine Learning Course (CampusX 100 Days)

The ML track follows the CampusX sequence. Each folder is a self-contained study unit:

- `my_notebook.ipynb` — practical work: concept headings, commented code, outputs, observations.
- `notes.md` — the full textbook-style chapter: intuition, mathematics, internals, implementation, interpretation, limitations, and interview preparation.
- `reference_notebooks/` — the original course notebook(s), preserved separately. These are references, not my working notebook.
- `README.md` — what the day covers, which files to open, and honest status.

`notes.md` follows the **Day 39 standard**: Part A is a complete main learning chapter; Part B is optional advanced reading and reference.

### Day map

| Days | Main focus |
|---|---|
| `Day_15`–`Day_22` | CSV files, JSON/SQL, APIs, web scraping, understanding data, univariate/bivariate EDA, pandas profiling |
| `Day_23`–`Day_30` | Feature engineering, standardization, normalization, ordinal/label encoding, one-hot encoding, `ColumnTransformer`, pipelines, `FunctionTransformer` |
| `DAY_31`–`DAY_36` | Power transforms, binning/binarization, mixed variables, dates/time, complete-case analysis, numerical imputation |
| `Day_37`–`Day_40` | Missing categorical data, missing indicators, random-sample imputation, automatic imputer selection, KNN imputer, iterative imputer |
| `Day_41` | Outlier theory (video-only; no notebook) |
| `Day_42`–`Day_45` | Z-score outlier handling, IQR method, percentile method, feature construction and splitting |
| `Day_47`–`Day_53` | PCA, simple/multiple linear regression, regression metrics, gradient descent, polynomial regression |
| `Day_55`–`Day_60` | Ridge, Lasso, ElasticNet, logistic regression, classification metrics, softmax |
| `Day_65`–`Day_68` | Random forest, AdaBoost, stacking and blending |

There are no empty future folders. Days without a folder are not represented locally. `Day_41` is intentionally theory-only because the source video has no notebook or dataset.

### How to study a day

1. Open `Day_XX/README.md` — what was studied, which files matter, and which order to use.
2. Work through `my_notebook.ipynb` — run the cells, read the outputs, and note what changed.
3. Revise from `Day_XX/notes.md` — theory, formulas, worked examples, code explanation, and interview questions.
4. Compare with `Day_XX/reference_notebooks/` when the course implementation differs from mine.
5. Record genuine progress in `PROGRESS.md`; never mark a day complete without checking the files.

Follow the days in numeric order. The course sequence is the prerequisite chain: data collection → EDA → feature engineering → preprocessing → model building.

### Practical notes for the ML track

- Kernels: `.venv` is used by most notebooks; `.venv312` is used by the ydata-profiling notebook.
- Dataset paths are mostly relative to each day folder, with some older notebooks using absolute `E:/MACHINE_LEARNING/...` paths.
- `vendor/100-days-of-machine-learning/` is the read-only master reference and is intentionally **not tracked** in Git.
- `reference_notebooks/` is preserved and never merged into `my_notebook.ipynb`.
- Do not commit API keys. `Day_17` uses `YOUR_TMDB_API_KEY` as a placeholder; set a real key through an environment variable locally.
- Binary learning artifacts (`.pdf`, `.png`, `.pkl`, `.zip`, `.sqlite`) are kept byte-exact on Windows via `.gitattributes`.

---

## How This Repo Is Organized

```text
MACHINE_LEARNING/
├── CORE_PYTHON/        → Python fundamentals (the "must-know" before ML)
├── OOPS/               → Object-Oriented Programming
├── PYTHON_LIBRARIES/   → NumPy and general Python practice
├── DSA/                → Data Structures & Algorithms
├── LECTURES/           → CampusX video material
├── Day_15 … Day_68/    → ML course days
├── EXTRA_*/            → independent ML practice
├── ML_Notes_Videos_31_40/ → generated chapter PDFs and supporting files
├── AGENT_STYLE/        → teaching style, roadmap, and formatting contracts
├── DATA_SETS/          → shared datasets
├── BASICS/             → earlier practice and frozen pipeline artifacts
├── vendor/             → read-only CampusX reference repository (untracked)
├── PROGRESS.md         → current project status
├── AGENTS.md           → permanent project contract
└── README.md           → this guide
```

---

## How Git Works (Beginner Guide)

### What is Git?

Git is a **version control system** — it tracks every change you make to files so you can go back, compare, or share your work.

### Key Concepts

| Term | Meaning |
|------|---------|
| **Repository (repo)** | A folder Git is tracking |
| **Working Directory** | Your actual files on disk |
| **Staging Area** | Files you've marked to be saved next |
| **Commit** | A snapshot of your staged changes |
| **Branch** | An independent line of work (default: `main`) |
| **Remote** | A copy of the repo hosted online (GitHub) |
| **origin** | The name Git gives your remote repo |

### The Workflow

```text
Edit files (Working Directory)
       │
       ▼
  git add .          (Stage changes)
       │
       ▼
  git commit -m "msg"   (Save a snapshot locally)
       │
       ▼
  git push origin main  (Upload to GitHub)
```

### Essential Commands

```bash
# --- Starting ---
git init                  # Turn a folder into a Git repo
git clone <url>           # Download an existing repo

# --- Daily workflow ---
git status                # See what changed
git add .                 # Stage ALL changes
git add file.py           # Stage one file
git commit -m "message"   # Commit staged changes
git push origin main      # Upload commits to GitHub

# --- Getting updates ---
git pull origin main      # Download latest changes from GitHub

# --- History ---
git log --oneline         # Short commit history
git diff                  # See what changed but isn't staged

# --- Undo ---
git checkout -- file.py   # Discard changes in a file (before staging)
git reset HEAD file.py    # Unstage a file (after git add, before commit)
```

### Step-by-Step: Pushing Your Work

```bash
# 1. Check what's changed
git status

# 2. Stage everything
git add .

# 3. Commit with a message
git commit -m "add numpy basics and broadcasting notes"

# 4. Push to GitHub
git push origin main

# 5. If the push is rejected, integrate the remote work first
git fetch origin
git merge origin/main
git push origin main
```

Never force-push shared `main`. If the histories are unrelated, merge with `--allow-unrelated-histories` and resolve conflicts deliberately.

### `.gitignore`

This repo uses `.gitignore` to skip files Git shouldn't track:

```text
__pycache__/          # Python bytecode cache
*.mp4                 # Large lecture videos
.vscode/              # Editor settings
tempCodeRunnerFile.py # Temp files
.venv/                # Local virtual environment
.venv312/             # Second local virtual environment
```

Model artifacts and learning figures inside tracked day folders are intentional and are preserved with `.gitattributes` rather than ignored.

### Common Mistakes & Fixes

| Problem | Fix |
|---------|-----|
| Committed wrong files | `git reset HEAD~1` (undo last commit, keep changes) |
| Forgot to pull before push | `git fetch origin` then `git merge origin/main` |
| Edited wrong file | `git checkout -- file.py` (discard changes) |
| Want to see past commits | `git log --oneline -10` |
| Push rejected as non-fast-forward | Integrate remote changes; do not force-push `main` |

---

## How to Use This Repo

```bash
# Clone it
git clone https://github.com/devSaumitr/machine-learning-lab.git
cd machine-learning-lab

# Run any Python practice file
python PYTHON_LIBRARIES/numpy/basics.py
python OOPS/codes/ATM.py

# Open an ML notebook
jupyter notebook Day_22/my_notebook.ipynb
```

### Requirements

- Python 3.x
- NumPy, Pandas, Matplotlib, Scikit-learn
- Jupyter Notebook or JupyterLab

Install packages as needed in the active virtual environment. There is no single `requirements.txt`; the two local environments (`.venv` and `.venv312`) were created separately.

---

## My Learning Process

1. **Watch** — CampusX lectures.
2. **Read** — topic notes and `notes.md` chapters.
3. **Practice** — write and run notebooks/scripts.
4. **Explain** — reconstruct the concept without looking at the notes.
5. **Revise** — use the chapter summary and interview questions.
6. **Push** — commit verified work to GitHub to track progress.

---

*Built while following the CampusX Data Science Mentorship Program and the 100 Days of Machine Learning course.*
