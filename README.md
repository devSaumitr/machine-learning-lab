# Machine Learning Lab

My personal learning repository for **Python, OOP, DSA, and Machine Learning** — built step by step, topic by topic.

**Repo:** [devSaumitr/machine-learning-lab](https://github.com/devSaumitr/machine-learning-lab)

---

## What I'm Learning

```
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
└── PYTHON/         → CampusX Data Science Mentorship Program videos (strings, OOP)
```

### Topic Roadmap

```
Day 1: Variables, data types, I/O, operators, conditions, loops, strings
  ↓
Day 2: Functions, *args/**kwargs, exceptions, mutability, scope/LEGB, truthy/falsy, None
  ↓
Day 3: Lambda, map/filter/reduce, recursion, iterators, generators, modules, __name__/__main__
  ↓
Day 4: OOP — classes, objects, encapsulation, inheritance, polymorphism, abstraction, magic methods
  ↓
NumPy → Pandas → Scikit-learn → ML models
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

```
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

# 5. If someone else pushed first, pull first
git pull origin main
git push origin main
```

### `.gitignore`

This repo uses `.gitignore` to skip files Git shouldn't track:

```
__pycache__/     # Python bytecode cache
*.mp4            # Large lecture videos
*.pkl            # ML model files
.vscode/         # Editor settings
tempCodeRunnerFile.py  # Temp files
```

### Common Mistakes & Fixes

| Problem | Fix |
|---------|-----|
| Committed wrong files | `git reset HEAD~1` (undo last commit, keep changes) |
| Forgot to pull before push | `git pull --rebase origin main` |
| Edited wrong file | `git checkout -- file.py` (discard changes) |
| Want to see past commits | `git log --oneline -10` |

---

## How This Repo Is Organized

```
Machine learning/
├── CORE_PYTHON/        → Python fundamentals (the "must-know" before ML)
│   ├── topics/Day2/    → Intermediate Python
│   ├── topics/Day3/    → Functional tools, recursion, modules
│   └── EXCEPTION_HANDELING/ → Error handling practice
│
├── OOPS/               → Object-Oriented Programming
│   ├── topics/         → 9 detailed markdown guides
│   ├── codes/          → Practice Python files
│   └── codes/notes/    → 11 deep-dive notes (gotchas, MRO, ABC, etc.)
│
├── PYTHON_LIBRARIES/   → Library practice
│   ├── numpy/          → 14+ files covering arrays to advanced
│   └── others/         → General Python tricks
│
├── DSA/                → Data Structures & Algorithms
│   └── red_black_tree  → Self-balancing BST (full guide + C++ code)
│
├── LECTURES/           → CampusX videos (excluded from Git via .gitignore)
│
└── self-explained.md   → OOP & `self` explained with diagrams
```

---

## How to Use This Repo

```bash
# Clone it
git clone https://github.com/devSaumitr/machine-learning-lab.git
cd machine-learning-lab

# Run any script
python PYTHON_LIBRARIES/numpy/basics.py
python OOPS/codes/ATM.py

# Read any topic
# Open topics/Day2/01-args-kwargs.md in your editor
```

### Requirements

- Python 3.x
- NumPy (`pip install numpy`)

---

## My Learning Process

1. **Watch** — CampusX lectures (stored in `LECTURES/`)
2. **Read** — Topic markdown files in `topics/` folders
3. **Practice** — Write Python scripts in `codes/` or `others/`
4. **Note** — Self-explained docs for tricky concepts (`self-explained.md`, `codes/notes/`)
5. **Push** — Commit everything to GitHub to track progress

---

*Built while following the CampusX Data Science Mentorship Program.*
