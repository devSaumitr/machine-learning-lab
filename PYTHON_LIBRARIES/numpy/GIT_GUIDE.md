# Git Guide — Add, Commit & Push

A quick reference for pushing your changes to GitHub.

## The Three Core Commands

### 1. `git add` — Stage your changes

```bash
# Stage ALL changed files
git add .

# Stage a specific file
git add filename.py

# Stage multiple files
git add file1.py file2.py
```

**What it does:** Puts your changes into a "staging area" (like putting items in a shopping cart before checkout).

---

### 2. `git commit` — Save a snapshot

```bash
# Commit with a message
git commit -m "add numpy basics script"

# Stage + commit in one step (only for tracked files)
git commit -am "update attributes script"
```

**What it does:** Saves the staged changes locally with a descriptive message. This is a **local** save — nothing goes to GitHub yet.

**Good commit messages:**
- `add array creation examples`
- `fix broadcasting typo`
- `update splitting notes`

**Bad commit messages:**
- `fix stuff`
- `asdfgh`
- `update`

---

### 3. `git push` — Send to GitHub

```bash
# Push to GitHub
git push origin main
```

**What it does:** Uploads your local commits to the remote repository (`origin`) on the `main` branch. Now it's live on GitHub.

---

## Full Workflow (Copy-Paste)

```bash
# Step 1: Check what changed
git status

# Step 2: Stage all changes
git add .

# Step 3: Commit with a message
git commit -m "your message here"

# Step 4: Push to GitHub
git push origin main
```

## Other Useful Commands

| Command | What It Does |
|---|---|
| `git status` | See which files are changed / staged |
| `git log --oneline` | View recent commit history |
| `git pull origin main` | Download latest changes from GitHub |
| `git diff` | See what changed in a file |
| `git checkout -- file.py` | Discard changes to a file |
| `git stash` | Temporarily save changes without committing |

## Common Errors & Fixes

### "Updates were rejected"
```bash
# Someone pushed changes first — pull first, then push
git pull origin main
git push origin main
```

### "Nothing to commit"
```bash
# No changes detected. Make sure you saved your files.
git status
```

### Wrong branch?
```bash
# Switch to main
git checkout main

# Or create and switch to a new branch
git checkout -b feature-name
```
