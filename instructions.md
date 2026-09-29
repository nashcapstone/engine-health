# Instructions

Rules for everyone working in this repo. Read before your first commit.

## Branch rules

1. Never push code to the `main` branch.
2. Make changes only in your own branch: `data`, `models` or `evaluation`.
3. Branches are merged into `main` later, together, at the sync points (end of day 3 and end of day 5).
4. Never force-push.

We all push through the same GitHub account, so GitHub will not stop a push to the wrong branch. Check with `git branch --show-current` before you push.

## The history log

The `history` branch holds one file, `history.md`. It is the team's running log: what each person changed, on which branch, and what the others need to know to build on it. The `history` branch is never merged into `main`.

### One-time setup: two folders

Keep two clones side by side, one for your work and one for the log.

```
git clone https://github.com/nashcapstone/engine-health.git
cd engine-health
git checkout -b models          # or: data, evaluation
cd ..

git clone -b history --single-branch https://github.com/nashcapstone/engine-health.git engine-health-history
```

Optional, so the log shows who wrote what. Run in both folders:

```
git config user.name "Your name"
```

### Every time you work

1. Before you implement anything, pull the log and read what is new:
   ```
   cd engine-health-history
   git pull
   ```
2. Implement your change in your working folder, on your own branch. Commit and push it there.
3. Add an entry at the bottom of `history.md` in the history folder, then push it:
   ```
   cd engine-health-history
   git pull
   # add your entry at the bottom of history.md
   git add history.md
   git commit -m "Log: <short summary>"
   git push
   ```
   If the push is rejected because someone else pushed first, run `git pull` and push again. Entries from two people merge automatically.

### What an entry contains

```
## YYYY-MM-DD | Member | branch

**Changed**
- What you added or changed, with file paths.

**Others need to know**
- Anything that affects another member: new functions they can call,
  changed shapes or formats, files they should pull, known problems.

**Commits:** short hashes
```

Log every change that another member could build on or be affected by. Add new entries; do not edit or delete someone else's.
