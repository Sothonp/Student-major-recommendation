# 12 — Git teamwork

Repository: `Sothonp/Student-major-recommendation` (keep it **private** until the project is graded).

## The rules

1. `main` changes only through a **pull request**, and only after `pytest` passes. Sothon reviews.
2. Each person works on **their own branch**, named `<name>/<what>`, for example `rachana/extra-figure`.
3. **A commit carries the name of the person who really wrote the change.** Each member uses their own GitHub
   account and sets their own `git config user.name` and `user.email`. The first version of every step was written by
   Sothon (with an AI assistant) and committed under Sothon's name; the commit message says whose step it is. Your own
   change, made after you read and understood your file, is committed under **your** name.
4. The survey spreadsheet and the saved models are never committed (`.gitignore`).

## Steps for your one change

```
git clone https://github.com/Sothonp/Student-major-recommendation.git
cd Student-major-recommendation
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest                      # all green before you start

git switch -c yourname/what-you-changed             # your branch
# ... edit your file and its test ...
.venv\Scripts\python -m pytest                      # all green again
git add <the files you changed>
git commit -m "step7: add per-track F1 chart (Rachana)"
git push -u origin yourname/what-you-changed
```

Then open a pull request on GitHub. Sothon (the repository owner) must first add each member as a collaborator
(Settings → Collaborators). In the pull request, write in one or two sentences what you changed and why.

## Before you merge, check

- `pytest` shows no failures.
- You can explain every line you changed.
- You did not touch someone else's file (that is where conflicts come from). If you must, tell the owner first.
