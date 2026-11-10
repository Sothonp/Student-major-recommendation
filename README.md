# AUPP IT Track Recommender

A machine-learning project (ITM-360, Group 7, American University of Phnom Penh) that suggests the
2 IT tracks that fit an AUPP student best (Software Development, Data Science / AI, Cybersecurity,
Networking) from their interests, and explains why.

It is trained on a Google Form survey of 257 AUPP IT students. Four models are compared with
stratified 5-fold cross-validation; the winner is evaluated once on 52 hidden students
(86.5% accuracy, 85.2% macro F1, 90.4% top-2 accuracy) and served by a small Flask web app with
SHAP-based reasons.

## Run it

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# put the survey at data\Student Major Recommendation (Responses).xlsx  (not in this repository)

.venv\Scripts\python -m pipeline.step5_train       # tune, compare, pick the winner, save models
.venv\Scripts\python -m pipeline.step6_evaluate    # final exam + ablation
.venv\Scripts\python -m pipeline.step7_analysis    # correlation, importance, figures
.venv\Scripts\python app\app.py                    # web app: http://127.0.0.1:5000
.venv\Scripts\python -m pytest                     # automatic checks
```

## Folders

| Folder | Contents |
|---|---|
| `pipeline/` | `config.py` and `step1_…` to `step8_…`, one file per step |
| `app/` | Flask web app |
| `tests/` | automatic checks (fake data only) |
| `results/` | result tables (CSV) and figures |
| `docs/` | design spec, one walkthrough per file, metrics guide, Q&A sheet, git guide |
| `report/` | report and slides drafts |

Start with `docs/01_overview_and_how_to_run.md`, then `docs/13_code_tour.md` to understand how the code fits together.
