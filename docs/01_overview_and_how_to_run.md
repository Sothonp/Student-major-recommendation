# 01 — Overview and how to run

## What this project does

A student answers 18 questions about their interests. A machine-learning model suggests the
2 IT tracks that fit best (Software Development, Data Science / AI, Cybersecurity,
Networking) and explains why. The model learned from **257 AUPP IT students** whose
survey answers and majors are known.

## The 13 steps

| # | Step | File | Owner |
|---|---|---|---|
| 1 | Load, clean and map the survey to 4 tracks → 257 students | `step1_preprocessing.py` | Sonika |
| 2 | Hide 52 students for the final exam, keep 205 for practice | `step1_preprocessing.py` | Sonika |
| 3 | Create the 4 models and the "always the most common track" baseline | `step3_models.py` | Sothon |
| 4 | Tune each model's settings (grid search + 5-fold) | `step4_tuning.py` | Thailong |
| 5 | Compare all models with 5-fold cross-validation | `step5_train.py` | Sothon |
| 6 | Pick the best model with the selection rule | `step5_train.py` | Sothon |
| 7 | Final exam: the winner predicts the 52 hidden students | `step6_evaluate.py` | Sothon |
| 8 | Ablation: which groups of questions help? | `step6_evaluate.py` | Sothon |
| 9 | Correlation, importance and the 6 figures | `step7_analysis.py` | Rachana |
| 10 | Refit the winner on all 257 students and save it | `step5_train.py` | Sothon |
| 11 | Top-2 tracks and the SHAP reasons | `step8_predict.py` | Thailong |
| 12 | The web app | `app/app.py` | Sothon |
| 13 | Report and slides | `report/` | everyone |

`step2_features.py` (Thailong) builds the 3 interest scores and the preprocessing that every
model uses. `config.py` (Sonika) holds the constants every file reads.

## Setup (once per laptop)

You need Python 3.14 (`py -3.14`). In the project folder:

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Put the survey spreadsheet at `data/Student Major Recommendation (Responses).xlsx`.
It is not in GitHub because it holds student answers.

## Run it (from the project folder)

```
.venv\Scripts\python -m pipeline.step5_train       # steps 1-4: tune, compare, pick the winner, save models
.venv\Scripts\python -m pipeline.step6_evaluate    # final exam + ablation
.venv\Scripts\python -m pipeline.step7_analysis    # correlation, importance, 6 figures
.venv\Scripts\python app\app.py                    # web app at http://127.0.0.1:5000
.venv\Scripts\python -m pytest                     # 71 checks; run before every merge
```

## Folder map

```
data/       the survey (not in GitHub)        models/   saved models (not in GitHub)
pipeline/   steps 1-8 + config                results/  tables (CSV) and figures/ (PNG)
app/        the web app                       tests/    automatic checks
docs/       these documents                   report/   report and slides
```

## The results in one minute

- 320 survey responses → 257 usable (CS and ICT students have no single track and are set aside).
- On the 205 practice students, all four models score about **0.91 macro F1**; the
  "most common track" baseline scores 0.12. The four models are statistically tied, so the
  selection rule picks the simplest: **Logistic Regression**.
- On the 52 hidden students the winner gets **86.5% accuracy** (95% interval 76.9%–96.2%),
  **85.2% macro F1** and **90.4% top-2 accuracy**. The baseline gets 32.7% accuracy.
- Scenario questions add about 6 points of macro F1. The 3 combined scores do not clearly help.

Details: `06_step6_evaluate.md`. Metrics explained: `10_metrics_explained.md`.
