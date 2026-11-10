# 00 — Design Spec: AUPP IT Track Recommender

Course: ITM-360 Machine Learning, American University of Phnom Penh (Group 7)
Advisor: Dr. Kuntha Pin
Spec date: 2026-11-10
Submission deadline: 2026-11-30 (report, slides, code). Presentations: Dec 1 / Dec 3 / Dec 8 (team leader registers a slot).

---

## 1. What we are building

A machine-learning system that recommends an IT track to an AUPP student from their **interests, work preferences and scenario choices** (not grades). The four tracks are:

| Track | Survey majors mapped to it |
|---|---|
| Software Development | Software Development, Interactive App Design and Development |
| Data Science / AI | Artificial Intelligence (AI), Data Analytics |
| Cybersecurity | Cybersecurity |
| Networking | Digital Infrastructure |

Computer Science (CS) and Information and Communication Technology (ICT) responses are set aside, because they do not map to a single track.

**Deliverables (professor's list):** final report (Canvas, course Word template), technical slides (Introduction, Objectives, Methods, Experiments, Results, Discussion/Conclusion, References), oral presentation (15 min + 5 min Q&A, every member presents and is graded individually), and source code (clear structure, separate files).

**Success criteria**
1. The pipeline runs end to end from the survey spreadsheet and produces every table and figure the report needs.
2. The best model is chosen by a fair, documented rule, and its performance is reported honestly (including the small-test-set caveat).
3. A Flask web app returns the top 2 tracks with plain-language reasons (SHAP).
4. Every team member can explain their own files and answer likely questions.

---

## 2. Decisions made so far

| # | Decision | Why |
|---|---|---|
| 1 | Data = the 320 Google Form responses, treated as real responses that were re-imported into a new sheet. The Timestamp column is dropped and no collection period is claimed from the file. | All 320 timestamps fall within 4.5 seconds on 2026-11-10, so they carry no information. The team confirmed the rows are real. |
| 2 | Year 1–2 vs Year 3–4 comparison from the proposal is **dropped**. Listed as a limitation and future work. | The spreadsheet has no year-of-study column. |
| 3 | Explanation method = **SHAP**, explained at the level of the **18 survey questions**. | Matches the approved proposal; one code path for every model. |
| 4 | Model selection rule = best mean CV **macro F1**; if the gap to a simpler model is smaller than one standard deviation, take the simpler model (order: Logistic Regression < KNN < Random Forest < Soft Voting). | 257 rows is small, so differences between models are often noise. |
| 5 | Added to the proposal: an ablation of **with vs without the 3 engineered scores**, across three model types. | The proposal claims the scores help, so the claim needs evidence. |
| 6 | Files are numbered `step1_…` to `step8_…`, inside `pipeline/`. | Easy for the team to follow; Python cannot import names that start with a digit. |
| 7 | AI-assisted code is allowed by the professor (verbal, reported by the team) on the condition that members understand it. The team keeps a record of her confirmation. | Course rule text says AI-generated code loses points; the professor relaxed it. |
| 8 | Git history must show **true authorship** (see section 12). | Each member is graded individually. |

---

## 3. The data

File: `data/Student Major Recommendation (Responses).xlsx` (sheet `Form responses 1`), 320 data rows, 20 columns. Not committed to GitHub.

| Columns | Meaning | Internal name |
|---|---|---|
| Timestamp | dropped | — |
| Q1–Q8 | interest ratings 1–5 | `q1`–`q8` |
| Q9–Q14 | enjoyment ratings 1–5 | `q9`–`q14` |
| Q15–Q18 | scenario multiple-choice answers | `q15`–`q18` |
| Q19 | current or intended major (label source) | `q19` |

Rating questions (shown to students exactly as in the Form):

| q | Short label (used in reasons) |
|---|---|
| q1 | writing computer programs and solving problems using code |
| q2 | artificial intelligence, machine learning and intelligent systems |
| q3 | analyzing data, statistics and finding patterns in datasets |
| q4 | cybersecurity, ethical hacking and protecting computer systems |
| q5 | computer networks, network communication and connectivity |
| q6 | servers, cloud computing and IT infrastructure |
| q7 | designing, developing, testing and improving software systems |
| q8 | designing user interfaces, user experiences and interactive applications |
| q9 | solving logical problems and figuring out how things work |
| q10 | investigating technical problems and finding their causes |
| q11 | working with numbers, datasets and discovering useful patterns |
| q12 | building and improving software applications |
| q13 | managing computer systems, networks, servers or infrastructure |
| q14 | combining technology and creativity to create digital products |

Scenario questions `q15`–`q18` and their answer options are stored in `config.py` exactly as worded in the Form (8, 5, 5 and 8 options, 26 in total).

**Observed class balance (as of 2026-11-10):**

| Track | Rows |
|---|---|
| Data Science / AI | 83 |
| Networking | 76 |
| Software Development | 62 |
| Cybersecurity | 36 |
| CS / ICT (set aside) | 63 |
| **Usable** | **257** |

---

## 4. Project layout

```
ML final project/
├── data/                         survey xlsx (not committed)
├── pipeline/
│   ├── __init__.py
│   ├── config.py                 constants only
│   ├── step1_preprocessing.py
│   ├── step2_features.py
│   ├── step3_models.py
│   ├── step4_tuning.py
│   ├── step5_train.py
│   ├── step6_evaluate.py
│   ├── step7_analysis.py
│   └── step8_predict.py
├── app/                          step 9: Flask web app
│   ├── app.py
│   ├── templates/ (index.html, result.html, error.html)
│   └── static/ (style.css)
├── models/                       saved models (not committed)
├── results/                      CSV tables + figures/
├── tests/
├── docs/                         numbered documents (this file is 00)
├── report/                       report and slides
├── requirements.txt
├── .gitignore
└── README.md
```

Run commands (from the project root):

```
python -m pipeline.step5_train       # steps 1–4, compare models, pick the winner, save models
python -m pipeline.step6_evaluate    # final exam on the 52 test students + ablations
python -m pipeline.step7_analysis    # correlation, importance, figures
python app/app.py                    # web app at http://127.0.0.1:5000
```

Each runnable step file (step5, step6, step7) ends with a small `if __name__ == "__main__":` block that calls its functions. These blocks are not counted in the function list in section 12.

---

## 5. Pipeline flow

| # | Step | File |
|---|---|---|
| 1 | Load, clean, map to 4 tracks → 257 records | step1 |
| 2 | Stratified split: 205 train / 52 test (seed 42) | step1 |
| 3 | Create the baseline and the 3 untuned models | step3 |
| 4 | Tune each model's settings (grid search, stratified 5-fold, macro F1) and build the voting ensemble from the 3 tuned models | step4 |
| 5 | 5-fold cross-validation of all 5 approaches → accuracy, macro F1, top-2 (mean ± std) | step5 |
| 6 | Select the best model by the rule in decision 4 | step5 |
| 7 | Evaluate once on the 52 hidden students | step6 |
| 8 | Ablation experiments | step6 |
| 9 | Correlation + importance analysis, charts, tables | step7 |
| 10 | Refit the winner on all 257 rows and save it | step5 |
| 11 | Top-2 recommendation + SHAP reasons | step8 |
| 12 | Web app | app/ |
| 13 | Report and slides | report/ |

The five approaches: **Baseline** (always the most common track), **Logistic Regression**, **KNN**, **Random Forest**, **Soft-voting ensemble** (of the three tuned models).

The test set is never used to tune or to select. Every random operation uses seed 42.

---

## 6. Features (43 total)

Input to every model is the 18 raw answers `q1`–`q18`. The preprocessing pipeline builds:

| Group | Count | How |
|---|---|---|
| Ratings `q1`–`q14` | 14 | numeric, standardized |
| Engineered scores | 3 | standardized (see below) |
| Scenarios `q15`–`q18` | 26 | one-hot, fixed category lists from `config.py`, unknown values ignored |

Engineered scores (computed inside the pipeline from the raw ratings):
- `build_vs_protect` = mean(q7, q12) − q4
- `data_score` = mean(q3, q11)
- `infra_score` = mean(q5, q6, q13)

The scaler is fitted **inside** each cross-validation fold (it is part of the sklearn pipeline), so there is no leakage.

---

## 7. Models and tuning

All models are scikit-learn pipelines (`preprocessor` + `classifier`).

| Model | Fixed settings | Grid searched |
|---|---|---|
| Baseline | `DummyClassifier(most_frequent)` | none |
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` | `C` ∈ {0.01, 0.1, 1, 10} |
| KNN | (no class weights available) | `n_neighbors` ∈ {3, 5, 7, 9, 11, 15}, `weights` ∈ {uniform, distance} |
| Random Forest | `class_weight="balanced"`, seed 42 | `n_estimators` ∈ {100, 300}, `max_depth` ∈ {None, 5, 10}, `min_samples_leaf` ∈ {1, 2, 4} |
| Soft-voting ensemble | built from the 3 tuned models, `voting="soft"` | not tuned |

Tuning: `GridSearchCV`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, scoring `f1_macro`, on the **205 training rows only**.

**Handoff between step4 (Thailong) and step5 (Sothon):**
`tune_candidates(candidates, X_train, y_train)` receives the untuned pipelines and returns a dict with 4 entries (`"Logistic Regression"`, `"KNN"`, `"Random Forest"`, `"Soft Voting"`). Each entry holds `estimator` (fitted on the training rows), `best_params` and `cv_macro_f1`. The baseline is not tuned and is not passed to this function: `build_candidates()` returns a dict with 4 entries (the baseline plus the 3 untuned models), and `main()` in step5 passes only the 3 untuned models to `tune_candidates`.

---

## 8. Evaluation

**Metrics:** accuracy; macro F1; top-2 accuracy (`top_k_accuracy_score`, k=2, with the 4 labels listed explicitly).

**Model comparison (step5):** 5-fold CV on the training rows for all 5 approaches. Output `results/cv_results.csv` with mean and std per metric. The winner is picked by `select_best_model` (decision 4). Exact rule: take the model with the highest mean macro F1; any model whose mean macro F1 is within one standard deviation (the best model's std across the 5 folds) of that best mean counts as tied; among the tied models, choose the simplest (Logistic Regression < KNN < Random Forest < Soft Voting). Note: CV scores are slightly optimistic because the same rows picked the settings; the test set is the clean estimate.

**Final exam (step6):** the winner, trained on the 205 training rows only (saved by step5 as `models/winner_train_only.joblib`), predicts the 52 test rows. Reported: accuracy, macro F1, top-2 accuracy, per-track precision and recall, confusion matrix, and a **95% bootstrap confidence interval** on accuracy (1000 resamples, seed 42). The baseline is reported next to it. Outputs: `results/test_results.csv`, `results/confusion_matrix.csv`.

**Ablation (step6):** for Logistic Regression, KNN and Random Forest, each with its tuned settings, run 5-fold CV on four feature sets:
1. Ratings only (q1–q14)
2. Ratings + engineered scores
3. Ratings + scenarios (q1–q18)
4. All (ratings + scores + scenarios)

Output `results/ablation.csv`. This answers: do scenario questions help, and do the engineered scores help. Three model types are used because a linear model cannot gain from scores that are sums of existing columns.

**Caveats to state in the report**
- Test set = 52 rows, with about 7 Cybersecurity rows. A one-row change moves test accuracy by about 2 points, hence the confidence interval and the weight on CV results.
- Scenario answer options closely mirror the major names (for example "Build and manage cloud/network infrastructure"), so scenario answers may nearly restate the label. The ratings-only result is the stricter, more honest number and must be reported next to the full-feature result.

---

## 9. Analysis outputs (step7)

- Correlation of each rating and each engineered score with each track (training rows only) → `results/correlation.csv`.
- Random Forest feature importance, summed per question → `results/feature_importance.csv`.
- Six figures in `results/figures/`: class balance, model comparison (CV macro F1 with std bars), confusion matrix, correlation heatmap, feature importance, ablation chart.

---

## 10. Prediction and explanation (step8)

- The **deployed model** is the winner refit on all 257 rows, saved to `models/best_model.joblib`.
- `models/shap_background.joblib` holds a random sample of 50 raw-answer rows from the training data (the "average student" for SHAP).
- `models/model_info.json` stores the winner's name, settings and CV scores.
- `predict_top2` returns the two most likely tracks with probabilities, highest first.
- `explain_prediction` uses SHAP to find the 3 questions whose answers pushed hardest toward a track, and returns them as plain sentences, for example: *You rated "computer networks, network communication and connectivity" 5 out of 5.* or *You chose "Build and manage cloud/network infrastructure" for the semester project question.*
- SHAP explains the 18 questions, not the 43 internal columns.
- **Implementation risk:** SHAP must work with mixed numeric and text inputs. If the standard explainer cannot take the raw DataFrame directly, use a permutation explainer with a small wrapper that rebuilds the DataFrame from an array. Behavior stays the same.
- Reasons that quote a scenario answer are partly circular (see caveat in section 8). The result page states that results are a suggestion based on interests.

---

## 11. Web app (`app/`, step 9)

- Flask, local only, `http://127.0.0.1:5000`. English only. No database, no login, nothing stored.
- `GET /` — `questionnaire`: shows the 18 questions with the same wording as the Form (ratings as 1–5 buttons, scenarios as multiple choice). The major question is not shown.
- `POST /recommend` — `recommend`: calls `validate_answers`, then `predict_top2` and `explain_prediction`, and renders `result.html`: the top 2 tracks with a percentage bar and 3 reasons each, a short "this is a suggestion, not a decision" note, and a Start over button.
- `validate_answers(form)` returns `(clean_answers, errors)`. It rejects missing answers, ratings outside 1–5, and scenario answers that are not a listed option. On error the form is shown again with a message.
- If `models/best_model.joblib` is missing, `error.html` tells the user to run `python -m pipeline.step5_train` first.

---

## 12. Functions and ownership (21 functions)

| File | Function | Job | Owner |
|---|---|---|---|
| config.py | (constants, no functions) | column names, track map, options, labels, paths, seed | Sonika |
| step1 | `load_responses(path)` | read xlsx, rename to q1–q19, drop timestamp | Sonika |
| step1 | `clean_responses(df)` | drop incomplete rows; ratings must be whole numbers 1–5; trim text | Sonika |
| step1 | `map_to_tracks(df)` | add `track`; drop CS/ICT → 257 rows | Sonika |
| step1 | `split_train_test(df)` | stratified 80/20, seed 42 → X_train, X_test, y_train, y_test | Sonika (Sothon co-owns) |
| step2 | `add_interest_scores(X)` | add the 3 engineered scores | Thailong |
| step2 | `build_preprocessor(feature_set)` | scores + scaler + one-hot; `feature_set` ∈ {ratings, ratings+scores, ratings+scenarios, all} | Thailong (Sothon co-owns) |
| step3 | `build_candidates()` | baseline + 3 untuned model pipelines | Sothon |
| step4 | `tune_candidates(candidates, X_train, y_train)` | grid search each model, build the voting ensemble | Thailong |
| step5 | `main()` | run steps 1–4, compare, select, refit, save | Sothon |
| step5 | `cross_validate_models(models, X_train, y_train)` | 5-fold scores table | Sothon |
| step5 | `select_best_model(cv_table)` | apply the selection rule | Sothon |
| step6 | `evaluate_on_test(model, X_test, y_test)` | test metrics, confusion matrix, bootstrap CI | Sothon |
| step6 | `run_ablation(X_train, y_train, best_params)` | the 3 × 4 ablation table | Sothon |
| step7 | `analyze_feature_signal(X_train, y_train)` | correlation + importance tables | Rachana |
| step7 | `plot_results()` | the six figures | Rachana |
| step8 | `load_artifacts()` | load model, background sample, info | Thailong |
| step8 | `predict_top2(model, answers)` | top 2 tracks with probabilities | Thailong |
| step8 | `explain_prediction(model, background, answers, track)` | 3 plain-language reasons | Thailong |
| app | `questionnaire()` | show the form | Sothon |
| app | `validate_answers(form)` | check input | Sothon |
| app | `recommend()` | predict, explain, render | Sothon |

Totals: Sothon 9, Thailong 6, Sonika 4, Rachana 2.

**Presenting:** Sonika — dataset and cleaning; Thailong — features, tuning, SHAP reasons; Sothon — models, comparison, results, app demo; Rachana — analysis, charts, conclusions. Final slide split is the team's decision.

**Git and authorship rules**
- One repository (`Sothonp/Student-major-recommendation`, created 2026-11-10 with a one-line README), `main` changed only by pull request after `pytest` passes; Sothon reviews.
- One branch per member (for example `sonika/step1-preprocessing`).
- Commits carry the name of whoever actually wrote or changed the code. Where the leader writes code for a step owned by Sonika or Rachana, it is committed under the leader's name with a message such as "Step 1 preprocessing (Sonika's step, written by Sothon, pending her review)".
- Once a member has reviewed their step, they make their own real changes and commit them under their own name. Each owner makes **one real change** to their file and can explain it (examples: a new data check, an extra score, a different grid range, a different chart).
- Raw survey data and saved models are never committed (`.gitignore`).

---

## 13. Tests (`tests/`, pytest)

Tests use small fake data built in `tests/conftest.py`. They never touch the real spreadsheet, and the fake data is never used for reported results.

- step1: the track map drops CS/ICT; cleaning removes missing or out-of-range ratings; the split keeps track proportions and the seed gives the same split twice.
- step2: the scores are exactly right on a hand-made row; the preprocessor outputs 43 columns for `all` and the right counts for the other feature sets; unseen scenario values do not crash.
- step4: `tune_candidates` returns 4 entries, each with `best_params` and `cv_macro_f1` (tiny grid).
- step5: `select_best_model` picks the simpler model inside one standard deviation, and the better model outside it.
- step8: `predict_top2` returns 2 items, highest first, with probabilities between 0 and 1; `explain_prediction` returns 3 strings.
- app: `validate_answers` rejects missing, out-of-range and unknown answers; `GET /` returns 200.

Verification on real data (done by running, then reported): the test set is untouched until step6; the baseline scores below every real model; the scaler is inside the CV pipeline; every figure and table in the report comes from a saved results file.

---

## 14. Documentation (`docs/`)

| File | Content |
|---|---|
| 00_design_spec.md | this document |
| 01_overview_and_how_to_run.md | 13-step flow, commands, setup |
| 02_step1_preprocessing.md … 09_app.md | one walkthrough per file: what it does and why, 5 likely questions with answers, the owner's suggested "one change" |
| 10_metrics_explained.md | accuracy, macro F1, top-2, tuning and cross-validation with worked examples |
| 11_qa_sheet.md | all likely professor questions with answers |
| 12_git_teamwork.md | branch, pull request and merge steps |

---

## 15. Setup

- Python 3.12 is installed. A project-local virtual environment `.venv` holds pandas, openpyxl, scikit-learn, shap, matplotlib, Flask, pytest and joblib at pinned versions listed in `requirements.txt`.
- Nothing is installed until this spec and the build plan are approved.

---

## 16. Build order and timeline

Build stages (check-in after each):
1. step1–step2 + tests
2. step3–step5 (training, comparison, winner)
3. step6–step7 (first real results)
4. step8 + app
5. docs, report draft, slides draft

| Dates | Work |
|---|---|
| Nov 10–16 | build stages 1–4; start docs |
| Nov 17–23 | each owner reads, runs and explains their files; makes their one real change; pull requests |
| Nov 24–28 | report, slides, rehearsal with mock Q&A |
| Nov 29–30 | final fixes and submit to Canvas |
| Dec 1 / 3 / 8 | presentation (the team leader registers a slot early; slots are first come, first served) |

---

## 17. Out of scope

Khmer user interface, hosting the app online, database or login, year-of-study analysis, deep learning, collecting more survey responses.

## 18. Open items for the team

1. Save the course report template (ITM-390 Machine Learning Proposal Format .docx) in the project folder so the report matches its sections.
2. Keep a record of the professor's confirmation about AI-assisted code.
3. Register a presentation date.
4. GitHub repository: created. Keep it private until graded, and add `.gitignore` (data, models, `.venv`) before the first real commit.
5. Optional: ask the professor about macro F1 and top-2 accuracy (draft message already prepared).
