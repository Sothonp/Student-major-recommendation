# 13 — Code tour: how everything fits together

Read this first if you want to understand the whole project in 20 minutes. Then open the files in the order below.

## 1. The big picture

```
 survey spreadsheet (data/…xlsx)
          │
   step1_preprocessing.py   load → clean → map majors to 4 tracks → split 205 train / 52 test
          │
   ┌──────┴───────────────────────────────────────────────┐
   │ every model is a "pipeline" of two parts:            │
   │   prep = step2_features.py  (3 scores, scaling,      │
   │                              one-hot scenario text)  │
   │   clf  = the classifier built in step3_models.py     │
   └──────┬───────────────────────────────────────────────┘
          │
   step4_tuning.py          try many settings per model (grid search + 5-fold) → voting team
          │
   step5_train.py           compare all models → pick the winner → save models/ and results/
          │
   ┌──────┴─────────────────┬──────────────────────────┐
   │ step6_evaluate.py      │ step7_analysis.py        │ step8_predict.py → app/app.py
   │ final exam on the 52   │ which questions matter,  │ top-2 tracks + SHAP reasons
   │ + ablation             │ 7 figures                │ for one student at a time
   └────────────────────────┴──────────────────────────┘
```

`config.py` is read by every file: question wording, answer options, the track names, the settings and the paths.

## 2. Reading order (about 2 hours in total)

| # | File | Minutes | What to look for |
|---|---|---|---|
| 1 | `pipeline/config.py` | 5 | only values, no logic; find the 4 tracks, the 18 questions, `PARAM_GRIDS` |
| 2 | `pipeline/step1_preprocessing.py` | 10 | the 4 functions run in order: load → clean → map → split |
| 3 | `pipeline/step2_features.py` | 15 | the 3 score formulas; why the scaler sits inside the pipeline |
| 4 | `pipeline/step3_models.py` | 5 | a dict of 4 pipelines (baseline + 3 models), each `prep` + `clf` |
| 5 | `pipeline/step4_tuning.py` | 10 | one `GridSearchCV` per model, then `VotingClassifier` |
| 6 | `pipeline/step5_train.py` | 15 | `main()` is the table of contents of the whole project |
| 7 | `pipeline/step6_evaluate.py` | 15 | the final exam; the bootstrap interval; the ablation loop |
| 8 | `pipeline/step7_analysis.py` | 10 | correlation, importance, then 7 figures one after another |
| 9 | `pipeline/step8_predict.py` | 15 | `predict_top2`, then SHAP in `explain_prediction` |
| 10 | `app/app.py` + `app/templates/` | 10 | two routes, `validate_answers`, three templates |

Start every file by reading the docstring at the top and the one under each function: they say in plain English what the code does.

## 3. What happens when you run each command

**`python -m pipeline.step5_train`** (`main()` in step 5 calls everything):
1. `load_responses` → `clean_responses(verbose=True)` → `map_to_tracks` → `split_train_test` (step 1)
2. `build_candidates` (step 3), then the baseline is set aside and fitted
3. `tune_candidates` (step 4) tunes the 3 models and builds the voting team
4. `cross_validate_models`, then `_tie_group` / `select_best_model` pick the winner (step 5)
5. the winner is refit on all 257 students and everything is saved

**`python -m pipeline.step6_evaluate`:** loads `models/winner_train_only.joblib`, runs `evaluate_on_test` and `run_ablation`, writes the CSVs.

**`python -m pipeline.step7_analysis`:** `analyze_feature_signal`, then `plot_results` draws the 7 figures from the CSVs.

**`python app\app.py`:** starts the web app at http://127.0.0.1:5000.

## 4. What happens when a student clicks "Get my recommendation"

1. The browser sends the 18 answers to `/recommend` → `recommend()` in `app/app.py`.
2. `validate_answers` checks them. Wrong or missing answers → the form is shown again with messages.
3. `_artifacts()` loads `models/best_model.joblib` the first time (`load_artifacts` in step 8).
4. `predict_top2` → the two most likely tracks with probabilities.
5. `explain_prediction` runs SHAP once per track → three sentences each.
6. `result.html` shows the two cards.

## 5. Files each command writes

| Command | Writes |
|---|---|
| step5_train | `results/cv_results.csv`, `results/tuned_params.json`, `models/*.joblib`, `models/model_info.json` |
| step6_evaluate | `results/test_results.csv`, `per_track.csv`, `confusion_matrix.csv`, `ablation.csv`, `class_balance.csv` |
| step7_analysis | `results/correlation.csv`, `feature_importance.csv`, `results/figures/*.png` (7 figures) |

Each later step only *reads* what an earlier step wrote, so you can re-run one step without redoing the others.

## 6. "I want to change …" cheat sheet

| I want to change | Edit |
|---|---|
| a question's wording, an answer option, a track name | `pipeline/config.py` |
| which majors belong to which track | `config.MAJOR_TO_TRACK` |
| the 3 combined scores | `step2_features.py` (`add_interest_scores`, `SCORE_COLS`) |
| the settings tuning tries | `config.PARAM_GRIDS` |
| add a model | `step3_models.py` + `config.PARAM_GRIDS` + `config.MODEL_SIMPLICITY` |
| how ties between models are decided | `step5_train.py` (`_tie_group`) |
| the seed, number of folds, test share | `config.py` (top) |
| the final-exam numbers | `step6_evaluate.py` |
| a figure | `step7_analysis.py` (`plot_results`) |
| how many reasons, their wording | `step8_predict.py` (`explain_prediction`) |
| how the app looks | `app/static/style.css`, `app/templates/` |
| what the app does with answers | `app/app.py` |

After any change: `python -m pytest`. Everything should stay green.

## 7. Small words you will meet

- **pipeline:** a chain of steps that always run in the same order (here: prepare the answers, then predict).
- **fit / predict / predict_proba:** learn from students / give a track / give a probability for each track.
- **DataFrame:** a table (pandas). `X` is the table of answers and `y` is the list of true tracks.
- **one-hot:** a text answer turned into 0/1 columns, one per possible answer.
- **cross-validation:** train on 4 groups of students, test on the 5th, rotate, average.
- **SHAP:** how much each answer pushed the result toward a track.
- **joblib:** the file format a trained model is saved in.

## 8. Private helper functions

Beyond the 21 main functions, a few small helpers start with an underscore (`_score`, `_tie_group`, `_encode`, `_decode`,
`_to_frame`, `_artifacts`, `_form_page`). They are used only inside their own file, so you can ignore them when you
want the big picture and read them when you want the details.

## 9. Three changes made for the owners (review them)

Each owner is expected to make one real change. These three were written by Sothon on their behalf and are on separate
branches; each owner must read it, run its tests, and be able to explain it.

| For | Change | Where | Tests |
|---|---|---|---|
| Sonika | `clean_responses(verbose=True)` prints how many rows each rule removed | `step1_preprocessing.py` | `test_step1.py` (2 new) |
| Rachana | a seventh figure: precision, recall and F1 per track | `step7_analysis.py` (`plot_results`) | `test_step7.py` |
| Sothon | training prints why the winner was chosen (best score, cut-off, tied models) | `step5_train.py` (`_tie_group`) | `test_step5.py` |

The walkthroughs `02`, `05` and `07` explain each change in detail.
