# AUPP IT Track Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline, evaluation, SHAP explanations and Flask app from the approved spec, then the team docs and report/slides drafts.

**Architecture:** Numbered modules `pipeline/step1_…` to `step8_…` plus `app/`. Every model is one scikit-learn `Pipeline` (preprocessing + classifier) that takes the 18 raw answers `q1`–`q18`, so scaling happens inside CV folds and SHAP explains the 18 questions directly. Steps pass results through `models/` and `results/` files.

**Tech Stack:** Python 3.12, pandas, openpyxl, scikit-learn, shap, matplotlib, Flask, joblib, pytest.

**Spec:** `docs/00_design_spec.md` (read it first; section numbers below refer to it).

## Global Constraints

- Python 3.12 in a project-local `.venv`; versions pinned in `requirements.txt` via `pip freeze`.
- Seed **42** everywhere; test size **0.2** (stratified); **5** folds, `StratifiedKFold(shuffle=True, random_state=42)`; tuning scorer `f1_macro`.
- Tracks (sorted, the order of `classes_`): `Cybersecurity`, `Data Science / AI`, `Networking`, `Software Development`.
- Expected real-data counts: 320 rows → 257 usable → **205 train / 52 test**; features = **43** (14 ratings + 3 scores + 26 one-hot).
- Top-2 accuracy = `top_k_accuracy_score(k=2, labels=TRACKS)`. Test bootstrap CI = 1000 resamples, seed 42, 95% percentile.
- Exactly the **21 functions** in spec section 12, with the names given there. Optional parameters may be added only where a task says so. Each runnable file ends with an `if __name__ == "__main__":` block (not counted).
- Step modules read paths from `pipeline.config` **at call time** (`config.DATA_PATH`, `config.MODELS_DIR`, `config.RESULTS_DIR`) so tests can monkeypatch them.
- The web app stores nothing, runs on `127.0.0.1:5000`, English only.
- Tests use fake data only; fake data never feeds reported results. `data/`, `models/`, `.venv/` are never committed.
- **Git routine (every task):** branch from `main` as `<owner>/<step>` (owners: sonika, thailong, sothon, rachana); commit message `<step>: <what> (<owner>'s step, written by Sothon; pending <owner>'s review)` ending with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; commits use Sothon's git identity (never another member's name); at task end run the full `pytest`, then `git checkout main && git merge --no-ff <branch>`. **Push to `origin` only at a stage check-in, after Sothon says yes.**
- Run Python as `.venv/Scripts/python` from the project root.

## Review Focus

1. A rating from the form that is not a whole number 1–5 (`"4.5"`, `"abc"`, `""`, `"6"`) → friendly error message, never a server error. (Task 11)
2. A student who answers all 1s or all 5s → still 2 tracks and 3 reasons. (Task 10)
3. Model files missing when the app starts or a request arrives → error page that names `python -m pipeline.step5_train`, not a stack trace. (Tasks 10, 11)
4. Spreadsheet quirks: ratings stored as floats (`4.0`), stray spaces in text, a blank answer, a filename with spaces and parentheses → cleaned or dropped, never a crash. (Task 2)
5. The rarest track (Cybersecurity) in tiny folds or a test set that lacks a track → metrics and top-2 still compute because labels are passed explicitly. (Tasks 6, 7)

---

### Task 1: Project setup

**Files:** Create `.gitignore`, `requirements.txt`, `pytest.ini`, `pipeline/__init__.py`, `tests/__init__.py`, `data/` (move the xlsx into it), `results/figures/`.

**Interfaces:** Produces: a working `.venv` and `pytest` run from the project root with `pythonpath = .`.

- [ ] **Step 1:** Create the venv and install: `python -m venv .venv`, then `.venv/Scripts/python -m pip install pandas openpyxl scikit-learn shap matplotlib flask pytest joblib`. Confirm `import shap, sklearn, flask` succeeds; if `shap` fails to import on Python 3.12, stop and report.
- [ ] **Step 2:** `.venv/Scripts/python -m pip freeze > requirements.txt`.
- [ ] **Step 3:** Write `.gitignore` (`.venv/`, `data/`, `*.xlsx`, `models/`, `__pycache__/`, `.pytest_cache/`), `pytest.ini` (`[pytest]` / `pythonpath = .` / `testpaths = tests`), empty `pipeline/__init__.py`, `tests/__init__.py`.
- [ ] **Step 4:** Move `Student Major Recommendation (Responses).xlsx` into `data/` with `mv`. Verify `git status --short` does not list it.
- [ ] **Step 5:** Run `.venv/Scripts/python -m pytest` → "no tests ran" and exit without errors.
- [ ] **Step 6:** Commit on `sothon/setup` per the git routine; merge to `main`.

---

### Task 2: config + step1_preprocessing (Sonika's files)

**Files:** Create `pipeline/config.py`, `pipeline/step1_preprocessing.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_step1.py`.

**Interfaces:**
- Produces `config` constants: `SEED, TEST_SIZE, N_FOLDS, TOP_K, PROJECT_ROOT, DATA_PATH, MODELS_DIR, RESULTS_DIR, TRACKS, MAJOR_TO_TRACK, EXCLUDED_MAJORS, RATING_COLS (q1–q14), SCENARIO_COLS (q15–q18), QUESTION_COLS (q1–q18), LABEL_COL ("q19"), RATING_LABELS (spec §3), SCENARIO_QUESTIONS, SCENARIO_OPTIONS (8/5/5/8 options, wording copied from the Form), SCENARIO_LABELS (short phrases: "the semester project question", "the dataset question", "the network failure question", "the team responsibility question"), FEATURE_SETS = ("ratings","ratings+scores","ratings+scenarios","all"), PARAM_GRIDS (spec §7, keys prefixed `clf__`), MODEL_SIMPLICITY = ["Logistic Regression","KNN","Random Forest","Soft Voting"]`.
- Produces step1 functions:
  - `load_responses(path: Path | str) -> pd.DataFrame` (columns `q1`–`q19`, timestamp dropped; rename by **position**)
  - `clean_responses(df) -> pd.DataFrame`
  - `map_to_tracks(df) -> pd.DataFrame` (adds `track`)
  - `split_train_test(df) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]` = `X_train, X_test, y_train, y_test` (X has only `q1`–`q18`)
- Produces fixtures: `fake_raw` (q1–q19 incl. CS/ICT rows, ≥ 40 rows per track + 30 CS/ICT, seeded, ratings int 1–5, scenario answers from `SCENARIO_OPTIONS`, planted signal: Cybersecurity rows have q4 ≥ 4 on average, Networking q5/q6/q13 high, Data q2/q3/q11 high, Software q1/q7/q12 high, scenario answer matches the track 70% of the time), `fake_labeled` (= `map_to_tracks(fake_raw)`), `fake_split` (= `split_train_test(fake_labeled)`).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
def test_config_shapes():
    assert len(config.TRACKS) == 4
    assert [len(config.SCENARIO_OPTIONS[q]) for q in config.SCENARIO_COLS] == [8, 5, 5, 8]
    assert set(config.MAJOR_TO_TRACK.values()) == set(config.TRACKS)
    assert len(config.RATING_LABELS) == 14
    sizes = {m: len(ParameterGrid(g)) for m, g in config.PARAM_GRIDS.items()}
    assert sizes == {"Logistic Regression": 4, "KNN": 12, "Random Forest": 18}

# tests/test_step1.py
def test_load_responses_renames_and_drops_timestamp(tmp_path):   # file name "Student Major Recommendation (Responses).xlsx"
    ...  # write a 20-column xlsx with arbitrary long headers; assert list(df.columns) == ["q1",...,"q19"], len(df) == rows

def test_clean_responses_handles_quirks():
    # rows: float rating 4.0 -> kept as int 4; rating 6 and 4.5 -> dropped; blank answer -> dropped;
    # " Cybersecurity " -> "Cybersecurity"
    assert out[config.RATING_COLS].dtypes.eq("int64").all()

def test_map_to_tracks():
    assert set(out["track"]) == set(config.TRACKS)
    assert not out["q19"].isin(config.EXCLUDED_MAJORS).any()
    # each of the 6 majors lands on the track in spec §1

def test_split_train_test(fake_labeled):
    X_tr, X_te, y_tr, y_te = split_train_test(fake_labeled)
    assert len(X_te) == math.ceil(0.2 * len(fake_labeled))
    assert list(X_tr.columns) == config.QUESTION_COLS
    # per-track share in test within 1 row of 20%; second call returns identical indices
```

- [ ] **Step 2:** Run `pytest tests/test_config.py tests/test_step1.py -v` → FAIL (modules missing).
- [ ] **Step 3:** Implement `config.py` (values copied verbatim from spec §1, §3, §7 and the Form wording; scenario options: take the distinct answer strings of q15–q18 from `data/Student Major Recommendation (Responses).xlsx`, which gives 8, 5, 5 and 8 options) and the four step1 functions. `clean_responses`: coerce ratings with `pd.to_numeric`, keep only whole numbers 1–5, strip text columns, drop rows with any blank among `q1`–`q19`. `split_train_test`: `train_test_split(test_size=config.TEST_SIZE, stratify=track, random_state=config.SEED)`.
- [ ] **Step 4:** Run the same tests → PASS.
- [ ] **Step 5 (real data check):** `.venv/Scripts/python -c "…"` using the functions on `config.DATA_PATH`. Expected: 320 loaded → 320 cleaned → 257 mapped with counts `{Data Science / AI: 83, Networking: 76, Software Development: 62, Cybersecurity: 36}` → split 205 / 52.
- [ ] **Step 6:** Commit on `sonika/step1-preprocessing`; merge.

---

### Task 3: step2_features (Thailong's file)

**Files:** Create `pipeline/step2_features.py`, `tests/test_step2.py`.

**Interfaces:**
- Consumes: `config.RATING_COLS, SCENARIO_COLS, SCENARIO_OPTIONS, FEATURE_SETS`; fixtures `fake_split`.
- Produces: `add_interest_scores(X: pd.DataFrame) -> pd.DataFrame` (copy of X plus `build_vs_protect`, `data_score`, `infra_score`); `build_preprocessor(feature_set: str = "all") -> sklearn.pipeline.Pipeline` taking raw `q1`–`q18`; invalid `feature_set` raises `ValueError`.

- [ ] **Step 1: Write failing tests**

```python
def test_add_interest_scores_exact():
    row = {"q3":4,"q11":2,"q4":5,"q5":1,"q6":3,"q7":5,"q12":3,"q13":5}   # other q's any valid value
    out = add_interest_scores(pd.DataFrame([row_with_all_q_cols]))
    assert out.loc[0,"build_vs_protect"] == -1.0 and out.loc[0,"data_score"] == 3.0 and out.loc[0,"infra_score"] == 3.0
    # input frame not modified

@pytest.mark.parametrize("fs,width", [("all",43),("ratings",14),("ratings+scores",17),("ratings+scenarios",40)])
def test_preprocessor_width(fake_split, fs, width):
    assert build_preprocessor(fs).fit_transform(X_train).shape[1] == width

def test_preprocessor_ignores_unseen_scenario_value(fake_split):
    # fitted on train; row with q15 = "Something new" -> no error, the 8 q15 columns are all zero

def test_scaler_fitted_on_train_only(fake_split):
    # train ratings columns have mean ~0 (abs < 1e-9) after fit_transform; a test set shifted by +2 does NOT have mean ~0

def test_invalid_feature_set():
    with pytest.raises(ValueError): build_preprocessor("nope")
```

- [ ] **Step 2:** Run `pytest tests/test_step2.py -v` → FAIL.
- [ ] **Step 3:** Implement. `build_preprocessor`: `Pipeline([("scores", FunctionTransformer(add_interest_scores)), ("columns", ColumnTransformer([...]))])`; ratings and score columns → `StandardScaler`; scenario columns → `OneHotEncoder(categories=[config.SCENARIO_OPTIONS[q] for q in SCENARIO_COLS], handle_unknown="ignore")`; which groups are included depends on `feature_set`. Output dense (`sparse_output=False`).
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `thailong/step2-features`; merge. **Stage 1 check-in:** run full `pytest`, show Sothon the real-data counts from Task 2 Step 5; push only if they say yes.

---

### Task 4: step3_models (Sothon's file)

**Files:** Create `pipeline/step3_models.py`, `tests/test_step3.py`.

**Interfaces:**
- Consumes: `build_preprocessor`.
- Produces: `build_candidates(feature_set: str = "all") -> dict[str, Pipeline]` with keys in order `"Baseline"`, `"Logistic Regression"`, `"KNN"`, `"Random Forest"`; every value is `Pipeline([("prep", build_preprocessor(feature_set)), ("clf", estimator)])`. (`feature_set` is an added optional argument used by the ablation.)
- Fixed settings: Baseline `DummyClassifier(strategy="most_frequent")`; LR `LogisticRegression(class_weight="balanced", max_iter=2000)`; KNN default `KNeighborsClassifier()`; RF `RandomForestClassifier(class_weight="balanced", random_state=42)`.

- [ ] **Step 1: Write failing tests:** `test_candidate_keys_and_order`; `test_fixed_settings` (asserts `class_weight == "balanced"`, `max_iter == 2000`, RF `random_state == 42`); `test_all_candidates_fit_and_predict(fake_split)` (every pipeline fits on train and returns predictions within `config.TRACKS`); `test_baseline_predicts_most_common_track` (all predictions equal `y_train.mode()[0]`).
- [ ] **Step 2:** Run `pytest tests/test_step3.py -v` → FAIL.
- [ ] **Step 3:** Implement `build_candidates`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `sothon/step3-models`; merge.

---

### Task 5: step4_tuning (Thailong's file)

**Files:** Create `pipeline/step4_tuning.py`, `tests/test_step4.py`.

**Interfaces:**
- Consumes: `build_candidates`, `config.PARAM_GRIDS, N_FOLDS, SEED`.
- Produces: `tune_candidates(candidates: dict[str, Pipeline], X_train, y_train, param_grids: dict | None = None) -> dict[str, dict]` (`param_grids=None` uses `config.PARAM_GRIDS`; the optional argument lets tests use tiny grids). Returns keys in order `"Logistic Regression"`, `"KNN"`, `"Random Forest"`, `"Soft Voting"`; each value `{"estimator": fitted model, "best_params": dict, "cv_macro_f1": float}`. For `"Soft Voting"`: `best_params = {}`, `cv_macro_f1 = None`.
- `candidates` passed in contains only the 3 single models (no Baseline).

- [ ] **Step 1: Write failing tests** (use tiny grids, e.g. LR `{"clf__C":[0.1,1]}`, KNN `{"clf__n_neighbors":[3,5]}`, RF `{"clf__n_estimators":[20], "clf__max_depth":[3,None]}`):
  `test_tune_returns_four_entries_in_order`; `test_best_params_come_from_grid`; `test_cv_macro_f1_between_0_and_1` (single models only); `test_soft_voting_probabilities` (`predict_proba(X_test)` has 4 columns and rows sum to 1 within 1e-9); `test_estimators_are_fitted` (every `estimator.predict(X_test)` runs).
- [ ] **Step 2:** Run `pytest tests/test_step4.py -v` → FAIL.
- [ ] **Step 3:** Implement: one `GridSearchCV(pipeline, grid, scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=42))` per model, `refit=True`; the ensemble is `VotingClassifier([(name, tuned.best_estimator_), …], voting="soft")` fitted on `X_train, y_train`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `thailong/step4-tuning`; merge.

---

### Task 6: step5_train (Sothon's file)

**Files:** Create `pipeline/step5_train.py`, `tests/test_step5.py`.

**Interfaces:**
- Consumes: steps 1–4 functions, `config.*`.
- Produces:
  - `cross_validate_models(models: dict[str, BaseEstimator], X_train, y_train) -> pd.DataFrame` — index = model name; columns `accuracy_mean, accuracy_std, macro_f1_mean, macro_f1_std, top2_mean, top2_std`; 5-fold stratified CV; top-2 scorer built from `top_k_accuracy_score(k=2, labels=config.TRACKS)` with `response_method="predict_proba"`.
  - `select_best_model(cv_table: pd.DataFrame) -> str` — spec §8 rule. Row `"Baseline"` is ignored. Highest `macro_f1_mean` is the best; models with `macro_f1_mean >= best_mean − best_std` are tied; return the tied model first in `config.MODEL_SIMPLICITY`.
  - `main() -> None` — loads `config.DATA_PATH`, runs steps 1–4, builds the CV table over Baseline + the 4 tuned models, selects the winner, writes: `results/cv_results.csv`, `results/tuned_params.json` ({model: best_params}), `models/winner_train_only.joblib` (winner fitted on train), `models/best_model.joblib` (clone of winner refit on all 257 rows), `models/shap_background.joblib` (50 random rows of `X_train`, seed 42), `models/model_info.json` (`winner`, `n_train`, `n_total`, the winner's CV row, `tuned_params`).

- [ ] **Step 1: Write failing tests**

```python
def test_select_best_model_clear_winner():          # RF 0.90±0.02, LR 0.70±0.02  -> "Random Forest"
def test_select_best_model_prefers_simpler_within_one_std():   # RF 0.80±0.04, LR 0.77±0.03 -> "Logistic Regression"
def test_select_best_model_ties_pick_first_simplest():         # KNN, RF tied, LR out -> "KNN"
def test_select_best_model_ignores_baseline():      # Baseline highest, others lower -> a non-Baseline name
def test_cross_validate_models_columns_and_ranges(fake_split):  # 6 columns, values in [0,1]; baseline macro_f1_mean < LR's
def test_cross_validate_models_rare_class():        # data where the rarest track has exactly 5 rows -> runs without error
def test_main_writes_all_artifacts(tmp_path, monkeypatch):
    # fake xlsx in tmp_path; monkeypatch config.DATA_PATH/MODELS_DIR/RESULTS_DIR and config.PARAM_GRIDS (tiny)
    # assert the 6 output files exist, model_info["winner"] in MODEL_SIMPLICITY, cv_results.csv has 5 rows,
    # best_model.joblib loads and predict_proba has 4 columns
```

- [ ] **Step 2:** Run `pytest tests/test_step5.py -v` → FAIL.
- [ ] **Step 3:** Implement the three functions and the `__main__` block (`main()`).
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `sothon/step5-train`; merge. **Stage 2 check-in:** run `python -m pipeline.step5_train` on real data; show Sothon the CV table and the winner; push only if they say yes.

---

### Task 7: step6_evaluate (Sothon's file)

**Files:** Create `pipeline/step6_evaluate.py`, `tests/test_step6.py`.

**Interfaces:**
- Consumes: `build_candidates(feature_set)`, `config.*`, `results/tuned_params.json`, `models/winner_train_only.joblib`.
- Produces:
  - `evaluate_on_test(model, X_test, y_test, baseline_model=None) -> dict` with keys `"metrics"` (DataFrame: rows `model` and optionally `baseline`; columns `accuracy, macro_f1, top2, accuracy_ci_low, accuracy_ci_high`), `"per_track"` (DataFrame indexed by the 4 tracks: `precision, recall, f1, support`), `"confusion"` (4×4 DataFrame, rows true, columns predicted, both in `config.TRACKS` order).
  - `run_ablation(X_train, y_train, best_params: dict[str, dict]) -> pd.DataFrame` — 12 rows (3 models × 4 feature sets), columns `model, feature_set, accuracy_mean, macro_f1_mean, macro_f1_std`; models are built with `build_candidates(feature_set)` then `set_params(**best_params[name])`.
- `__main__` block: loads the train-only winner, fits the baseline on train, calls both functions, writes `results/test_results.csv`, `results/per_track.csv`, `results/confusion_matrix.csv`, `results/ablation.csv`, `results/class_balance.csv`.

- [ ] **Step 1: Write failing tests**

```python
def test_evaluate_metrics_ranges_and_top2(fitted_model, fake_split):    # top2 >= accuracy; all in [0,1]; CI low <= accuracy <= high
def test_bootstrap_ci_is_deterministic():                               # two calls give identical CI
def test_confusion_matrix_shape_and_total():                            # 4x4, index/columns == config.TRACKS, sum == len(y_test)
def test_baseline_reported_next_to_model():                             # metrics has a "baseline" row when baseline_model is given
def test_evaluate_when_test_set_lacks_a_track():                        # y_test with only 3 tracks -> per_track still has 4 rows, no exception
def test_run_ablation_grid():                                           # 12 rows, 3 models, 4 feature sets, macro_f1_mean in [0,1]
```
  (`fitted_model` fixture: the Logistic Regression candidate fitted on `fake_split` train; add to `conftest.py`.)
- [ ] **Step 2:** Run `pytest tests/test_step6.py -v` → FAIL.
- [ ] **Step 3:** Implement both functions and the `__main__` block. Per-track precision/recall via `precision_recall_fscore_support(labels=TRACKS, zero_division=0)`; bootstrap resamples the 52 rows with `np.random.default_rng(42)`.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `sothon/step6-evaluate`; merge.

---

### Task 8: step7_analysis (Rachana's file)

**Files:** Create `pipeline/step7_analysis.py`, `tests/test_step7.py`.

**Interfaces:**
- Consumes: `config.*`, the CSV files from steps 5–6.
- Produces:
  - `analyze_feature_signal(X_train, y_train) -> dict[str, pd.DataFrame]` with `"correlation"` (17 rows: q1–q14 + the 3 scores; 4 columns, one per track; Pearson correlation with the 0/1 indicator of that track) and `"importance"` (18 rows, one per question q1–q18; Random Forest `feature_importances_` summed per question; sums to 1; sorted descending).
  - `plot_results() -> list[Path]` — reads `config.RESULTS_DIR` CSVs (`class_balance, cv_results, confusion_matrix, correlation, feature_importance, ablation`) and saves 6 PNGs in `results/figures/`: `class_balance.png, model_comparison.png, confusion_matrix.png, correlation_heatmap.png, feature_importance.png, ablation.png`. Uses the `Agg` backend.
- `__main__` block: runs `analyze_feature_signal` on the real train split, writes `results/correlation.csv`, `results/feature_importance.csv`, then calls `plot_results()`.

- [ ] **Step 1: Write failing tests:** `test_correlation_shape_and_range` (17×4, values in [−1, 1]); `test_correlation_planted_signal` (q4 vs Cybersecurity > 0.3 on `fake_split`); `test_importance_sums_to_one_and_sorted` (18 rows, sum within 1e-6 of 1, descending); `test_plot_results_creates_six_nonempty_pngs(tmp_path, monkeypatch)` (write small fake CSVs into `tmp_path`, point `config.RESULTS_DIR` there, assert the 6 files exist with size > 0).
- [ ] **Step 2:** Run `pytest tests/test_step7.py -v` → FAIL.
- [ ] **Step 3:** Implement both functions and the `__main__` block. Heatmap and bars with matplotlib only; titles in plain English; saved at dpi 150.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5:** Commit on `rachana/step7-analysis`; merge.

---

### Task 9: Real-data run and verification

**Files:** Modify none (unless a bug is found); outputs go to `models/` and `results/`.

**Interfaces:** Consumes everything from Tasks 1–8.

- [ ] **Step 1:** Run the full test suite: `.venv/Scripts/python -m pytest -v` → all pass.
- [ ] **Step 2:** `.venv/Scripts/python -m pipeline.step5_train`, then `-m pipeline.step6_evaluate`, then `-m pipeline.step7_analysis`. Each must exit 0 and create its files (list them with `ls results results/figures models`).
- [ ] **Step 3: Verify the spec's checks** and report each as pass/fail with the evidence:
  1. `results/cv_results.csv` has 5 rows; the Baseline's `macro_f1_mean` is below every real model's.
  2. `grep -n "X_test" pipeline/step3_models.py pipeline/step4_tuning.py pipeline/step5_train.py` shows `X_test` used only to build the final all-257 refit in `main()`.
  3. `models/model_info.json` shows `n_train = 205`, `n_total = 257`, and a `winner` from the 4 models.
  4. `results/test_results.csv` has `model` and `baseline` rows and a CI; `results/confusion_matrix.csv` sums to 52.
  5. `results/ablation.csv` has 12 rows.
  6. `results/figures/` has 6 non-empty PNGs.
- [ ] **Step 4:** Summarize the real numbers (CV table, test metrics with CI, ablation) for Sothon, including the caveats in spec §8. If any check fails, fix the cause before moving on.
- [ ] **Step 5:** Commit any fixes (`sothon/verification`) and merge. **Stage 3 check-in:** show the results; push only if Sothon says yes.

---

### Task 10: step8_predict (Thailong's file)

**Files:** Create `pipeline/step8_predict.py`, `tests/test_step8.py`.

**Interfaces:**
- Consumes: `models/best_model.joblib`, `models/shap_background.joblib`, `models/model_info.json`, `config.RATING_LABELS, SCENARIO_LABELS`.
- Produces:
  - `load_artifacts() -> tuple[BaseEstimator, pd.DataFrame, dict]` — raises `FileNotFoundError` whose message contains `python -m pipeline.step5_train` when a file is missing.
  - `predict_top2(model, answers: dict) -> list[tuple[str, float]]` — `answers` has keys `q1`–`q18`; returns 2 `(track, probability)` tuples, highest first.
  - `explain_prediction(model, background: pd.DataFrame, answers: dict, track: str, n_reasons: int = 3) -> list[str]` — the `n_reasons` questions with the largest SHAP value for `track`, as sentences: ratings → `You rated "<RATING_LABELS[q]>" <value> out of 5.`; scenarios → `You chose "<answer>" for <SCENARIO_LABELS[q]>.`

- [ ] **Step 1: Spike (verify SHAP works):** on the fitted model from `conftest.py`, run a SHAP explainer on one raw-answer row with a background of 20 rows. Expected: 18 values per class (shape `(1, 18, 4)` or equivalent), no dtype errors. If the standard explainer rejects the mixed numeric/text DataFrame, wrap the model in a function that rebuilds a DataFrame from an array and use the permutation explainer (spec §10). Record which worked in the file's docstring.
- [ ] **Step 2: Write failing tests**

```python
def test_predict_top2_shape_and_order():           # 2 items, distinct tracks in TRACKS, p1 >= p2, both in [0,1]
def test_explain_returns_three_sentences():        # 3 non-empty str; each contains "out of 5" or a quoted scenario answer
def test_extreme_answers_still_work():             # all q1-q14 = 1, then = 5 (scenarios = first option): 2 tracks, 3 reasons, no exception
def test_load_artifacts_missing_file(tmp_path, monkeypatch):   # MODELS_DIR -> empty dir: FileNotFoundError mentioning step5_train
def test_load_artifacts_roundtrip(tmp_path, monkeypatch):      # save fake artifacts, load them, predict works
```
- [ ] **Step 3:** Run `pytest tests/test_step8.py -v` → FAIL.
- [ ] **Step 4:** Implement the three functions. Take the top `n_reasons` by SHAP value for the asked track (values may be ≤ 0 for atypical answers; still return 3).
- [ ] **Step 5:** Run tests → PASS.
- [ ] **Step 6:** Commit on `thailong/step8-predict`; merge.

---

### Task 11: Flask app (Sothon's files)

**Files:** Create `app/__init__.py`, `app/app.py`, `app/templates/index.html`, `app/templates/result.html`, `app/templates/error.html`, `app/static/style.css`, `tests/test_app.py`.

**Interfaces:**
- Consumes: `load_artifacts, predict_top2, explain_prediction`, `config.*`.
- Produces: module-level `app = Flask(__name__)`; `questionnaire()` (`GET /`); `recommend()` (`POST /recommend`); `validate_answers(form: Mapping[str, str]) -> tuple[dict, list[str]]` returning `(clean_answers, error_messages)` where ratings are `int` 1–5 (only the strings `"1"`–`"5"` are accepted) and scenario answers are strings from `config.SCENARIO_OPTIONS`.
- Artifacts are loaded lazily on the first `POST` (not at import). Missing model → `error.html` with status 503 and the text `python -m pipeline.step5_train`. `app.py` inserts the project root into `sys.path` so `python app/app.py` works; running it serves `http://127.0.0.1:5000`.

- [ ] **Step 1: Write failing tests** (Flask test client; `conftest.py` gets a fixture `saved_artifacts(tmp_path, monkeypatch)` that saves a fake model/background/info into a temp `MODELS_DIR`)

```python
def test_validate_answers_accepts_valid_form():     # ints for q1-q14, strings for q15-q18, errors == []
@pytest.mark.parametrize("bad", ["4.5", "abc", "", "6", "0"])
def test_validate_answers_rejects_bad_rating(bad): # one error message, answers incomplete
def test_validate_answers_rejects_missing_and_unknown():   # missing q3, q15 = "Not an option" -> 2 errors
def test_get_form_shows_18_questions():             # 200; 18 question groups; no q19 field
def test_post_valid_shows_two_tracks_and_note(saved_artifacts):   # 200; two track names; 3 reasons each (6 list items); text "suggestion"
def test_post_invalid_re_renders_form_with_messages(saved_artifacts):   # 200 (not 500); error text visible; submitted values kept
def test_post_all_ones_and_all_fives(saved_artifacts):      # both return 200 with 2 tracks
def test_post_without_model_shows_friendly_error():         # 503; body contains "python -m pipeline.step5_train"; GET / still 200
```
- [ ] **Step 2:** Run `pytest tests/test_app.py -v` → FAIL.
- [ ] **Step 3:** Implement `validate_answers` first, then the routes and templates. `index.html`: 14 rating questions as radio buttons 1–5 using `RATING_LABELS` text from the Form, 4 scenario questions as radio groups, a submit button "Get my recommendation". `result.html`: two cards (track, percentage bar, 3 reasons), the note "This is a suggestion based on your interests, not a decision.", and a "Start over" link. CSS: one file, readable at phone width.
- [ ] **Step 4:** Run tests → PASS.
- [ ] **Step 5 (real-model smoke):** start the app in the background (`python app/app.py`), `curl` the form (expect 200), then POST a Networking-style profile (q5, q6, q13 = 5, q4 = 2, scenario answers about servers/networks). Expected: Networking is among the two suggestions with 3 reasons each. Stop the server.
- [ ] **Step 6:** Commit on `sothon/app`; merge. **Stage 4 check-in:** show Sothon the app output; push only if they say yes.

---

### Task 12: Team docs and README

**Files:** Create `docs/01_overview_and_how_to_run.md`, `docs/02_step1_preprocessing.md`, `docs/03_step2_features.md`, `docs/04_step3_models_and_step4_tuning.md`, `docs/05_step5_train.md`, `docs/06_step6_evaluate.md`, `docs/07_step7_analysis.md`, `docs/08_step8_predict.md`, `docs/09_app.md`, `docs/10_metrics_explained.md`, `docs/11_qa_sheet.md`, `docs/12_git_teamwork.md`; replace `README.md`.

**Interfaces:** Consumes the real results from Task 9 and the code as built.

- [ ] **Step 1:** Write `01`: the 13-step flow table, the four run commands, setup steps, and who owns which file.
- [ ] **Step 2:** Write each walkthrough `02`–`09` with exactly these headings: `What this file does`, `Function by function` (plain-language explanation for each function in that file), `Five likely questions` (numbered Q1–Q5 with short answers), `Your one change` (a concrete suggestion for that owner, small enough to do and explain). Use the owner's name at the top. Write in plain language for readers new to coding.
- [ ] **Step 3:** Write `10` from the conversation's worked examples (accuracy, macro F1 with the 20-student scoreboard, top-2 with the 5-student table, tuning and 5-fold with the KNN example); `11`: every question from the five-question lists plus questions about leakage, the small test set, why SHAP, why macro F1, the scenario-mirrors-label caveat, and the dropped year comparison; `12`: branch, commit, pull request, `pytest` before merge, and the authorship rule from spec §12.
- [ ] **Step 4:** Replace the one-line `README.md` with: what the project is, setup, run commands, folder map, link to `docs/`.
- [ ] **Step 5 (verify):** `grep -c "## Five likely questions" docs/0[2-9]_*.md` shows 1 for each of the 8 walkthroughs; each has Q1–Q5.
- [ ] **Step 6:** Commit on `sothon/docs`; merge.

---

### Task 13: Report and slides drafts

**Files:** Create `report/draft_report.md`, `report/slides_outline.md`.

**Interfaces:** Consumes the results files and, if available, the course template `ITM-390-Machine-Learning-Proposal-Format-1.docx` saved in `report/`.

- [ ] **Step 1:** If the template docx is in `report/`, read its section headings and follow its order and instructions; otherwise use the proposal's sections (Introduction, Goals and Objectives, Literature Review, Methodology, Results, Conclusion, References) and flag it.
- [ ] **Step 2:** Write `draft_report.md`: methodology exactly as built; a results section whose every number is copied from `results/*.csv`; limitations from spec §8 and §2 (re-imported timestamps, small test set, scenario options mirroring the label, dropped year comparison); references reused from the proposal.
- [ ] **Step 3:** Write `slides_outline.md` with the professor's sections (Introduction, Objectives, Methods, Experiments, Results, Discussion/Conclusion, References), a 15-minute budget, and the per-member split from spec §12.
- [ ] **Step 4 (verify):** every metric quoted in `draft_report.md` appears in the matching results CSV (check each by `grep`).
- [ ] **Step 5:** Commit on `sothon/report-draft`; merge. **Stage 5 check-in:** hand the drafts to the team for review and ask whether to convert them to Word and PowerPoint.
