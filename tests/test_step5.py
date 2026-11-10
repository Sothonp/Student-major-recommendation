import json

import joblib
import pandas as pd
import pytest

from pipeline import config
from pipeline.step3_models import build_candidates
from pipeline.step5_train import cross_validate_models, main, select_best_model

TINY_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1]},
    "KNN": {"clf__n_neighbors": [3, 5]},
    "Random Forest": {"clf__n_estimators": [20], "clf__max_depth": [3, None]},
}
CV_COLUMNS = ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "top2_mean", "top2_std"]


def _table(rows):
    """rows: {model: (macro_f1_mean, macro_f1_std)}"""
    return pd.DataFrame(
        {"macro_f1_mean": {m: v[0] for m, v in rows.items()},
         "macro_f1_std": {m: v[1] for m, v in rows.items()}}
    )


def test_select_best_model_clear_winner():
    table = _table({"Logistic Regression": (0.70, 0.02), "KNN": (0.65, 0.02),
                    "Random Forest": (0.90, 0.02), "Soft Voting": (0.80, 0.02)})
    assert select_best_model(table) == "Random Forest"


def test_select_best_model_prefers_simpler_within_one_std():
    table = _table({"Logistic Regression": (0.77, 0.03), "KNN": (0.60, 0.03),
                    "Random Forest": (0.80, 0.04), "Soft Voting": (0.75, 0.03)})
    assert select_best_model(table) == "Logistic Regression"


def test_select_best_model_tie_picks_the_first_simplest():
    table = _table({"Logistic Regression": (0.70, 0.03), "KNN": (0.82, 0.03),
                    "Random Forest": (0.83, 0.03), "Soft Voting": (0.79, 0.03)})
    assert select_best_model(table) == "KNN"   # KNN and Random Forest tie; Logistic Regression is out


def test_select_best_model_ignores_the_baseline():
    table = _table({"Baseline": (0.95, 0.01), "Logistic Regression": (0.70, 0.02),
                    "Random Forest": (0.80, 0.02)})
    assert select_best_model(table) == "Random Forest"


def test_cross_validate_models_columns_and_ranges(fake_split):
    X_train, _, y_train, _ = fake_split
    candidates = build_candidates()
    models = {"Baseline": candidates["Baseline"], "Logistic Regression": candidates["Logistic Regression"]}

    table = cross_validate_models(models, X_train, y_train)

    assert list(table.columns) == CV_COLUMNS
    assert list(table.index) == ["Baseline", "Logistic Regression"]
    assert ((table >= 0) & (table <= 1)).all().all()
    assert table.loc["Baseline", "macro_f1_mean"] < table.loc["Logistic Regression", "macro_f1_mean"]
    assert (table["top2_mean"] >= table["accuracy_mean"]).all()


def test_cross_validate_models_rare_track(fake_labeled):
    # the rarest track has exactly 5 rows, so each of the 5 folds holds just one of them
    others = fake_labeled[fake_labeled["track"] != "Cybersecurity"]
    rare = fake_labeled[fake_labeled["track"] == "Cybersecurity"].head(5)
    data = pd.concat([others, rare])
    models = {"Logistic Regression": build_candidates()["Logistic Regression"]}

    table = cross_validate_models(models, data[config.QUESTION_COLS], data["track"])

    assert 0 <= table.loc["Logistic Regression", "top2_mean"] <= 1


@pytest.fixture
def trained(fake_xlsx_path, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PATH", fake_xlsx_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "PARAM_GRIDS", TINY_GRIDS)
    main()
    return tmp_path


def test_main_writes_all_artifacts(trained):
    for name in ["winner_train_only.joblib", "best_model.joblib", "shap_background.joblib", "model_info.json"]:
        assert (trained / "models" / name).exists(), name
    for name in ["cv_results.csv", "tuned_params.json"]:
        assert (trained / "results" / name).exists(), name


def test_main_results_are_consistent(trained):
    info = json.loads((trained / "models" / "model_info.json").read_text())
    cv = pd.read_csv(trained / "results" / "cv_results.csv", index_col=0)

    assert info["winner"] in config.MODEL_SIMPLICITY
    assert info["n_total"] == 160 and info["n_train"] == 128   # 160 fake usable rows, 32 held out
    assert list(cv.index) == ["Baseline"] + config.MODEL_SIMPLICITY
    assert cv.loc["Baseline", "macro_f1_mean"] < cv.loc[info["winner"], "macro_f1_mean"]
    assert set(json.loads((trained / "results" / "tuned_params.json").read_text())) == {
        "Logistic Regression", "KNN", "Random Forest", "Soft Voting"}


def test_main_saved_models_work(trained):
    final = joblib.load(trained / "models" / "best_model.joblib")
    background = joblib.load(trained / "models" / "shap_background.joblib")

    assert list(background.columns) == config.QUESTION_COLS
    assert len(background) == 50
    assert final.predict_proba(background).shape == (50, 4)
    train_only = joblib.load(trained / "models" / "winner_train_only.joblib")
    assert train_only.predict(background).shape == (50,)
