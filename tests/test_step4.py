import numpy as np
import pytest

from pipeline import config
from pipeline.step3_models import build_candidates
from pipeline.step4_tuning import tune_candidates

TINY_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1]},
    "KNN": {"clf__n_neighbors": [3, 5]},
    "Random Forest": {"clf__n_estimators": [20], "clf__max_depth": [3, None]},
}
SINGLE_MODELS = ["Logistic Regression", "KNN", "Random Forest"]


def _three_untuned_models():
    candidates = build_candidates()
    candidates.pop("Baseline")          # the baseline has no settings to tune
    return candidates


@pytest.fixture
def tuned(fake_split):
    X_train, _, y_train, _ = fake_split
    return tune_candidates(_three_untuned_models(), X_train, y_train, param_grids=TINY_GRIDS)


def test_tune_returns_four_entries_in_order(tuned):
    assert list(tuned) == SINGLE_MODELS + ["Soft Voting"]
    for entry in tuned.values():
        assert set(entry) == {"estimator", "best_params", "cv_macro_f1"}


def test_best_params_come_from_the_grid(tuned):
    for name in SINGLE_MODELS:
        best = tuned[name]["best_params"]
        assert set(best) == set(TINY_GRIDS[name])
        for key, value in best.items():
            assert value in TINY_GRIDS[name][key]


def test_cv_macro_f1_is_a_score_for_single_models_only(tuned):
    for name in SINGLE_MODELS:
        assert 0.0 <= tuned[name]["cv_macro_f1"] <= 1.0
    assert tuned["Soft Voting"]["best_params"] == {}
    assert tuned["Soft Voting"]["cv_macro_f1"] is None


def test_soft_voting_gives_probabilities_for_the_four_tracks(tuned, fake_split):
    X_test = fake_split[1]
    voting = tuned["Soft Voting"]["estimator"]
    proba = voting.predict_proba(X_test)
    assert proba.shape == (len(X_test), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(voting.classes_) == config.TRACKS


def test_every_estimator_is_already_fitted(tuned, fake_split):
    X_test = fake_split[1]
    for entry in tuned.values():
        assert len(entry["estimator"].predict(X_test)) == len(X_test)


def test_default_grids_are_read_from_config_at_call_time(fake_split, monkeypatch):
    X_train, _, y_train, _ = fake_split
    monkeypatch.setattr(config, "PARAM_GRIDS", TINY_GRIDS)
    result = tune_candidates(_three_untuned_models(), X_train, y_train)   # no param_grids given
    assert set(result["KNN"]["best_params"]) == {"clf__n_neighbors"}
