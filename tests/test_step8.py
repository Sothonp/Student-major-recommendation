import json

import joblib
import pytest

from pipeline import config
from pipeline.step8_predict import explain_prediction, load_artifacts, predict_top2


@pytest.fixture
def background(fake_split):
    return fake_split[0].sample(20, random_state=0)


@pytest.fixture
def answers(fake_split):
    """One student's 18 answers, as the app passes them: {"q1": 4, ..., "q18": "..."}."""
    return fake_split[1].iloc[0].to_dict()


def _extreme(rating):
    row = {q: rating for q in config.RATING_COLS}
    row.update({q: config.SCENARIO_OPTIONS[q][0] for q in config.SCENARIO_COLS})
    return row


def test_predict_top2_shape_and_order(fitted_model, answers):
    top2 = predict_top2(fitted_model, answers)

    assert len(top2) == 2
    (first, p1), (second, p2) = top2
    assert first != second
    assert {first, second} <= set(config.TRACKS)
    assert p1 >= p2
    assert 0 <= p2 <= p1 <= 1
    assert p1 + p2 <= 1 + 1e-9


def test_explain_returns_three_distinct_sentences(fitted_model, background, answers):
    reasons = explain_prediction(fitted_model, background, answers, "Networking")

    assert len(reasons) == 3
    assert len(set(reasons)) == 3
    for reason in reasons:
        assert isinstance(reason, str) and reason
        rating_sentence = "out of 5" in reason
        scenario_sentence = any(f'"{answers[q]}"' in reason for q in config.SCENARIO_COLS)
        assert rating_sentence or scenario_sentence


def test_explanation_is_repeatable(fitted_model, background, answers):
    first = explain_prediction(fitted_model, background, answers, "Cybersecurity")
    second = explain_prediction(fitted_model, background, answers, "Cybersecurity")
    assert first == second


@pytest.mark.parametrize("rating", [1, 5])
def test_extreme_answers_still_work(fitted_model, background, rating):
    answers = _extreme(rating)

    top2 = predict_top2(fitted_model, answers)
    reasons = explain_prediction(fitted_model, background, answers, top2[0][0])

    assert len(top2) == 2
    assert len(reasons) == 3


def test_load_artifacts_names_the_missing_step(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")    # does not exist
    with pytest.raises(FileNotFoundError, match="step5_train"):
        load_artifacts()


def test_load_artifacts_round_trip(tmp_path, monkeypatch, fitted_model, background, answers):
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(fitted_model, models / "best_model.joblib")
    joblib.dump(background, models / "shap_background.joblib")
    (models / "model_info.json").write_text(json.dumps({"winner": "Logistic Regression"}))
    monkeypatch.setattr(config, "MODELS_DIR", models)

    model, loaded_background, info = load_artifacts()

    assert info == {"winner": "Logistic Regression"}
    assert loaded_background.shape == background.shape
    assert len(predict_top2(model, answers)) == 2
