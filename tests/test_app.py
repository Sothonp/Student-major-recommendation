import json
import re

import joblib
import pytest

from app.app import app, validate_answers
from pipeline import config


def _valid_form(rating="3"):
    form = {q: rating for q in config.RATING_COLS}
    form.update({q: config.SCENARIO_OPTIONS[q][0] for q in config.SCENARIO_COLS})
    return form


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def saved_artifacts(tmp_path, monkeypatch, fitted_model, fake_split):
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(fitted_model, models / "best_model.joblib")
    joblib.dump(fake_split[0].sample(20, random_state=0), models / "shap_background.joblib")
    (models / "model_info.json").write_text(json.dumps({"winner": "Logistic Regression"}))
    monkeypatch.setattr(config, "MODELS_DIR", models)


# ---- validate_answers -------------------------------------------------------

def test_validate_answers_accepts_a_valid_form():
    answers, errors = validate_answers(_valid_form("4"))

    assert errors == []
    assert answers["q1"] == 4 and isinstance(answers["q1"], int)
    assert answers["q15"] == config.SCENARIO_OPTIONS["q15"][0]
    assert set(answers) == set(config.QUESTION_COLS)


@pytest.mark.parametrize("bad", ["4.5", "abc", "", "6", "0"])
def test_validate_answers_rejects_a_bad_rating(bad):
    form = _valid_form()
    form["q1"] = bad

    answers, errors = validate_answers(form)

    assert len(errors) == 1 and "Question 1" in errors[0]
    assert "q1" not in answers


def test_validate_answers_rejects_missing_and_unknown_answers():
    form = _valid_form()
    del form["q3"]                       # missing rating
    form["q15"] = "Not an option"        # not one of the listed choices

    _, errors = validate_answers(form)

    assert len(errors) == 2
    assert "Question 3" in errors[0] and "Question 15" in errors[1]


# ---- pages ------------------------------------------------------------------

def test_get_form_shows_the_18_questions_and_no_major_question(client):
    response = client.get("/")

    assert response.status_code == 200
    names = set(re.findall(r'name="(q\d+)"', response.get_data(as_text=True)))
    assert names == set(config.QUESTION_COLS)      # q1..q18, no q19


def test_post_valid_form_shows_two_tracks_reasons_and_note(client, saved_artifacts):
    response = client.post("/recommend", data=_valid_form())
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert sum(track in page for track in config.TRACKS) >= 2
    assert page.count("<li>") == 6                 # 3 reasons for each of the 2 tracks
    assert "suggestion" in page
    assert 'href="/"' in page                      # the Start over link


def test_post_invalid_form_shows_the_form_again_with_messages(client, saved_artifacts):
    form = _valid_form("5")
    form["q3"] = "abc"

    response = client.post("/recommend", data=form)
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Question 3" in page
    assert re.search(r'name="q1" value="5"[^>]*checked', page)    # earlier answers are kept


@pytest.mark.parametrize("rating", ["1", "5"])
def test_post_all_ones_and_all_fives(client, saved_artifacts, rating):
    response = client.post("/recommend", data=_valid_form(rating))
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert sum(track in page for track in config.TRACKS) >= 2


def test_post_without_a_trained_model_shows_a_friendly_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "no_models_here")

    response = client.post("/recommend", data=_valid_form())

    assert response.status_code == 503
    assert "python -m pipeline.step5_train" in response.get_data(as_text=True)
    assert client.get("/").status_code == 200       # the form itself still works
