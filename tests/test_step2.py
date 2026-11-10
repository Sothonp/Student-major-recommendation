import pandas as pd
import pytest

from pipeline import config
from pipeline.step2_features import add_interest_scores, build_preprocessor


def _one_student(**overrides):
    """One student with all 18 answers; ratings default to 3, scenarios to the first option."""
    row = {q: 3 for q in config.RATING_COLS}
    row.update({q: config.SCENARIO_OPTIONS[q][0] for q in config.SCENARIO_COLS})
    row.update(overrides)
    return pd.DataFrame([row])


def test_add_interest_scores_exact():
    X = _one_student(q3=4, q11=2, q4=5, q5=1, q6=3, q7=5, q12=3, q13=5)
    before = X.copy()

    out = add_interest_scores(X)

    assert out.loc[0, "build_vs_protect"] == -1.0   # mean(5, 3) - 5
    assert out.loc[0, "data_score"] == 3.0          # mean(4, 2)
    assert out.loc[0, "infra_score"] == 3.0         # mean(1, 3, 5)
    pd.testing.assert_frame_equal(X, before)         # the input is not changed


@pytest.mark.parametrize(
    "feature_set, width",
    [("all", 43), ("ratings", 14), ("ratings+scores", 17), ("ratings+scenarios", 40)],
)
def test_preprocessor_width(fake_split, feature_set, width):
    X_train = fake_split[0]
    assert build_preprocessor(feature_set).fit_transform(X_train).shape[1] == width


def test_preprocessor_ignores_unseen_scenario_value(fake_split):
    X_train, X_test = fake_split[0], fake_split[1]
    prep = build_preprocessor("all").fit(X_train)
    new_student = X_test.head(1).copy()
    new_student["q15"] = "Something new"

    out = prep.transform(new_student)   # must not raise

    names = list(prep.named_steps["columns"].get_feature_names_out())
    q15_cols = [i for i, n in enumerate(names) if n.startswith("scenarios__q15_")]
    assert len(q15_cols) == 8
    assert (out[0, q15_cols] == 0).all()


def test_scaler_is_fitted_on_train_only(fake_split):
    X_train, X_test = fake_split[0], fake_split[1]
    prep = build_preprocessor("all")

    train_out = prep.fit_transform(X_train)
    assert abs(train_out[:, :14].mean()) < 1e-9      # training ratings are centred

    shifted = X_test.copy()
    shifted[config.RATING_COLS] = shifted[config.RATING_COLS] + 2
    assert prep.transform(shifted)[:, :14].mean() > 1   # not re-centred on the test data


def test_invalid_feature_set():
    with pytest.raises(ValueError):
        build_preprocessor("nope")
