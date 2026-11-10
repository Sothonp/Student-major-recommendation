import pytest

from pipeline import config
from pipeline.step3_models import build_candidates
from pipeline.step6_evaluate import evaluate_on_test, run_ablation

METRIC_COLUMNS = ["accuracy", "macro_f1", "top2", "accuracy_ci_low", "accuracy_ci_high"]


def test_metrics_ranges_and_top2(fitted_model, fake_split):
    X_test, y_test = fake_split[1], fake_split[3]

    metrics = evaluate_on_test(fitted_model, X_test, y_test)["metrics"]

    assert list(metrics.columns) == METRIC_COLUMNS
    row = metrics.loc["model"]
    assert ((row >= 0) & (row <= 1)).all()
    assert row["accuracy"] == pytest.approx((fitted_model.predict(X_test) == y_test).mean())
    assert row["top2"] >= row["accuracy"]
    assert row["accuracy_ci_low"] <= row["accuracy"] <= row["accuracy_ci_high"]


def test_bootstrap_interval_is_repeatable(fitted_model, fake_split):
    X_test, y_test = fake_split[1], fake_split[3]
    first = evaluate_on_test(fitted_model, X_test, y_test)["metrics"]
    second = evaluate_on_test(fitted_model, X_test, y_test)["metrics"]
    assert first.equals(second)


def test_confusion_matrix_shape_and_total(fitted_model, fake_split):
    X_test, y_test = fake_split[1], fake_split[3]

    confusion = evaluate_on_test(fitted_model, X_test, y_test)["confusion"]

    assert confusion.shape == (4, 4)
    assert list(confusion.index) == config.TRACKS
    assert list(confusion.columns) == config.TRACKS
    assert confusion.to_numpy().sum() == len(y_test)


def test_per_track_table(fitted_model, fake_split):
    X_test, y_test = fake_split[1], fake_split[3]

    per_track = evaluate_on_test(fitted_model, X_test, y_test)["per_track"]

    assert list(per_track.index) == config.TRACKS
    assert list(per_track.columns) == ["precision", "recall", "f1", "support"]
    assert per_track["support"].sum() == len(y_test)


def test_baseline_is_reported_next_to_the_model(fitted_model, fake_split):
    X_train, X_test, y_train, y_test = fake_split
    baseline = build_candidates()["Baseline"].fit(X_train, y_train)

    with_baseline = evaluate_on_test(fitted_model, X_test, y_test, baseline_model=baseline)["metrics"]
    without = evaluate_on_test(fitted_model, X_test, y_test)["metrics"]

    assert list(with_baseline.index) == ["model", "baseline"]
    assert list(without.index) == ["model"]
    assert with_baseline.loc["baseline", "macro_f1"] < with_baseline.loc["model", "macro_f1"]


def test_test_set_that_lacks_a_track(fitted_model, fake_split):
    X_test, y_test = fake_split[1], fake_split[3]
    keep = y_test != "Cybersecurity"

    result = evaluate_on_test(fitted_model, X_test[keep], y_test[keep])   # must not raise

    assert list(result["per_track"].index) == config.TRACKS
    assert result["per_track"].loc["Cybersecurity", "support"] == 0
    assert 0 <= result["metrics"].loc["model", "top2"] <= 1


def test_run_ablation_grid(fake_split):
    X_train, _, y_train, _ = fake_split
    best_params = {
        "Logistic Regression": {"clf__C": 1},
        "KNN": {"clf__n_neighbors": 5, "clf__weights": "uniform"},
        "Random Forest": {"clf__n_estimators": 20, "clf__max_depth": 3},
    }

    table = run_ablation(X_train, y_train, best_params)

    assert list(table.columns) == ["model", "feature_set", "accuracy_mean", "macro_f1_mean", "macro_f1_std"]
    assert len(table) == 12
    assert set(table["model"]) == set(best_params)
    assert set(table["feature_set"]) == set(config.FEATURE_SETS)
    assert table["macro_f1_mean"].between(0, 1).all()
