from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from pipeline import config
from pipeline.step3_models import build_candidates


def test_candidate_keys_and_order():
    assert list(build_candidates()) == ["Baseline", "Logistic Regression", "KNN", "Random Forest"]


def test_every_candidate_is_a_prep_plus_clf_pipeline():
    for pipe in build_candidates().values():
        assert isinstance(pipe, Pipeline)
        assert list(pipe.named_steps) == ["prep", "clf"]   # the grids use the name "clf__..."


def test_fixed_settings():
    c = build_candidates()
    baseline = c["Baseline"].named_steps["clf"]
    lr = c["Logistic Regression"].named_steps["clf"]
    knn = c["KNN"].named_steps["clf"]
    rf = c["Random Forest"].named_steps["clf"]

    assert isinstance(baseline, DummyClassifier) and baseline.strategy == "most_frequent"
    assert isinstance(lr, LogisticRegression)
    assert lr.class_weight == "balanced" and lr.max_iter == 2000
    assert isinstance(knn, KNeighborsClassifier)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.class_weight == "balanced" and rf.random_state == config.SEED


def test_all_candidates_fit_and_predict(fake_split):
    X_train, X_test, y_train, _ = fake_split
    for name, pipe in build_candidates().items():
        predictions = pipe.fit(X_train, y_train).predict(X_test)
        assert len(predictions) == len(X_test), name
        assert set(predictions) <= set(config.TRACKS), name


def test_baseline_predicts_most_common_track(fake_split):
    X_train, X_test, y_train, _ = fake_split
    baseline = build_candidates()["Baseline"].fit(X_train, y_train)
    assert set(baseline.predict(X_test)) == {y_train.mode()[0]}


def test_feature_set_reaches_the_preprocessor(fake_split):
    X_train, _, y_train, _ = fake_split
    pipe = build_candidates("ratings")["KNN"]
    assert pipe.named_steps["prep"].fit_transform(X_train).shape[1] == 14
