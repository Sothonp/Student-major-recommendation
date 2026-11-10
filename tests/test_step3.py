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

    assert isinstance(baseline, DummyClassifier) and baseline.strategy == "prior"   # always the most common track, with class shares as probabilities
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


def test_baseline_top2_is_the_share_of_the_two_most_common_tracks(fake_split):
    # Review fix: a "most common track" baseline must give real probabilities (the class shares),
    # otherwise its top-2 score is decided by column order and means nothing.
    import numpy as np
    from sklearn.metrics import top_k_accuracy_score

    X_train, X_test, y_train, y_test = fake_split
    sizes = [("Data Science / AI", 30), ("Networking", 20), ("Software Development", 10), ("Cybersecurity", 5)]
    keep = np.concatenate([np.flatnonzero(y_train == track)[:n] for track, n in sizes])
    baseline = build_candidates()["Baseline"].fit(X_train.iloc[keep], y_train.iloc[keep])

    proba = baseline.predict_proba(X_test)

    # columns follow config.TRACKS (sorted): Cybersecurity, Data Science / AI, Networking, Software Development
    assert np.allclose(proba[0], [5 / 65, 30 / 65, 20 / 65, 10 / 65])
    top2 = top_k_accuracy_score(y_test, proba, k=2, labels=config.TRACKS)
    assert abs(top2 - y_test.isin(["Data Science / AI", "Networking"]).mean()) < 1e-9
