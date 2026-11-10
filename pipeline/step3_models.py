"""Step 3: create the models (not yet tuned).

Owner: Sothon. Each model is one scikit-learn Pipeline: first the preprocessing from
step 2 ("prep"), then the classifier ("clf"). The tuning grids in config use "clf__".
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from pipeline import config
from pipeline.step2_features import build_preprocessor


def build_candidates(feature_set="all"):
    """Return the baseline and the 3 untuned models, in this order.

    - Baseline: always answers the most common track (a floor the real models must beat)
    - Logistic Regression and Random Forest use balanced class weights, so the small
      Cybersecurity group is not ignored. KNN has no class weights.

    feature_set picks which question groups the models may use (see step 2); the
    ablation experiment in step 6 uses the non-default groups.
    """
    classifiers = {
        # "prior" always answers the most common track, like "most_frequent", but also reports each
        # track's share as its probability. That makes the baseline's top-2 score meaningful:
        # it means "always suggest the two most common tracks".
        "Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=config.SEED),
    }
    return {
        name: Pipeline([("prep", build_preprocessor(feature_set)), ("clf", clf)])
        for name, clf in classifiers.items()
    }
