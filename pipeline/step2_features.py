"""Step 2: features. Build the 3 interest scores and the preprocessing pipeline.

Owner: Thailong (Sothon co-owns build_preprocessor, where data leakage matters).
The preprocessor takes the 18 raw answers q1..q18, so every model sees the same input
and the app can explain the 18 questions directly.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from pipeline import config

SCORE_COLS = ["build_vs_protect", "data_score", "infra_score"]


def add_interest_scores(X):
    """Return a copy of X with 3 combined scores added.

    - build_vs_protect = mean(q7, q12) - q4   (software building vs. cybersecurity)
    - data_score       = mean(q3, q11)        (data and patterns)
    - infra_score      = mean(q5, q6, q13)    (networks, servers, systems)
    """
    out = X.copy()
    out["build_vs_protect"] = (X["q7"] + X["q12"]) / 2 - X["q4"]
    out["data_score"] = (X["q3"] + X["q11"]) / 2
    out["infra_score"] = (X["q5"] + X["q6"] + X["q13"]) / 3
    return out


def build_preprocessor(feature_set="all"):
    """Build the preprocessing pipeline for one group of features.

    feature_set is one of config.FEATURE_SETS:
      "ratings" (14), "ratings+scores" (17), "ratings+scenarios" (40), "all" (43).

    The scaler lives inside this pipeline, so inside cross-validation it is fitted
    on the training folds only (no data leakage). Scenario answers become 0/1 columns;
    an answer that was never seen in training is ignored instead of crashing.
    """
    if feature_set not in config.FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {config.FEATURE_SETS}, got {feature_set!r}")

    parts = [("ratings", StandardScaler(), config.RATING_COLS)]
    if feature_set in ("ratings+scores", "all"):
        parts.append(("scores", StandardScaler(), SCORE_COLS))
    if feature_set in ("ratings+scenarios", "all"):
        options = [config.SCENARIO_OPTIONS[q] for q in config.SCENARIO_COLS]
        parts.append((
            "scenarios",
            OneHotEncoder(categories=options, handle_unknown="ignore", sparse_output=False),
            config.SCENARIO_COLS,
        ))

    return Pipeline([
        ("scores", FunctionTransformer(add_interest_scores)),
        ("columns", ColumnTransformer(parts)),   # columns not listed are dropped
    ])
