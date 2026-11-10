import numpy as np
import pandas as pd

from pipeline import config
from pipeline.step2_features import SCORE_COLS
from pipeline.step7_analysis import analyze_feature_signal, plot_results

FIGURES = [
    "class_balance.png",
    "model_comparison.png",
    "confusion_matrix.png",
    "correlation_heatmap.png",
    "feature_importance.png",
    "ablation.png",
    "per_track_scores.png",
]


def test_correlation_shape_and_range(fake_split):
    X_train, y_train = fake_split[0], fake_split[2]

    corr = analyze_feature_signal(X_train, y_train)["correlation"]

    assert corr.shape == (17, 4)
    assert list(corr.index) == config.RATING_COLS + SCORE_COLS
    assert list(corr.columns) == config.TRACKS
    assert corr.to_numpy().min() >= -1 and corr.to_numpy().max() <= 1


def test_correlation_finds_the_planted_signal(fake_split):
    X_train, y_train = fake_split[0], fake_split[2]
    corr = analyze_feature_signal(X_train, y_train)["correlation"]
    assert corr.loc["q4", "Cybersecurity"] > 0.3        # Cybersecurity rows rate q4 high
    assert corr.loc["infra_score", "Networking"] > 0.3


def test_importance_sums_to_one_and_is_sorted(fake_split):
    X_train, y_train = fake_split[0], fake_split[2]

    importance = analyze_feature_signal(X_train, y_train)["importance"]

    assert sorted(importance.index) == sorted(config.QUESTION_COLS)    # one row per question
    assert abs(importance["importance"].sum() - 1) < 1e-6
    assert importance["importance"].is_monotonic_decreasing


def _write_fake_results(folder):
    models = ["Baseline", "Logistic Regression", "KNN", "Random Forest", "Soft Voting"]
    pd.DataFrame({"track": config.TRACKS, "students": [36, 83, 76, 62]}).to_csv(
        folder / "class_balance.csv", index=False)
    cv = pd.DataFrame({"accuracy_mean": 0.8, "macro_f1_mean": [0.2, 0.8, 0.7, 0.8, 0.8],
                       "macro_f1_std": 0.03, "top2_mean": 0.9}, index=models)
    cv.index.name = "model"
    cv.to_csv(folder / "cv_results.csv")
    pd.DataFrame(np.eye(4, dtype=int) * 10, index=config.TRACKS, columns=config.TRACKS).to_csv(
        folder / "confusion_matrix.csv")
    features = config.RATING_COLS + SCORE_COLS
    pd.DataFrame(np.linspace(-0.5, 0.5, 17 * 4).reshape(17, 4), index=features, columns=config.TRACKS).to_csv(
        folder / "correlation.csv")
    pd.DataFrame({"importance": np.full(18, 1 / 18)}, index=config.QUESTION_COLS).to_csv(
        folder / "feature_importance.csv")
    rows = [{"model": m, "feature_set": fs, "accuracy_mean": 0.8, "macro_f1_mean": 0.7, "macro_f1_std": 0.03}
            for m in ["Logistic Regression", "KNN", "Random Forest"] for fs in config.FEATURE_SETS]
    pd.DataFrame(rows).to_csv(folder / "ablation.csv", index=False)
    pd.DataFrame({"precision": [0.67, 1.0, 0.86, 0.87], "recall": [0.86, 0.82, 0.8, 1.0],
                  "f1": [0.75, 0.9, 0.83, 0.93], "support": [7, 17, 15, 13]}, index=config.TRACKS).to_csv(
        folder / "per_track.csv")


def test_plot_results_creates_seven_png_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_fake_results(tmp_path)

    paths = plot_results()

    assert [p.name for p in paths] == FIGURES
    for path in paths:
        assert path.parent == tmp_path / "figures"
        assert path.read_bytes()[:4] == b"\x89PNG"     # a real, non-empty PNG file
