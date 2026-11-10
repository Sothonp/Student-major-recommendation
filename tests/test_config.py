from sklearn.model_selection import ParameterGrid

from pipeline import config


def test_tracks_are_four_and_sorted():
    assert len(config.TRACKS) == 4
    assert config.TRACKS == sorted(config.TRACKS)


def test_question_columns():
    assert config.RATING_COLS == [f"q{i}" for i in range(1, 15)]
    assert config.SCENARIO_COLS == ["q15", "q16", "q17", "q18"]
    assert config.QUESTION_COLS == config.RATING_COLS + config.SCENARIO_COLS
    assert len(config.QUESTION_COLS) == 18
    assert config.LABEL_COL == "q19"


def test_scenario_options_counts_and_labels():
    sizes = [len(config.SCENARIO_OPTIONS[q]) for q in config.SCENARIO_COLS]
    assert sizes == [8, 5, 5, 8]
    assert sum(sizes) == 26
    assert set(config.SCENARIO_QUESTIONS) == set(config.SCENARIO_COLS)
    assert set(config.SCENARIO_LABELS) == set(config.SCENARIO_COLS)


def test_major_to_track_matches_spec():
    assert config.MAJOR_TO_TRACK == {
        "Software Development": "Software Development",
        "Interactive App Design and Development": "Software Development",
        "Artificial Intelligence (AI)": "Data Science / AI",
        "Data Analytics": "Data Science / AI",
        "Cybersecurity": "Cybersecurity",
        "Digital Infrastructure": "Networking",
    }
    assert set(config.MAJOR_TO_TRACK.values()) == set(config.TRACKS)
    assert config.EXCLUDED_MAJORS == [
        "Computer Science (CS)",
        "Information and Communication Technology (ICT)",
    ]


def test_rating_labels_cover_q1_to_q14():
    assert set(config.RATING_LABELS) == set(config.RATING_COLS)


def test_run_settings():
    assert (config.SEED, config.TEST_SIZE, config.N_FOLDS, config.TOP_K) == (42, 0.2, 5, 2)
    assert config.FEATURE_SETS == ("ratings", "ratings+scores", "ratings+scenarios", "all")
    assert config.MODEL_SIMPLICITY == ["Logistic Regression", "KNN", "Random Forest", "Soft Voting"]
    assert config.DATA_PATH.name == "Student Major Recommendation (Responses).xlsx"


def test_param_grid_sizes():
    sizes = {m: len(ParameterGrid(g)) for m, g in config.PARAM_GRIDS.items()}
    assert sizes == {"Logistic Regression": 4, "KNN": 12, "Random Forest": 18}
    for grid in config.PARAM_GRIDS.values():
        assert all(key.startswith("clf__") for key in grid)
