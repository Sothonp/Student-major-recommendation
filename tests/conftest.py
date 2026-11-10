"""Shared test fixtures.

Everything here is FAKE data made for tests only. It is never used for results.
Fixtures are added task by task; each one is built from the real pipeline functions.
"""
import numpy as np
import pandas as pd
import pytest


def make_fake_responses(n_per_track=40, n_aux=30, seed=0):
    """Fake survey rows with columns q1..q19 and a planted signal.

    - Each track has its own typical ratings (for example Cybersecurity rows rate q4 high).
    - Scenario answers match the track 70% of the time.
    - n_aux rows have a CS or ICT major (these get set aside by map_to_tracks).
    """
    from pipeline import config

    rng = np.random.default_rng(seed)
    majors = {
        "Software Development": ["Software Development", "Interactive App Design and Development"],
        "Data Science / AI": ["Artificial Intelligence (AI)", "Data Analytics"],
        "Cybersecurity": ["Cybersecurity"],
        "Networking": ["Digital Infrastructure"],
    }
    # typical rating (out of 5) for the questions that matter for each track
    typical = {
        "Software Development": {"q1": 4, "q7": 4, "q12": 4, "q8": 3.5},
        "Data Science / AI": {"q2": 4, "q3": 4, "q11": 4},
        "Cybersecurity": {"q4": 4.5, "q5": 3.5, "q6": 3.5, "q9": 4.5, "q10": 4.5},
        "Networking": {"q5": 4.5, "q6": 4.5, "q13": 4.5, "q10": 4.5},
    }

    def ratings(profile):
        row = {}
        for q in config.RATING_COLS:
            mean = profile.get(q, 2.5)
            row[q] = int(np.clip(round(rng.normal(mean, 0.8)), 1, 5))
        return row

    def scenarios(track_index):
        row = {}
        for q in config.SCENARIO_COLS:
            options = config.SCENARIO_OPTIONS[q]
            if rng.random() < 0.7:
                row[q] = options[track_index % len(options)]
            else:
                row[q] = options[int(rng.integers(len(options)))]
        return row

    rows = []
    for i, track in enumerate(config.TRACKS):
        for _ in range(n_per_track):
            row = ratings(typical[track])
            row.update(scenarios(i))
            row["q19"] = str(rng.choice(majors[track]))
            rows.append(row)
    for _ in range(n_aux):
        row = ratings({})
        row.update(scenarios(int(rng.integers(4))))
        row["q19"] = str(rng.choice(config.EXCLUDED_MAJORS))
        rows.append(row)

    df = pd.DataFrame(rows)[config.QUESTION_COLS + ["q19"]]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture
def fake_raw():
    """190 fake rows: 4 tracks x 40 plus 30 CS/ICT rows."""
    return make_fake_responses()


@pytest.fixture
def fake_labeled(fake_raw):
    from pipeline.step1_preprocessing import map_to_tracks

    return map_to_tracks(fake_raw)


@pytest.fixture
def fake_split(fake_labeled):
    from pipeline.step1_preprocessing import split_train_test

    return split_train_test(fake_labeled)


@pytest.fixture
def fake_xlsx_path(tmp_path, fake_raw):
    """A fake survey spreadsheet in the same shape as the real Google Form export."""
    sheet = fake_raw.copy()
    sheet.columns = [f"Question {i} with a long header" for i in range(1, 20)]
    sheet.insert(0, "Timestamp", 46336.6)
    path = tmp_path / "Student Major Recommendation (Responses).xlsx"
    sheet.to_excel(path, index=False)
    return path
