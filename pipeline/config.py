"""Constants shared by every step. No functions here, only values.

Owner: Sonika. If you change a value here, every step that reads it changes too.
"""
from pathlib import Path

# ---- run settings -----------------------------------------------------------
SEED = 42            # every random step uses this, so results repeat exactly
TEST_SIZE = 0.2      # 20% of students are hidden for the final exam
N_FOLDS = 5          # 5-fold cross-validation
TOP_K = 2            # the app shows 2 suggestions, so we also score "top 2"

# ---- paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Student Major Recommendation (Responses).xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- tracks and majors ------------------------------------------------------
# sorted, which is the same order scikit-learn uses for its predicted classes
TRACKS = ["Cybersecurity", "Data Science / AI", "Networking", "Software Development"]

MAJOR_TO_TRACK = {
    "Software Development": "Software Development",
    "Interactive App Design and Development": "Software Development",
    "Artificial Intelligence (AI)": "Data Science / AI",
    "Data Analytics": "Data Science / AI",
    "Cybersecurity": "Cybersecurity",
    "Digital Infrastructure": "Networking",
}
# these two majors do not belong to one track, so they are set aside
EXCLUDED_MAJORS = [
    "Computer Science (CS)",
    "Information and Communication Technology (ICT)",
]

# ---- survey columns ---------------------------------------------------------
RATING_COLS = [f"q{i}" for i in range(1, 15)]       # q1-q14: ratings from 1 to 5
SCENARIO_COLS = ["q15", "q16", "q17", "q18"]         # q15-q18: multiple choice
QUESTION_COLS = RATING_COLS + SCENARIO_COLS          # the 18 questions a student answers
LABEL_COL = "q19"                                    # current or intended major

# short wording of each rating question (used in the app's reasons)
RATING_LABELS = {
    "q1": "writing computer programs and solving problems using code",
    "q2": "artificial intelligence, machine learning and intelligent systems",
    "q3": "analyzing data, statistics and finding patterns in datasets",
    "q4": "cybersecurity, ethical hacking and protecting computer systems",
    "q5": "computer networks, network communication and connectivity",
    "q6": "servers, cloud computing and IT infrastructure",
    "q7": "designing, developing, testing and improving software systems",
    "q8": "designing user interfaces, user experiences and interactive applications",
    "q9": "solving logical problems and figuring out how things work",
    "q10": "investigating technical problems and finding their causes",
    "q11": "working with numbers, datasets and discovering useful patterns",
    "q12": "building and improving software applications",
    "q13": "managing computer systems, networks, servers or infrastructure",
    "q14": "combining technology and creativity to create digital products",
}

# full wording of the 4 scenario questions, as in the Google Form
SCENARIO_QUESTIONS = {
    "q15": "You have one semester to build a technology project. Which would you choose?",
    "q16": "You are given a large dataset containing information about students. "
           "What would you most like to do?",
    "q17": "A company's network suddenly stops working. Which task would interest you the most?",
    "q18": "You join a technology team. Which responsibility would you most like to have?",
}

# short phrase for each scenario question, used inside a reason sentence
SCENARIO_LABELS = {
    "q15": "the semester project question",
    "q16": "the dataset question",
    "q17": "the network failure question",
    "q18": "the team responsibility question",
}

# every allowed answer, exactly as worded in the Form (8 + 5 + 5 + 8 = 26 options)
SCENARIO_OPTIONS = {
    "q15": [
        "Build and manage cloud/network infrastructure",
        "Analyze data and create a dashboard",
        "Build an AI model that recognizes objects in images",
        "Build a system that detects cyberattacks",
        "Develop a software application",
        "Build a large-scale software platform",
        "Design an interactive mobile application",
        "Develop an organization's information system",
    ],
    "q16": [
        "Analyze the data and discover trends",
        "Protect the data from security threats",
        "Develop an application that uses the data",
        "Build a machine-learning model to make predictions",
        "Design an interface for users to explore the data",
    ],
    "q17": [
        "Analyze system data and logs to identify the cause",
        "Develop software to monitor and diagnose the problem",
        "Manage the servers and cloud infrastructure",
        "Determine whether it was caused by a cyberattack",
        "Investigate the network infrastructure",
    ],
    "q18": [
        "Develop software applications and systems",
        "Build and maintain networks, servers, and infrastructure",
        "Analyze data and produce useful insights",
        "Protect systems against cyber threats",
        "Develop information systems for organizations",
        "Design interactive applications and user experiences",
        "Develop AI and machine-learning solutions",
        "Work across different areas of information and communication technology",
    ],
}

# ---- modelling --------------------------------------------------------------
# which feature groups a model may use (the ablation compares these)
FEATURE_SETS = ("ratings", "ratings+scores", "ratings+scenarios", "all")

# settings the computer tries for each model ("clf__" = the classifier step of the pipeline)
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
}

# simplest first: used to break ties when models score about the same
MODEL_SIMPLICITY = ["Logistic Regression", "KNN", "Random Forest", "Soft Voting"]
