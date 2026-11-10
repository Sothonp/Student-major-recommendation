# English Students' Major Recommendation — Final Report (draft)

**Course:** ITM-360 Machine Learning, American University of Phnom Penh, School of Digital Technologies
**Advisor:** Dr. Kuntha Pin
**Team (Group 7):** Hor Thailong (2025184), Yeong Vechakasy Sothon (2025341), Ro Sokharachana (2025284), Bunroeung Sonika (2025667)

> **Draft notes for the team (delete before submitting).** (1) The professor's Word template
> (`ITM-390-Machine-Learning-Proposal-Format-1.docx`) was not available when this draft was written; the sections below follow
> your proposal. Copy the text into the template and delete its instructions and example images. (2) Every number in
> Sections 4–6 comes from `results/*.csv`; if you re-run the pipeline, re-check them. (3) Check the numbers quoted from
> reference [5] against the paper itself. (4) Edit Section 8 so it matches what each member really did.

---

## 1. Introduction

This project builds a machine-learning model that recommends an IT track to university students: Software Development,
Data Science / AI, Cybersecurity or Networking. The recommendation is based on a survey of students' interests and
preferences, not on their grades.

Cambodia needs more IT specialists. The *Roadmap for Digital Skills Development in Cambodia 2024–2035* reports a shortage
of software developers and of data and network technicians, and says digital recruitment must grow by about 30% per year
[1]. A CDRI study found that ICT graduates earn about 10% more than similar non-ICT graduates [2]. Yet Cambodian higher
education faces a skills mismatch, and one recognized cause is weak career guidance [3]. Without guidance, students often
choose majors for practicality, family influence and job prospects [4]. Choosing *within* IT is harder still, because the
options share many foundation courses.

Machine learning has been used to help students choose a field, for example by predicting an engineering major from work
preferences [5], from course records [6], or STEM major choice from national survey data [7]. These studies predict broad
majors, and most depend on academic records. In Cambodia, research on major choice has explained STEM versus non-STEM
decisions [8], but no study has built a model that recommends a major within IT.

We address this gap with an original survey of AUPP IT students, a comparison of four classifiers, and a web app that returns
a student's two most likely tracks with the reasons behind them.

## 2. Goals and objectives

**Main goal.** Develop a model that predicts the most suitable IT track for AUPP students from their interests, work
preferences and scenario choices, and serve it in a web app that gives a data-based suggestion before they choose.

| Objective (from the proposal) | Status |
|---|---|
| a. Collect survey data on interests, preferences, scenario choices and major | Done: 320 responses, 257 usable |
| b. Find which questions best separate the tracks (correlation, Random Forest importance) | Done: Section 5.4 |
| c. Build combined features (build-vs-protect, data, infrastructure scores) | Done; their benefit is tested in Section 5.3 |
| d. Train and compare Logistic Regression, KNN, Random Forest and a soft-voting ensemble with stratified 5-fold cross-validation, accuracy, macro F1 and top-2 accuracy | Done: Sections 5.1–5.2 |
| e. Build the "Recommendation" web app with top-2 tracks and reasons | Done: Section 5.5 |

One planned analysis was dropped: comparing accuracy for Year 1–2 and Year 3–4 students. The survey did not ask for year of
study, so the comparison is not possible with this data (see Section 7).

## 3. Literature review

**3.1 Cambodia's need for IT specialists.** The Digital Economy and Society Policy Framework 2021–2035 lists digital skills,
data governance and cybersecurity among its focus areas [9]. The 2024–2035 roadmap reports that only about 10% of Cambodia's
roughly 200,000 higher-education students study digital fields and that guiding students in choosing their study and career
direction is one of its purposes [1]. A CDRI study with CADT reports that ICT enrolment remains low [2].

**3.2 How Cambodian students choose majors.** Eam et al. [8] surveyed 2,016 first-year students at 15 universities and found that
students who chose STEM were more likely to want a STEM career, to have high self-efficacy and to have planned their major since
high school. Kao and Shimizu [10] found similar factors at high-school level. Interviews with 31 students in Phnom Penh showed that
choices follow practicality, desirability and convenience [4]. Heng and Sol [3] identified weak career guidance as one cause of the
skills mismatch. These studies explain group patterns; none gives a recommendation to an individual student or looks inside IT.

**3.3 Machine learning for major recommendation.** Beaulac and Rosenthal [6] predicted majors from course records in Canada and
Chang et al. [7] predicted STEM major choice from U.S. survey data; both rely mainly on academic records. The closest study is Jubair
et al. [5], who surveyed 734 engineering students in Jordan about their work preferences and predicted their major with 91.3% accuracy
for two majors and 77.7% for five. It shows that interest-based surveys can predict a student's field without grades.

**3.4 Research gap.** No study has built a model to recommend an IT major for Cambodian students from their interests. This project
addresses that gap.

## 4. Methodology

### 4.1 Dataset

We built an original Google Form survey for AUPP IT students. It has 14 rating questions (1–5; interest in eight technology areas and
enjoyment of six activities), 4 scenario questions (for example "A company's network suddenly stops working. Which task would interest
you the most?") and the student's current or intended major, which is the label to predict.

The export has 320 responses. Its timestamps do not reflect the original submission times (the responses were copied into a new sheet),
so we do not use them and make no claim about the collection period. Cleaning kept all 320 rows (every rating was a whole number
from 1 to 5 and no answer was blank). Majors were mapped to four tracks; Computer Science (CS) and Information and Communication Technology
(ICT) students were set aside because they do not belong to a single track.

| Track | Survey majors | Students |
|---|---|---|
| Data Science / AI | Artificial Intelligence, Data Analytics | 83 |
| Networking | Digital Infrastructure | 76 |
| Software Development | Software Development, Interactive App Design and Development | 62 |
| Cybersecurity | Cybersecurity | 36 |
| (set aside) | Computer Science, ICT | 63 |

That leaves **257 usable students**. A stratified 80/20 split (seed 42) gives **205 training** and **52 hidden test** students; the test set
holds 17 Data Science / AI, 15 Networking, 13 Software Development and 7 Cybersecurity students.

### 4.2 Features

The 14 ratings are standardized. The four scenario questions are one-hot encoded (26 columns). Three engineered scores combine related
answers: **build-vs-protect** = mean(q7, q12) − q4; **data score** = mean(q3, q11); **infrastructure score** = mean(q5, q6, q13). This gives
43 features. The scaler is part of the model pipeline, so it is fitted only on training data in every cross-validation round (no leakage).

### 4.3 Models and tuning

We compared a majority-class baseline, Logistic Regression, K-Nearest Neighbors, Random Forest and a soft-voting ensemble of the three tuned
models. Logistic Regression and Random Forest use balanced class weights; KNN has no class-weight option. Settings were tuned by grid search
with stratified 5-fold cross-validation on the 205 training students, scored by macro F1.

| Model | Settings searched | Best found |
|---|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10} | C = 0.1 |
| KNN | neighbors ∈ {3, 5, 7, 9, 11, 15}; weights ∈ {uniform, distance} | 5 neighbors, uniform |
| Random Forest | trees ∈ {100, 300}; depth ∈ {none, 5, 10}; min leaf ∈ {1, 2, 4} | 100 trees, no depth limit, min leaf 1 |

### 4.4 Evaluation

Models are compared by stratified 5-fold cross-validation on the training students, reporting accuracy, **macro F1** (each track weighted
equally, important because Cybersecurity is small) and **top-2 accuracy** (the real track is one of the two suggested tracks, matching what the
app shows). **Selection rule:** choose the highest mean macro F1; any model within one standard deviation of the best counts as tied, and among tied
models we choose the simplest (Logistic Regression < KNN < Random Forest < Soft Voting). The test set is used once, for the winner. Its accuracy
interval is a 95% bootstrap interval (1000 resamples). An ablation re-runs cross-validation with groups of questions removed.

### 4.5 Explanations and web app

For each suggested track the app shows the three answers that pushed most toward it, computed with SHAP (a permutation explainer over the 18
questions). The app is a Flask web page; it stores no answers.

### 4.6 Implementation and tools

Python with pandas, scikit-learn, SHAP, matplotlib and Flask. An AI coding assistant helped write the first version of the code; the team
reviewed, ran and modified it, and automatic tests (78 checks) cover each step.

## 5. Results

### 5.1 Model comparison (5-fold cross-validation, 205 training students)

| Model | Accuracy | Macro F1 | Top-2 |
|---|---|---|---|
| Baseline (most common track) | 0.322 ± 0.011 | 0.122 ± 0.003 | 0.620 ± 0.013 |
| Logistic Regression | 0.907 ± 0.032 | 0.910 ± 0.028 | 0.966 ± 0.013 |
| KNN | 0.907 ± 0.044 | 0.912 ± 0.038 | 0.966 ± 0.037 |
| Random Forest | 0.902 ± 0.039 | 0.905 ± 0.034 | 0.951 ± 0.030 |
| Soft Voting | 0.902 ± 0.039 | 0.905 ± 0.034 | 0.966 ± 0.013 |

The baseline always suggests the most common track; for top-2 it suggests the two most common tracks (Data Science / AI and Networking). All four models are far above the baseline and are tied (their differences are smaller than one standard deviation of the fold scores). Random Forest and Soft Voting have identical accuracy and macro F1 because the voting team chose the same track as Random Forest for all 205 training students; their top-2 scores differ. Applying the
selection rule, the cut-off is 0.912 − 0.038 = 0.874, every model passes it, and the simplest, **Logistic Regression**, is chosen.
(Figure: `results/figures/model_comparison.png`.)

### 5.2 Final exam (52 hidden students)

| | Accuracy | Macro F1 | Top-2 |
|---|---|---|---|
| Logistic Regression | 0.865 (95% interval 0.769–0.962) | 0.852 | 0.904 |
| Baseline | 0.327 (95% interval 0.212–0.462) | 0.123 | 0.615 |

| Track | Precision | Recall | Students |
|---|---|---|---|
| Cybersecurity | 0.667 | 0.857 | 7 |
| Data Science / AI | 1.000 | 0.824 | 17 |
| Networking | 0.857 | 0.800 | 15 |
| Software Development | 0.867 | 1.000 | 13 |

The confusion matrix (Figure `confusion_matrix.png`; per-track scores in `per_track_scores.png`) shows that Software Development students were never missed; the main errors are two
Networking students and one Data Science / AI student labeled Cybersecurity. With only 52 test students, one student is about two points of
accuracy, which is why the interval is wide; the cross-validation accuracy (0.907) lies inside it.

### 5.3 Ablation: which questions help (macro F1, 5-fold cross-validation)

| Model | Ratings only | + scores | + scenarios | All |
|---|---|---|---|---|
| Logistic Regression | 0.847 | 0.841 | 0.910 | 0.910 |
| KNN | 0.807 | 0.832 | 0.902 | 0.912 |
| Random Forest | 0.823 | 0.824 | 0.900 | 0.905 |

Scenario questions add about six points. The engineered scores gave no consistent benefit: KNN gained 0.025, the other two models did not, and
all changes are within the fold spread (±0.03 to ±0.09). (Figure `ablation.png`.)

### 5.4 Which questions separate the tracks

Correlations (training students) match the design of the scores: the data score correlates 0.729 with Data Science / AI, the infrastructure
score 0.629 with Networking, and build-vs-protect 0.567 with Software Development and −0.471 with Cybersecurity. Individually, q3 (data and
statistics) correlates 0.697 with Data Science / AI and q4 (cybersecurity) 0.562 with Cybersecurity. Random Forest importance is highest for
the scenario questions q18 (0.171), q15 (0.140) and q17 (0.103), followed by q5 (0.077), q4 (0.071) and q3 (0.060).
(Figures `correlation_heatmap.png`, `feature_importance.png`, `class_balance.png`.)

### 5.5 The web app

A student answers the 18 questions and receives two tracks with percentages and three reasons each. For example, a student who rated servers,
cloud computing and infrastructure 5 out of 5 and chose "Build and manage cloud/network infrastructure" for the project question received
Networking (88%) and Data Science / AI (8%), with those two answers among the reasons. [Insert a screenshot of the result page.]

## 6. Discussion

The four models perform alike, which suggests the interest survey carries the signal and the choice of classifier matters little at this data
size. We chose the simplest model because it is easiest to explain and the evidence does not support a more complex one.

Our accuracy is higher than the 70–80% we expected in the proposal, but it must be read with care. The scenario questions describe activities that
mirror the majors, so some students are, in effect, restating their major. The ratings-only result (macro F1 about 0.85 in cross-validation) is the
stricter figure. Direct comparison with Jubair et al. [5], who report 77.7% for five majors, is not meaningful because the data, the number of
classes and the setting differ.

The SHAP reasons show which answers the model relied on; they are not causes. A low rating can appear as a reason (a low software interest points
toward Networking); the app words such reasons as "only 2 out of 5" so they read as reasons and not as contradictions.

## 7. Limitations

- One university and 257 students; the Cybersecurity group has only 36 students (7 in the test set), so its scores are noisy.
- The label is the student's *current or intended* major, which may itself reflect friends or trends. The model learns from students who have already chosen.
- Scenario answers partly restate the major (Section 6).
- The survey has no year-of-study question, so the planned Year 1–2 vs Year 3–4 comparison was not possible.
- The model has not been tested on students who have not yet chosen a track, so we do not know how well it guides them.

## 8. Conclusion and future work

We built and compared four classifiers on an original survey of AUPP IT students. All reach about 0.91 macro F1 in cross-validation (baseline 0.12);
the chosen Logistic Regression model scores 86.5% accuracy, 85.2% macro F1 and 90.4% top-2 accuracy on 52 unseen students, and powers a web app that
explains each suggestion. Future work: more students and universities, a year-of-study question, skills and career interests, checking recommendations
against students' later satisfaction, and a Khmer interface.

## 9. Students' roles and experience

| Student | Role in project | Experience |
|---|---|---|
| Hor Thailong | Features (step 2), tuning (step 4), predictions and SHAP reasons (step 8) | Python, NumPy, Pandas, Matplotlib, scikit-learn |
| Yeong Vechakasy Sothon | Models, training, evaluation, web app (steps 3, 5, 6, app) | Python, Flask, HTML, CSS, JavaScript, Git/GitHub, React |
| Bunroeung Sonika | Survey design and collection, configuration and data preprocessing (step 1) | Python, Excel, Microsoft Word |
| Ro Sokharachana | Feature analysis, figures and documentation (step 7) | Python, Pandas, Matplotlib, Excel, Microsoft Word |

*(Edit to match what each member really did.)*

## 10. References

[1] Ministry of Posts and Telecommunications and Cambodia Academy of Digital Technology, *Roadmap for Digital Skills Development in Cambodia 2024–2035*. Phnom Penh, Cambodia, 2024.

[2] V. Roth, S. Song, P. Chea, S. Ven, S. Seng, and S. Hean, *Demand for and Supply of Digital Skills in Cambodia*, Research Report. Phnom Penh, Cambodia: Cambodia Development Resource Institute, Nov. 2021.

[3] K. Heng and K. Sol, "Skills mismatches in Cambodian higher education: Major causes and possible solutions," *Cambodian Journal of Education Research*, vol. 3, no. 1, pp. 76–91, 2023.

[4] C. Peou, "On Cambodian higher education and skills mismatch: Young people choosing university majors in a context of risk and uncertainty," *Journal of Education and Work*, vol. 30, no. 1, pp. 26–38, 2017.

[5] F. Jubair, T. Ali, K. A. Darabkh, and I. F. Jafar, "An innovative machine learning mechanism for predicting the best fit engineering major for Jordanian undergraduate students," *SN Computer Science*, vol. 6, no. 7, Art. no. 900, 2025, doi: 10.1007/s42979-025-04464-6.

[6] C. Beaulac and J. S. Rosenthal, "Predicting university students' academic success and major using random forests," *Research in Higher Education*, vol. 60, pp. 1048–1064, 2019, doi: 10.1007/s11162-019-09546-y.

[7] C.-N. Chang, S. Lin, O.-M. Kwok, and G. K. Saw, "Predicting STEM major choice: A machine learning classification and regression tree approach," *Journal of STEM Education Research*, vol. 6, no. 2, pp. 358–374, 2023, doi: 10.1007/s41979-023-00099-5.

[8] P. Eam, B. Keo, P. Leng, S. Song, and S. Khieng, "Correlates of STEM major choice: A quantitative look at Cambodian university freshmen," *Research in Science & Technological Education*, vol. 39, no. 2, pp. 206–224, 2021, doi: 10.1080/02635143.2019.1682987.

[9] Royal Government of Cambodia, *Cambodia Digital Economy and Society Policy Framework 2021–2035*. Phnom Penh, Cambodia, 2021.

[10] S. Kao and K. Shimizu, "Factors affecting Cambodian upper secondary school students' choice of science track," *International Journal of Sociology of Education*, vol. 9, no. 3, pp. 262–292, 2020, doi: 10.17583/rise.2020.4823.
