# 11 — Q&A sheet: questions the professor may ask

Every member should be able to answer every question here in their own words. Numbers are from the real run.
Each walkthrough (`02`–`09`) has five more questions about its own file. If you do not know an answer, say what you
do know and think it through; a wrong confident answer is worse.

## The problem and the data

**1. What problem does the project solve?** Cambodia needs more IT specialists, but students often choose a major
because of friends, family or popularity, with little guidance. We recommend an IT track (Software Development,
Data Science / AI, Cybersecurity, Networking) from a student's interests.

**2. What is new compared with earlier work?** Earlier studies predict broad majors from grades or explain STEM vs
non-STEM choices. None recommends a major *within IT* for Cambodian students from interests, and none has our dataset.

**3. Where did the data come from?** Our own Google Form survey of AUPP IT students: 14 rating questions, 4 scenario
questions, and their current or intended major (the label). We have 320 responses. The spreadsheet was re-imported
into a new sheet, so its timestamps were reset; we do not claim a collection period from it.

**4. Why 257 and not 320 students?** 63 students are Computer Science or ICT majors. They are general majors and do
not belong to one of our four tracks, so there is no correct label. Tracks: Data Science / AI 83, Networking 76,
Software Development 62, Cybersecurity 36.

**5. How did you map majors to tracks?** Software Development and Interactive App Design → Software Development;
Artificial Intelligence and Data Analytics → Data Science / AI; Cybersecurity → Cybersecurity; Digital
Infrastructure → Networking.

## Method

**6. Why interests and not grades?** Grades say what a student can do now; interests say what they would enjoy. The
IT majors share many first-year courses, so grades barely separate them.

**7. What are the engineered scores?** `build_vs_protect` = average(q7, q12) − q4; `data_score` = average(q3, q11);
`infra_score` = average(q5, q6, q13). They combine related answers to separate similar tracks.

**8. Why these models?** Logistic Regression is a simple, explainable baseline; KNN works from similar students;
Random Forest captures interactions; soft voting combines them. We compare them fairly and also report a
"most common track" baseline.

**9. What is cross-validation?** Split the 205 practice students into 5 groups; train on 4, test on the 5th, rotate,
average. It gives a stable score without touching the hidden test students.

**10. What is tuning?** Choosing a model's settings (for example how many neighbors KNN uses) by trying many and
keeping the best cross-validation score.

**11. Why macro F1 and top-2?** Macro F1 treats each track equally, so the small Cybersecurity group cannot be ignored.
Top-2 matches the app, which shows two suggestions. See `10_metrics_explained.md`.

**12. How did you avoid data leakage?** We split first, fitted the scaler inside each cross-validation round, tuned only
on the practice students, and used the 52 test students once at the end.

**13. KNN scored highest. Why did Logistic Regression win?** The differences (0.912 vs 0.910 macro F1) are much smaller
than the fold spread (±0.03–0.04), so the models are tied. Our rule then prefers the simplest model.

## Results

**14. What are the main results?** On the 52 hidden students: 86.5% accuracy (95% interval 76.9%–96.2%), 85.2% macro F1,
90.4% top-2 accuracy. The baseline: 32.7% accuracy, 12.3% macro F1. Cross-validation macro F1 on the practice students
is about 0.91 for all four models.

**15. Why is the test score lower than cross-validation?** The test set has only 52 students, so the score is noisy (one
student ≈ 2 points). The cross-validation value (90.7%) lies inside the test interval.

**16. Your proposal expected 70–80%. Why is it higher?** The scenario answers describe activities that match the majors,
so they are very informative. Ratings only give about 85% macro F1 in cross-validation. We report both.

**17. Which track is hardest?** Cybersecurity: precision 0.67, because 3 students from other tracks were labeled
Cybersecurity. It also has the fewest students (36; 7 in the test set), so the number is noisy.

**18. Do the scenario questions help?** Yes: macro F1 0.847 → 0.910 for Logistic Regression. But their wording mirrors the
major names, so part of the gain is students restating their major. That is a limitation we state.

**19. Did the engineered scores help?** Not clearly. KNN gained about 2.5 points, Logistic Regression and Random Forest did
not. The changes are within the fold spread.

**20. Which questions matter most?** q18, q15 and q17 (scenario questions), then q5 (networks), q4 (cybersecurity), q3 (data).

## Limitations and next steps

**21. What are the limitations?** One university and 257 students; the label is the student's *current or intended* major,
which may itself come from friends or trends (the very problem we want to reduce); the Cybersecurity group is small; scenario
answers partly restate the major; the data has no year of study, so we dropped the Year 1–2 vs 3–4 comparison; students already
inside IT majors may not represent students who have not chosen yet.

**22. Would it work for a student who has not started IT?** We do not know. The model learned from students already in a track.
Checking it on new students is future work.

**23. What would you do next?** More universities and students, a year-of-study question, skills and career interests, a
follow-up asking whether students were happy with the track suggested, and a Khmer interface.

## SHAP, the app and the process

**24. What is SHAP?** It shows how much each answer pushed a prediction toward a track compared with an average student. The
three strongest answers become the reasons shown to the student.

**25. What do the percentages mean?** The model's confidence from the patterns in our survey, not a guarantee. The app calls
the result a suggestion, not a decision.

**26. What happens to a student's answers in the app?** Nothing is stored. There is no database and no login.

**27. Did you use AI to write the code?** Yes, an AI assistant helped write the first version. Our professor allowed AI help as
long as we understand the code. Each member read, ran and changed their own part, and can explain it. Answer honestly if asked.

**28. Random Forest and Soft Voting have exactly the same scores. Did you really run the ensemble?** Yes. The voting team
chose the same track as Random Forest for all 205 practice students (we checked), so accuracy and macro F1 are identical. Their
top-2 scores differ (0.951 vs 0.966), which shows their probabilities are not the same. Logistic Regression and KNN also agree with
the team on 204 of 205 students: the models mostly make the same decisions.

**29. How can a constant baseline get 62% top-2 accuracy?** The baseline always suggests the two most common tracks (Data
Science / AI and Networking). Those two tracks hold about 62% of the students, so it is right 62% of the time in top-2. That is
why top-2 accuracy must be compared with this baseline, not with zero. Our models reach 0.95–0.97.

**30. Who did what?** Sonika: `config.py`, step 1 (data). Thailong: step 2 (features), step 4 (tuning), step 8 (predictions and
SHAP reasons). Sothon: step 3 (models), step 5 (training), step 6 (evaluation), the web app. Rachana: step 7 (analysis and figures).
Edit this to match what each person really did.
