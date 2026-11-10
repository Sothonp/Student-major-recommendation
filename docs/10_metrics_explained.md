# 10 — Metrics, tuning and cross-validation explained

## Accuracy

The percentage of students the model got right. Simple, but it hides problems with small groups.

## Macro F1 (the fair score)

For **each track** we compute two counts:
- **Precision:** of the students the model *labeled* this track, how many really are?
- **Recall:** of the students who *really are* this track, how many did the model find?

**F1** mixes the two into one grade for that track. It is not a plain average: it is pulled toward the lower
of the two. **Macro F1** adds the 4 tracks' grades and divides by 4, so a small track counts as much as a big one.

**Why not only accuracy?** A class of 10 students, 9 Networking and 1 Cybersecurity. A lazy model that always says
"Networking" gets 9 of 10 right: **90% accuracy**, but it never finds the Cybersecurity student. Networking's
grade is about 95%, Cybersecurity's is 0%, so macro F1 = (95 + 0) ÷ 2 ≈ **47%**. Accuracy gives every *student* a
vote, so the big group wins; macro F1 gives every *track* a vote.

**A scoreboard (20 students: 8 Data/AI, 6 Networking, 4 Software, 2 Cybersecurity; made-up numbers):**

| Model | Accuracy | Data/AI | Networking | Software | Cyber | Macro F1 |
|---|---|---|---|---|---|---|
| X: always "Data/AI" | 40% | 57 | 0 | 0 | 0 | 14% |
| Y: ignores Cyber | **90%** | 100 | 86 | 100 | **0** | 71% |
| Z: balanced | 80% | 88 | 83 | 75 | 50 | **74%** |

By accuracy Y wins; by macro F1 Z wins. For an app that should help Cybersecurity students, Z is the model you want.

For the presentation: *"Macro F1 gives every track the same weight, so a model cannot hide poor results on a small
track like Cybersecurity."*

## Top-2 accuracy

Right if the student's real track is **either** of the model's two best guesses. The app shows 2 suggestions, so this
is the score for what the student sees.

| Real track | 1st guess | 2nd guess | Right on 1st? | Right in top 2? |
|---|---|---|---|---|
| Networking | Networking 70% | Cybersecurity 20% | ✅ | ✅ |
| Cybersecurity | Networking 55% | Cybersecurity 30% | ❌ | ✅ |
| Software | Software 60% | Data/AI 25% | ✅ | ✅ |
| Data/AI | Software 45% | Data/AI 35% | ❌ | ✅ |
| Cybersecurity | Networking 50% | Software 30% | ❌ | ❌ |

Accuracy = 2/5 = 40%; top-2 accuracy = 4/5 = 80%. Top-2 is always at least as high as accuracy.

## Tuning (finding the best settings)

Models have **settings** (like oven temperature). They are not learned from the data; we choose them. Example, KNN:
K = 1 copies one student (too jumpy); K = 100 lets the big groups always win; the best K is in between. **Grid search**
tries every candidate K, scores each with cross-validation, and keeps the best. Only the 205 practice students are used.

## 5-fold cross-validation

Split the 205 practice students into 5 groups. Train on 4, test on the 5th, rotate until each group has been the test
group, then average the 5 scores. It uses all the practice students for both training and testing without ever testing
on a student the model trained on. The spread (standard deviation) of the 5 scores shows how stable the model is.

## Data leakage

When information from the test students leaks into training, the score looks better than the truth. We prevent it by
splitting first, fitting the scaler only on training students, tuning only on practice students, and using the 52
hidden students once, at the end.
