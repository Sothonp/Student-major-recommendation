# 02 — `config.py` and `step1_preprocessing.py`

**Owner: Sonika** (Sothon co-owns `split_train_test`).

## What this file does

`step1_preprocessing.py` takes the raw Google Form spreadsheet and turns it into clean,
labeled students that every later step can use. `config.py` holds the fixed values (track
names, question wording, answer options, settings) so nothing is typed twice.

Numbers on the real data: 320 rows read → 320 kept after cleaning → **257** after mapping to
tracks → **205** practice students + **52** hidden test students.

## Function by function

- **`load_responses(path)`** reads the spreadsheet. The first column is the Timestamp. All 320
  timestamps are within 4.5 seconds because the sheet was re-imported, so they carry no
  information and are dropped. The other 19 columns are renamed `q1` to `q19` by position,
  because the real headers are long sentences.
- **`clean_responses(df)`** keeps only valid rows: text has no stray spaces, ratings must be
  the whole numbers 1 to 5 (Excel stores them as `4.0`, which is fine; `4.5` or `6` is not), and
  any row with a blank answer is dropped. (On the real data nothing was dropped.)
- **`map_to_tracks(df)`** adds a `track` column using the table in `config.py`. Computer Science
  and ICT students are removed, because they do not belong to one of the four tracks (63 students).
- **`split_train_test(df)`** hides 20% of the students (52) for the final exam. "Stratified"
  means each track keeps the same share in both groups. The seed 42 makes the split identical
  every time.

## Five likely questions

1. **Why did you drop the timestamp?** After the sheet was re-imported, every timestamp became
   the same few seconds, so it says nothing about when students answered. We do not use it as
   a feature and we do not claim a collection period.
2. **Why were CS and ICT students set aside?** They are general majors, not one of the four
   tracks, so there is no correct label to learn. That removed 63 students and left 257.
3. **What does "stratified" mean and why use it?** Cybersecurity is the smallest track (36
   students). A random split could put almost none of them in the test set. Stratified keeps the
   proportions equal (7 Cybersecurity students in the test set).
4. **Why a fixed seed?** So anyone running the code gets the same split and the same numbers.
5. **Why split before doing anything else?** The 52 test students must stay unseen. If
   they influenced cleaning, scaling or tuning, the final score would look better than it is.
   That mistake is called data leakage.

## Your change (already made for you by Sothon: read it, run its tests, then explain it)

`clean_responses` now accepts `verbose=True`, and then it prints how many rows each rule removed. `step5_train.py`
turns it on, so every training run starts with this line:

```
Cleaning: 320 rows read, 0 dropped for missing answers, 0 dropped for ratings that are not whole numbers from 1 to 5, 320 kept.
```

**How it works.** The function counts the rows (`len(out)`) at three moments: when it starts, after it drops rows with a
blank answer, and after it drops rows whose ratings are not whole numbers from 1 to 5. The difference between two moments is
how many rows that rule removed. It is silent by default, because the app and the tests also call this function and should
not print.

**Tests:** `test_clean_responses_reports_what_it_removed` (a table with 2 blank answers and 2 bad ratings must report
exactly that) and `test_clean_responses_is_quiet_by_default`, both in `tests/test_step1.py`. Run them with
`python -m pytest tests/test_step1.py`.

**Be ready to say:** "0 dropped" means every response was complete and valid, so all 320 reached the next step; the 63
students who are lost later are CS and ICT students set aside by `map_to_tracks`, not bad data.

**Your next step, if you want one of your own:** add a count of how many students each major has before mapping, and print it
in the same style. Commit it under your own name.
