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

## Your one change

Make `clean_responses` print how many rows each rule removed (blank answers, out-of-range
ratings). The result on the real data is "0 removed", which is itself worth saying in the report.
Add a test in `tests/test_step1.py`, run `pytest`, and explain it.
