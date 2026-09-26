# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Zenvix  
**Team Members:** Pranav A S (Team Lead), Aibin K Jayan, B Abhinav Ram, Ajsal Ashraf  
**Submission Date:** 25 September 2026

---

## 1. Executive Summary

We normalise noisy multilingual records, then find candidates with a country-scoped, hashed TF-IDF top-K sparse search from each Source 2/3 record into Source 1. A LightGBM classifier scores every candidate pair using 44 similarity and context features. Each record is then assigned to at most one S1 entity: its best-scoring candidate, if the score clears a threshold tuned for F0.5.

The main ideas are:
* **Record-centric assignment**, based on the observed fact that every S2/S3 record belongs to at most one S1 entity.
* **A transliteration map learned from the training pairs** for Devanagari and Telugu records.
* **A density-preserving validation design** that samples whole states and takes token statistics from the full training set.

The out-of-fold macro F0.5 on a 250k-entity validation universe is **0.975**.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data (2.2M S1, 5.0M S2 and 5.3M S3 records) showed the following:

* **One owner per record.** 7.64M of 10.32M S2/S3 records are matched, and **no record matches more than one S1 entity**. The remaining ~26% match nothing and act as distractors.
* **Country is a perfect block.** Across all 7.64M matched pairs, a record's `country` equals its S1 entity's `country`. We use this as a generic key and never enumerate the country values, so France (test only) is handled without special cases.
* **Singletons.** 5.6% of S1 entities have no match. Matches per entity range from 0 to 11, with a mean of 3.5.
* **Name noise:**
  * legal-suffix variation and reordering (`Pvt Pi Technologies Ltd`, `L.L.C. Superior Financial International`)
  * OCR-style digit substitutions (`6reen`, `5uperior`, `Techno1ogy`)
  * random typos and doubled tokens (`XXZ XXZ Umbrella`)
  * honorific prefixes (`Dr`, `Smt`, `Shri`) and junk prefixes (`...`, `--`, `<<`)
  * web-domain names (`o7green.com`, `GanganthsummitCom`)
  * aliases (`a/k/a`, `d/b/a`)
  * unrelated noise words appended (`… Center`, `… Services`)
  * whole names in **Devanagari or Telugu script** (`सदर्न हॉस्पिटैलिटी प्राइवेट लिमिटेड` = *Southern Hospitality Private Limited*)
* **Address noise:**
  * component reordering, and upper vs. title case
  * `##` prefixes and `null` components
  * leading zeros (`0534`)
  * abbreviations (`Rd`, `Ct`, `Ave`)
  * state as full name, code or native script (`Maharashtra`, `MH`, `महाराष्ट्र`)
  * missing components, or an empty address (~3% of records)
  * municipal numbering variants (`3-6-667/4` vs `3-6-667/4/9`)
* **France (test only)** follows the same noise patterns, plus French street forms (`R.` = rue, `AV`, `BD`, `Allée`) and legal forms (SARL, SAS, SASU, EURL, SCI).

### 2.2 Solution Strategy

**Approach Type:** Hybrid: normalisation → sparse blocking → gradient-boosted pair classifier → constrained assignment.  
**Core Innovation:**
1. **Record-centric blocking and arg-max assignment.** This uses the "each record has at most one owner" structure, which removes most multi-match false merges by construction.
2. **A transliteration token map learned from the training labels.** This bridges native-script names to their English S1 forms with no external resources.
3. **A validation design that matches test-scale density:**
   * geo-stratified sampling (whole states), so every validation entity keeps its real look-alike neighbours;
   * IDF and frequency caps computed on the full training S1, so blocking in validation behaves as it does on the 1.7M-entity test set.

---

## 3. Candidate Generation (Blocking)

**Normalisation (applied to every record first):**
* `anyascii` transliteration, followed by the learned token map (for example `praivet → private`, `hospitailiti → hospitality`, `mharastr → maharashtra`). It has 551 entries, kept only when a token occurs across ≥ 5 distinct S1 entities, so no proper names leak into validation.
* OCR digit repair inside alphabetic tokens (0→o, 1→l, 3→e, 4→a, 5→s, 6→g, 8→b).
* Domain names reduced to their label, and `a/k/a` / `d/b/a` alias splitting.
* Abbreviation expansion for names (`pvt`, `ltd`, `corp`, `inc`, `co`, …) and addresses (US, Indian and French street types).
* State canonicalisation through a lookup keyed by the country label (countries without an entry pass through unchanged).
* Removal of `##`, `null`, `No.`, `PO Box`, and leading zeros in numbers.
* A **core name**: the name with legal forms and honorifics removed.

**Blocking keys used:** for each record, three hashed token sets (crc32, 2^24 buckets), all prefixed with the country:
* *name*: core-name tokens, adjacent-token bigrams, the concatenated core (catches `ganganthsummit.com`-style names), and alias parts;
* *address*: word tokens (≥ 3 characters) and number→next-word bigrams (`125_polk`);
* *numbers*: the set of numbers in the address.

Each field is TF-IDF weighted, using IDF from S1. Tokens found in more than 3,000 S1 rows are dropped from the blocking vector. The fields are combined with weights 0.6 (name), 0.3 (address) and 0.1 (numbers), then l2-normalised.

**Search:** `sparse_dot_topn` (a multi-threaded sparse matrix product that keeps only the top-K) returns, for **each S2/S3 record**, its top-8 S1 rows with cosine ≥ 0.08. We keep only candidates whose cosine is at least 25% of the record's best candidate.

**Candidate pairs generated:**
* Validation universe: 5.37M pairs for 1.17M records (4.6 per record). The reduction ratio against the full cross product is 0.99998.
* **Test: 62.9M pairs for 9.97M records (6.3 per record).** Only 1,574 of 1.73M S1 entities receive no candidate.

**How we ensured true matches were not lost:**
* Name and address are *both* keys, so a record survives if either one is intact. This covers native-script names (the address matches) and empty addresses (the name matches).
* Bigrams and the concatenated name recover names made only of common words.
* The low absolute floor and relative cut favour recall; precision is left to the matcher.
* Blocking recall is measured directly. On the validation universe, **97.2% of true pairs** are retained, which puts the macro-F0.5 ceiling at **0.991** for a perfect matcher on these candidates.

---

## 4. Matching Model

**Features used:** 44 per (record, S1) pair, all computed in C++ via `rapidfuzz.process.cpdist` or as sparse row-wise dot products:
- **Name features:**
  * rapidfuzz `ratio` and `token_set_ratio` on the full normalised name
  * `ratio`, `token_sort`, `token_set`, `partial_ratio` and Jaro-Winkler on the core name
  * `ratio` and `partial_ratio` on the concatenated core
  * `ratio` on the consonant skeleton (vowels removed, repeats collapsed), which is robust to transliteration and typos
  * best `token_set` across alias parts
  * IDF-weighted name cosine
  * name-token intersection, Jaccard and coverage
  * core-name lengths
- **Address features:**
  * `ratio`, `token_set`, `token_sort` and `partial_token_set`
  * address TF-IDF cosine
  * number-set cosine, Jaccard, intersection and coverage, plus the size of each number set
  * address-token Jaccard and coverage
  * empty-address flag
- **Other:**
  * *competition inside the record's candidate list*: blocking cosine, its rank, gap and ratio to the best candidate, number of candidates, and the gap to the best candidate for `c_tset`, `a_tset` and `name_cos`;
  * *S1-side ambiguity*: how many S1 rows share the exact core name or address;
  * record source (S2/S3) and a non-Latin-script flag.

No feature is a one-hot of country, so France uses the same model.

The features with the highest LightGBM gain are, in order:
* candidate rank
* address token-set gap to the best candidate
* number Jaccard
* number cosine
* address token-set score
* number-set size
* name ratio
* blocking cosine

**Model type:** LightGBM binary classifier (MIT licence; about 660 trees, 127 leaves, learning rate 0.08). No pretrained or neural models are used.

**Training data:** all candidate pairs from a geo-stratified 10% sample of training S1 entities. Whole states are sampled, which gives 250k S1 entities and 1.17M S2/S3 records. That sample includes every record matched to a sampled entity, plus the unmatched records whose address maps to a sampled state. The component→state map behind this is learned from matched pairs.

**Threshold selection method:**
1. Produce out-of-fold predictions from 3-fold GroupKFold, grouped by real-world entity.
2. Apply the decision rule: *assign each record to its arg-max S1 if p ≥ τ*.
3. Grid-search τ ∈ [0.20, 0.95] to maximise **macro F0.5**, which gives τ = 0.70.

The curve is flat between 0.60 and 0.80 (F0.5 0.9748–0.9753), so the choice is robust.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold, 250k S1 entities):** **0.9753**
  * US: 0.9750 (212k entities)
  * India: 0.9769 (39k entities)
  * Singletons correctly left empty: 97.3%
  * Ceiling with perfect matching on our candidates: 0.991
  * Predicting nothing: 0.056
- **Test set:** 5.7M matches. 5.8% of S1 entities are predicted as singletons, close to the 5.6% training base rate. By country, the share of S2/S3 records matched is France 61.2%, India 56.3% and US 56.9%, and the predicted singleton rate is France 5.1%, India 5.9% and US 5.9%. France looks like the labelled countries, which suggests the threshold transfers to the unseen country.
- **Common false positives (wrong merges):**
  * **Same name, nearly the same street number.** Example: `Gant and Alvarez Inc, 2136 Barefoot Park Lane` vs `… 2149 Barefoot Park Ln`. The data contains deliberately near-duplicate distinct businesses, so a one-digit house-number change is often a different entity rather than a typo.
  * **Generic names in the same building or sector.** Example: `Noida Projects Clinic, A-95 Sector-63` vs `Noida Ambika Clinic, A-99 Sector-63`.
  * **Name-only records with an empty address** that point to a plausible but wrong entity.
- **Common false negatives (missed matches):**
  * about 2.8% of true pairs are never retrieved: heavy typos in both name and address, or a random replacement name (`Brixquokelo`, `Deltawex`) together with a partial address;
  * native-script names for proper nouns absent from the learned map;
  * matches that score below τ because the record carries only the name, or only the address.

---

## 6. Conclusion

A normalisation → sparse blocking → gradient-boosting pipeline reaches 0.975 out-of-fold macro F0.5 while running on a 12-core CPU with 16 GB RAM, taking about 50 minutes end to end.

The largest gains came from two decisions:
* exploiting the one-owner-per-record structure (arg-max assignment and rank/gap features);
* learning transliteration from the labels instead of relying on generic romanisation.

We also learned that validation must reproduce test-scale density. Our first design used document frequencies from the small dev index, and blocking behaved differently in validation than at test scale.

Next steps:
* sharper house-number mismatch features aimed at the near-duplicate hard negatives;
* S1-side consistency features (agreement among the records assigned to the same entity);
* a larger training sample and more boosting rounds, since every fold was still improving at 600 rounds.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
* `src/run_pipeline.py` is the full pipeline as one script. Run it from `student_resource/` with `python code/business_entity_resolution/src/run_pipeline.py`. It writes `output/matching_results.tsv`, `output/candidate_pairs.tsv` and `output/lgb_matcher.txt`, then runs `utils/validate_submission.py`.
* `src/business_entity_resolution.ipynb` is the same pipeline as a notebook, with the outputs of the submitted run.
* `README.md` gives setup, run instructions, runtime, configuration and licences.
* `requirements.txt` pins the dependencies for Python 3.12.4: numpy, pandas, scipy, scikit-learn, lightgbm, rapidfuzz, anyascii, joblib, sparse_dot_topn.

The script and notebook are organised in these sections:
0. configuration
1. I/O and metric
2. data loading, EDA and geo-stratified dev universe
3. normalisation and the learned transliteration map
4. S1 blocking index
5. candidate generation and pair features
6. out-of-fold LightGBM and threshold tuning
7. final model
8. streaming test inference
9. validation

### B. Additional Results

Threshold sweep (out-of-fold macro F0.5):

| τ | 0.30 | 0.40 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| F0.5 | 0.9675 | 0.9712 | 0.9734 | 0.9748 | 0.9750 | **0.9753** | 0.9752 | 0.9749 | 0.9727 |

Pipeline funnel:

| Stage | Validation | Test |
|---|---|---|
| S1 entities | 250,249 | 1,732,544 |
| S2/S3 records | 1,169,998 | 9,969,589 |
| Candidate pairs | 5,370,989 | 62,911,416 |
| Predicted matches | — | 5,703,341 |
| Pair-level blocking recall | 0.972 | — |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
