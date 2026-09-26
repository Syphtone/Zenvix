# Output manifest — submitted run

The two output TSVs are **not stored in this repository**. `candidate_pairs.tsv` is
833 MB and GitHub hard-rejects any single file over 100 MB. This manifest records
their exact size, content hash and row statistics so the submitted run stays
auditable, and so a regenerated run can be checked against it byte-for-byte.

Regenerate both files with:

```bash
python code/business_entity_resolution/src/run_pipeline.py
```

Run from the challenge's `student_resource/` directory. The pipeline is fully
seeded (`SEED = 42`), so a rerun on the same data reproduces these hashes.

---

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `matching_results.tsv` | 95,941,513 | `dbfb998ea8f7f2aca23d4fee382c9152e0fde1927a4a6682691bd79fd6ec9e96` |
| `candidate_pairs.tsv` | 833,187,361 | `f98e6de49c508788ce5a041c170d98ef75e31e11d3e97903dcf42496a6fc8878` |
| `Zenvix_submission.zip` | 395,346,231 | `5b96f71335b0150a7bacb362444700554b244ce6f0efbc73b944a97da9a73adf` |

## Row statistics

| Metric | `matching_results.tsv` | `candidate_pairs.tsv` |
|---|---:|---:|
| Rows (excl. header) | 1,732,544 | 1,732,544 |
| Total IDs listed | 5,703,341 | 62,911,416 |
| Empty rows | 100,159 | 1,574 |
| Empty share | 5.781 % | 0.091 % |

Row count equals the 1,732,544 Source 1 entities in `test_source1.tsv` exactly —
one row per entity, as the format requires. Predicted singleton rate (5.781 %) sits
close to the 5.6 % training base rate. Candidates average 6.31 per S2/S3 record
across the 9,969,589 test records.

## Validation

`utils/validate_submission.py` (the organisers' script, stdlib only) was run twice
against these files:

**1. Default run** — both files, all format rules:

```
required S1 entities: 1732544
matching_results.tsv: 1732544 rows (100159 empty, 1632385 non-empty).
candidate_pairs.tsv:  1732544 rows (1574 empty, 1730970 non-empty).
PASS - no blocking issues found. Safe to submit.
```

**2. With `--check-ids`** — additionally verifies every matched ID exists in the
test set (loads all 9,969,589 Source 2/3 IDs):

```
required S1 entities: 1732544
valid S2/S3 match IDs: 9969589
matching_results.tsv: 1732544 rows (100159 empty, 1632385 non-empty).
PASS - no blocking issues found. Safe to submit.
```

Both passes confirm: one row per S1 entity, no duplicate `source1_entity_id`, no
duplicate IDs within any list, Source 2/3 IDs only, every matched ID exists in the
test set, and every match also appears in `candidate_pairs.tsv` (matches are a
strict subset of candidates).

## Model

`output/lgb_matcher.txt` (the trained LightGBM model, 9.2 MB) is also excluded from
git. Properties read back from the saved model:

| Property | Value |
|---|---|
| Trees | 658 |
| `num_leaves` | 127 |
| `learning_rate` | 0.08 |
| Features (`max_feature_idx`) | 43 → **44 features** |

The 44 feature names stored in the model match the feature set built by
`pair_features()` exactly.

## Scores

| Split | Macro F<sub>0.5</sub> |
|---|---|
| Out-of-fold validation (250,249 S1 entities) | **0.9753** |
| — US (212 k entities) | 0.9750 |
| — India (39 k entities) | 0.9769 |
| Ceiling with a perfect matcher on our candidates | 0.991 |
| Predicting nothing (all singletons) | 0.056 |

Pair-level blocking recall on the validation universe: 0.972.
