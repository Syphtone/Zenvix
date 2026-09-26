# Business Entity Resolution: reproduction guide

This pipeline regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data. It uses blocking plus a LightGBM matcher, runs on CPU only, and uses no external data or APIs.

## Contents

| Path | Purpose |
|---|---|
| `src/run_pipeline.py` | End-to-end pipeline as a plain Python script (data → normalisation → blocking → features → model → outputs → validation) |
| `src/business_entity_resolution.ipynb` | The same pipeline as a notebook, with the outputs of the run that produced the submitted files |
| `requirements.txt` | Pinned dependencies (Python 3.12.4) |

## Setup

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Run

Run from the challenge's `student_resource/` directory, which contains `dataset/train`, `dataset/test` and `utils/`:

```bash
python code/business_entity_resolution/src/run_pipeline.py
```

To run the notebook instead, open `src/business_entity_resolution.ipynb` with `student_resource/` as the working directory and run all cells.

Paths can be overridden with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `ER_DATA_DIR` | `dataset` | Folder containing `train/` and `test/` |
| `ER_OUT_DIR` | `output` | Where the TSVs and the model are written |
| `ER_VALIDATOR` | `utils/validate_submission.py` | Validator run at the end (skipped if the file is missing) |

### Outputs

* `output/matching_results.tsv`: final matches, one row per test S1 entity.
* `output/candidate_pairs.tsv`: exactly the (S1, S2/S3) pairs the model scored. The matches are a subset of it.
* `output/lgb_matcher.txt`: the trained LightGBM model.

The run finishes by calling `utils/validate_submission.py`, which should print `PASS`.

### Resources and runtime

Reference machine: 12 logical cores, 16 GB RAM, no GPU.

| Stage | Time |
|---|---|
| Load data and build the geo-stratified dev universe | ~2 min |
| Learn the transliteration map and full-train document frequencies | ~2 min |
| Dev blocking and features (1.17M records → 5.4M pairs) | ~3 min |
| 3-fold LightGBM with out-of-fold threshold tuning, then the final model | ~8 min |
| Test inference, streamed in 250k-record chunks (9.97M records → 62.9M pairs) | ~34 min |

Peak memory is about 5.5 GB. Every random step uses `SEED = 42`.

## Configuration

These constants are at the top of the script and notebook. The submitted run used the values below.

| Constant | Value | Meaning |
|---|---|---|
| `DEV_FRAC` | 0.10 | Share of training S1 entities (whole states) used for training and validation |
| `HASH_BITS` | 24 | Hashed vocabulary size |
| `DF_CAP` | 3000 | Tokens in more S1 rows than this are ignored during blocking |
| `TOP_K` | 8 | S1 candidates retrieved per S2/S3 record |
| `MIN_COS` | 0.08 | Absolute cosine floor for a candidate |
| `REL_COS` | 0.25 | Relative cosine floor (share of the record's best candidate) |
| `CHUNK` | 250,000 | Records per streaming chunk |
| `RUN_TEST` | True | Set to False to stop after validation |

## Licences

All dependencies are permissively licensed:

| Licence | Packages |
|---|---|
| MIT | LightGBM, rapidfuzz |
| Apache 2.0 | sparse_dot_topn |
| BSD | numpy, pandas, scipy, scikit-learn, joblib |
| ISC | anyascii |

No pretrained language models are used.
