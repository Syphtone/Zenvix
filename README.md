# Zenvix — Amazon ML Challenge 2026

**Business Entity Resolution.** Given business records from three independent,
noisy sources with no shared identifiers, find every Source 2 / Source 3 record
that refers to the same real-world business as each Source 1 entity.

**Team:** Pranav A S (lead), Aibin K Jayan, B Abhinav Ram, Ajsal Ashraf
**Out-of-fold macro F<sub>0.5</sub>:** **0.9753** (250,249-entity validation universe)

---

## Submission order

The challenge asks for two separate deliverables, in this order:

### 1. Leaderboard upload — during the challenge window

Upload **`output/matching_results.tsv`** (95.9 MB) in the Portal. This single file
drives both the public and the private leaderboard. Maximum 5 submissions per day.

> Not in this repo — see [`output/MANIFEST.md`](output/MANIFEST.md) for its size,
> SHA-256 and row statistics. Regenerate with the pipeline below.

### 2. Final submission package — one zip per team

`Zenvix_submission.zip` (395 MB), in exactly this layout:

```
Zenvix_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file as the leaderboard upload)
│   └── candidate_pairs.tsv         # the blocking candidate set fed to the model
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # run_pipeline.py + the notebook
│       ├── README.md               # end-to-end reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # methodology write-up
```

Rebuild it from this repository with:

```bash
python tools/build_submission_zip.py \
    --matching   path/to/matching_results.tsv \
    --candidates path/to/candidate_pairs.tsv \
    --out        Zenvix_submission.zip
```

The tool packs exactly the seven files the specification lists — nothing else — and
verifies the archive afterwards.

### 3. If shortlisted (top 100)

[`Documentation_template.md`](Documentation_template.md) already covers everything
requested at that stage: methodology, candidate generation / blocking strategy,
model architecture and feature engineering.

---

## What's in this repository

| Path | Contents |
|---|---|
| [`Documentation_template.md`](Documentation_template.md) | Methodology write-up — the filled-in official template |
| [`code/business_entity_resolution/`](code/business_entity_resolution/) | Self-contained, runnable pipeline |
| [`code/business_entity_resolution/src/run_pipeline.py`](code/business_entity_resolution/src/run_pipeline.py) | The whole pipeline as one script — the reproduction entry point |
| [`code/business_entity_resolution/src/business_entity_resolution.ipynb`](code/business_entity_resolution/src/business_entity_resolution.ipynb) | Same pipeline as a notebook, carrying the stored outputs of the submitted run |
| [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) | Setup, run instructions, runtime, configuration, licences |
| [`output/MANIFEST.md`](output/MANIFEST.md) | Checksums, row statistics, validator transcripts and model properties for the submitted run |
| [`tools/build_submission_zip.py`](tools/build_submission_zip.py) | Rebuilds the submission zip to spec and verifies it |

### Deliberately not committed

The challenge dataset (~2.5 GB of Amazon-provided data), the two output TSVs and
the trained model are all excluded — see [`.gitignore`](.gitignore).
`candidate_pairs.tsv` alone is 833 MB and GitHub hard-rejects any file over 100 MB.
Everything needed to regenerate them is here, and
[`output/MANIFEST.md`](output/MANIFEST.md) pins the expected hashes.

---

## Approach at a glance

| Stage | What happens |
|---|---|
| **1. Normalisation** | Transliterate Devanagari / Telugu via `anyascii` plus a token map **learned from the training labels**, expand abbreviations, canonicalise state names through a country-keyed lookup, repair OCR digits (`6reen` → `green`), strip `##` / `null` / domain noise |
| **2. Blocking** | Country-scoped hashed TF-IDF (2²⁴ buckets) over name tokens, name bigrams, address tokens, house numbers and number+street bigrams, with over-frequent tokens dropped. Sparse top-K cosine search (`sparse_dot_topn`) retrieves the best Source 1 candidates **for each Source 2/3 record** |
| **3. Pair features** | 44 features: rapidfuzz similarities (name, core name, concatenated core, consonant skeleton, aliases, address), per-field TF-IDF cosines, number-set overlap, rank/gap within the record's own candidate list, Source 1 ambiguity counts |
| **4. Matcher** | LightGBM binary classifier (658 trees, 127 leaves), scored out-of-fold with `GroupKFold` grouped by real-world entity |
| **5. Decision** | Each Source 2/3 record belongs to **at most one** Source 1 entity — verified across all 7.64 M matched training pairs. Assign each record to its arg-max candidate when *p* ≥ τ, with τ = 0.70 tuned for macro F<sub>0.5</sub> |

Three ideas did most of the work:

- **Record-centric arg-max assignment.** Because a record has at most one owner,
  resolving per *record* rather than per *entity* removes most multi-match false
  merges by construction — which matters under a precision-heavy metric.
- **Transliteration learned from the labels**, not from an external resource. A
  mapping is kept only when its token spans ≥ 5 distinct Source 1 entities, so
  generic words (`private`, `limited`, state names) are learned and one-off brand
  names never leak into validation.
- **Density-preserving validation.** Whole states are sampled, and document
  frequencies come from the *full* training Source 1, so blocking at validation
  scale behaves as it does on the 1.73 M-entity test set.

`country` is treated as an open set of string labels throughout — no feature
one-hots it and no country list is hard-coded — so **France**, which appears only
in the test set, flows through the same model. Its predicted singleton rate (5.1 %)
lands close to US (5.9 %) and India (5.9 %), which suggests τ transfers to the
unseen country.

### Results

| Metric | Value |
|---|---|
| Out-of-fold macro F<sub>0.5</sub> | **0.9753** |
| Pair-level blocking recall | 0.972 |
| F<sub>0.5</sub> ceiling given our candidates | 0.991 |
| Singletons correctly left empty | 97.3 % |
| Candidate reduction ratio vs. full cross product | 0.99998 |

Runs on **CPU only** — 12 logical cores, 16 GB RAM, ~50 minutes end to end, peak
memory ~5.5 GB.

---

## Reproducing

Full instructions, runtime breakdown and every tunable constant are in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md).
In short, from the challenge's `student_resource/` directory:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py
```

The run ends by invoking the organisers' `utils/validate_submission.py`, which
should print `PASS`.

---

## Compliance

- **No external data.** No external database, API, geocoding service or internet
  augmentation is used anywhere in the pipeline. The only hard-coded knowledge is
  generic abbreviation and state-name normalisation; the transliteration map is
  learned from the provided training labels.
- **Model licence and size.** The final model is LightGBM (MIT), a gradient-boosted
  tree ensemble — no pretrained or neural model is used, so the 8-billion-parameter
  ceiling is not approached. Every dependency is permissive: MIT (LightGBM,
  rapidfuzz), Apache 2.0 (`sparse_dot_topn`), BSD (numpy, pandas, scipy,
  scikit-learn, joblib), ISC (`anyascii`).
- **Reproducibility.** Every random step is seeded with `SEED = 42`.

Licensed under Apache 2.0 — see [`LICENSE`](LICENSE).
