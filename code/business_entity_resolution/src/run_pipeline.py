"""Business Entity Resolution: end-to-end pipeline (script version of business_entity_resolution.ipynb).

Run from the challenge's student_resource/ directory (the one containing dataset/ and utils/):
    python code/business_entity_resolution/src/run_pipeline.py
Paths can be overridden with ER_DATA_DIR, ER_OUT_DIR and ER_VALIDATOR environment variables.
"""
# %% [markdown]
# # Business Entity Resolution — Blocking + LightGBM Matcher
#
# **Task.** For every Source 1 (S1) entity, find all Source 2 / Source 3 records that describe the same real-world business. Scored with **macro F0.5**, averaged over every S1 entity. Singletons count too: an empty prediction for an entity with no matches scores 1.0.
#
# **Pipeline**
#
# | Stage | What happens |
# |---|---|
# | 1. Normalisation | Transliterate Indic scripts (`anyascii` plus a token map *learned from training pairs*), expand abbreviations, canonicalise state names, repair OCR digits (`6reen` → `green`), strip junk (`##`, `null`, `www.`/`.com`) |
# | 2. Blocking | Country-scoped hashed TF-IDF over name tokens, name bigrams, address tokens, house numbers and number+street bigrams, with very frequent tokens dropped. Sparse top-K cosine search (`sparse_dot_topn`) finds the best S1 candidates for **each S2/S3 record** |
# | 3. Pair features | 44 features: rapidfuzz similarities (name, core name, alias, no-space, consonant skeleton, address), TF-IDF cosines per field, number overlap, rank/gap within the record's candidate list, S1 ambiguity counts |
# | 4. Matcher | LightGBM binary classifier, evaluated with out-of-fold predictions (GroupKFold by entity) |
# | 5. Decision | Each S2/S3 record matches **at most one** S1 entity (verified in training data). Assign each record to its arg-max S1 if p ≥ τ; tune τ for macro F0.5 |
#
# **Key facts from EDA** (see Section 2): a record's `country` always equals its S1 entity's `country`; each S2/S3 record belongs to at most one S1 entity; about 26% of S2/S3 records match nothing; about 5.6% of S1 entities are singletons.
#
# **Scale.** The test set has 1.7M S1 entities and 10M S2/S3 records. Everything here is CPU-only and streams S2/S3 in chunks, so it fits in 16 GB RAM.
#
# **Fair play.** No external data or APIs are used. The only "knowledge" hard-coded is generic abbreviation/state-name normalisation. The transliteration map is learned from the training labels.

# %% [markdown]
# ## 0. Setup & configuration

# %%
# pip install -r requirements.txt   (all dependencies are MIT / BSD / ISC / Apache licensed)
import os, re, csv, gc, time, zlib, math, subprocess, sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anyascii import anyascii
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sparse_dot_topn import sp_matmul_topn
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

DATA_DIR   = os.environ.get("ER_DATA_DIR", "dataset")          # folder with train/ and test/
OUT_DIR    = os.environ.get("ER_OUT_DIR", "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED       = 42
N_JOBS     = max(1, (os.cpu_count() or 2) - 1)

# Dev universe: a *geo-stratified* sample of training S1 entities (whole states kept together so the
# local density of look-alike businesses matches the test set). Raise it if you have the RAM/time.
DEV_FRAC   = 0.10

# Blocking
HASH_BITS  = 24          # hashed vocabulary size 2^24
DF_CAP     = 3000        # tokens occurring in more than DF_CAP S1 rows are ignored during blocking
TOP_K      = 8           # S1 candidates retrieved per S2/S3 record
MIN_COS    = 0.08        # absolute cosine floor for a candidate
REL_COS    = 0.25        # keep a candidate only if cos >= REL_COS * best cos of that record
CHUNK      = 250_000     # S2/S3 records processed per streaming chunk

RUN_TEST   = True        # set False to stop after validation
rng = np.random.default_rng(SEED)
print("workers:", N_JOBS)

# %% [markdown]
# ## 1. I/O helpers
# All files are TSV and are read with `quoting=QUOTE_NONE`, so a stray `"` inside a business name can never swallow a line. Entity IDs are also encoded as int64 (`source * 1e10 + number`) for cheap joins. The original strings are kept for output.

# %%
def read_tsv(path, **kw):
    """Read a challenge TSV. Every column stays a string and quoting is disabled, so a
    stray double quote inside a business name can never swallow the rest of the line."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, **kw)

NONLATIN = re.compile(r"[^\x00-\u024F\u1E00-\u1EFF]")   # anything outside Latin / Latin-extended

def id2int(s: pd.Series) -> np.ndarray:
    """Encode `S<src>-<number>` entity ids as int64 (`src * 1e10 + number`) for cheap joins
    and array indexing. The original id strings are kept for the output files."""
    src = s.str.slice(1, 2).astype(np.int64).to_numpy()
    num = s.str.slice(3).astype(np.int64).to_numpy()
    return src * 10**10 + num

def fbeta_macro(pred: dict, truth: dict, s1_ids, beta=0.5):
    """Official metric: per-S1 F_beta, macro averaged. Empty/empty = 1, one-sided empty = 0."""
    b2 = beta * beta
    tot = 0.0
    for s in s1_ids:
        t = truth.get(s, ()); p = pred.get(s, ())
        if not t and not p: tot += 1.0; continue
        if not t or not p: continue
        tp = len(set(t) & set(p))
        if tp == 0: continue
        pr, rc = tp / len(p), tp / len(t)
        tot += (1 + b2) * pr * rc / (b2 * pr + rc)
    return tot / len(s1_ids)

# %% [markdown]
# ## 2. Load training data & quick EDA

# %%
t0 = time.time()
s1_all = read_tsv(f"{DATA_DIR}/train/train_source1.tsv")
gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
print("S1 train:", s1_all.shape, "| ground truth rows:", len(gt))
print(s1_all.country.value_counts().to_dict())

n_match = gt.matched_entity_ids.map(lambda x: x.count(",") + 1 if x else 0)
print(f"singleton share: {(n_match == 0).mean():.3f}  | mean matches per S1: {n_match.mean():.2f}")
print("matches-per-entity distribution:", n_match.value_counts().sort_index().to_dict())

# record -> S1 mapping (int64 ids)
g = gt[gt.matched_entity_ids != ""]
ex = g.assign(r=g.matched_entity_ids.str.split(",")).explode("r")
rec2s1 = pd.Series(id2int(ex.source1_entity_id), index=id2int(ex.r))
assert rec2s1.index.is_unique, "a record is matched to more than one S1 entity"
print(f"{len(rec2s1):,} matched S2/S3 records — each belongs to exactly one S1 entity")
del g, ex, n_match; gc.collect()
s1_all["iid"] = id2int(s1_all.entity_id)
print(f"loaded in {time.time()-t0:.0f}s")

# %%
pd.set_option("display.max_colwidth", 80, "display.width", 200)
print(s1_all.sample(8, random_state=1)[["entity_id", "business_name", "business_address", "country"]].to_string())

# %% [markdown]
# ### 2.1 Geo-stratified dev universe
# A plain random sample of S1 entities would thin out the look-alike businesses around each entity. Blocking and precision would then look better than they really are. Instead we
# 1. give every S1 entity a coarse geo key (its state: the last address component that is a frequent state-like value),
# 2. sample **whole geo keys** until `DEV_FRAC` of S1 is covered,
# 3. keep every record matched to a sampled S1, and every unmatched record whose address maps to a sampled geo key. The component→geo map is learned from matched pairs, so it also covers `Texas`/`TX`/`महाराष्ट्र`. Unmatched records with no recognisable geo are kept with probability `DEV_FRAC`.

# %%
def comps_of(addr: str):
    """Split an address on commas into lowercase ASCII components, dropping blanks."""
    return [anyascii(c).strip().lower() for c in addr.split(",") if c.strip()]

last = s1_all.business_address.str.rsplit(",", n=1).str[-1].str.strip().str.lower()
vc = (s1_all.country + "|" + last).value_counts()
geo_vocab = set(vc[vc >= 500].index)

def s1_geo(addr, country):
    """Coarse geo key (`country|state`) for an S1 row: the last address component that is a
    frequent state-like value. Returns `country|?` when no component is recognised."""
    for c in reversed(comps_of(addr)):
        if f"{country}|{c}" in geo_vocab: return f"{country}|{c}"
    return f"{country}|?"

s1_all["geo"] = [s1_geo(a, c) for a, c in zip(s1_all.business_address, s1_all.country)]
geo_sizes = s1_all.geo.value_counts()
geo_order = rng.permutation(geo_sizes.index.to_numpy())
picked, covered = [], 0
for gk in geo_order:
    if covered >= DEV_FRAC * len(s1_all): break
    if geo_sizes[gk] > 0.5 * DEV_FRAC * len(s1_all):  # skip keys too big for the budget
        continue
    picked.append(gk); covered += geo_sizes[gk]
picked = set(picked)
dev_s1 = s1_all[s1_all.geo.isin(picked)].reset_index(drop=True)
print(f"dev S1: {len(dev_s1):,} entities from {len(picked)} geo keys "
      f"({dev_s1.country.value_counts().to_dict()})")
dev_s1_set = set(dev_s1.iid.tolist())

# %%
def load_dev_records(split="train"):
    """Stream S2/S3, keep records of sampled S1s plus geo-matched unmatched records.
    Also returns *all* matched records written partly in a non-Latin script (used in 3.1)."""
    s1_geo_map = dict(zip(s1_all.iid.to_numpy(), s1_all.geo.to_numpy()))
    comp2geo = None
    keep, nonlatin_rows = [], []
    nl_pat = NONLATIN.pattern
    for src in (2, 3):
        for ch in read_tsv(f"{DATA_DIR}/{split}/{split}_source{src}.tsv", chunksize=1_000_000):
            ch["iid"] = id2int(ch.entity_id)
            true_s1 = rec2s1.reindex(ch.iid.to_numpy()).to_numpy()
            ch["true_s1"] = np.nan_to_num(true_s1, nan=-1).astype(np.int64)
            if comp2geo is None:   # learn component -> geo from the first chunk's matched records
                cnt = defaultdict(Counter)
                m = ch[ch.true_s1 >= 0].sample(min(300_000, (ch.true_s1 >= 0).sum()), random_state=SEED)
                for a, c, s in zip(m.business_address, m.country, m.true_s1):
                    gk = s1_geo_map.get(s)
                    for comp in set(comps_of(a)):
                        cnt[f"{c}|{comp}"][gk] += 1
                comp2geo = {}
                for k, cc in cnt.items():
                    tot = sum(cc.values()); gk, n = cc.most_common(1)[0]
                    if tot >= 20 and n / tot >= 0.95 and not gk.endswith("|?"):
                        comp2geo[k] = gk
                print(f"learned {len(comp2geo):,} address-component -> geo mappings")
            matched = ch.true_s1 >= 0
            nl = matched & (ch.business_name.str.contains(nl_pat) | ch.business_address.str.contains(nl_pat))
            nonlatin_rows.append(ch.loc[nl, ["business_name", "business_address", "true_s1"]])
            keep_m = matched & ch.true_s1.isin(dev_s1_set)
            um = ch[~matched]
            um_geo = []
            for a, c in zip(um.business_address, um.country):
                gk = None
                for comp in reversed(comps_of(a)):
                    gk = comp2geo.get(f"{c}|{comp}")
                    if gk: break
                um_geo.append(gk)
            um_geo = pd.Series(um_geo, index=um.index)
            coin = pd.Series(rng.random(len(um)) < DEV_FRAC, index=um.index)
            keep_u = um_geo.isin(picked) | (um_geo.isna() & coin)
            keep.append(pd.concat([ch[keep_m], um[keep_u]]))
    return pd.concat(keep, ignore_index=True), pd.concat(nonlatin_rows, ignore_index=True)

t0 = time.time()
dev_rec, nonlatin_train = load_dev_records()
print(f"non-Latin matched training records (for 3.1): {len(nonlatin_train):,}")
print(f"dev records: {len(dev_rec):,} (matched {int((dev_rec.true_s1 >= 0).sum()):,}) in {time.time()-t0:.0f}s")

# ground truth restricted to the dev universe
dev_truth = defaultdict(list)
for r, s in zip(dev_rec.iid.to_numpy(), dev_rec.true_s1.to_numpy()):
    if s >= 0: dev_truth[s].append(r)
print(f"dev singletons: {1 - len(dev_truth)/len(dev_s1):.3f}")

# %% [markdown]
# ## 3. Normalisation
#
# Rules are applied through dictionary lookups keyed by the `country` label. A country without an entry, such as France, simply passes through, so the country set stays open. French street and legal forms are included because the test set contains France.

# %%
US_STATES = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co",
 "connecticut":"ct","delaware":"de","district of columbia":"dc","florida":"fl","georgia":"ga","hawaii":"hi",
 "idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la",
 "maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms",
 "missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj",
 "new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok",
 "oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd",
 "tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa",
 "west virginia":"wv","wisconsin":"wi","wyoming":"wy"}
IN_STATES = {"andhra pradesh":["ap"],"arunachal pradesh":["ar"],"assam":["as"],"bihar":["br"],
 "chhattisgarh":["cg","ct"],"goa":["ga"],"gujarat":["gj"],"haryana":["hr"],"himachal pradesh":["hp"],
 "jharkhand":["jh"],"karnataka":["ka"],"kerala":["kl"],"madhya pradesh":["mp"],"maharashtra":["mh"],
 "manipur":["mn"],"meghalaya":["ml"],"mizoram":["mz"],"nagaland":["nl"],"odisha":["od","or","orissa"],
 "punjab":["pb"],"rajasthan":["rj"],"sikkim":["sk"],"tamil nadu":["tn"],"telangana":["tg","ts"],
 "tripura":["tr"],"uttar pradesh":["up"],"uttarakhand":["uk","ut","uttaranchal"],"west bengal":["wb"],
 "delhi":["dl"],"jammu and kashmir":["jk"],"chandigarh":["ch"],"puducherry":["py","pondicherry"]}
STATE_MAP = {"US": {**{k: v for k, v in US_STATES.items()}, **{v: v for v in US_STATES.values()}}, "India": {}}
for full, abbrs in IN_STATES.items():
    canon = full.replace(" ", "")
    for k in [full, *abbrs]: STATE_MAP["India"][k] = canon

ADDR_ABBR = {"st":"street","str":"street","rd":"road","ave":"avenue","av":"avenue","blvd":"boulevard",
 "bd":"boulevard","bld":"boulevard","dr":"drive","ln":"lane","ct":"court","pl":"place","sq":"square",
 "hwy":"highway","pkwy":"parkway","cir":"circle","trl":"trail","ter":"terrace","ste":"suite",
 "apt":"apartment","fl":"floor","flr":"floor","bldg":"building","n":"north","s":"south","e":"east",
 "w":"west","ne":"northeast","nw":"northwest","se":"southeast","sw":"southwest","mt":"mount",
 "ft":"fort","hts":"heights","nr":"near","opp":"opposite","sec":"sector","pkt":"pocket","ngr":"nagar",
 "mkt":"market","clny":"colony","r":"rue","ch":"chemin","imp":"impasse","all":"allee","rte":"route",
 "fg":"faubourg","crs":"cours","qu":"quai","sq":"square","res":"residence"}
ADDR_DROP = {"no","number","null","none","nan","na","po","box"}
NAME_ABBR = {"pvt":"private","prvt":"private","ltd":"limited","ltda":"limited","corp":"corporation",
 "inc":"incorporated","incorp":"incorporated","co":"company","cie":"company","intl":"international",
 "mfg":"manufacturing","svcs":"services","svc":"services","bros":"brothers","assoc":"associates",
 "assn":"association","dept":"department","ctr":"center","centre":"center","mgmt":"management",
 "tech":"technology","et":"and","n":"and"}
LEGAL = {"private","limited","llc","llp","lp","incorporated","corporation","company","plc","pllc","pc",
 "sarl","sas","sasu","eurl","sa","sci","snc","scop","the","and","of","de","du","des","la","le","les",
 "d","l","dr","smt","shri","sri","mr","mrs","ms","m","s","opc"}
OCR = str.maketrans("0134568", "oleasgb")
DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9-]{3,})\.(?:co\.in|com|in|net|org|fr|biz|info|co|us)\b")
ALIAS = re.compile(r"\b(?:a/k/a|aka|d/b/a|dba|t/a|f/k/a|fka|formerly)\b")

def translit(text, tmap):
    """Transliterate non-Latin tokens (Devanagari, Telugu, ...) and map them with the learned table."""
    if not NONLATIN.search(text): return text, False
    out = []
    for tok in text.split():
        if NONLATIN.search(tok):
            t = re.sub(r"[^a-z0-9]", "", anyascii(tok).lower())
            out.append(tmap.get(t, t))
        else:
            out.append(tok)
    return " ".join(out), True

def fix_ocr(tok):
    """Repair OCR-style digit-for-letter substitutions inside a mixed token (`6reen` -> `green`).
    Pure-alpha and pure-digit tokens are left alone, so house numbers survive untouched."""
    if tok.isalpha() or tok.isdigit(): return tok
    letters = sum(ch.isalpha() for ch in tok)
    return tok.translate(OCR) if letters >= 2 else tok

def norm_name(raw, tmap):
    """Normalise a business name.

    Transliterates non-Latin script, pulls the label out of a web domain, splits `a/k/a` and
    `d/b/a` aliases, repairs OCR digits and expands legal/business abbreviations.

    Returns `(full, core, alias, nonlatin)`: the full normalised token string, the *core* name
    (legal forms and honorifics removed), the alias parts joined by ` | `, and a flag saying
    whether the original text contained non-Latin script."""
    s, nonlat = translit(raw, tmap)
    s = anyascii(s).lower()
    s = DOMAIN.sub(r" \1 ", s)
    s = s.replace("&", " and ").replace("+", " ")
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", s)   # l.l.c. -> llc,  s.a.s -> sas
    s = re.sub(r"[^a-z0-9/ ]", " ", s)
    s = ALIAS.sub(" | ", s).replace("/", " ")
    parts, full = [], []
    for part in s.split("|"):
        toks = [NAME_ABBR.get(t, t) for t in (fix_ocr(t) for t in part.split())]
        if toks: parts.append(toks); full.extend(toks)
    cores = [[t for t in p if t not in LEGAL] or p for p in parts] or [[]]
    core = max(cores, key=len)
    alias = " | ".join(" ".join(c) for c in cores)
    return " ".join(full), " ".join(core), alias, nonlat

def norm_addr(raw, country, tmap):
    """Normalise an address: transliterate, strip `##` / `null` / `No.` / `PO Box` noise, expand
    street-type abbreviations, canonicalise the state through the country-keyed lookup, split
    digit-letter runs and drop leading zeros from numbers.

    A country with no entry in `STATE_MAP` (France) passes straight through, which keeps the
    country set open as the problem statement requires."""
    s, _ = translit(raw, tmap)
    s = anyascii(s).lower().replace("##", " ")
    smap = STATE_MAP.get(country, {})
    comps = []
    for c in s.split(","):
        c = re.sub(r"[^a-z0-9 ]", " ", c)
        c = " ".join(ADDR_ABBR.get(t, t) for t in c.split() if t not in ADDR_DROP)
        if not c: continue
        comps.append(smap.get(c, c))
    s = " ".join(comps)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)
    toks = [t.lstrip("0") or "0" if t.isdigit() else t for t in s.split()]
    return " ".join(toks)

print(norm_name("Orbiumbra+ a/k/a O 7 6reen Fitness, L.L.C.", {}))
print(norm_name("sadarn hospitality pvt. ltd. www.southernhosp.com", {}))
print(norm_addr("##0534 Conrad Dr, null, Kalispell, Montana", "US", {}))
print(norm_addr("No. 3-6-667/4/9 Flat No. 304, Somajiguda, Hyderabad, AP", "India", {}))
print(norm_addr("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", {}))

# %% [markdown]
# ### 3.1 Learn a transliteration token map from training pairs
# Many India records write the name (or the state) in Devanagari or Telugu. `anyascii` gives rough Latin text (`प्राइवेट` → `praivet`, `महाराष्ट्र` → `mharastr`). We learn `praivet → private` by counting how often each transliterated token co-occurs with each Latin token of the matched S1 record. We keep the pair with the best co-occurrence × string-similarity score.
#
# The map is learned from **all** non-Latin matched training records, not just the dev universe. To avoid leaking proper names into validation, a mapping is kept only if the token appears across **≥ 5 different S1 entities**. That keeps generic words (`private`, `limited`, `technologies`, state names) and drops one-off brand names.

# %%
def learn_translit_map(rec_df, s1_df, min_entities=5):
    """Learn a transliteration token map from the training labels only (no external resource).

    For every non-Latin token of a matched record, count co-occurrence with the Latin tokens of
    its S1 entity and keep the target with the best `co-occurrence x string-similarity` score.
    A mapping survives only when its token appears across >= `min_entities` distinct S1
    entities: that keeps generic words (`private`, `limited`, state names) and drops one-off
    brand names, so no proper noun leaks from the labels into validation.

    Returns `{transliterated_token: english_token}`."""
    s1_text = dict(zip(s1_df.iid.to_numpy(), (s1_df.business_name + " " + s1_df.business_address).to_numpy()))
    tok_ents = defaultdict(set); co = defaultdict(Counter)
    for name, addr, s in zip(rec_df.business_name, rec_df.business_address, rec_df.true_s1):
        text = name + " " + addr
        if s < 0 or not NONLATIN.search(text): continue
        src = {re.sub(r"[^a-z0-9]", "", anyascii(t).lower()) for t in text.split() if NONLATIN.search(t)}
        src.discard("")
        tgt = set(re.sub(r"[^a-z0-9 ]", " ", anyascii(s1_text.get(s, "")).lower()).split())
        for t in src:
            tok_ents[t].add(s); co[t].update(tgt)
    tmap = {}
    for t, ents in tok_ents.items():
        n = len(ents)
        if n < min_entities: continue
        best, best_sc = None, 0.0
        for cand, c in co[t].most_common(15):
            if c < 0.5 * n: break
            sc = (c / n) * fuzz.ratio(t, cand) / 100
            if sc > best_sc: best, best_sc = cand, sc
        if best and best_sc >= 0.35 and best != t: tmap[t] = best
    return tmap

t0 = time.time()
TMAP = learn_translit_map(nonlatin_train, s1_all)
del nonlatin_train; gc.collect()
print(f"learned {len(TMAP):,} transliteration mappings in {time.time()-t0:.0f}s; examples:")
print(dict(list(sorted(TMAP.items(), key=lambda kv: -len(kv[0])))[:25]))

# %% [markdown]
# ### 3.2 Parallel normalisation + hashed token extraction
# One pass per record produces the normalised strings used by the feature stage and three hashed token-id lists:
# * **name** — core-name tokens, adjacent-token bigrams, and the concatenated core (`ganganthsummit`, which catches web-domain names),
# * **addr** — address word tokens (≥ 3 chars) plus number→next-word bigrams (`125_polk`),
# * **num** — the set of numbers in the address.
#
# Every token is prefixed with the country and field before hashing (crc32 → 2^24 buckets). Country therefore acts as a hard block without hard-coding any country list.

# %%
HMASK = (1 << HASH_BITS) - 1

def _h(s):
    """Hash a country-and-field-prefixed token into one of 2^HASH_BITS buckets (crc32)."""
    return zlib.crc32(s.encode()) & HMASK

def process_rows(names, addrs, countries, tmap):
    """Normalise one block of records and extract their hashed blocking tokens.

    Runs inside a joblib worker. Returns the normalised string columns plus three token-id lists
    per record: *name* (core tokens, adjacent bigrams, the concatenated core, alias parts),
    *addr* (word tokens of >= 3 chars and number->next-word bigrams) and *num* (address
    numbers). Every token is prefixed with the country before hashing, so country acts as a
    hard block without any country list being hard-coded."""
    out_name, out_core, out_alias, out_addr, out_nl = [], [], [], [], []
    nid, aid, uid = [], [], []
    for nm, ad, c in zip(names, addrs, countries):
        full, core, alias, nl = norm_name(nm, tmap)
        a = norm_addr(ad, c, tmap)
        out_name.append(full); out_core.append(core); out_alias.append(alias); out_addr.append(a); out_nl.append(nl)
        ct = core.split()
        ntok = {f"{c}|n|{t}" for t in ct if len(t) >= 2}
        ntok |= {f"{c}|nb|{x}_{y}" for x, y in zip(ct, ct[1:])}
        cat = "".join(ct)
        if len(cat) >= 6: ntok.add(f"{c}|nc|{cat}")
        for p in alias.split(" | ")[:3]:                       # alias parts (a/k/a, d/b/a)
            ntok |= {f"{c}|n|{t}" for t in p.split() if len(t) >= 2}
        at = a.split()
        atok = {f"{c}|a|{t}" for t in at if len(t) >= 3 and not t.isdigit()}
        atok |= {f"{c}|ab|{x}_{y}" for x, y in zip(at, at[1:]) if x.isdigit() and not y.isdigit()}
        utok = {f"{c}|#|{t}" for t in at if t.isdigit()}
        nid.append(sorted({_h(t) for t in ntok})); aid.append(sorted({_h(t) for t in atok}))
        uid.append(sorted({_h(t) for t in utok}))
    return out_name, out_core, out_alias, out_addr, out_nl, nid, aid, uid

def _to_csr(lists, n_cols=1 << HASH_BITS):
    """Build a binary CSR matrix from per-row sorted token-id lists, with no per-cell Python loop."""
    indptr = np.zeros(len(lists) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(x) for x in lists])
    indices = np.fromiter((i for x in lists for i in x), dtype=np.int32, count=indptr[-1])
    return sp.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(lists), n_cols))

def normalise_frame(df, tmap, n_jobs=N_JOBS, block=20_000):
    """Returns a dict of normalised columns and binary CSR matrices (name, addr, num)."""
    blocks = [(df.business_name.to_numpy()[i:i+block], df.business_address.to_numpy()[i:i+block],
               df.country.to_numpy()[i:i+block]) for i in range(0, len(df), block)]
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(process_rows)(n, a, c, tmap) for n, a, c in blocks)
    cols = [sum((r[k] for r in res), []) for k in range(8)]
    return dict(name=np.array(cols[0], dtype=object), core=np.array(cols[1], dtype=object),
                alias=np.array(cols[2], dtype=object), addr=np.array(cols[3], dtype=object),
                nonlatin=np.array(cols[4], dtype=bool),
                Mn=_to_csr(cols[5]), Ma=_to_csr(cols[6]), Mu=_to_csr(cols[7]))

# %% [markdown]
# ## 4. Blocking index over Source 1
# IDF comes from the S1 document frequency. **For the dev universe the document frequencies come from the *full* training S1 (2.2M rows).** Otherwise the small dev index would keep tokens that `DF_CAP` removes at test scale (1.7M rows), and blocking would behave differently in validation than on test. Each field is TF-IDF weighted and l2-normalised, which gives the per-field cosines used later as features. The **blocking** vector concatenates the three fields with weights (name 0.6, address 0.3, numbers 0.1), drops tokens with df > `DF_CAP`, and is l2-normalised again.

# %%
FIELD_W = {"Mn": 0.6, "Ma": 0.3, "Mu": 0.1}

def l2norm(M):
    """Row-wise L2 normalisation of a sparse matrix; all-zero rows stay zero."""
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n).dot(M).tocsr()

def df_stats(s1_df, tmap, block=400_000):
    """Document frequency of every hashed token, per field (memory-light: strings are discarded)."""
    dfs = {f: np.zeros(1 << HASH_BITS, dtype=np.int32) for f in FIELD_W}
    for i in range(0, len(s1_df), block):
        N = normalise_frame(s1_df.iloc[i:i+block], tmap)
        for f in FIELD_W: dfs[f] += np.bincount(N[f].indices, minlength=1 << HASH_BITS).astype(np.int32)
    return dfs, len(s1_df)

class S1Index:
    def __init__(self, s1_df, tmap, stats=None):
        """Build the blocking index over Source 1.

        `stats` supplies document frequencies computed elsewhere. For the dev universe we pass the
        *full* training S1 frequencies, so `DF_CAP` drops the same tokens it drops at test scale
        and blocking behaves in validation as it does on the 1.7M-entity test set."""
        t0 = time.time()
        self.ids = s1_df.entity_id.to_numpy(); self.iid = s1_df.iid.to_numpy()
        self.N = normalise_frame(s1_df, tmap)
        dfs, n = stats if stats is not None else (
            {f: np.bincount(self.N[f].indices, minlength=1 << HASH_BITS) for f in FIELD_W}, len(s1_df))
        self.idf, self.cap_mask = {}, {}
        for f in FIELD_W:
            df = dfs[f]
            self.idf[f] = np.log((n + 1) / (df + 1)).astype(np.float32) + 1
            self.idf[f][df == 0] = 0
            self.cap_mask[f] = (df <= DF_CAP).astype(np.float32)
        self.F = {f: self.tfidf(self.N[f], f) for f in FIELD_W}
        self.B_T = self.block_vec(self.N).T.tocsr()
        # S1-side ambiguity: how many S1 rows share the exact same core name / address
        self.core_dup = pd.Series(self.N["core"]).map(pd.Series(self.N["core"]).value_counts()).to_numpy()
        self.addr_dup = pd.Series(self.N["addr"]).map(pd.Series(self.N["addr"]).value_counts()).to_numpy()
        print(f"S1 index: {len(s1_df):,} rows, blocking nnz {self.B_T.nnz:,}, built in {time.time()-t0:.0f}s")

    def tfidf(self, M, f):
        """TF-IDF weight field `f` of a binary matrix and L2-normalise it, giving the per-field cosines."""
        return l2norm(M.dot(sp.diags(self.idf[f])))

    def block_vec(self, N):
        """Build the blocking vector: weight each field by TF-IDF with over-frequent tokens masked
        out, L2-normalise, scale by the field weight (name .6 / address .3 / numbers .1), sum the
        three and L2-normalise again."""
        parts = [N[f].dot(sp.diags(self.idf[f] * self.cap_mask[f])) for f in FIELD_W]
        parts = [l2norm(P) * w for P, w in zip(parts, FIELD_W.values())]
        return l2norm(parts[0] + parts[1] + parts[2])

    def search(self, Q):
        """Top-K S1 candidates for each query row -> (q_row, s1_row, cos) arrays."""
        C = sp_matmul_topn(self.block_vec(Q), self.B_T, top_n=TOP_K, threshold=MIN_COS,
                           sort=True, n_threads=N_JOBS).tocoo()
        q, s, v = C.row, C.col, C.data
        best = np.zeros(Q["Mn"].shape[0], dtype=np.float32)
        np.maximum.at(best, q, v)
        keep = v >= REL_COS * best[q]
        return q[keep], s[keep], v[keep]

t0 = time.time()
train_stats = df_stats(s1_all, TMAP)
print(f"full-train df stats in {time.time()-t0:.0f}s")
dev_index = S1Index(dev_s1, TMAP, stats=train_stats)
del train_stats; gc.collect()

# %% [markdown]
# ## 5. Candidate generation + pair features
# Each S2/S3 record looks up its top-K S1 candidates. The features for each (record, S1) pair come from rapidfuzz's C++ `cpdist`, which is element-wise and multi-threaded, plus sparse row-wise dot products. No Python loop over pairs is needed.

# %%
def rowdot(A, B, ia, ib):
    """Row-wise dot products A[ia] . B[ib] for aligned index arrays."""
    return np.asarray(A[ia].multiply(B[ib]).sum(1)).ravel().astype(np.float32)

def skeleton(arr):
    """Consonant skeleton of each name: vowels removed and repeated letters collapsed. Robust to
    transliteration spelling drift and to vowel typos."""
    return np.array([re.sub(r"(.)\1+", r"\1", re.sub(r"[aeiouy ]", "", x)) for x in arr], dtype=object)

def pair_features(Q, qsrc, idx, q, s, cos):
    """Build the feature matrix for the candidate pairs `(q[i], s[i])` with blocking cosine `cos[i]`.

    Four groups of features:
      1. rapidfuzz string similarities on the name, core name, concatenated core, consonant
         skeleton, aliases and address, all through the C++ element-wise `process.cpdist`;
      2. per-field TF-IDF cosines and set overlaps (numbers, name tokens, address tokens),
         computed as sparse row-wise dot products;
      3. *competition* features describing where this candidate sits in the record's own
         candidate list (rank, gap and ratio to the best candidate, list size);
      4. S1-side ambiguity counts and record-level flags (source, empty address, non-Latin).

    No Python loop runs over pairs. Returns a DataFrame with one row per candidate pair."""
    S = idx.N
    f = {}
    f["cos"] = cos.astype(np.float32)
    d = pd.DataFrame({"q": q, "cos": cos})
    grp = d.groupby("q")["cos"]
    f["cos_rank"] = grp.rank(ascending=False, method="first").to_numpy(np.float32)
    best = grp.transform("max").to_numpy(); f["cos_gap"] = (best - cos).astype(np.float32)
    f["cos_ratio"] = (cos / best).astype(np.float32)
    f["n_cand"] = grp.transform("size").to_numpy(np.float32)

    qn, sn = Q["name"][q], S["name"][s]
    qc, sc = Q["core"][q], S["core"][s]
    qa, sa = Q["addr"][q], S["addr"][s]
    W = dict(workers=N_JOBS, dtype=np.float32)
    f["n_ratio"] = process.cpdist(qn, sn, scorer=fuzz.ratio, **W)
    f["n_tset"] = process.cpdist(qn, sn, scorer=fuzz.token_set_ratio, **W)
    f["c_ratio"] = process.cpdist(qc, sc, scorer=fuzz.ratio, **W)
    f["c_tsort"] = process.cpdist(qc, sc, scorer=fuzz.token_sort_ratio, **W)
    f["c_tset"] = process.cpdist(qc, sc, scorer=fuzz.token_set_ratio, **W)
    f["c_partial"] = process.cpdist(qc, sc, scorer=fuzz.partial_ratio, **W)
    f["c_jw"] = process.cpdist(qc, sc, scorer=JaroWinkler.normalized_similarity, **W)
    qcat = np.array([x.replace(" ", "") for x in qc], dtype=object)
    scat = np.array([x.replace(" ", "") for x in sc], dtype=object)
    f["cat_ratio"] = process.cpdist(qcat, scat, scorer=fuzz.ratio, **W)
    f["cat_partial"] = process.cpdist(qcat, scat, scorer=fuzz.partial_ratio, **W)
    f["skel_ratio"] = process.cpdist(skeleton(qc), skeleton(sc), scorer=fuzz.ratio, **W)
    # alias: best token_set over "a/k/a" parts of the record
    qal = Q["alias"][q]
    has_alias = np.array(["|" in x for x in qal])
    f["alias_best"] = f["c_tset"].copy()
    if has_alias.any():
        ii = np.where(has_alias)[0]
        best_al = np.zeros(len(ii), dtype=np.float32)
        for k in range(3):
            part = np.array([(x.split(" | ") + [""] * 3)[k] for x in qal[ii]], dtype=object)
            best_al = np.maximum(best_al, process.cpdist(part, sc[ii], scorer=fuzz.token_set_ratio, **W))
        f["alias_best"][ii] = np.maximum(f["alias_best"][ii], best_al)
    f["a_ratio"] = process.cpdist(qa, sa, scorer=fuzz.ratio, **W)
    f["a_tset"] = process.cpdist(qa, sa, scorer=fuzz.token_set_ratio, **W)
    f["a_tsort"] = process.cpdist(qa, sa, scorer=fuzz.token_sort_ratio, **W)
    f["a_partial"] = process.cpdist(qa, sa, scorer=fuzz.partial_token_set_ratio, **W)

    # field-level TF-IDF cosines
    for fld, nm in (("Mn", "name_cos"), ("Ma", "addr_cos"), ("Mu", "num_cos")):
        f[nm] = rowdot(Q["_tf"][fld], idx.F[fld], q, s)
    # set overlaps on numbers / name tokens (binary matrices)
    for fld, nm in (("Mu", "num"), ("Mn", "ntok"), ("Ma", "atok")):
        inter = rowdot(Q[fld], S[fld], q, s)
        lq = np.diff(Q[fld].indptr)[q].astype(np.float32); ls = np.diff(S[fld].indptr)[s].astype(np.float32)
        f[f"{nm}_inter"] = inter
        f[f"{nm}_jacc"] = inter / np.maximum(lq + ls - inter, 1)
        f[f"{nm}_q_cov"] = inter / np.maximum(lq, 1)
        if nm == "num": f["num_q_len"], f["num_s_len"] = lq, ls
    f["q_addr_empty"] = (np.fromiter(map(len, qa), np.int32, len(qa)) == 0).astype(np.float32)
    f["q_nonlatin"] = Q["nonlatin"][q].astype(np.float32)
    f["q_src"] = qsrc[q].astype(np.float32)
    f["len_q_core"] = np.fromiter(map(len, qc), np.float32, len(qc))
    f["len_s_core"] = np.fromiter(map(len, sc), np.float32, len(sc))
    f["s_core_dup"] = idx.core_dup[s].astype(np.float32)
    f["s_addr_dup"] = idx.addr_dup[s].astype(np.float32)
    F = pd.DataFrame(f)
    # competition inside the record's candidate list
    for col in ("c_tset", "a_tset", "name_cos"):
        g = F.groupby(q)[col]
        F[f"{col}_gap"] = g.transform("max") - F[col]
    return F

def candidates_and_features(rec_df, idx, tmap):
    """Normalise a chunk of S2/S3 records, block against the S1 index, compute pair features."""
    Q = normalise_frame(rec_df, tmap)
    Q["_tf"] = {f: idx.tfidf(Q[f], f) for f in FIELD_W}
    q, s, cos = idx.search(Q)
    qsrc = rec_df.entity_id.str.slice(1, 2).astype(np.int8).to_numpy()
    F = pair_features(Q, qsrc, idx, q, s, cos)
    return q, s, F

# %%
t0 = time.time()
parts = []
for i in range(0, len(dev_rec), CHUNK):
    ch = dev_rec.iloc[i:i+CHUNK]
    q, s, F = candidates_and_features(ch, dev_index, TMAP)
    F["q_iid"] = ch.iid.to_numpy()[q]; F["s_iid"] = dev_index.iid[s]
    F["y"] = (ch.true_s1.to_numpy()[q] == F["s_iid"].to_numpy()).astype(np.int8)
    F["grp"] = np.where(ch.true_s1.to_numpy()[q] >= 0, ch.true_s1.to_numpy()[q], ch.iid.to_numpy()[q])
    parts.append(F)
    print(f"  chunk {i//CHUNK}: {len(ch):,} records -> {len(F):,} pairs  ({time.time()-t0:.0f}s)")
pairs = pd.concat(parts, ignore_index=True); del parts; gc.collect()

n_matched = int((dev_rec.true_s1 >= 0).sum())
found = int(pairs.y.sum())
print(f"\ncandidate pairs: {len(pairs):,}  ({len(pairs)/len(dev_rec):.2f} per record)")
print(f"blocking recall (pair level): {found/n_matched:.4f}")
print(f"reduction ratio vs. full cross product: {1 - len(pairs)/(len(dev_rec)*len(dev_s1)):.6f}")

# %% [markdown]
# ### 5.1 Blocking quality: recall ceiling in F0.5 terms
# The score we would get with a perfect matcher applied to these candidates.

# %%
dev_s1_ids = dev_s1.iid.tolist()
oracle = defaultdict(list)
for qi, si in zip(pairs.q_iid.to_numpy()[pairs.y.to_numpy() == 1], pairs.s_iid.to_numpy()[pairs.y.to_numpy() == 1]):
    oracle[si].append(qi)
print(f"F0.5 ceiling given candidates: {fbeta_macro(oracle, dev_truth, dev_s1_ids):.4f}")
print(f"F0.5 of predicting nothing:   {fbeta_macro({}, dev_truth, dev_s1_ids):.4f}")

# %% [markdown]
# ## 6. LightGBM matcher, out-of-fold evaluation
# GroupKFold keeps every record of the same real-world entity in the same fold, so the scores are honest out-of-fold estimates.

# %%
FEATS = [c for c in pairs.columns if c not in ("q_iid", "s_iid", "y", "grp")]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=N_JOBS, verbose=-1, seed=SEED)
N_ROUNDS = 600

oof = np.zeros(len(pairs), dtype=np.float32); best_iters = []
X = pairs[FEATS].to_numpy(np.float32); y = pairs.y.to_numpy()
for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, pairs.grp.to_numpy())):
    t0 = time.time()
    m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=FEATS), N_ROUNDS,
                  valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va] = m.predict(X[va], num_iteration=m.best_iteration); best_iters.append(m.best_iteration)
    print(f"fold {k}: best_iter={m.best_iteration}  ({time.time()-t0:.0f}s)")
pairs["p"] = oof

# %% [markdown]
# ### 6.1 Decision rule & threshold tuning
# Each S2/S3 record belongs to **at most one** S1 entity, so we keep only the record's arg-max candidate. It becomes a match if `p ≥ τ`. τ is chosen to maximise macro F0.5 on the out-of-fold predictions.

# %%
def assign(pairs_df, tau):
    """Apply the decision rule: assign each S2/S3 record to its single best-scoring S1 candidate,
    and only when that score clears `tau`. This uses the EDA fact that a record belongs to at
    most one S1 entity, which removes most multi-match false merges by construction.

    Returns `{s1_iid: [record_iid, ...]}`."""
    top = pairs_df.loc[pairs_df.groupby("q_iid")["p"].idxmax()]
    top = top[top.p >= tau]
    pred = defaultdict(list)
    for qi, si in zip(top.q_iid.to_numpy(), top.s_iid.to_numpy()): pred[si].append(qi)
    return pred

top_all = pairs.loc[pairs.groupby("q_iid")["p"].idxmax(), ["q_iid", "s_iid", "p"]]
res = []
for tau in np.round(np.arange(0.2, 0.96, 0.05), 2):
    res.append((tau, fbeta_macro(assign(top_all, tau), dev_truth, dev_s1_ids)))
res = pd.DataFrame(res, columns=["tau", "F0.5"]); print(res.to_string(index=False))
TAU = float(res.loc[res["F0.5"].idxmax(), "tau"])
print(f"\nbest tau = {TAU}  ->  OOF macro F0.5 = {res['F0.5'].max():.4f}")

# %%
# per-country breakdown and error types
pred = assign(top_all, TAU)
for c in dev_s1.country.unique():
    ids = dev_s1.iid[dev_s1.country == c].tolist()
    print(f"{c:8s} F0.5 = {fbeta_macro(pred, dev_truth, ids):.4f}  (n={len(ids):,})")
sing = [s for s in dev_s1_ids if s not in dev_truth]
print(f"singletons correctly left empty: {np.mean([s not in pred for s in sing]):.4f}")

imp = pd.Series(m.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print(imp.head(20).round(0).to_string())

# %%
# a few false positives for inspection
rec_lookup = dev_rec.set_index("iid")[["business_name", "business_address"]]
s1_lookup = dev_s1.set_index("iid")[["business_name", "business_address"]]
fp = top_all[(top_all.p >= TAU)].merge(pairs.loc[pairs.y == 0, ["q_iid", "s_iid"]])
for _, r in fp.sample(min(8, len(fp)), random_state=0).iterrows():
    a, b = s1_lookup.loc[r.s_iid], rec_lookup.loc[r.q_iid]
    print(f"p={r.p:.2f}\n  S1 : {a.business_name} | {a.business_address}\n  rec: {b.business_name} | {b.business_address}")

# %% [markdown]
# ## 7. Final model on all dev pairs

# %%
best_iter = int(np.mean(best_iters) * 1.1) or N_ROUNDS
final_model = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=FEATS), best_iter)
final_model.save_model(f"{OUT_DIR}/lgb_matcher.txt")
print(f"final model: {best_iter} rounds, tau = {TAU}")
# free training-side memory before the (large) test pass
del X, y, pairs, top_all, dev_index, s1_all, rec2s1, dev_rec, rec_lookup, s1_lookup, fp; gc.collect()

# %% [markdown]
# ## 8. Test inference (streaming)
# The S1 test index is built once. S2 and S3 are then streamed in chunks of `CHUNK` records, and for each chunk we block, compute features, score, and take the arg-max with the threshold. Only per-S1 id lists stay in memory.
#
# `candidate_pairs.tsv` holds exactly the pairs the model scored. `matching_results.tsv` is a subset of it by construction.

# %%
if RUN_TEST:
    t_all = time.time()
    test_s1 = read_tsv(f"{DATA_DIR}/test/test_source1.tsv")
    test_s1["iid"] = id2int(test_s1.entity_id)
    print("test S1:", len(test_s1), test_s1.country.value_counts().to_dict())
    test_index = S1Index(test_s1, TMAP)
    cand_lists = [[] for _ in range(len(test_s1))]
    match_lists = [[] for _ in range(len(test_s1))]
    n_rec = n_pairs = n_match = 0
    for src in (2, 3):
        for ch in read_tsv(f"{DATA_DIR}/test/test_source{src}.tsv", chunksize=CHUNK):
            ch = ch.reset_index(drop=True)
            q, s, F = candidates_and_features(ch, test_index, TMAP)
            p = final_model.predict(F[FEATS].to_numpy(np.float32), num_threads=N_JOBS)
            rid = ch.entity_id.to_numpy()
            for qi, si in zip(q, s): cand_lists[si].append(rid[qi])
            d = pd.DataFrame({"q": q, "s": s, "p": p})
            top = d.loc[d.groupby("q")["p"].idxmax()]
            top = top[top.p >= TAU]
            for qi, si in zip(top.q.to_numpy(), top.s.to_numpy()): match_lists[si].append(rid[qi])
            n_rec += len(ch); n_pairs += len(d); n_match += len(top)
            print(f"  S{src}: {n_rec:,} records, {n_pairs:,} pairs, {n_match:,} matches ({time.time()-t_all:.0f}s)")
            del F, d, top; gc.collect()

    def write_lists(path, col, lists):
        """Write one output TSV: one row per test S1 entity, ids comma-joined.

        `dict.fromkeys` de-duplicates while preserving order, which satisfies the
        no-duplicate-ids-in-a-list rule. Entities with nothing to write get an empty second
        field, exactly as the output format requires for singletons."""
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for sid, L in zip(test_index.ids, lists):
                fh.write(f"{sid}\t{','.join(dict.fromkeys(L))}\n")

    write_lists(f"{OUT_DIR}/matching_results.tsv", "matched_entity_ids", match_lists)
    write_lists(f"{OUT_DIR}/candidate_pairs.tsv", "candidate_entity_ids", cand_lists)
    n_empty = sum(1 for L in match_lists if not L)
    print(f"done in {time.time()-t_all:.0f}s — {n_empty/len(match_lists):.3f} of S1 predicted as singletons")

# %% [markdown]
# ## 9. Validate the submission files

# %%
VALIDATOR = os.environ.get("ER_VALIDATOR", "utils/validate_submission.py")
if RUN_TEST and os.path.isfile(VALIDATOR):
    r = subprocess.run([sys.executable, VALIDATOR,
                        "--matching", f"{OUT_DIR}/matching_results.tsv",
                        "--candidate", f"{OUT_DIR}/candidate_pairs.tsv",
                        "--test-dir", f"{DATA_DIR}/test"], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])

# %% [markdown]
# ## 10. Ideas for further gains
# * **Bigger dev universe** (`DEV_FRAC` 0.3–1.0): more training pairs and a more faithful density.
# * **S1-side consistency**: second-stage features such as "how many other records the model assigned to this S1" and their mutual similarity. Matched records of one entity agree with each other, so they can vouch for each other.
# * **Char n-gram blocking pass** for records whose name *and* address are heavily typo'd (these are the misses from Section 5.1).
# * **Cross-encoder re-ranking** of the uncertain band (0.3 < p < 0.8) with a small Apache/MIT model, e.g. `multilingual-e5-small` or `bge-m3` (both MIT), fine-tuned on training pairs. CPU inference is only feasible for that narrow band.
# * **France**: no labels exist. Check the predicted singleton rate and score distribution against US/India to confirm that τ transfers.
