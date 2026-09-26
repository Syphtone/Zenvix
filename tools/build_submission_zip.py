#!/usr/bin/env python3
"""Build the Amazon ML Challenge 2026 submission zip to spec, then verify it.

The archive must contain exactly these members at its root:

    output/matching_results.tsv        final matches (also the leaderboard upload)
    output/candidate_pairs.tsv         the blocking candidate set fed to the model
    code/business_entity_resolution/src/run_pipeline.py
    code/business_entity_resolution/src/business_entity_resolution.ipynb
    code/business_entity_resolution/README.md
    code/business_entity_resolution/requirements.txt
    Documentation_template.md          methodology write-up

The two TSVs live outside this repository (``candidate_pairs.tsv`` is 833 MB and
GitHub rejects files over 100 MB), so their paths are passed in. Everything else is
read from the repository itself.

Usage, from the repository root::

    python tools/build_submission_zip.py \
        --matching   path/to/matching_results.tsv \
        --candidates path/to/candidate_pairs.tsv \
        --out        Zenvix_submission.zip

Nothing beyond the seven members above is packed, so stray notebooks, caches or
editor files can never leak into the submission.
"""
import argparse
import hashlib
import os
import sys
import time
import zipfile

# Repository-relative source -> path inside the archive.
REPO_MEMBERS = {
    "Documentation_template.md":
        "Documentation_template.md",
    "code/business_entity_resolution/README.md":
        "code/business_entity_resolution/README.md",
    "code/business_entity_resolution/requirements.txt":
        "code/business_entity_resolution/requirements.txt",
    "code/business_entity_resolution/src/run_pipeline.py":
        "code/business_entity_resolution/src/run_pipeline.py",
    "code/business_entity_resolution/src/business_entity_resolution.ipynb":
        "code/business_entity_resolution/src/business_entity_resolution.ipynb",
}


def sha256(path, block=1 << 24):
    """Content hash of a file, read in blocks so an 833 MB TSV never lands in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path):
    """Return (data_rows, total_ids, empty_rows) for an output TSV, streaming it.

    The header is skipped. ``total_ids`` counts comma-separated entity ids across
    every row, and ``empty_rows`` counts entities predicted as singletons (or, in
    ``candidate_pairs.tsv``, entities for which blocking found nothing).
    """
    rows = ids = empty = 0
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            rows += 1
            _, _, val = line.rstrip("\n").partition("\t")
            if val:
                ids += val.count(",") + 1
            else:
                empty += 1
    return rows, ids, empty


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, help="path to matching_results.tsv")
    ap.add_argument("--candidates", required=True, help="path to candidate_pairs.tsv")
    ap.add_argument("--out", default="Zenvix_submission.zip", help="archive to write")
    ap.add_argument("--repo", default=".", help="repository root (default: cwd)")
    ap.add_argument("--stats", action="store_true",
                    help="also report row counts and SHA-256 of the two TSVs")
    args = ap.parse_args()

    # Resolve every member up front, so a missing file fails before we write anything.
    members = [(args.matching, "output/matching_results.tsv"),
               (args.candidates, "output/candidate_pairs.tsv")]
    for rel, arc in REPO_MEMBERS.items():
        members.append((os.path.join(args.repo, rel), arc))

    missing = [src for src, _ in members if not os.path.isfile(src)]
    if missing:
        sys.exit("ERROR: missing required file(s):\n  " + "\n  ".join(missing))

    if os.path.exists(args.out):
        os.remove(args.out)

    print("packing %d members -> %s" % (len(members), args.out))
    t0 = time.time()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for src, arc in sorted(members, key=lambda m: m[1]):
            print("  + %-62s %9.1f MB" % (arc, os.path.getsize(src) / 1e6), flush=True)
            z.write(src, arc)
    print("packed in %.0fs" % (time.time() - t0))

    # Verify: every member present, no corrupt entry, and the two TSVs byte-identical
    # to the sources (CRC32 is stored per member, so this needs no decompression).
    with zipfile.ZipFile(args.out) as z:
        bad = z.testzip()
        if bad is not None:
            sys.exit("ERROR: corrupt member in archive: %s" % bad)
        names = set(z.namelist())
        expected = {arc for _, arc in members}
        if names != expected:
            sys.exit("ERROR: archive contents differ from spec\n"
                     "  unexpected: %s\n  missing: %s"
                     % (sorted(names - expected), sorted(expected - names)))
        raw = sum(i.file_size for i in z.infolist())

    print("\nARCHIVE OK - %d members, exactly to spec" % len(expected))
    print("  uncompressed : %8.1f MB" % (raw / 1e6))
    print("  archive      : %8.1f MB" % (os.path.getsize(args.out) / 1e6))

    if args.stats:
        print("\nOutput statistics (compare against output/MANIFEST.md):")
        for label, path in (("matching_results.tsv", args.matching),
                            ("candidate_pairs.tsv", args.candidates)):
            rows, ids, empty = count_rows(path)
            print("  %-22s rows=%-10s ids=%-12s empty=%-9s (%.3f%%)"
                  % (label, f"{rows:,}", f"{ids:,}", f"{empty:,}", 100 * empty / max(rows, 1)))
            print("  %-22s sha256=%s" % ("", sha256(path)))
        print("  %-22s sha256=%s" % (os.path.basename(args.out), sha256(args.out)))

    print("\nReminder: upload output/matching_results.tsv to the Portal for the "
          "leaderboard; the zip is the separate final submission package.")


if __name__ == "__main__":
    main()
