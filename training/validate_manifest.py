#!/usr/bin/env python3
"""Fail-closed validator for a JSONL consent manifest."""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

SHA256 = re.compile(r"^[a-f0-9]{64}$")
ALLOWED_LICENSES = {"performer-release", "commercial-dataset", "synthetic"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def validate(path: Path):
    errors, subjects, hashes = [], defaultdict(set), {}
    records = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip(): continue
        try: record = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc}"); continue
        records.append(record); prefix = f"line {number}"
        for field in ("subject_id", "consent_id", "adult_confirmed", "license", "assets", "split"):
            if field not in record: errors.append(f"{prefix}: missing {field}")
        if record.get("adult_confirmed") is not True: errors.append(f"{prefix}: adult confirmation required")
        if record.get("revoked", False): errors.append(f"{prefix}: revoked subject must not enter training")
        if record.get("license") not in ALLOWED_LICENSES: errors.append(f"{prefix}: invalid license")
        split = record.get("split")
        if split not in ALLOWED_SPLITS: errors.append(f"{prefix}: invalid split")
        sid = record.get("subject_id")
        if sid and split: subjects[sid].add(split)
        for asset in record.get("assets", []):
            digest = asset.get("sha256", "")
            if not SHA256.fullmatch(digest): errors.append(f"{prefix}: invalid asset sha256")
            elif digest in hashes and hashes[digest] != sid: errors.append(f"{prefix}: asset hash shared by subjects {hashes[digest]} and {sid}")
            else: hashes[digest] = sid
    for sid, splits in subjects.items():
        if len(splits) > 1: errors.append(f"subject {sid} leaks across splits: {sorted(splits)}")
    return records, errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument("manifest",type=Path);args=parser.parse_args()
    records, errors=validate(args.manifest)
    if errors:
        print("MANIFEST REJECTED", file=sys.stderr)
        print("\n".join(f"- {e}" for e in errors), file=sys.stderr);raise SystemExit(1)
    print(f"MANIFEST ACCEPTED: {len(records)} records")

if __name__ == "__main__": main()
