#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert features CSV → JSON for sonar payload builder")
    parser.add_argument("--csv", required=True, help="Input CSV file (e.g. data/triaged_15_features2.csv)")
    parser.add_argument("--json", required=True, help="Output JSON file (e.g. data/triaged_15_features2.json)")
    args = parser.parse_args()

    features = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            features.append({
                "feature_id": f"feature_{idx}",  # auto-generated ID
                "feature_name": row.get("feature_name", "").strip(),
                "feature_description": row.get("feature_description", "").strip(),
            })

    Path(args.json).write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(features)} features to {args.json}")

if __name__ == "__main__":
    main()
