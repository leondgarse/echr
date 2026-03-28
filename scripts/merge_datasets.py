"""
Merge a newly preprocessed CSV into the existing enlarged processed.csv.
Deduplicates on item_id. Prints before/after stats.

Usage:
  python scripts/merge_datasets.py \
    --existing  data_v1/processed/processed.csv \
    --new       data_new/processed/processed.csv \
    --output    data_v1/processed/processed.csv
"""

import argparse
import pandas as pd


def main(args):
    old = pd.read_csv(args.existing)
    new = pd.read_csv(args.new)

    print(f"Existing dataset : {len(old)} cases  (violations: {old['label'].sum()}, "
          f"non-violations: {(old['label']==0).sum()})")
    print(f"New batch        : {len(new)} cases  (violations: {new['label'].sum()}, "
          f"non-violations: {(new['label']==0).sum()})")

    # Filter new to Art.6 only (same scope as existing enlarged dataset)
    art6_mask = (
        new['violation_articles'].fillna('').str.contains(r'(?:^|;)6(?:$|;|-)', regex=True) |
        new['nonviolation_articles'].fillna('').str.contains(r'(?:^|;)6(?:$|;|-)', regex=True)
    )
    new_art6 = new[art6_mask].copy()
    print(f"New batch Art.6  : {len(new_art6)} cases")

    merged = pd.concat([old, new_art6], ignore_index=True)
    before = len(merged)
    merged.drop_duplicates(subset=['item_id'], inplace=True)
    dupes = before - len(merged)
    if dupes:
        print(f"Dropped {dupes} duplicate item_ids")

    v = merged['label'].sum()
    nv = (merged['label'] == 0).sum()
    print(f"\nMerged dataset   : {len(merged)} cases  "
          f"(violations: {v} [{v/len(merged)*100:.1f}%], "
          f"non-violations: {nv} [{nv/len(merged)*100:.1f}%])")

    print("\nCountry breakdown:")
    for country, grp in merged.groupby('respondent'):
        v_c = grp['label'].sum()
        nv_c = (grp['label'] == 0).sum()
        print(f"  {country:6s}: {len(grp):4d}  V={v_c}  NV={nv_c}")

    merged.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True)
    parser.add_argument("--new",      required=True)
    parser.add_argument("--output",   required=True)
    main(parser.parse_args())
