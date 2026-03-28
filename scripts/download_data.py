
import argparse
from echr_extractor import get_echr_extra, get_echr
from echr_extractor.ECHR_html_downloader import download_full_text_main
import os
import pandas as pd
import json
import re

TARGET_ARTICLES = ['3', '5', '6', '8']

def has_article_violation(row, articles=TARGET_ARTICLES):
    """Check if the case has a violation of any target article."""
    v = str(row.get('violation', 'nan')).lower()
    if v in ['nan', 'false', '', '0']:
        return False
    for art in articles:
        # Match exact article number (e.g., '3' but not '13', '3' matches '3;5' or '5;3')
        if re.search(rf'(?:^|;){art}(?:$|;|-)', v):
            return True
    return False

def has_article_nonviolation(row, articles=TARGET_ARTICLES):
    """Check if the case has a non-violation of any target article."""
    nv = str(row.get('nonviolation', 'nan')).lower()
    if nv in ['nan', 'false', '', '0']:
        return False
    for art in articles:
        if re.search(rf'(?:^|;){art}(?:$|;|-)', nv):
            return True
    return False

def fetch_batch(query, label_name, count, fetch_multiplier):
    print(f"--- Fetching {label_name} candidates (target={count}, fetching={count * fetch_multiplier}) ---")
    df = get_echr(
        count=count * fetch_multiplier,
        language=['ENG'],
        verbose=True,
        query_payload=query,
        save_file='n'
    )
    return df

def download_for_country(country_code, per_label_count, fetch_multiplier, articles):
    """Download balanced violation/non-violation cases for a single country, filtered by article."""
    print(f"\n{'='*60}")
    print(f"Downloading for country: {country_code} ({per_label_count} per label)")
    print(f"Target articles: {articles}")
    print(f"{'='*60}")

    # Build article-specific HUDOC queries
    # violation:"3" OR violation:"5" OR violation:"6" OR violation:"8"
    article_v_filter = " OR ".join([f'(violation:"{a}")' for a in articles])
    article_nv_filter = " OR ".join([f'(nonviolation:"{a}")' for a in articles])

    query_v = (
        f'(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") '
        f'AND ({article_v_filter}) AND (respondent:"{country_code}") AND (languageisocode:"ENG")'
    )
    query_nv = (
        f'(contentsitename=ECHR) AND (documentcollectionid2:"JUDGMENTS") '
        f'AND ({article_nv_filter}) AND (respondent:"{country_code}") AND (languageisocode:"ENG")'
    )

    df_v = fetch_batch(query_v, f"{country_code} VIOLATION (Art {','.join(articles)})", per_label_count, fetch_multiplier)
    df_nv = fetch_batch(query_nv, f"{country_code} NON-VIOLATION (Art {','.join(articles)})", per_label_count, fetch_multiplier)

    if df_v is False: df_v = pd.DataFrame()
    if df_nv is False: df_nv = pd.DataFrame()

    print(f"  Retrieved {len(df_v)} violation candidates and {len(df_nv)} non-violation candidates.")

    # Filter: ensure the case actually has the target article in the right field
    df_v_filtered = df_v[df_v.apply(has_article_violation, axis=1)].copy()
    df_v_filtered['download_label'] = 'violation'
    df_v_final = df_v_filtered.head(per_label_count)

    df_nv_filtered = df_nv[df_nv.apply(has_article_nonviolation, axis=1)].copy()
    df_nv_filtered['download_label'] = 'non-violation'
    df_nv_final = df_nv_filtered.head(per_label_count)

    print(f"  Filtered to {len(df_v_final)} Violations and {len(df_nv_final)} Non-Violations (Art {','.join(articles)}).")

    if len(df_v_final) < per_label_count or len(df_nv_final) < per_label_count:
        print(f"  WARNING: Could not find enough cases for {country_code}. Proceeding with available.")

    combined = pd.concat([df_v_final, df_nv_final], ignore_index=True)
    combined.drop_duplicates(subset=['itemid'], inplace=True)
    return combined

def download_data(args):
    countries = [c.strip() for c in args.countries.split(',')]
    per_label = args.per_country_count
    articles = [a.strip() for a in args.articles.split(',')]
    total_target = len(countries) * per_label * 2

    print(f"Data Plan: {len(countries)} countries × {per_label} per label × 2 labels = {total_target} total cases")
    print(f"Countries: {countries}")
    print(f"Articles: {articles}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    for country in countries:
        country_df = download_for_country(country, per_label, args.fetch_multiplier, articles)
        if len(country_df) > 0:
            all_dfs.append(country_df)

    if not all_dfs:
        print("No cases found for any country.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=['itemid'], inplace=True)

    print(f"\n{'='*60}")
    print(f"Combined dataset: {len(combined_df)} cases")
    print(f"Breakdown:")
    for country in countries:
        subset = combined_df[combined_df['respondent'] == country]
        v_count = len(subset[subset['download_label'] == 'violation'])
        nv_count = len(subset[subset['download_label'] == 'non-violation'])
        print(f"  {country}: {v_count} violations, {nv_count} non-violations")
    print(f"{'='*60}")

    # Download Full Text
    print("Downloading full texts for combined dataset...")
    json_list = download_full_text_main(combined_df, threads=10)

    # Save
    metadata_path = os.path.join(output_dir, "metadata.csv")
    combined_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")

    text_path = os.path.join(output_dir, "full_text.json")
    with open(text_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Full text saved to {text_path}")

    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download balanced ECHR cases per country, filtered by article")
    parser.add_argument("--countries", type=str, default="RUS,TUR,GBR",
                        help="Comma-separated list of respondent country codes (default: RUS,TUR,GBR)")
    parser.add_argument("--per_country_count", type=int, default=305,
                        help="Number of cases per label per country (default: 305)")
    parser.add_argument("--fetch_multiplier", type=int, default=5,
                        help="Fetch N times more candidates to filter (default: 5)")
    parser.add_argument("--articles", type=str, default="3,5,6,8",
                        help="Comma-separated list of ECHR articles to filter (default: 3,5,6,8)")
    # Legacy argument for backward compatibility
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Directory to save metadata.csv and full_text.json")
    parser.add_argument("--count", type=int, default=None,
                        help="(Legacy) Total cases per class. Overrides --per_country_count if set.")
    args = parser.parse_args()

    if args.count is not None:
        print("WARNING: Using legacy --count mode.")
        args.per_country_count = args.count

    download_data(args)
