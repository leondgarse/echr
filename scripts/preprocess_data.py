
import os
import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split

TARGET_ARTICLES = ['3', '5', '6', '8']

def clean_text(text):
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_facts(text):
    """Extract the FACTS section from an ECHR judgment (English only)."""
    text_upper = text.upper()
    
    # Start markers, ordered by specificity
    start_markers = [
        "AS TO THE FACTS",
        "THE FACTS",
        "I.  THE CIRCUMSTANCES OF THE CASE",
        "THE CIRCUMSTANCES OF THE CASE",
    ]
    
    start_idx = -1
    for marker in start_markers:
        idx = text_upper.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break
            
    if start_idx == -1:
        return None
        
    # End markers
    end_markers = [
        "THE LAW",
        "RELEVANT DOMESTIC LAW",
        "RELEVANT LEGAL FRAMEWORK",
        "PROCEEDINGS BEFORE THE COMMISSION",
    ]
    
    best_end_idx = len(text)
    for marker in end_markers:
        idx = text_upper.find(marker, start_idx)
        if idx != -1 and idx < best_end_idx:
            best_end_idx = idx
            
    return text[start_idx:best_end_idx].strip()

TARGET_COUNTRIES = ['RUS', 'TUR', 'GBR']

def normalize_respondent(respondent, target_countries=TARGET_COUNTRIES):
    """Normalize multi-country respondents like 'MDA;RUS' to the target country 'RUS'."""
    if ';' in str(respondent):
        for country in target_countries:
            if country in str(respondent):
                return country
    return str(respondent)

def preprocess(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    full_text_path = os.path.join(data_dir, "full_text.json")
    
    if not os.path.exists(metadata_path) or not os.path.exists(full_text_path):
        print("Data files not found.")
        return

    df_meta = pd.read_csv(metadata_path)
    with open(full_text_path, 'r') as f:
        full_texts = json.load(f)
        
    # Convert full_texts list to dict keyed by item_id
    text_map = {item['item_id']: item['full_text'] for item in full_texts}
    
    processed_data = []
    
    label_count = 0
    for _, row in df_meta.iterrows():
        item_id = row['itemid']
        if item_id not in text_map:
            # print(f"Skipping {item_id}: No text JSON entry")
            continue
            
        text = text_map[item_id]
        if not text:
            # print(f"Skipping {item_id}: Empty text")
            continue
            
        facts = extract_facts(text)
        if not facts or len(facts) < 100: # Filter simplistic/empty extractions
            # print(f"Skipping {item_id}: No FACTS section found")
            continue
            
        # Determine label
        # Violation if 'violation' column is not empty/null/FALSE
        # No Violation if 'nonviolation' is not empty/null/FALSE
        # Ignore if ambiguous or neither (e.g. inadmissable without merit judgment)
        
        # Determine label based on trusted download tag if available
        download_label = str(row.get('download_label', '')).lower()
        
        label = -1
        if download_label == 'violation':
             label = 1
        elif download_label == 'non-violation':
             label = 0
        else:
             # Fallback logic
             is_violation = str(row['violation']).strip().lower()
             is_nonviolation = str(row['nonviolation']).strip().lower()
             
             if is_violation not in ['nan', 'false', ''] and is_violation != '0':
                  label = 1
             elif is_nonviolation not in ['nan', 'false', ''] and is_nonviolation != '0':
                  label = 0
        
        if label == -1:
            # print(f"Skipping {item_id}: No label (v='{row['violation']}', nv='{row['nonviolation']}')")
            continue
             
        if label != -1:
            label_count += 1
            
            # Extract which target articles are involved
            v_arts = str(row.get('violation', '')).strip()
            nv_arts = str(row.get('nonviolation', '')).strip()
            
            def extract_target_articles(field_val):
                if not field_val or field_val.lower() in ['nan', 'false', '0']:
                    return []
                found = []
                for art in TARGET_ARTICLES:
                    if re.search(rf'(?:^|;){art}(?:$|;|-)', field_val):
                        found.append(art)
                return found
            
            processed_data.append({
                'text': clean_text(facts),
                'label': label,
                'item_id': item_id,
                'respondent': normalize_respondent(row.get('respondent', 'UNKNOWN')),
                'violation_articles': ';'.join(extract_target_articles(v_arts)),
                'nonviolation_articles': ';'.join(extract_target_articles(nv_arts)),
            })
            
    df_proc = pd.DataFrame(processed_data)
    print(f"Processed {len(df_proc)} valid cases out of {len(df_meta)} total. Found {label_count} labeled text-having cases.")
    
    if len(df_proc) == 0:
        print("No valid cases found. Check text extraction or labeling logic.")
        return

    # Merge year from metadata
    df_proc = pd.merge(df_proc, df_meta[['itemid', 'judgementdate']].rename(columns={'itemid': 'item_id'}),
                       on='item_id', how='left')
    df_proc['year'] = pd.to_datetime(df_proc['judgementdate'], dayfirst=True, errors='coerce').dt.year
    df_proc.drop(columns=['judgementdate'], inplace=True)

    # Summary
    print(f"\n--- Per-country breakdown ---")
    for country in sorted(df_proc['respondent'].unique()):
        for label_val in [0, 1]:
            subset = df_proc[(df_proc['respondent'] == country) & (df_proc['label'] == label_val)]
            if len(subset) == 0:
                continue
            label_name = 'V' if label_val == 1 else 'NV'
            art_col = 'violation_articles' if label_val == 1 else 'nonviolation_articles'
            art_counts = {}
            for arts in subset[art_col].dropna():
                for a in str(arts).split(';'):
                    if a and a in TARGET_ARTICLES:
                        art_counts[f'Art{a}'] = art_counts.get(f'Art{a}', 0) + 1
            print(f"  {country} {label_name}: {len(subset):>4} | Years {subset['year'].min()}-{subset['year'].max()} | {art_counts}")

    output_dir = os.path.join(os.path.dirname(data_dir), "processed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed.csv")
    df_proc.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_proc)} cases to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess ECHR data (extract FACTS, label, no splitting)")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()
    preprocess(args.data_dir)
