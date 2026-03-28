"""
Analyse which TF-IDF features drive SVM predictions.
Outputs top-N positive/negative coefficients for:
  - violation vs non-violation (linear SVM weights)
  - features stable across year quintiles (temporal stability)

Usage:
  python scripts/svm_features.py \
    --data_path data_v1/processed/processed.csv \
    --top_n 30
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation', 'articles': 'article',
}
LEGAL_STOP = set(stopwords.words('english')) | {'court', 'case', 'mr', 'mrs', 'ms'}


def legal_tokenizer(text):
    wnl = WordNetLemmatizer()
    tokens = []
    for tok in text.lower().split():
        tok = tok.strip(".,;:()[]\"'")
        if not tok or tok in LEGAL_STOP or len(tok) < 2:
            continue
        tok = MANUAL_LEMMA_MAP.get(tok, wnl.lemmatize(tok, pos='n'))
        tokens.append(tok)
    return tokens


def load_art6(data_path):
    df = pd.read_csv(data_path)
    art6_mask = (
        df['violation_articles'].fillna('').str.contains(r'(?:^|;)6(?:$|;|-)', regex=True) |
        df['nonviolation_articles'].fillna('').str.contains(r'(?:^|;)6(?:$|;|-)', regex=True)
    )
    df = df[art6_mask].dropna(subset=['text', 'label', 'year']).copy()
    df['year'] = df['year'].astype(int)
    return df


def build_pipeline():
    vec = TfidfVectorizer(
        tokenizer=legal_tokenizer, token_pattern=None, lowercase=False,
        min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2)
    )
    clf = LinearSVC(C=0.1, class_weight='balanced', max_iter=2000)
    return Pipeline([('tfidf', vec), ('clf', clf)])


def print_top_features(coef, feature_names, top_n, label=''):
    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]
    print(f"\n{label}")
    print(f"  Top {top_n} VIOLATION features (positive weight):")
    for i in top_pos_idx:
        print(f"    {feature_names[i]:40s}  {coef[i]:.4f}")
    print(f"  Top {top_n} NON-VIOLATION features (negative weight):")
    for i in top_neg_idx:
        print(f"    {feature_names[i]:40s}  {coef[i]:.4f}")


def temporal_stability(df, feature_names, top_n=50):
    """
    Train SVM on each year quintile, collect top-N features by abs weight.
    Report features present in all 5 quintiles (stable) vs only some (unstable).
    """
    df = df.copy()
    df['quintile'] = pd.qcut(df['year'], q=5, labels=False, duplicates='drop')
    quintile_top = []
    for q in sorted(df['quintile'].unique()):
        sub = df[df['quintile'] == q]
        if sub['label'].nunique() < 2 or len(sub) < 20:
            continue
        pipe = build_pipeline()
        pipe.fit(sub['text'], sub['label'])
        coef = pipe.named_steps['clf'].coef_[0]
        fn = pipe.named_steps['tfidf'].get_feature_names_out()
        top_idx = np.argsort(np.abs(coef))[-top_n:]
        quintile_top.append(set(fn[top_idx]))

    if len(quintile_top) < 2:
        print("Not enough quintiles for stability analysis.")
        return

    stable = quintile_top[0].intersection(*quintile_top[1:])
    print(f"\n--- Temporal stability (top-{top_n} features per quintile) ---")
    print(f"  Features stable across all {len(quintile_top)} quintiles ({len(stable)}):")
    for f in sorted(stable):
        print(f"    {f}")


def main(args):
    df = load_art6(args.data_path)
    print(f"Art.6 cases: {len(df)}  violation rate: {df['label'].mean():.1%}")

    # --- Full dataset SVM weights ---
    pipe = build_pipeline()
    pipe.fit(df['text'], df['label'])
    coef = pipe.named_steps['clf'].coef_[0]
    fn = pipe.named_steps['tfidf'].get_feature_names_out()
    print_top_features(coef, fn, args.top_n, label="=== Full-corpus SVM weights ===")

    # --- Country-stratified: do violation features differ by country? ---
    print("\n=== Top VIOLATION features per country (top-5 absolute) ===")
    for country, grp in df.groupby('respondent'):
        if grp['label'].nunique() < 2 or len(grp) < 30:
            continue
        p = build_pipeline()
        p.fit(grp['text'], grp['label'])
        c = p.named_steps['clf'].coef_[0]
        f = p.named_steps['tfidf'].get_feature_names_out()
        top5 = [f[i] for i in np.argsort(c)[-5:][::-1]]
        bot5 = [f[i] for i in np.argsort(c)[:5]]
        print(f"  {country:6s} ({len(grp):4d} cases, {grp['label'].mean():.0%} viol)  "
              f"V={top5}  NV={bot5}")

    # --- Temporal stability ---
    temporal_stability(df, fn, top_n=args.top_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--top_n', type=int, default=20)
    main(parser.parse_args())
