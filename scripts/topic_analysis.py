"""
topic_analysis.py — Topic model interpretation for ECHR Art.6 corpus.

Fits NMF (primary), LDA, and BERTopic on the full Art.6 corpus (no train/test
split — this is EDA, not classification). Reports:
  1. Top words per topic (default: 20 words)
  2. Topic distribution by violation vs non-violation
  3. Topic distribution by respondent country (RUS / TUR / GBR)
  4. Representative cases per topic (highest topic-weight documents)

Saves a markdown report to results/topic_analysis/topic_report.md

Usage:
    python scripts/topic_analysis.py                          # NMF (default)
    python scripts/topic_analysis.py --model nmf
    python scripts/topic_analysis.py --model lda
    python scripts/topic_analysis.py --model bertopic
    python scripts/topic_analysis.py --model all
    python scripts/topic_analysis.py --n_topics 12 --top_words 20
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

for pkg_path, pkg_name in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet',   'wordnet'),
    ('corpora/omw-1.4',  'omw-1.4'),
]:
    try:
        nltk.data.find(pkg_path)
    except LookupError:
        nltk.download(pkg_name)

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Legal tokenizer (identical to train_svm.ipynb)
# ---------------------------------------------------------------------------

wnl = WordNetLemmatizer()
LEGAL_EXTRA_STOPWORDS = {"mr", "mrs", "ms", "court", "case"}
ALL_STOPWORDS = set(nltk_stopwords.words("english")) | LEGAL_EXTRA_STOPWORDS
MANUAL_LEMMA_MAP = {
    "applicants": "applicant", "authorities": "authority",
    "proceedings": "proceeding", "judges": "judge",
    "lawyers": "lawyer", "officers": "officer",
    "children": "child", "women": "woman", "men": "man",
}

def normalize_token(tok):
    tok = tok.lower().strip()
    tok = MANUAL_LEMMA_MAP.get(tok, tok)
    if not re.search(r"[a-z]", tok):
        return ""
    return wnl.lemmatize(tok, pos="n")

def legal_tokenizer(text):
    tokens = re.findall(r"[a-z']+", str(text).lower())
    return [
        t for tok in tokens
        if (t := normalize_token(tok)) and len(t) > 2 and t not in ALL_STOPWORDS
    ]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_art6(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'processed.csv'))
    def contains_art6(s):
        return s.astype(str).str.split(';').apply(lambda a: '6' in a)
    v  = contains_art6(df['violation_articles'])
    nv = contains_art6(df['nonviolation_articles'])
    df6 = df[v | nv].copy()
    df6['label_art6'] = v[df6.index].astype(int)
    df6 = df6.reset_index(drop=True)
    print(f"Art.6: {len(df6)} cases | violation: {df6['label_art6'].mean():.1%}")
    return df6

# ---------------------------------------------------------------------------
# NMF analysis (primary — professor recommendation)
# ---------------------------------------------------------------------------

def run_nmf(df, n_topics, top_n_words=20):
    """
    NMF with TF-IDF input. Produces sparser, more interpretable topics than
    LDA for legal text — words have non-negative additive contributions and
    topics tend to be more coherent / less mixed.
    """
    tfidf = TfidfVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, ngram_range=(1, 1),
        sublinear_tf=True,
    )
    T     = tfidf.fit_transform(df['text'].fillna('').values)
    vocab = np.array(tfidf.get_feature_names_out())
    print(f'TF-IDF matrix: {T.shape[0]} docs | {T.shape[1]} terms')

    nmf = NMF(n_components=n_topics, init='nndsvda', random_state=RANDOM_STATE,
              max_iter=300, l1_ratio=0.1)
    doc_topics_raw = nmf.fit_transform(T)  # (N, k), non-negative, unnormalised

    # Normalise rows to [0,1] so each doc's weights sum to 1 (probability-like)
    row_sums = doc_topics_raw.sum(axis=1, keepdims=True)
    doc_topics = doc_topics_raw / np.maximum(row_sums, 1e-9)

    topics = []
    for i, comp in enumerate(nmf.components_):
        top_idx   = comp.argsort()[:-top_n_words-1:-1]
        top_words = vocab[top_idx].tolist()
        topics.append({'id': i, 'top_words': top_words, 'weight': comp[top_idx].tolist()})

    recon_err = nmf.reconstruction_err_
    print(f'NMF reconstruction error: {recon_err:.4f}')
    return nmf, doc_topics, topics


# ---------------------------------------------------------------------------
# LDA analysis
# ---------------------------------------------------------------------------

def run_lda(df, n_topics, top_n_words=20):
    cv = CountVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, ngram_range=(1, 1),
    )
    C = cv.fit_transform(df['text'].values)
    vocab = np.array(cv.get_feature_names_out())

    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=50, learning_method='batch',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    doc_topics = lda.fit_transform(C)  # (N, k)

    topics = []
    for i, comp in enumerate(lda.components_):
        top_idx   = comp.argsort()[:-top_n_words-1:-1]
        top_words = vocab[top_idx].tolist()
        topics.append({'id': i, 'top_words': top_words, 'weight': comp[top_idx].tolist()})

    return lda, doc_topics, topics

# ---------------------------------------------------------------------------
# BERTopic analysis
# ---------------------------------------------------------------------------

def run_bertopic(df, n_topics, device='cpu'):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    emb_model   = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    umap_model  = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                       metric='cosine', random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5,
                             metric='euclidean', prediction_data=True)

    bt = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        calculate_probabilities=True,
        verbose=False,
    )
    topic_ids, probs = bt.fit_transform(df['text'].tolist())
    probs = np.nan_to_num(np.array(probs, dtype=float), nan=0.0)
    if probs.ndim == 1:
        probs = probs.reshape(len(df), -1)

    # Build topic word list from BERTopic's internal representation
    topics = []
    for tid in sorted(set(topic_ids)):
        if tid == -1:
            continue
        words_scores = bt.get_topic(tid)
        if not words_scores:
            continue
        top_words = [w for w, _ in words_scores[:12]]
        topics.append({'id': tid, 'top_words': top_words})

    n_outliers = sum(1 for t in topic_ids if t == -1)
    print(f"  BERTopic: {len(topics)} topics, {n_outliers} outlier docs")
    return bt, np.array(topic_ids), probs, topics

# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def topic_by_group(doc_topics, group_series, topic_ids=None):
    """
    For each topic, compute mean probability by group value.
    Returns DataFrame: rows=topics, cols=group values.
    """
    df_probs = pd.DataFrame(doc_topics)
    df_probs['group'] = group_series.values
    return df_probs.groupby('group').mean().T  # (n_topics, n_groups)


def dominant_topic(doc_topics):
    """Return the index of the highest-probability topic per document."""
    return np.argmax(doc_topics, axis=1)


def representative_cases(doc_topics, df, topics, top_k=3):
    """
    For each topic, find the top_k documents with highest weight.
    Enables 'close reading' — identify which real cases exemplify each topic.
    """
    label_col = 'label_art6' if 'label_art6' in df.columns else 'label'
    lines = ["### Representative cases per topic (close reading)\n"]
    lines.append("*Top cases by topic weight — use item_id to retrieve from HUDOC*\n")
    for t in topics:
        tid = t['id']
        top_idx = doc_topics[:, tid].argsort()[::-1][:top_k]
        lines.append(f"**Topic {tid}** — _{', '.join(t['top_words'][:8])}_")
        lines.append("| Case ID | Country | Year | Outcome | Weight |")
        lines.append("|---|---|---|---|---|")
        for idx in top_idx:
            row = df.iloc[idx]
            outcome = 'Violation' if row[label_col] == 1 else 'No Violation'
            weight  = doc_topics[idx, tid]
            year    = int(row['year']) if 'year' in df.columns else '?'
            lines.append(f"| {row['item_id']} | {row['respondent']} | {year} | {outcome} | {weight:.3f} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_report(model_name, n_topics, topics, doc_topics, df, top_k_cases=3):
    label_col = 'label_art6' if 'label_art6' in df.columns else 'label'
    lines = [f"## {model_name} — {n_topics} topics\n"]

    # 1. Top words per topic (20 words)
    lines.append("### Top words per topic\n")
    lines.append("| Topic | Top words |")
    lines.append("|---|---|")
    for t in topics:
        lines.append(f"| {t['id']} | {', '.join(t['top_words'][:20])} |")
    lines.append("")

    # 2. Mean topic weight by violation label
    lines.append("### Mean topic probability — Violation vs Non-Violation\n")
    by_label = topic_by_group(doc_topics, df[label_col])
    by_label.columns = ['Non-Violation', 'Violation']
    by_label['Δ (V−NV)'] = (by_label['Violation'] - by_label['Non-Violation']).round(4)
    by_label = by_label.round(4).sort_values('Δ (V−NV)', ascending=False)
    lines.append("| Topic | Non-Violation | Violation | Δ (V−NV) |")
    lines.append("|---|---|---|---|")
    for tid, row in by_label.iterrows():
        top_words = next((t['top_words'][:5] for t in topics if t['id'] == tid), [])
        label = ', '.join(top_words) if top_words else str(tid)
        lines.append(f"| {tid} ({label}) | {row['Non-Violation']} | {row['Violation']} | {row['Δ (V−NV)']} |")
    lines.append("")

    # 3. Mean topic weight by country
    lines.append("### Mean topic probability — by Respondent Country\n")
    by_country = topic_by_group(doc_topics, df['respondent'])
    by_country = by_country.round(4)
    country_cols = sorted(by_country.columns.tolist())
    lines.append("| Topic | " + " | ".join(country_cols) + " |")
    lines.append("|---|" + "|".join(["---"] * len(country_cols)) + "|")
    for tid in by_country.index:
        top_words = next((t['top_words'][:3] for t in topics if t['id'] == tid), [])
        label = ', '.join(top_words) if top_words else str(tid)
        vals = " | ".join(str(by_country.loc[tid, c]) for c in country_cols)
        lines.append(f"| {tid} ({label}) | {vals} |")
    lines.append("")

    # 4. Dominant topic distribution per class
    lines.append("### Dominant topic frequency — Violation vs Non-Violation\n")
    dom = dominant_topic(doc_topics)
    df_dom = pd.DataFrame({'topic': dom, 'label': df[label_col].values})
    ct = pd.crosstab(df_dom['topic'], df_dom['label'],
                     colnames=['label'], normalize='columns').round(3)
    ct.columns = ['Non-Violation', 'Violation']
    ct['top_words'] = [
        ', '.join(next((t['top_words'][:5] for t in topics if t['id'] == i), []))
        for i in ct.index
    ]
    lines.append("| Topic | Non-Violation | Violation | Top words |")
    lines.append("|---|---|---|---|")
    for tid, row in ct.iterrows():
        lines.append(f"| {tid} | {row['Non-Violation']} | {row['Violation']} | {row['top_words']} |")
    lines.append("")

    # 5. Representative cases per topic (close reading)
    lines.append(representative_cases(doc_topics, df, topics, top_k=top_k_cases))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import torch
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    df = load_art6(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    label_col = 'label_art6' if 'label_art6' in df.columns else 'label'
    report_lines = [
        "# Topic Analysis — ECHR Art.6 Corpus\n",
        f"**Corpus:** {len(df)} cases | {df[label_col].mean():.1%} violation rate\n",
        f"**Countries:** {df['respondent'].value_counts().to_dict()}\n",
        "---\n",
    ]

    if args.model in ('nmf', 'all'):
        print(f"\nFitting NMF (k={args.n_topics}, top_words={args.top_words}) ...")
        _, doc_topics_nmf, topics_nmf = run_nmf(df, args.n_topics, top_n_words=args.top_words)
        report_lines.append(build_report(
            "NMF", args.n_topics, topics_nmf, doc_topics_nmf, df,
            top_k_cases=args.top_k_cases,
        ))
        np.save(os.path.join(args.output_dir, f'nmf_doc_topics_k{args.n_topics}.npy'), doc_topics_nmf)
        print(f"\nNMF top words per topic:")
        for t in topics_nmf:
            print(f"  T{t['id']}: {', '.join(t['top_words'][:10])}")

    if args.model in ('lda', 'all'):
        print(f"\nFitting LDA (k={args.n_topics}) ...")
        _, doc_topics_lda, topics_lda = run_lda(df, args.n_topics, top_n_words=args.top_words)
        report_lines.append(build_report(
            "LDA", args.n_topics, topics_lda, doc_topics_lda, df,
            top_k_cases=args.top_k_cases,
        ))
        np.save(os.path.join(args.output_dir, f'lda_doc_topics_k{args.n_topics}.npy'), doc_topics_lda)

    if args.model in ('bertopic', 'all'):
        print(f"\nFitting BERTopic (nr_topics={args.n_topics}, device={device}) ...")
        _, _, doc_topics_bt, topics_bt = run_bertopic(df, args.n_topics, device=device)
        report_lines.append(build_report(
            "BERTopic", args.n_topics, topics_bt, doc_topics_bt, df,
            top_k_cases=args.top_k_cases,
        ))
        np.save(os.path.join(args.output_dir, f'bertopic_doc_topics_k{args.n_topics}.npy'), doc_topics_bt)

    report_path = os.path.join(args.output_dir, 'topic_report.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str, default='data')
    parser.add_argument('--n_topics',    type=int, default=10)
    parser.add_argument('--top_words',   type=int, default=20)
    parser.add_argument('--top_k_cases', type=int, default=3,
                        help='Representative cases per topic for close reading')
    parser.add_argument('--model',       type=str, default='nmf',
                        choices=['nmf', 'lda', 'bertopic', 'all'])
    parser.add_argument('--cpu',         action='store_true', default=False)
    parser.add_argument('--output_dir',  type=str, default='results/topic_analysis')
    main(parser.parse_args())
