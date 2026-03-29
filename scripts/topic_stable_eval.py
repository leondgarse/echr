"""
topic_stable_eval.py — Multi-seed stability evaluation for key topic-augmented models.

Runs the same models across N different 75/25 stratified splits and reports
mean ± std for each metric. With only 109 test samples per split, single-run
numbers have high variance; this gives a reliable estimate.

Also runs BERTopic with multiple model seeds per data split to capture
UMAP/HDBSCAN stochasticity independently from data split variance.

Models evaluated:
  - Dummy (majority class baseline)
  - TF-IDF / SVM
  - TF-IDF / LR
  - TF-IDF + LDA (best k from single-run sweep: k=30) / SVM
  - TF-IDF + LDA (k=30) / LR
  - TF-IDF + BERTopic (MiniLM, k=10) / SVM   ← best single-run model
  - TF-IDF + BERTopic (MiniLM, k=10) / LR

Usage:
    python scripts/topic_stable_eval.py
    python scripts/topic_stable_eval.py --data_seeds 0 1 2 3 4 5 6 7 8 9 --bertopic_seeds 0 1 2
    python scripts/topic_stable_eval.py --no_bertopic  # LDA + TF-IDF only, fast
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
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]:
    try:
        nltk.data.find(pkg_path)
    except LookupError:
        nltk.download(pkg_name)

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

# ---------------------------------------------------------------------------
# Legal tokenizer
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

def make_tfidf():
    return TfidfVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, sublinear_tf=True,
        ngram_range=(1, 2),
    )

def make_count_vec():
    return CountVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, ngram_range=(1, 1),
    )

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_art6(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'processed', 'processed.csv'))
    def contains_art6(s):
        return s.astype(str).str.split(';').apply(lambda a: '6' in a)
    v  = contains_art6(df['violation_articles'])
    nv = contains_art6(df['nonviolation_articles'])
    df6 = df[v | nv].copy()
    df6['label_art6'] = v[df6.index].astype(int)
    return df6.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Topic model helpers
# ---------------------------------------------------------------------------

def fit_lda(X_train, n_topics, rs):
    cv  = make_count_vec()
    C   = cv.fit_transform(X_train)
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=30, learning_method='batch',
        random_state=rs, n_jobs=-1,
    )
    lda.fit(C)
    Ztr = lda.transform(C)
    Zte = lda.transform(cv.transform(X_train))   # placeholder — overwritten below
    return cv, lda

def lda_features(X_train, X_test, n_topics, rs):
    cv, lda = fit_lda(X_train, n_topics, rs)
    Ztr = lda.transform(cv.transform(X_train))
    Zte = lda.transform(cv.transform(X_test))
    return Ztr, Zte

def bertopic_features(X_train, X_test, n_topics, bt_seed, device='cpu'):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    emb   = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    umap  = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                 metric='cosine', random_state=bt_seed)
    hdb   = HDBSCAN(min_cluster_size=10, min_samples=5,
                    metric='euclidean', prediction_data=True)
    bt    = BERTopic(embedding_model=emb, umap_model=umap, hdbscan_model=hdb,
                     nr_topics=n_topics, calculate_probabilities=True, verbose=False)

    _, Ptr = bt.fit_transform(list(X_train))
    _, Pte = bt.transform(list(X_test))
    Ptr = np.nan_to_num(np.array(Ptr, dtype=float), nan=0.0)
    Pte = np.nan_to_num(np.array(Pte, dtype=float), nan=0.0)
    if Ptr.ndim == 1: Ptr = Ptr.reshape(len(X_train), -1)
    if Pte.ndim == 1: Pte = Pte.reshape(len(X_test),  -1)
    return Ptr, Pte

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def tune_and_eval(label, Xtr, Xte, ytr, yte, cv_rs):
    """Returns dict of metrics for one (feature-set, data-split, [bt-seed]) combo."""
    param_grid = {
        'clf__C':            [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'clf__class_weight': [None, 'balanced'],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_rs)
    rows = []
    for clf_name, clf_class, extra in [
        ('SVM', LinearSVC,         {'max_iter': 2000}),
        ('LR',  LogisticRegression, {'max_iter': 1000}),
    ]:
        gs = GridSearchCV(
            Pipeline([('clf', clf_class(**extra, random_state=cv_rs))]),
            param_grid, cv=skf, scoring='f1_macro', n_jobs=-1, refit=True,
        )
        gs.fit(Xtr, ytr)
        pred = gs.predict(Xte)
        rows.append({
            'model':    f"{label} / {clf_name}",
            'macro_f1': f1_score(yte, pred, average='macro'),
            'acc':      accuracy_score(yte, pred),
            'f1_v':     f1_score(yte, pred, pos_label=1, average='binary'),
            'f1_nv':    f1_score(yte, pred, pos_label=0, average='binary'),
        })
    return rows

# ---------------------------------------------------------------------------
# One full evaluation pass for a single data seed
# ---------------------------------------------------------------------------

def eval_one_split(X, y, data_seed, args, device):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=data_seed
    )

    records = []

    # Dummy baseline
    dum = DummyClassifier(strategy='most_frequent')
    dum.fit(X_train, y_train)
    pred = dum.predict(X_test)
    records.append({
        'model': 'Dummy', 'data_seed': data_seed, 'bt_seed': -1,
        'macro_f1': f1_score(y_test, pred, average='macro'),
        'acc':      accuracy_score(y_test, pred),
        'f1_v':     f1_score(y_test, pred, pos_label=1, average='binary'),
        'f1_nv':    f1_score(y_test, pred, pos_label=0, average='binary'),
    })

    # TF-IDF features (shared across models this split)
    tfidf = make_tfidf()
    Atr = tfidf.fit_transform(X_train)
    Ate = tfidf.transform(X_test)

    for r in tune_and_eval('TF-IDF only', Atr, Ate, y_train, y_test, data_seed):
        records.append({**r, 'data_seed': data_seed, 'bt_seed': -1})

    if not args.no_lda:
        Ztr, Zte = lda_features(X_train, X_test, args.lda_k, rs=data_seed)
        Xtr_lda = hstack([Atr, csr_matrix(Ztr)])
        Xte_lda = hstack([Ate, csr_matrix(Zte)])
        for r in tune_and_eval(f'TF-IDF + LDA (k={args.lda_k})',
                               Xtr_lda, Xte_lda, y_train, y_test, data_seed):
            records.append({**r, 'data_seed': data_seed, 'bt_seed': -1})

    if not args.no_bertopic:
        for bt_seed in args.bertopic_seeds:
            print(f"    BERTopic bt_seed={bt_seed} ...")
            Ptr, Pte = bertopic_features(X_train, X_test, args.bt_k, bt_seed, device)
            Xtr_bt = hstack([Atr, csr_matrix(Ptr)])
            Xte_bt = hstack([Ate, csr_matrix(Pte)])
            for r in tune_and_eval(f'TF-IDF + BERTopic (k={args.bt_k})',
                                   Xtr_bt, Xte_bt, y_train, y_test, data_seed):
                records.append({**r, 'data_seed': data_seed, 'bt_seed': bt_seed})

    return records

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import torch
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    df = load_art6(args.data_dir)
    X  = df['text'].values
    y  = df['label_art6'].values
    print(f"Art.6: {len(df)} cases | {y.mean():.1%} violation")
    print(f"Data seeds: {args.data_seeds}")
    if not args.no_bertopic:
        print(f"BERTopic seeds: {args.bertopic_seeds}  k={args.bt_k}")
    print()

    all_records = []
    for i, ds in enumerate(args.data_seeds):
        print(f"{'='*60}")
        print(f"Data seed {ds}  ({i+1}/{len(args.data_seeds)})")
        print(f"{'='*60}")
        recs = eval_one_split(X, y, ds, args, device)
        all_records.extend(recs)

    df_raw = pd.DataFrame(all_records)
    os.makedirs(args.output_dir, exist_ok=True)
    df_raw.to_csv(os.path.join(args.output_dir, 'raw_runs.csv'), index=False)

    # Aggregate: for BERTopic models, average over bt_seeds first within each
    # data_seed, then average over data_seeds — correctly separating the two
    # variance sources.
    print(f"\n{'='*60}")
    print("STABILITY RESULTS (mean ± std over data splits)")
    print(f"{'='*60}")

    summary_rows = []
    for model in df_raw['model'].unique():
        sub = df_raw[df_raw['model'] == model]

        # For BERTopic: average over bt_seeds per data_seed first
        if sub['bt_seed'].nunique() > 1:
            per_split = sub.groupby('data_seed')[
                ['macro_f1', 'acc', 'f1_v', 'f1_nv']
            ].mean()
        else:
            per_split = sub.set_index('data_seed')[['macro_f1', 'acc', 'f1_v', 'f1_nv']]

        summary_rows.append({
            'Model':        model,
            'Macro-F1':     f"{per_split['macro_f1'].mean():.3f} ± {per_split['macro_f1'].std():.3f}",
            'Accuracy':     f"{per_split['acc'].mean():.3f} ± {per_split['acc'].std():.3f}",
            'F1 (V)':       f"{per_split['f1_v'].mean():.3f} ± {per_split['f1_v'].std():.3f}",
            'F1 (NV)':      f"{per_split['f1_nv'].mean():.3f} ± {per_split['f1_nv'].std():.3f}",
            'macro_f1_mean': per_split['macro_f1'].mean(),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('macro_f1_mean', ascending=False)
    df_summary = df_summary.drop(columns='macro_f1_mean')
    print(df_summary.to_string(index=False))

    df_summary.to_csv(os.path.join(args.output_dir, 'stability_summary.csv'), index=False)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      type=str, default='data')
    parser.add_argument('--data_seeds',    type=int, nargs='+',
                        default=list(range(10)),
                        help='Random seeds for 75/25 data splits (default: 0..9)')
    parser.add_argument('--bertopic_seeds', type=int, nargs='+',
                        default=[0, 1, 2],
                        help='Random seeds for BERTopic UMAP init per data split')
    parser.add_argument('--lda_k',         type=int, default=30,
                        help='LDA topic count (best from single-run sweep)')
    parser.add_argument('--bt_k',          type=int, default=10,
                        help='BERTopic nr_topics (best from single-run sweep)')
    parser.add_argument('--no_lda',        action='store_true', default=False)
    parser.add_argument('--no_bertopic',   action='store_true', default=False)
    parser.add_argument('--cpu',           action='store_true', default=False)
    parser.add_argument('--output_dir',    type=str,
                        default='results/topic_stable_eval')
    main(parser.parse_args())
