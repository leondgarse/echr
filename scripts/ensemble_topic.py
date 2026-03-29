"""
ensemble_topic.py — Stacking ensemble: fine-tuned BERT + TF-IDF + BERTopic.

Uses the same Art.6 random 75/25 split (RANDOM_STATE=42).

Signals stacked:
  - Fine-tuned chunked LegalBERT probabilities  (2-d softmax from test_probs.npz)
  - TF-IDF features                             (sparse, fitted on train)
  - BERTopic topic distributions                (dense, fitted on train)

Meta-learner: LogisticRegression (5-fold CV tuned) on concatenated signals.
Also evaluates each signal alone and all pairwise combinations for ablation.

Usage:
    python scripts/ensemble_topic.py \
        --bert_probs results/legalbert_chunked_encoder/test_probs.npz \
        --data_dir data --n_topics 10
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
import torch

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42

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
    return df6.reset_index(drop=True)

# ---------------------------------------------------------------------------
# BERTopic helper
# ---------------------------------------------------------------------------

def fit_bertopic_signal(X_train, X_test, n_topics, device='cpu'):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    emb_model     = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    umap_model    = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                         metric='cosine', random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5,
                             metric='euclidean', prediction_data=True)
    bt = BERTopic(
        embedding_model=emb_model, umap_model=umap_model,
        hdbscan_model=hdbscan_model, nr_topics=n_topics,
        calculate_probabilities=True, verbose=False,
    )
    _, Ptr = bt.fit_transform(list(X_train))
    _, Pte = bt.transform(list(X_test))
    Ptr = np.nan_to_num(np.array(Ptr, dtype=float), nan=0.0)
    Pte = np.nan_to_num(np.array(Pte, dtype=float), nan=0.0)
    if Ptr.ndim == 1: Ptr = Ptr.reshape(len(X_train), -1)
    if Pte.ndim == 1: Pte = Pte.reshape(len(X_test), -1)
    return Ptr, Pte

# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def tune_clf(X_train, y_train, clf_name='LR'):
    param_grid = {
        'clf__C':            [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'clf__class_weight': [None, 'balanced'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if clf_name == 'LR':
        base = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        base = LinearSVC(max_iter=2000, random_state=RANDOM_STATE)
    gs = GridSearchCV(Pipeline([('clf', base)]), param_grid,
                      cv=cv, scoring='f1_macro', n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_score_

def eval_row(name, clf, X_test, y_test):
    pred = clf.predict(X_test)
    return {
        'Model':             name,
        'Accuracy':          round(accuracy_score(y_test, pred), 3),
        'Macro-F1':          round(f1_score(y_test, pred, average='macro'), 3),
        'F1 (Violation)':    round(f1_score(y_test, pred, pos_label=1, average='binary'), 3),
        'F1 (No Violation)': round(f1_score(y_test, pred, pos_label=0, average='binary'), 3),
    }

def run(label, Xtr, Xte, y_train, y_test):
    print(f"\n  [{label}] tuning ...")
    clf, cv_f1 = tune_clf(Xtr, y_train)
    r = eval_row(label, clf, Xte, y_test)
    print(f"    cv_f1={cv_f1:.3f} → test macro-F1={r['Macro-F1']}  "
          f"F1(V)={r['F1 (Violation)']}  F1(NV)={r['F1 (No Violation)']}")
    return r

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    df = load_art6(args.data_dir)
    X  = df['text'].values
    y  = df['label_art6'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    rows = []

    # --- Signal A: TF-IDF ---
    print("\n" + "="*60)
    print("Signal A: TF-IDF")
    tfidf = make_tfidf()
    Atr = tfidf.fit_transform(X_train)
    Ate = tfidf.transform(X_test)
    rows.append(run("TF-IDF only", Atr, Ate, y_train, y_test))

    # --- Signal B: Fine-tuned BERT probs (train-set rows only) ---
    print("\n" + "="*60)
    print("Signal B: Fine-tuned BERT probs")
    bert_data = np.load(args.bert_probs)
    # test_probs.npz stores probs for the test set from training run —
    # same split (RANDOM_STATE=42), so idx_test aligns directly.
    # For training signal we re-encode using the saved encoder.
    from transformers import AutoTokenizer, AutoModel
    import importlib.util, sys as _sys
    _spec = importlib.util.spec_from_file_location(
        "topic_hybrid", os.path.join(os.path.dirname(__file__), "topic_hybrid.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    extract_bert_embeddings = _mod.extract_bert_embeddings

    checkpoint = args.bert_checkpoint if args.bert_checkpoint else \
                 os.path.join(os.path.dirname(args.bert_probs), 'hf_encoder')
    bert_device = f'cuda:{args.bert_gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"  Loading encoder from {checkpoint}  device={bert_device}")
    tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
    bert_model = AutoModel.from_pretrained(checkpoint).to(bert_device)
    bert_model.eval()

    print("  Encoding train ...")
    Btr = extract_bert_embeddings(X_train, tokenizer, bert_model,
                                  chunk_size=510, max_chunks=8,
                                  batch_size=args.batch_size, device=bert_device)
    print("  Encoding test ...")
    Bte = extract_bert_embeddings(X_test, tokenizer, bert_model,
                                  chunk_size=510, max_chunks=8,
                                  batch_size=args.batch_size, device=bert_device)
    del bert_model
    torch.cuda.empty_cache()
    rows.append(run("BERT only", Btr, Bte, y_train, y_test))

    # --- Signal C: BERTopic ---
    print("\n" + "="*60)
    print(f"Signal C: BERTopic (k={args.n_topics})")
    Ctr, Cte = fit_bertopic_signal(X_train, X_test, args.n_topics, device=device)
    rows.append(run(f"BERTopic only (k={args.n_topics})", Ctr, Cte, y_train, y_test))

    # --- Pairwise combinations ---
    print("\n" + "="*60)
    print("Pairwise combinations")
    rows.append(run("TF-IDF + BERT",
                    hstack([Atr, csr_matrix(Btr)]), hstack([Ate, csr_matrix(Bte)]),
                    y_train, y_test))
    rows.append(run(f"TF-IDF + BERTopic (k={args.n_topics})",
                    hstack([Atr, csr_matrix(Ctr)]), hstack([Ate, csr_matrix(Cte)]),
                    y_train, y_test))
    rows.append(run(f"BERT + BERTopic (k={args.n_topics})",
                    np.hstack([Btr, Ctr]), np.hstack([Bte, Cte]),
                    y_train, y_test))

    # --- Full ensemble: TF-IDF + BERT + BERTopic ---
    print("\n" + "="*60)
    print("Full ensemble: TF-IDF + BERT + BERTopic")
    Xtr_full = hstack([Atr, csr_matrix(Btr), csr_matrix(Ctr)])
    Xte_full = hstack([Ate, csr_matrix(Bte), csr_matrix(Cte)])
    rows.append(run(f"TF-IDF + BERT + BERTopic (k={args.n_topics})",
                    Xtr_full, Xte_full, y_train, y_test))

    # --- Summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'ensemble_results.csv')
    df_res.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str, default='data')
    parser.add_argument('--bert_probs',     type=str,
                        default='results/legalbert_chunked_encoder/test_probs.npz')
    parser.add_argument('--bert_checkpoint',type=str,
                        default='results/legalbert_chunked_encoder/hf_encoder')
    parser.add_argument('--n_topics',       type=int, default=10)
    parser.add_argument('--bert_gpu',       type=int, default=0)
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--cpu',            action='store_true', default=False)
    parser.add_argument('--output_dir',     type=str, default='results/ensemble_topic')
    main(parser.parse_args())
