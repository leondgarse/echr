"""
topic_hybrid.py — LDA / BERTopic features + chunked Legal-BERT embeddings for ECHR violation prediction.

Article 6 only, random 75/25 stratified split (RANDOM_STATE=42).

Feature combinations compared
-------------------------------
  tfidf              — TF-IDF (1-2gram) baseline (replicates train_svm.ipynb)
  lda                — LDA topic distributions only
  tfidf+lda          — TF-IDF concatenated with LDA topic distributions
  bertopic           — BERTopic soft topic distributions only
  tfidf+bertopic     — TF-IDF concatenated with BERTopic distributions
  bert               — Frozen chunked Legal-BERT CLS mean-pool (768-d)
  bert+lda           — Legal-BERT embedding + LDA topic distributions
  bert+bertopic      — Legal-BERT embedding + BERTopic distributions

Classifiers: LinearSVC, LogisticRegression
  - C and class_weight tuned via 5-fold GridSearchCV (macro-F1) on training set

Usage
-----
    python scripts/topic_hybrid.py --data_dir data
    python scripts/topic_hybrid.py --data_dir data --n_topics 10 20 30 --no_bert
    python scripts/topic_hybrid.py --data_dir data --no_lda   # BERTopic + BERT only
    python scripts/topic_hybrid.py --data_dir data --no_bertopic  # LDA + BERT only
    python scripts/topic_hybrid.py --data_dir data --n_topics 20 --model_name nlpaueb/legal-bert-base-uncased
"""

import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import warnings

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

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Legal tokenizer  (identical to train_svm.ipynb for reproducibility)
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
# TF-IDF helper
# ---------------------------------------------------------------------------

def make_tfidf():
    return TfidfVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, sublinear_tf=True,
        ngram_range=(1, 2),
    )


def make_count_vec():
    """CountVectorizer (unigrams, raw counts) for LDA input."""
    return CountVectorizer(
        tokenizer=legal_tokenizer, preprocessor=None, token_pattern=None,
        lowercase=False, min_df=3, max_df=0.90, ngram_range=(1, 1),
    )


# ---------------------------------------------------------------------------
# Load data — Article 6 subset
# ---------------------------------------------------------------------------

def load_art6(data_dir):
    path = os.path.join(data_dir, 'processed', 'processed.csv')
    df = pd.read_csv(path)

    def contains_art6(series):
        return series.astype(str).str.split(';').apply(lambda arts: '6' in arts)

    v  = contains_art6(df['violation_articles'])
    nv = contains_art6(df['nonviolation_articles'])
    df6 = df[v | nv].copy()
    df6['label_art6'] = v[df6.index].astype(int)
    df6 = df6.reset_index(drop=True)
    print(f"Art.6 subset: {len(df6)} cases  |  violation rate: {df6['label_art6'].mean():.2%}")
    return df6


# ---------------------------------------------------------------------------
# Chunked Legal-BERT embedding extraction (frozen — no fine-tuning)
# ---------------------------------------------------------------------------

def extract_bert_embeddings(texts, tokenizer, model, chunk_size=510, max_chunks=8,
                             batch_size=8, device='cpu'):
    """
    For each document:
      1. Tokenise (no special tokens).
      2. Split into consecutive chunks of `chunk_size` tokens.
      3. Prepend [CLS] / append [SEP], pad to chunk_size+2.
      4. Forward through Legal-BERT, take CLS hidden state per chunk.
      5. Mean-pool CLS embeddings across *real* chunks → 768-d vector.

    Returns np.ndarray of shape (len(texts), hidden_size).
    """
    model.eval()
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        batch_embs  = []

        for text in batch_texts:
            tok_ids = tokenizer.encode(str(text), add_special_tokens=False)

            raw_chunks = [
                tok_ids[i: i + chunk_size]
                for i in range(0, max(len(tok_ids), 1), chunk_size)
            ]
            n_real     = min(len(raw_chunks), max_chunks)
            raw_chunks = raw_chunks[:max_chunks]

            seq_len = chunk_size + 2
            input_ids_list      = []
            attention_mask_list = []

            for chunk in raw_chunks:
                ids  = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                ids  = ids[:seq_len]
                pad  = seq_len - len(ids)
                amsk = [1] * len(ids) + [0] * pad
                ids  = ids + [tokenizer.pad_token_id] * pad
                input_ids_list.append(ids)
                attention_mask_list.append(amsk)

            # Pad to max_chunks with dummy all-pad sequences
            while len(input_ids_list) < max_chunks:
                input_ids_list.append([tokenizer.pad_token_id] * seq_len)
                attention_mask_list.append([0] * seq_len)

            ids_t  = torch.tensor(input_ids_list,      dtype=torch.long,  device=device)
            amsk_t = torch.tensor(attention_mask_list, dtype=torch.long,  device=device)
            B, C, L = 1, max_chunks, seq_len

            with torch.no_grad():
                out = model(
                    input_ids=ids_t.view(C, L),
                    attention_mask=amsk_t.view(C, L),
                )
            cls_embs = out.last_hidden_state[:, 0, :].cpu().numpy()  # (C, H)

            # Mean-pool only real chunks
            doc_emb = cls_embs[:n_real].mean(axis=0)  # (H,)
            batch_embs.append(doc_emb)

        all_embs.extend(batch_embs)
        print(f"  Encoded {min(start + batch_size, len(texts))}/{len(texts)} docs", end='\r')

    print()
    return np.array(all_embs)  # (N, H)


# ---------------------------------------------------------------------------
# LDA helpers
# ---------------------------------------------------------------------------

def fit_lda(X_train_text, n_topics):
    cv  = make_count_vec()
    C   = cv.fit_transform(X_train_text)
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=30, learning_method='batch',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lda.fit(C)
    return cv, lda


def lda_transform(cv, lda, texts):
    C = cv.transform(texts)
    return lda.transform(C)  # (N, k)  — document-topic probabilities


# ---------------------------------------------------------------------------
# BERTopic helpers
# ---------------------------------------------------------------------------

def fit_bertopic(X_train_text, n_topics, device='cpu', precomputed_embeddings=None):
    """
    Fit BERTopic on training documents.

    If `precomputed_embeddings` (N_train, H) is provided, BERTopic skips its
    internal sentence-transformer step and uses those embeddings directly for
    UMAP + HDBSCAN — useful for reusing chunked Legal-BERT doc vectors.
    Otherwise falls back to sentence-transformers/all-MiniLM-L6-v2.

    `n_topics` sets nr_topics for automatic topic reduction after fitting.

    Returns (topic_model, train_probs) where train_probs is (N_train, n_topics).
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    # Conservative UMAP / HDBSCAN settings for small corpora (~327 docs)
    umap_model    = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                         metric='cosine', random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5,
                             metric='euclidean', prediction_data=True)

    if precomputed_embeddings is not None:
        # Pass Legal-BERT embeddings directly; set embedding_model=None to skip re-encoding
        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=n_topics,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(
            list(X_train_text), embeddings=precomputed_embeddings
        )
    else:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2', device=device
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=n_topics,
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probs = topic_model.fit_transform(list(X_train_text))

    n_real     = len(set(t for t in topics if t >= 0))
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"    BERTopic discovered {n_real} real topics (outliers: {n_outliers})")
    return topic_model, probs


def bertopic_transform(topic_model, texts, precomputed_embeddings=None):
    """
    Transform new texts using a fitted BERTopic model.
    If precomputed_embeddings provided, skips internal re-encoding.
    Returns soft probability matrix (N, n_topics).
    """
    if precomputed_embeddings is not None:
        topics, probs = topic_model.transform(
            list(texts), embeddings=precomputed_embeddings
        )
    else:
        topics, probs = topic_model.transform(list(texts))
    return probs


def _safe_probs(probs):
    """Ensure probs is a 2-D float array (BERTopic can return 1-D for single docs)."""
    p = np.array(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    # Replace any NaN with 0 (outlier docs may have no topic assignment)
    p = np.nan_to_num(p, nan=0.0)
    return p


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def tune_and_train(X_train, y_train):
    """GridSearchCV over C × class_weight for SVM and LR; returns fitted models."""
    param_grid = {
        'clf__C':            [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'clf__class_weight': [None, 'balanced'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, clf_class, extra in [
        ('SVM',          LinearSVC,         {'max_iter': 2000}),
        ('Logistic Reg', LogisticRegression, {'max_iter': 1000}),
    ]:
        pipe = Pipeline([('clf', clf_class(**extra, random_state=RANDOM_STATE))])
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro',
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        results[name] = gs.best_estimator_
        print(f"    {name}: best C={gs.best_params_['clf__C']}, "
              f"cw={gs.best_params_['clf__class_weight']}, "
              f"cv_macro-F1={gs.best_score_:.3f}")
    return results


def eval_report(name, y_true, y_pred):
    return {
        'Model':             name,
        'Accuracy':          round(accuracy_score(y_true, y_pred), 3),
        'Macro-F1':          round(f1_score(y_true, y_pred, average='macro'), 3),
        'F1 (Violation)':    round(f1_score(y_true, y_pred, pos_label=1, average='binary'), 3),
        'F1 (No Violation)': round(f1_score(y_true, y_pred, pos_label=0, average='binary'), 3),
    }


# ---------------------------------------------------------------------------
# Run one feature-set evaluation
# ---------------------------------------------------------------------------

def run_experiment(label, X_train, X_test, y_train, y_test):
    """Tune, train, predict; return list of result dicts."""
    print(f"\n  [{label}] tuning classifiers ...")
    models = tune_and_train(X_train, y_train)
    rows   = []
    for clf_name, clf in models.items():
        pred = clf.predict(X_test)
        r = eval_report(f"{label} / {clf_name}", y_test, pred)
        rows.append(r)
        print(f"    → {clf_name}: acc={r['Accuracy']}  macro-F1={r['Macro-F1']}  "
              f"F1(V)={r['F1 (Violation)']}  F1(NV)={r['F1 (No Violation)']}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- data ----------------------------------------------------------------
    df6    = load_art6(args.data_dir)
    X      = df6['text'].values
    y      = df6['label_art6'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"Test violation rate: {y_test.mean():.2%}\n")

    all_rows = []

    # ---- 1. TF-IDF baseline --------------------------------------------------
    print("=" * 60)
    print("1. TF-IDF only (baseline)")
    print("=" * 60)
    tfidf = make_tfidf()
    Xtr_tfidf = tfidf.fit_transform(X_train)
    Xte_tfidf = tfidf.transform(X_test)
    all_rows += run_experiment("TF-IDF only", Xtr_tfidf, Xte_tfidf, y_train, y_test)

    # cache per-k topic vectors to reuse in BERT combinations
    lda_vecs     = {}   # k -> (Ztr, Zte)
    bertopic_vecs = {}  # k -> (Ptr, Pte)

    # ---- 2. LDA only & TF-IDF + LDA ------------------------------------------
    if not args.no_lda:
        for k in args.n_topics:
            print("=" * 60)
            print(f"2. LDA (k={k})")
            print("=" * 60)
            cv_lda, lda_model = fit_lda(X_train, k)
            Ztr = lda_transform(cv_lda, lda_model, X_train)
            Zte = lda_transform(cv_lda, lda_model, X_test)
            lda_vecs[k] = (Ztr, Zte)

            all_rows += run_experiment(f"LDA only (k={k})", Ztr, Zte, y_train, y_test)

            Xtr_comb = hstack([Xtr_tfidf, csr_matrix(Ztr)])
            Xte_comb = hstack([Xte_tfidf, csr_matrix(Zte)])
            all_rows += run_experiment(f"TF-IDF + LDA (k={k})", Xtr_comb, Xte_comb, y_train, y_test)

    # ---- 3. BERTopic only & TF-IDF + BERTopic ---------------------------------
    if not args.no_bertopic:
        bt_device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        for k in args.n_topics:
            print("=" * 60)
            print(f"3. BERTopic (nr_topics={k})  device={bt_device}")
            print("=" * 60)
            bt_model, Ptr_raw = fit_bertopic(X_train, k, device=bt_device)
            Pte_raw           = bertopic_transform(bt_model, X_test)
            Ptr = _safe_probs(Ptr_raw)
            Pte = _safe_probs(Pte_raw)
            bertopic_vecs[k] = (Ptr, Pte)

            all_rows += run_experiment(f"BERTopic only (k={k})", Ptr, Pte, y_train, y_test)

            Xtr_comb = hstack([Xtr_tfidf, csr_matrix(Ptr)])
            Xte_comb = hstack([Xte_tfidf, csr_matrix(Pte)])
            all_rows += run_experiment(f"TF-IDF + BERTopic (k={k})", Xtr_comb, Xte_comb, y_train, y_test)

    # ---- 4. Legal-BERT (frozen) + LDA / BERTopic ------------------------------
    if not args.no_bert:
        print("=" * 60)
        print(f"4. Chunked Legal-BERT (frozen) — {args.model_name}")
        print("=" * 60)
        from transformers import AutoTokenizer, AutoModel

        bert_device = f'cuda:{args.bert_gpu}' if (torch.cuda.is_available() and not args.cpu) else 'cpu'
        print(f"  Device: {bert_device}")

        checkpoint = args.bert_checkpoint if args.bert_checkpoint else args.model_name
        print(f"  Checkpoint: {checkpoint}")
        tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
        bert_model = AutoModel.from_pretrained(checkpoint).to(bert_device)
        bert_model.eval()

        print("  Encoding training set ...")
        Btr = extract_bert_embeddings(X_train, tokenizer, bert_model,
                                      chunk_size=510, max_chunks=args.max_chunks,
                                      batch_size=args.batch_size, device=bert_device)
        print("  Encoding test set ...")
        Bte = extract_bert_embeddings(X_test, tokenizer, bert_model,
                                      chunk_size=510, max_chunks=args.max_chunks,
                                      batch_size=args.batch_size, device=bert_device)
        del bert_model
        torch.cuda.empty_cache()

        all_rows += run_experiment("BERT only", Btr, Bte, y_train, y_test)

        if not args.no_lda:
            for k in args.n_topics:
                Ztr, Zte = lda_vecs[k]
                all_rows += run_experiment(
                    f"BERT + LDA (k={k})",
                    np.hstack([Btr, Ztr]), np.hstack([Bte, Zte]),
                    y_train, y_test,
                )

        if not args.no_bertopic:
            for k in args.n_topics:
                Ptr, Pte = bertopic_vecs[k]
                all_rows += run_experiment(
                    f"BERT + BERTopic (k={k})",
                    np.hstack([Btr, Ptr]), np.hstack([Bte, Pte]),
                    y_train, y_test,
                )

        # BERTopic using Legal-BERT embeddings as backbone (no MiniLM re-encoding)
        if args.bertopic_use_bert_embs and not args.no_bertopic:
            print("=" * 60)
            print("5. BERTopic with Legal-BERT embeddings as backbone")
            print("=" * 60)
            for k in args.n_topics:
                print(f"  Fitting BERTopic (nr_topics={k}) on Legal-BERT embeddings ...")
                bt_model, Ptr_raw = fit_bertopic(
                    X_train, k, precomputed_embeddings=Btr
                )
                Pte_raw = bertopic_transform(bt_model, X_test, precomputed_embeddings=Bte)
                Ptr_lb  = _safe_probs(Ptr_raw)
                Pte_lb  = _safe_probs(Pte_raw)

                all_rows += run_experiment(
                    f"BERTopic-LegalBERT only (k={k})", Ptr_lb, Pte_lb, y_train, y_test
                )
                Xtr_comb = hstack([Xtr_tfidf, csr_matrix(Ptr_lb)])
                Xte_comb = hstack([Xte_tfidf, csr_matrix(Pte_lb)])
                all_rows += run_experiment(
                    f"TF-IDF + BERTopic-LegalBERT (k={k})", Xtr_comb, Xte_comb, y_train, y_test
                )
                all_rows += run_experiment(
                    f"BERT + BERTopic-LegalBERT (k={k})",
                    np.hstack([Btr, Ptr_lb]), np.hstack([Bte, Pte_lb]),
                    y_train, y_test,
                )

    # ---- Summary table -------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY — Article 6, Random Split (75/25)")
    print("=" * 60)
    df_res = pd.DataFrame(all_rows)
    print(df_res.to_string(index=False))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'topic_hybrid_results.csv')
        df_res.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default='data',
                        help='Root data directory (contains processed/processed.csv)')
    parser.add_argument('--model_name', type=str,   default='nlpaueb/legal-bert-base-uncased',
                        help='HuggingFace model name for chunked BERT embeddings')
    parser.add_argument('--n_topics',   type=int,   nargs='+', default=[10, 20, 30],
                        help='LDA topic counts to sweep')
    parser.add_argument('--max_chunks', type=int,   default=8,
                        help='Max 510-token chunks per document')
    parser.add_argument('--batch_size', type=int,   default=16,
                        help='Docs per batch during BERT encoding')
    parser.add_argument('--no_bert',           action='store_true', default=False,
                        help='Skip BERT embedding extraction (LDA/TF-IDF/BERTopic experiments only)')
    parser.add_argument('--bert_checkpoint',   type=str,   default='',
                        help='Path to fine-tuned HF encoder (overrides --model_name for BERT encoding)')
    parser.add_argument('--bertopic_use_bert_embs', action='store_true', default=False,
                        help='Use chunked Legal-BERT embeddings as BERTopic backbone (requires BERT encoding)')
    parser.add_argument('--no_lda',       action='store_true', default=False,
                        help='Skip LDA experiments')
    parser.add_argument('--no_bertopic',  action='store_true', default=False,
                        help='Skip BERTopic experiments')
    parser.add_argument('--bert_gpu',     type=int,   default=0,
                        help='CUDA device index for Legal-BERT encoding (default 0)')
    parser.add_argument('--cpu',          action='store_true', default=False,
                        help='Force CPU even if GPU available')
    parser.add_argument('--output_dir',   type=str,   default='results/topic_hybrid',
                        help='Directory for saving result CSV')
    main(parser.parse_args())
