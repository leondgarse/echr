"""
svm_truncated.py — Compare SVM on full text vs DeBERTa head_tail truncated text.

Isolates whether SVM's advantage over DeBERTa comes purely from seeing the full
document (vs 512 tokens). Runs SVM on three text versions:
  1. Full text (baseline)
  2. head_tail truncated (same 512-token window as DeBERTa)
  3. head only (first 512 tokens)

Uses the DeBERTa tokenizer so token counts exactly match what DeBERTa sees.

Usage:
  python scripts/svm_truncated.py \
    --data_dir data_v1
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation', 'articles': 'article',
}
BASE_STOP = set(stopwords.words('english')) | {'court', 'case', 'mr', 'mrs', 'ms'}
wnl = WordNetLemmatizer()


def legal_tokenizer(text):
    tokens = [t for t in text.lower().split() if t.isalpha() and t not in BASE_STOP]
    return [MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n')) for t in tokens]


def _contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )


def truncate_head_tail(text, tokenizer, max_len=512, n_head=128):
    """Reproduce DeBERTa head_tail truncation, then decode back to text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tail = max_len - n_head - 2  # 382
    if len(tokens) <= max_len - 2:
        kept = tokens
    else:
        kept = tokens[:n_head] + tokens[-n_tail:]
    return tokenizer.decode(kept, skip_special_tokens=True)


def truncate_head(text, tokenizer, max_len=512):
    """Keep only first max_len tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_len - 2]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def run_svm(train_texts, test_texts, train_labels, test_labels, name):
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=legal_tokenizer, token_pattern=None, lowercase=False,
            min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2),
        )),
        ('clf', LinearSVC(C=0.1, class_weight='balanced', max_iter=2000)),
    ])
    pipe.fit(train_texts, train_labels)
    y_pred = pipe.predict(test_texts)
    mf1 = f1_score(test_labels, y_pred, average='macro')
    f1s = f1_score(test_labels, y_pred, average=None, labels=[0, 1])
    acc = (test_labels == y_pred).mean()
    print(f"  {name:<45} macro-F1={mf1:.4f}  acc={acc:.3f}  "
          f"F1(NV)={f1s[0]:.3f}  F1(V)={f1s[1]:.3f}")
    return mf1


def main(args):
    data_path = os.path.join(args.data_dir, 'processed', 'processed.csv')
    df = pd.read_csv(data_path)

    # Article 6 subset with Art.6-specific label
    mask = (_contains_article(df['violation_articles'], '6') |
            _contains_article(df['nonviolation_articles'], '6'))
    df = df[mask].copy()
    df['label'] = _contains_article(df['violation_articles'], '6').astype(int)
    print(f"Art.6 subset: {len(df)} cases, {df['label'].mean():.1%} violation")

    train_val, test_df = train_test_split(
        df, test_size=0.25, stratify=df['label'], random_state=args.data_seed
    )
    val_size = 0.15 / 0.75
    train_df, _ = train_test_split(
        train_val, test_size=val_size, stratify=train_val['label'],
        random_state=args.data_seed
    )
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}\n")

    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    # --- 1. Full text (SVM baseline) ---
    print("Running SVM variants:")
    run_svm(train_df['text'].fillna('').tolist(),
            test_df['text'].fillna('').tolist(),
            y_train, y_test, "Full text (SVM baseline)")

    # --- 2. head_tail truncated (same as DeBERTa) ---
    print("Loading DeBERTa tokenizer for truncation...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    print("Truncating texts (head_tail 512 tokens)...")
    train_ht = [truncate_head_tail(t, tokenizer) for t in train_df['text'].fillna('')]
    test_ht  = [truncate_head_tail(t, tokenizer) for t in test_df['text'].fillna('')]
    run_svm(train_ht, test_ht, y_train, y_test,
            "head_tail 512 tok (same as DeBERTa)")

    # --- 3. head only ---
    train_h = [truncate_head(t, tokenizer) for t in train_df['text'].fillna('')]
    test_h  = [truncate_head(t, tokenizer) for t in test_df['text'].fillna('')]
    run_svm(train_h, test_h, y_train, y_test,
            "head only 512 tok")

    # --- 4. Extended: 1024 and 2048 tokens (head_tail) ---
    for max_len in [1024, 2048]:
        n_head = 256 if max_len == 1024 else 512
        train_ext = [truncate_head_tail(t, tokenizer, max_len=max_len, n_head=n_head)
                     for t in train_df['text'].fillna('')]
        test_ext  = [truncate_head_tail(t, tokenizer, max_len=max_len, n_head=n_head)
                     for t in test_df['text'].fillna('')]
        run_svm(train_ext, test_ext, y_train, y_test,
                f"head_tail {max_len} tok")

    # --- Token coverage stats ---
    print("\nToken coverage stats (DeBERTa tokenizer):")
    lengths = [len(tokenizer.encode(t, add_special_tokens=False))
               for t in df['text'].fillna('')]
    lengths = np.array(lengths)
    for pct in [50, 75, 90, 95, 100]:
        print(f"  p{pct}: {np.percentile(lengths, pct):.0f} tokens")
    print(f"  mean: {lengths.mean():.0f}  >512: {(lengths>512).mean():.1%}"
          f"  >1024: {(lengths>1024).mean():.1%}"
          f"  >2048: {(lengths>2048).mean():.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  type=str, required=True)
    parser.add_argument('--data_seed', type=int, default=42)
    main(parser.parse_args())
