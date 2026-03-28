"""
Quick TF-IDF + LinearSVC comparison on a given processed.csv.
Uses best hyperparameters from train_svm.ipynb (C=0.1, class_weight='balanced').
Reports random split and temporal split macro-F1, matching SVM notebook methodology.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# ---- Legal tokenizer (ported from train_svm.ipynb / EDA.ipynb) ----
MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation', 'articles': 'article',
}
LEGAL_STOP = set(stopwords.words('english')) | {'court', 'case', 'mr', 'mrs', 'ms'}
wnl = WordNetLemmatizer()

def legal_tokenizer(text):
    tokens = [t for t in text.lower().split() if t.isalpha() and t not in LEGAL_STOP]
    return [MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n')) for t in tokens]

def contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )

def evaluate(name, y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average='macro')
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    acc = (y_true == y_pred).mean()
    print(f"  {name:<30} macro-F1={mf1:.3f}  acc={acc:.3f}  "
          f"F1(no-viol)={f1s[0]:.3f}  F1(viol)={f1s[1]:.3f}")
    return mf1

def run(data_path, article6_only, seed=42):
    df = pd.read_csv(data_path)

    if article6_only:
        mask = contains_article(df['violation_articles'], '6') | \
               contains_article(df['nonviolation_articles'], '6')
        df = df[mask].copy()
        df['label'] = contains_article(df['violation_articles'], '6').astype(int)
        print(f"\nArticle 6 subset: {len(df)} cases, {df['label'].mean():.1%} violation")
    else:
        print(f"\nAll articles: {len(df)} cases, {df['label'].mean():.1%} violation")

    tfidf_params = dict(
        tokenizer=legal_tokenizer, token_pattern=None, lowercase=False,
        min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2),
    )

    pipe_dummy = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                            ('clf',  DummyClassifier(strategy='most_frequent'))])
    pipe_svm   = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                            ('clf',  LinearSVC(C=0.1, class_weight='balanced', max_iter=2000))])
    pipe_lr    = Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                            ('clf',  LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000))])

    # --- Random split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        df['text'], df['label'], test_size=0.25, stratify=df['label'], random_state=seed
    )
    print(f"\n  [Random split 75/25]  train={len(X_tr)}  test={len(X_te)}")
    for name, pipe in [('Dummy', pipe_dummy), ('SVM', pipe_svm), ('LogReg', pipe_lr)]:
        pipe.fit(X_tr, y_tr)
        evaluate(name, y_te.values, pipe.predict(X_te))

    # --- Temporal split ---
    cutoff = int(df['year'].quantile(0.75))
    train_t = df[df['year'] <  cutoff]
    test_t  = df[df['year'] >= cutoff]
    print(f"\n  [Temporal split year<{cutoff}]  train={len(train_t)}  test={len(test_t)}")
    for name, pipe in [('Dummy', pipe_dummy), ('SVM', pipe_svm), ('LogReg', pipe_lr)]:
        pipe.fit(train_t['text'], train_t['label'])
        evaluate(name, test_t['label'].values, pipe.predict(test_t['text']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--article6_only', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args.data_path, args.article6_only)
