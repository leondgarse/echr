"""
Debiasing probe: progressively remove spurious features from TF-IDF SVM.

Three conditions compared on the same train/test split:
  baseline   — legal_tokenizer with isalpha() (years already excluded)
  no_places  — baseline + ECHR country/city names added to stopwords
  no_places_no_years_raw — also strip 4-digit years from raw text before tokenising
                           (catches "1997." or "1997," that survive isalpha)

If performance drops sharply under no_places, place names were load-bearing.
If performance is stable, the model has learned other generalisable features.

Usage:
  python scripts/svm_debiased.py \
    --data_path data/processed/processed.csv --article6_only
"""

import argparse
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Stopword sets ────────────────────────────────────────────────────────────

MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation', 'articles': 'article',
}

BASE_STOP = set(stopwords.words('english')) | {'court', 'case', 'mr', 'mrs', 'ms'}

# Country names, demonyms, major cities for all ECHR-relevant states
PLACE_STOP = {
    # Russia
    'russia', 'russian', 'moscow', 'chelyabinsk', 'novosibirsk', 'omsk',
    'yekaterinburg', 'volgograd', 'kazan', 'rostov', 'tula', 'saratov',
    'samara', 'ufa', 'perm', 'voronezh', 'krasnoyarsk', 'vladivostok',
    # Turkey
    'turkey', 'turkish', 'ankara', 'istanbul', 'izmir', 'bursa', 'adana',
    'gaziantep', 'konya', 'antalya', 'diyarbakir', 'mersin',
    # UK / Great Britain
    'uk', 'britain', 'british', 'england', 'english', 'wales', 'welsh',
    'scotland', 'scottish', 'london', 'edinburgh', 'cardiff', 'belfast',
    'manchester', 'birmingham', 'liverpool', 'leeds', 'sheffield',
    # Poland
    'poland', 'polish', 'warsaw', 'krakow', 'lodz', 'wroclaw', 'poznan',
    'gdansk', 'szczecin', 'bydgoszcz', 'lublin', 'katowice',
    # Romania
    'romania', 'romanian', 'bucharest', 'cluj', 'timisoara', 'iasi',
    'constanta', 'craiova', 'brasov', 'galati', 'ploiesti',
    # Germany
    'germany', 'german', 'berlin', 'hamburg', 'munich', 'cologne',
    'frankfurt', 'stuttgart', 'dusseldorf', 'dortmund', 'essen',
    'bremen', 'leipzig', 'dresden', 'hannover', 'nuremberg',
    # France
    'france', 'french', 'paris', 'marseille', 'lyon', 'toulouse',
    'nice', 'nantes', 'strasbourg', 'montpellier', 'bordeaux',
    'lille', 'rennes',
    # Belgium
    'belgium', 'belgian', 'brussels', 'antwerp', 'ghent', 'liege',
    'bruges', 'leuven',
    # Austria
    'austria', 'austrian', 'vienna', 'graz', 'linz', 'salzburg',
    'innsbruck', 'klagenfurt',
    # Netherlands
    'netherlands', 'dutch', 'amsterdam', 'rotterdam', 'hague', 'utrecht',
    'eindhoven', 'groningen', 'tilburg',
    # Sweden
    'sweden', 'swedish', 'stockholm', 'gothenburg', 'malmo', 'uppsala',
    'vasteras', 'orebro', 'linkoping',
    # Norway
    'norway', 'norwegian', 'oslo', 'bergen', 'trondheim', 'stavanger',
    # Ukraine
    'ukraine', 'ukrainian', 'kyiv', 'kiev', 'kharkiv', 'odessa',
    'dnipro', 'donetsk', 'zaporizhzhia', 'lviv', 'mykolaiv',
    # Greece
    'greece', 'greek', 'athens', 'thessaloniki', 'piraeus', 'patras',
    # Spain
    'spain', 'spanish', 'madrid', 'barcelona', 'valencia', 'seville',
    'zaragoza', 'malaga', 'bilbao',
    # Italy
    'italy', 'italian', 'rome', 'milan', 'naples', 'turin', 'palermo',
    'genoa', 'bologna', 'florence', 'venice',
    # Switzerland
    'switzerland', 'swiss', 'zurich', 'geneva', 'bern', 'berne',
    'basel', 'lausanne',
    # Hungary
    'hungary', 'hungarian', 'budapest', 'debrecen', 'miskolc', 'pecs',
    # Moldova
    'moldova', 'moldovan', 'chisinau',
    # Generic ECHR/geographic terms
    'european', 'europe', 'council', 'strasbourg', 'echr',
    # Polish-language artefacts
    'sąd', 'sad',
    # Romanian-language artefacts
    'judecătoriei', 'judecatoria',
}

# ── Tokenisers ────────────────────────────────────────────────────────────────

wnl = WordNetLemmatizer()

def make_tokenizer(extra_stop):
    stop = BASE_STOP | extra_stop
    def tokenizer(text):
        tokens = [t for t in text.lower().split() if t.isalpha() and t not in stop]
        return [MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n')) for t in tokens]
    return tokenizer

def strip_years(text):
    """Remove 4-digit year tokens (catches e.g. '1997.' that survive isalpha)."""
    return re.sub(r'\b(19|20)\d{2}\b', ' ', text)

# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average='macro')
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    acc = (y_true == y_pred).mean()
    print(f"    {name:<35} macro-F1={mf1:.3f}  acc={acc:.3f}  "
          f"F1(no-viol)={f1s[0]:.3f}  F1(viol)={f1s[1]:.3f}")
    return mf1

def make_pipe(tokenizer):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenizer, token_pattern=None, lowercase=False,
            min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2),
        )),
        ('clf', LinearSVC(C=0.1, class_weight='balanced', max_iter=2000)),
    ])

def contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )

# ── Main ─────────────────────────────────────────────────────────────────────

def run(data_path, article6_only):
    df = pd.read_csv(data_path)

    if article6_only:
        mask = contains_article(df['violation_articles'], '6') | \
               contains_article(df['nonviolation_articles'], '6')
        df = df[mask].copy()
        df['label'] = contains_article(df['violation_articles'], '6').astype(int)
        print(f"\nArticle 6 subset: {len(df)} cases, {df['label'].mean():.1%} violation\n")
    else:
        print(f"\nAll articles: {len(df)} cases, {df['label'].mean():.1%} violation\n")

    conditions = [
        ('baseline (years already excl.)',    df['text'],                    make_tokenizer(set())),
        ('+ remove place names',               df['text'],                    make_tokenizer(PLACE_STOP)),
        ('+ remove places + strip years',      df['text'].apply(strip_years), make_tokenizer(PLACE_STOP)),
    ]

    cutoff = int(df['year'].quantile(0.75))

    for label, texts, tokenizer in conditions:
        df_work = df.copy()
        df_work['text'] = texts

        X_tr, X_te, y_tr, y_te = train_test_split(
            df_work['text'], df_work['label'],
            test_size=0.25, stratify=df_work['label'], random_state=42
        )
        train_t = df_work[df_work['year'] <  cutoff]
        test_t  = df_work[df_work['year'] >= cutoff]

        pipe = make_pipe(tokenizer)

        print(f"  [{label}]")
        pipe.fit(X_tr, y_tr)
        evaluate('Random split', y_te.values, pipe.predict(X_te))
        pipe.fit(train_t['text'], train_t['label'])
        evaluate(f'Temporal split (year<{cutoff})', test_t['label'].values, pipe.predict(test_t['text']))
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--article6_only', action=argparse.BooleanOptionalAction, default=True)
    main_args = parser.parse_args()
    run(main_args.data_path, main_args.article6_only)
