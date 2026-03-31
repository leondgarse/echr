"""
test_preproc_bert.py — Compare text preprocessing variants for LegalBERT.

Answers: does year-masking / isalpha / SVM-style preprocessing help or hurt
LegalBERT-512 and ChunkedBERT macro-F1 on the ECHR Article-6 task?

Training stack mirrors echr.ipynb exactly:
  - Focal loss (gamma=2)
  - LLRD optimizer (decay=0.90)
  - Gradient accumulation (chunked: phys_bs=2, ga=4)
  - Linear warmup schedule (10% / 15% steps)
  - 4-seed ensemble + val-tuned threshold
  - 5 epochs, best-val-F1 checkpoint per seed

Usage:
    # Token-length analysis only (fast, no GPU):
    python scripts/test_preproc_bert.py

    # Full fine-tuning comparison:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_preproc_bert.py \\
        --train_bert --arch chunked --variant raw

    # All variants / archs:
    python scripts/test_preproc_bert.py --train_bert
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup, set_seed as hf_set_seed)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _SW = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _SW = set(stopwords.words('english'))

# ---------------------------------------------------------------------------
# Hyperparameters (mirrors echr.ipynb HP dict)
# ---------------------------------------------------------------------------
HP = dict(
    model_name  = 'nlpaueb/legal-bert-base-uncased',
    seeds       = [0, 1, 2, 42],
    epochs      = 5,
    batch_size  = 8,       # logical batch size
    lr          = 2e-5,
    wd          = 0.01,
    llrd_decay  = 0.90,
    focal_gamma = 2.0,
    max_len     = 512,
    n_head      = 128,     # head tokens for 512-tok head+tail truncation
    chunk_size  = 510,
    n_chunks    = 4,       # overridable via --n_chunks
    warmup_frac_512     = 0.10,
    warmup_frac_chunked = 0.15,
    phys_bs_chunked     = 2,   # physical batch for chunked (grad-accum)
)

# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation',  'articles': 'article',
}
LEGAL_STOP = _SW | {'court', 'case', 'mr', 'mrs', 'ms'}
wnl = WordNetLemmatizer()


def legal_tokenizer(text: str) -> list:
    tokens = [t for t in text.lower().split() if t.isalpha() and t not in LEGAL_STOP]
    return [MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n')) for t in tokens]

TFIDF_PARAMS = dict(
    tokenizer=legal_tokenizer, token_pattern=None, lowercase=False,
    min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2),
)


def build_vocab(texts: list, min_df: int = 3, max_df: float = 0.90) -> set:
    from collections import Counter
    n = len(texts)
    df_cnt: Counter = Counter()
    for text in texts:
        for tok in set(legal_tokenizer(text)):
            df_cnt[tok] += 1
    return {t for t, c in df_cnt.items() if c >= min_df and c / n <= max_df}


def apply_preprocessed(text: str, vocab: set) -> str:
    """Full SVM-equivalent: isalpha + stop words + lemmatise + vocab filter."""
    return ' '.join(t for t in legal_tokenizer(text) if t in vocab)


def apply_isalpha(text: str) -> str:
    """Keep only purely alphabetic whitespace-split tokens; preserve casing."""
    return ' '.join(t for t in text.split() if t.isalpha())


_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

def apply_year_masked(text: str) -> str:
    """Remove 4-digit years (1900-2099); keep everything else."""
    return ' '.join(_YEAR_RE.sub('', text).split())


# ---------------------------------------------------------------------------
# Token-length analysis
# ---------------------------------------------------------------------------

def analyse_token_lengths(df: pd.DataFrame, tokenizer, vocab: set):
    print('\n' + '=' * 65)
    print('Token-length analysis')
    print('=' * 65)
    print(f'Vocab size (min_df=3, max_df=0.90): {len(vocab):,}')

    modes = [
        ('raw',              df['facts'].tolist()),
        ('year_masked',      [apply_year_masked(t)    for t in df['facts']]),
        ('isalpha',          [apply_isalpha(t)         for t in df['facts']]),
        ('preprocessed',     [apply_preprocessed(t, vocab) for t in df['facts']]),
    ]
    print(f'\n{"Mode":<18} {"Med":>5} {"P75":>5} {"P90":>5} {"P95":>5} '
          f'{"Max":>6}  {"≤512":>7}  {"≤2040":>7}')
    print('-' * 65)
    for mode, texts in modes:
        lens = np.array([
            len(tokenizer(t, add_special_tokens=True,
                          truncation=False, return_attention_mask=False)['input_ids'])
            for t in texts
        ])
        print(f'{mode:<18} {int(np.median(lens)):>5} {int(np.percentile(lens,75)):>5} '
              f'{int(np.percentile(lens,90)):>5} {int(np.percentile(lens,95)):>5} '
              f'{int(lens.max()):>6}  {(lens<=512).mean()*100:>6.1f}%  '
              f'{(lens<=2040).mean()*100:>6.1f}%')


# ---------------------------------------------------------------------------
# Datasets — exact copies of echr.ipynb
# ---------------------------------------------------------------------------

class ECHRDataset512(Dataset):
    """Head+tail 512-token truncation (n_head=128)."""
    def __init__(self, texts, labels, tokenizer, max_len=512, n_head=128):
        self.items = []
        n_tail = max_len - n_head - 2
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_len - 2:
                ids = ids[:n_head] + ids[-n_tail:]
            enc = tokenizer.prepare_for_model(
                ids, add_special_tokens=True,
                max_length=max_len, padding='max_length',
                truncation=False, return_tensors='pt')
            self.items.append(({k: v.squeeze(0) for k, v in enc.items()},
                               torch.tensor(label, dtype=torch.long)))

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


class ECHRChunkedDataset(Dataset):
    """4×510-token chunked dataset (mean-pool CLS)."""
    def __init__(self, texts, labels, tokenizer, chunk_size=510, n_chunks=4):
        self.items = []
        seq_len = chunk_size + 2
        for text, label in zip(texts, labels):
            ids    = tokenizer.encode(text, add_special_tokens=False)
            raw    = [ids[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
            n_real = sum(1 for c in raw if c)
            input_ids_list, attn_mask_list = [], []
            for chunk_toks in raw:
                full = [tokenizer.cls_token_id] + chunk_toks + [tokenizer.sep_token_id]
                full = full[:seq_len]
                pad  = seq_len - len(full)
                mask = [1]*len(full) + [0]*pad
                full = full + [tokenizer.pad_token_id]*pad
                input_ids_list.append(torch.tensor(full,  dtype=torch.long))
                attn_mask_list.append(torch.tensor(mask,  dtype=torch.long))
            self.items.append({
                'input_ids':      torch.stack(input_ids_list),
                'attention_mask': torch.stack(attn_mask_list),
                'chunk_mask':     torch.tensor(
                    [1]*n_real + [0]*(n_chunks - n_real), dtype=torch.float),
                'label':          torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, **enc):
        cls = self.bert(**enc).last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))


class ChunkedBERT(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, chunk_mask):
        B, C, L  = input_ids.shape
        cls_emb  = self.encoder(
            input_ids=input_ids.view(B*C, L),
            attention_mask=attention_mask.view(B*C, L),
        ).last_hidden_state[:, 0].view(B, C, -1)
        cm     = chunk_mask.unsqueeze(-1)
        pooled = (cls_emb * cm).sum(1) / cm.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Training utilities (exact notebook copies)
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, class_weights, gamma=2.0):
    log_p = F.log_softmax(logits, dim=-1)
    ce    = F.nll_loss(log_p, labels, weight=class_weights, reduction='none')
    p_t   = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)
    return ((1 - p_t) ** gamma * ce).mean()


def build_llrd_optimizer(encoder, classifier_params, lr, wd, decay):
    layers = encoder.encoder.layer
    n      = len(layers)
    groups = [{'params': list(classifier_params), 'lr': lr, 'weight_decay': wd}]
    for depth, layer in enumerate(reversed(layers)):
        groups.append({'params': list(layer.parameters()),
                       'lr': lr * (decay ** (depth + 1)), 'weight_decay': wd})
    groups.append({'params': list(encoder.embeddings.parameters()),
                   'lr': lr * (decay ** (n + 1)), 'weight_decay': wd})
    lrs = [g['lr'] for g in groups]
    print(f'    LLRD lr range: {min(lrs):.2e} – {max(lrs):.2e}')
    return AdamW(groups, eps=1e-8)


def tune_threshold(val_probs, y_val):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.15, 0.86, 0.01):
        f1 = f1_score(y_val, (val_probs[:, 1] > t).astype(int), average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def get_probs(model, loader, device, arch):
    model.eval()
    ps = []
    with torch.no_grad():
        for batch in loader:
            if arch == 'bert512':
                enc = {k: v.to(device) for k, v in batch[0].items()}
                logits = model(**enc)
            else:
                logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['chunk_mask'].to(device),
                )
            ps.append(torch.softmax(logits, -1).cpu().numpy())
    return np.vstack(ps)


# ---------------------------------------------------------------------------
# Per-seed training
# ---------------------------------------------------------------------------

def train_seed_512(seed, train_texts, val_texts, test_texts,
                   y_train, y_val, y_test, hp, device, class_weights):
    hf_set_seed(seed)
    tok   = AutoTokenizer.from_pretrained(hp['model_name'])
    model = BertClassifier(hp['model_name']).to(device)

    tr_ds = ECHRDataset512(train_texts, y_train, tok, hp['max_len'], hp['n_head'])
    vl_ds = ECHRDataset512(val_texts,   y_val,   tok, hp['max_len'], hp['n_head'])
    te_ds = ECHRDataset512(test_texts,  y_test,  tok, hp['max_len'], hp['n_head'])
    tr_ld = DataLoader(tr_ds, batch_size=hp['batch_size'],   shuffle=True,  num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=hp['batch_size']*2, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=hp['batch_size']*2, shuffle=False, num_workers=0)

    cw  = class_weights.to(device)
    opt = build_llrd_optimizer(model.bert, model.classifier.parameters(),
                               hp['lr'], hp['wd'], hp['llrd_decay'])
    total = len(tr_ld) * hp['epochs']
    sched = get_linear_schedule_with_warmup(
        opt, int(hp['warmup_frac_512'] * total), total)

    best_vf1, best_state = 0.0, None
    for ep in range(hp['epochs']):
        model.train()
        for batch in tr_ld:
            enc = {k: v.to(device) for k, v in batch[0].items()}
            lbl = batch[1].to(device)
            loss = focal_loss(model(**enc), lbl, cw, hp['focal_gamma'])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        vp  = get_probs(model, vl_ld, device, 'bert512')
        vf1 = f1_score(y_val, vp.argmax(1), average='macro')
        print(f'    ep{ep+1}: val_f1={vf1:.3f}', end='  ', flush=True)
        if vf1 > best_vf1:
            best_vf1  = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f'→ best={best_vf1:.3f}')
    model.load_state_dict(best_state)
    return get_probs(model, vl_ld, device, 'bert512'), \
           get_probs(model, te_ld, device, 'bert512')


def train_seed_chunked(seed, train_texts, val_texts, test_texts,
                       y_train, y_val, y_test, hp, device, class_weights):
    hf_set_seed(seed)
    tok   = AutoTokenizer.from_pretrained(hp['model_name'])
    model = ChunkedBERT(hp['model_name']).to(device)

    tr_ds = ECHRChunkedDataset(train_texts, y_train, tok, hp['chunk_size'], hp['n_chunks'])
    vl_ds = ECHRChunkedDataset(val_texts,   y_val,   tok, hp['chunk_size'], hp['n_chunks'])
    te_ds = ECHRChunkedDataset(test_texts,  y_test,  tok, hp['chunk_size'], hp['n_chunks'])

    phys_bs = hp['phys_bs_chunked']
    ga      = max(1, hp['batch_size'] // phys_bs)
    tr_ld = DataLoader(tr_ds, batch_size=phys_bs,   shuffle=True,  num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=phys_bs*2, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=phys_bs*2, shuffle=False, num_workers=0)

    cw  = class_weights.to(device)
    opt = build_llrd_optimizer(model.encoder, model.classifier.parameters(),
                               hp['lr'], hp['wd'], hp['llrd_decay'])
    total = (len(tr_ld) // ga) * hp['epochs']
    sched = get_linear_schedule_with_warmup(
        opt, int(hp['warmup_frac_chunked'] * total), total)

    best_vf1, best_state = 0.0, None
    for ep in range(hp['epochs']):
        model.train(); opt.zero_grad()
        for step, batch in enumerate(tr_ld):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            cmask = batch['chunk_mask'].to(device)
            lbl   = batch['label'].to(device)
            loss  = focal_loss(model(ids, mask, cmask), lbl, cw, hp['focal_gamma']) / ga
            loss.backward()
            if (step + 1) % ga == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        vp  = get_probs(model, vl_ld, device, 'chunked')
        vf1 = f1_score(y_val, vp.argmax(1), average='macro')
        print(f'    ep{ep+1}: val_f1={vf1:.3f}', end='  ', flush=True)
        if vf1 > best_vf1:
            best_vf1  = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f'→ best={best_vf1:.3f}')
    model.load_state_dict(best_state)
    return get_probs(model, vl_ld, device, 'chunked'), \
           get_probs(model, te_ld, device, 'chunked')


# ---------------------------------------------------------------------------
# Ensemble run (4 seeds → average probs → tuned threshold)
# ---------------------------------------------------------------------------

def run_ensemble(arch, variant, train_texts, val_texts, test_texts,
                 y_train, y_val, y_test, hp, device):
    cw = torch.tensor(
        compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(y_train)),
        dtype=torch.float32)

    train_fn = train_seed_512 if arch == 'bert512' else train_seed_chunked
    all_vp, all_tp = [], []

    for seed in hp['seeds']:
        print(f'\n  [{arch} / {variant}] seed={seed}')
        vp, tp = train_fn(seed, train_texts, val_texts, test_texts,
                          y_train, y_val, y_test, hp, device, cw)
        all_vp.append(vp); all_tp.append(tp)
        torch.cuda.empty_cache()

    ens_val  = np.mean(all_vp, axis=0)
    ens_test = np.mean(all_tp, axis=0)
    best_t, best_vf1 = tune_threshold(ens_val, y_val)
    pred = (ens_test[:, 1] > best_t).astype(int)
    mf1  = f1_score(y_test, pred, average='macro')
    f1s  = f1_score(y_test, pred, average=None, labels=[0, 1])
    print(f'\n  Ensemble macro-F1={mf1:.3f}  '
          f'F1(NV)={f1s[0]:.3f}  F1(V)={f1s[1]:.3f}  '
          f'threshold={best_t:.2f}')
    return mf1, f1s, best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  default='data/processed/processed.csv')
    parser.add_argument('--train_bert', action='store_true')
    parser.add_argument('--arch',    default=None, choices=['bert512', 'chunked'])
    parser.add_argument('--variant', default=None,
                        choices=['raw', 'year_masked', 'isalpha', 'preprocessed'])
    parser.add_argument('--epochs',     type=int, default=HP['epochs'])
    parser.add_argument('--seeds',      type=int, nargs='+', default=HP['seeds'])
    parser.add_argument('--batch_size', type=int, default=HP['batch_size'])
    parser.add_argument('--n_chunks',   type=int, default=HP['n_chunks'],
                        help='Number of 510-token chunks (2=1020 tok, 4=2040 tok)')
    parser.add_argument('--device',     default=None,
                        help='cuda device index e.g. 0, or "cpu"')
    args = parser.parse_args()

    hp = {**HP, 'epochs': args.epochs, 'seeds': args.seeds,
          'batch_size': args.batch_size, 'n_chunks': args.n_chunks}

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_path).rename(columns={'text': 'facts'})
    df = df.dropna(subset=['facts'])

    def art6_label(row):
        viol   = [a.strip() for a in str(row.get('violation_articles',   '')).split(';')]
        noviol = [a.strip() for a in str(row.get('nonviolation_articles', '')).split(';')]
        if '6' in viol:   return 1
        if '6' in noviol: return 0
        return -1

    df['label'] = df.apply(art6_label, axis=1)
    df = df[df['label'] >= 0].copy()

    # Exact same split as echr.ipynb (SEED=42, 75/25 then 15% of 75% for val)
    train_val, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df['label'])
    train_df, val_df   = train_test_split(
        train_val, test_size=0.15/0.75, random_state=42, stratify=train_val['label'])

    print(f'Dataset: {len(df)} cases  '
          f'(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})  '
          f'{df["label"].mean():.1%} violation')

    # ── Token-length analysis ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hp['model_name'])
    vocab = build_vocab(train_df['facts'].tolist(), min_df=3, max_df=0.90)
    analyse_token_lengths(
        pd.concat([train_df.assign(split='train'),
                   val_df.assign(split='val'),
                   test_df.assign(split='test')]),
        tokenizer, vocab)

    if not args.train_bert:
        print('\n(Pass --train_bert to run fine-tuning comparison)')
        return

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(f'cuda:{args.device}' if args.device.isdigit()
                              else args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    # ── Prepare text variants ─────────────────────────────────────────────────
    def make_texts(raw_list):
        return {
            'raw':          raw_list,
            'year_masked':  [apply_year_masked(t)        for t in raw_list],
            'isalpha':      [apply_isalpha(t)             for t in raw_list],
            'preprocessed': [apply_preprocessed(t, vocab) for t in raw_list],
        }

    tr_texts = make_texts(train_df['facts'].tolist())
    va_texts = make_texts(val_df['facts'].tolist())
    te_texts = make_texts(test_df['facts'].tolist())
    y_train  = train_df['label'].tolist()
    y_val    = val_df['label'].tolist()
    y_test   = test_df['label'].tolist()

    variants_to_run = ['raw', 'year_masked', 'isalpha', 'preprocessed']
    if args.variant:
        variants_to_run = [args.variant]

    archs_to_run = ['bert512', 'chunked']
    if args.arch:
        archs_to_run = [args.arch]

    # ── SVM baselines (CPU, fast) ─────────────────────────────────────────────
    def run_svm(variant):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
            ('clf',   LinearSVC(C=0.1, class_weight='balanced', max_iter=3000)),
        ])
        # SVM always uses its own tokenizer — for 'preprocessed' variant the text
        # is already lemmatised/filtered, so re-tokenising is safe (idempotent).
        pipe.fit(tr_texts[variant], y_train)
        pred = pipe.predict(te_texts[variant])
        mf1  = f1_score(y_test, pred, average='macro')
        f1s  = f1_score(y_test, pred, average=None, labels=[0, 1])
        return mf1, f1s

    print(f'\n{"="*65}')
    print('SVM BASELINES')
    print(f'{"="*65}')
    print(f'{"Variant":<16} {"Macro-F1":>9} {"F1(NV)":>7} {"F1(V)":>7}')
    print('-' * 44)
    svm_results = {}
    for variant in variants_to_run:
        mf1, f1s = run_svm(variant)
        svm_results[variant] = {'macro_f1': mf1, 'f1_nv': f1s[0], 'f1_v': f1s[1]}
        print(f'{variant:<16} {mf1:>9.3f} {f1s[0]:>7.3f} {f1s[1]:>7.3f}')

    # ── Run ───────────────────────────────────────────────────────────────────
    results = []
    for variant in variants_to_run:
        for arch in archs_to_run:
            print(f'\n{"="*60}')
            print(f'  {arch.upper()}  |  variant={variant}')
            print(f'{"="*60}')
            mf1, f1s, thr = run_ensemble(
                arch, variant,
                tr_texts[variant], va_texts[variant], te_texts[variant],
                y_train, y_val, y_test, hp, device)
            results.append({
                'arch': arch, 'variant': variant,
                'macro_f1': mf1, 'f1_nv': f1s[0], 'f1_v': f1s[1],
                'threshold': thr,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_chunks_label = f'chunked-{hp["n_chunks"]}x ({hp["n_chunks"]*hp["chunk_size"]} tok)'
    print(f'\n{"="*65}')
    print(f'SUMMARY  [n_chunks={hp["n_chunks"]} → {hp["n_chunks"]*hp["chunk_size"]} tok coverage]')
    print(f'{"="*65}')
    print(f'{"Model":<22} {"Variant":<16} {"Macro-F1":>9} {"F1(NV)":>7} {"F1(V)":>7}')
    print('-' * 62)
    for variant in variants_to_run:
        sv = svm_results.get(variant)
        if sv:
            print(f'{"SVM (full text)":<22} {variant:<16} {sv["macro_f1"]:>9.3f} '
                  f'{sv["f1_nv"]:>7.3f} {sv["f1_v"]:>7.3f}')
        for r in results:
            if r['variant'] != variant:
                continue
            arch_label = n_chunks_label if r['arch'] == 'chunked' else 'bert512 (512 tok)'
            print(f'{arch_label:<22} {variant:<16} {r["macro_f1"]:>9.3f} '
                  f'{r["f1_nv"]:>7.3f} {r["f1_v"]:>7.3f}')
        print()

    print('\nDelta vs raw:')
    for arch in ['bert512', 'chunked']:
        raw = next((r for r in results if r['arch'] == arch
                    and r['variant'] == 'raw'), None)
        if not raw:
            continue
        for r in results:
            if r['arch'] != arch or r['variant'] == 'raw':
                continue
            delta = r['macro_f1'] - raw['macro_f1']
            tag   = '(+)' if delta > 0.005 else ('(-)' if delta < -0.005 else '(~)')
            print(f'  {arch:<10} {r["variant"]:<16}  '
                  f'{raw["macro_f1"]:.3f} → {r["macro_f1"]:.3f}  '
                  f'{delta:+.3f} {tag}')


if __name__ == '__main__':
    main()
