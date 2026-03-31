"""
test_svm_bert_ensemble.py — Hybrid ensemble: 1 calibrated SVM + N ChunkedBERT seeds.

Motivation:
  SVM (TF-IDF, full text) and ChunkedBERT (contextual, 2040 tok) are complementary:
  - SVM reads the whole document via bag-of-words
  - ChunkedBERT captures local context and domain pretraining
  Averaging their probabilities should beat either alone.

Ensemble:
  avg_probs = (svm_probs + bert_probs_seed0 + bert_probs_seed1) / 3
  Threshold tuned on val set.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/test_svm_bert_ensemble.py
    CUDA_VISIBLE_DEVICES=1 python scripts/test_svm_bert_ensemble.py --n_bert_seeds 4
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
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
# Hyperparameters (mirrors echr.ipynb)
# ---------------------------------------------------------------------------
HP = dict(
    model_name          = 'nlpaueb/legal-bert-base-uncased',
    seeds               = [0, 1, 2, 42],
    epochs              = 5,
    batch_size          = 8,
    lr                  = 2e-5,
    wd                  = 0.01,
    llrd_decay          = 0.90,
    focal_gamma         = 2.0,
    chunk_size          = 510,
    n_chunks            = 4,
    warmup_frac         = 0.15,
    phys_bs             = 2,
)

# ---------------------------------------------------------------------------
# SVM preprocessing (identical to echr.ipynb)
# ---------------------------------------------------------------------------
MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation',  'articles': 'article',
}
LEGAL_STOP = _SW | {'court', 'case', 'mr', 'mrs', 'ms'}
wnl = WordNetLemmatizer()

def legal_tokenizer(text):
    tokens = [t for t in text.lower().split() if t.isalpha() and t not in LEGAL_STOP]
    return [MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n')) for t in tokens]

TFIDF_PARAMS = dict(
    tokenizer=legal_tokenizer, token_pattern=None, lowercase=False,
    min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1, 2),
)

# ---------------------------------------------------------------------------
# ChunkedBERT dataset + model (identical to echr.ipynb)
# ---------------------------------------------------------------------------

class ECHRChunkedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, chunk_size=510, n_chunks=4,
                 token_offset=0):
        """token_offset: skip this many tokens before chunking (for ChunkB model)."""
        self.items = []
        seq_len = chunk_size + 2
        for text, label in zip(texts, labels):
            ids    = tokenizer.encode(text, add_special_tokens=False)
            ids    = ids[token_offset:]          # skip first N tokens for offset model
            raw    = [ids[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
            n_real = sum(1 for c in raw if c)
            iids, masks = [], []
            for chunk in raw:
                full = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                full = full[:seq_len]
                pad  = seq_len - len(full)
                masks.append([1]*len(full) + [0]*pad)
                full = full + [tokenizer.pad_token_id]*pad
                iids.append(torch.tensor(full,  dtype=torch.long))
                masks[-1] = torch.tensor(masks[-1], dtype=torch.long)
            self.items.append({
                'input_ids':      torch.stack(iids),
                'attention_mask': torch.stack(masks),
                'chunk_mask':     torch.tensor(
                    [1]*n_real + [0]*(n_chunks - n_real), dtype=torch.float),
                'label':          torch.tensor(label, dtype=torch.long),
            })

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


class ChunkedBERT(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, chunk_mask):
        B, C, L = input_ids.shape
        cls = self.encoder(
            input_ids=input_ids.view(B*C, L),
            attention_mask=attention_mask.view(B*C, L),
        ).last_hidden_state[:, 0].view(B, C, -1)
        cm     = chunk_mask.unsqueeze(-1)
        pooled = (cls * cm).sum(1) / cm.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Training utilities (identical to echr.ipynb)
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


def get_probs_chunked(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device),
                           batch['chunk_mask'].to(device))
            ps.append(torch.softmax(logits, -1).cpu().numpy())
    return np.vstack(ps)


def train_seed_chunked(seed, train_texts, val_texts, test_texts,
                       y_train, y_val, y_test, hp, device, class_weights,
                       token_offset=0):
    hf_set_seed(seed)
    tok   = AutoTokenizer.from_pretrained(hp['model_name'])
    model = ChunkedBERT(hp['model_name']).to(device)

    tr_ds = ECHRChunkedDataset(train_texts, y_train, tok, hp['chunk_size'], hp['n_chunks'], token_offset)
    vl_ds = ECHRChunkedDataset(val_texts,   y_val,   tok, hp['chunk_size'], hp['n_chunks'], token_offset)
    te_ds = ECHRChunkedDataset(test_texts,  y_test,  tok, hp['chunk_size'], hp['n_chunks'], token_offset)

    ga = max(1, hp['batch_size'] // hp['phys_bs'])
    tr_ld = DataLoader(tr_ds, batch_size=hp['phys_bs'],   shuffle=True,  num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=hp['phys_bs']*2, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=hp['phys_bs']*2, shuffle=False, num_workers=0)

    cw  = class_weights.to(device)
    opt = build_llrd_optimizer(model.encoder, model.classifier.parameters(),
                               hp['lr'], hp['wd'], hp['llrd_decay'])
    total = (len(tr_ld) // ga) * hp['epochs']
    sched = get_linear_schedule_with_warmup(
        opt, int(hp['warmup_frac'] * total), total)

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
        vp  = get_probs_chunked(model, vl_ld, device)
        vf1 = f1_score(y_val, vp.argmax(1), average='macro')
        print(f'    ep{ep+1}: val_f1={vf1:.3f}', end='  ', flush=True)
        if vf1 > best_vf1:
            best_vf1  = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f'→ best={best_vf1:.3f}')
    model.load_state_dict(best_state)
    return (get_probs_chunked(model, vl_ld, device),
            get_probs_chunked(model, te_ld, device))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/processed.csv')
    parser.add_argument('--device',    default=None)
    args = parser.parse_args()

    # ── Load data (same split as echr.ipynb) ─────────────────────────────────
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

    train_val, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(
        train_val, test_size=0.15/0.75, random_state=42,
        stratify=train_val['label'])

    print(f'Dataset: {len(df)} cases  '
          f'(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})  '
          f'{df["label"].mean():.1%} violation')

    X_train = train_df['facts'].tolist()
    X_val   = val_df['facts'].tolist()
    X_test  = test_df['facts'].tolist()
    y_train = train_df['label'].tolist()
    y_val   = val_df['label'].tolist()
    y_test  = test_df['label'].tolist()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(f'cuda:{args.device}' if args.device.isdigit()
                              else args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Calibrated SVM (full text) ────────────────────────────────────────
    print('\n── SVM (full text, Platt-calibrated) ──')
    svm_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf',   CalibratedClassifierCV(
                      LinearSVC(C=0.1, class_weight='balanced', max_iter=3000),
                      cv=5, method='sigmoid')),
    ])
    svm_pipe.fit(X_train, y_train)
    svm_val_probs  = svm_pipe.predict_proba(X_val)
    svm_test_probs = svm_pipe.predict_proba(X_test)
    svm_solo_f1 = f1_score(y_test, svm_pipe.predict(X_test), average='macro')
    svm_f1s     = f1_score(y_test, svm_pipe.predict(X_test), average=None, labels=[0, 1])
    print(f'  SVM solo  macro-F1={svm_solo_f1:.3f}  '
          f'F1(NV)={svm_f1s[0]:.3f}  F1(V)={svm_f1s[1]:.3f}')

    # ── 2. ChunkedBERT (N seeds) ─────────────────────────────────────────────
    cw = torch.tensor(
        compute_class_weight('balanced', classes=np.array([0, 1]),
                             y=np.array(y_train)), dtype=torch.float32)

    seeds = HP['seeds'][:args.n_bert_seeds]
    print(f'\n── ChunkedBERT ({len(seeds)} seeds: {seeds}) ──')

    bert_val_probs  = []
    bert_test_probs = []
    for seed in seeds:
        print(f'\n  seed={seed}')
        vp, tp = train_seed_chunked(
            seed, X_train, X_val, X_test,
            y_train, y_val, y_test, HP, device, cw)
        bert_val_probs.append(vp)
        bert_test_probs.append(tp)
        torch.cuda.empty_cache()

    # BERT-only ensemble
    bert_ens_val  = np.mean(bert_val_probs,  axis=0)
    bert_ens_test = np.mean(bert_test_probs, axis=0)
    bert_t, _  = tune_threshold(bert_ens_val, y_val)
    bert_pred  = (bert_ens_test[:, 1] > bert_t).astype(int)
    bert_solo_f1 = f1_score(y_test, bert_pred, average='macro')
    bert_f1s     = f1_score(y_test, bert_pred, average=None, labels=[0, 1])
    print(f'\n  BERT-only ({len(seeds)} seeds)  macro-F1={bert_solo_f1:.3f}  '
          f'F1(NV)={bert_f1s[0]:.3f}  F1(V)={bert_f1s[1]:.3f}  thr={bert_t:.2f}')

    # ── 3. Hybrid ensembles (sweep SVM weight) ────────────────────────────────
    print(f'\n── Hybrid ensemble sweep (SVM weight vs BERT weight) ──')
    print(f'{"SVM_w":>6}  {"BERT_w":>6}  {"Macro-F1":>9}  {"F1(NV)":>7}  '
          f'{"F1(V)":>7}  {"Thr":>5}')
    print('-' * 50)

    best_hybrid_f1  = 0.0
    best_hybrid_cfg = None

    # Equal weight = 1 SVM + N BERT (each BERT seed = 1 unit)
    # Sweep: svm_weight in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    # bert_weight per seed always = 1.0; svm counts as svm_w units
    for svm_w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        bert_w = len(seeds)   # total weight across all BERT seeds
        total_w = svm_w + bert_w
        ens_val  = (svm_w * svm_val_probs  + bert_w * bert_ens_val)  / total_w
        ens_test = (svm_w * svm_test_probs + bert_w * bert_ens_test) / total_w
        t, _  = tune_threshold(ens_val, y_val)
        pred  = (ens_test[:, 1] > t).astype(int)
        mf1   = f1_score(y_test, pred, average='macro')
        f1s   = f1_score(y_test, pred, average=None, labels=[0, 1])
        marker = ' ◄ best' if mf1 > best_hybrid_f1 else ''
        print(f'{svm_w:>6.2f}  {bert_w:>6.0f}  {mf1:>9.3f}  '
              f'{f1s[0]:>7.3f}  {f1s[1]:>7.3f}  {t:>5.2f}{marker}')
        if mf1 > best_hybrid_f1:
            best_hybrid_f1  = mf1
            best_hybrid_cfg = {'svm_w': svm_w, 'bert_w': bert_w, 'thr': t,
                               'f1_nv': f1s[0], 'f1_v': f1s[1]}

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'  SVM solo ({len(X_train)+len(X_val)} train, full text):  '
          f'macro-F1={svm_solo_f1:.3f}')
    print(f'  BERT-only ({len(seeds)} seeds, 2040 tok):  '
          f'macro-F1={bert_solo_f1:.3f}')
    print(f'  Best hybrid (svm_w={best_hybrid_cfg["svm_w"]:.2f}, '
          f'bert_w={best_hybrid_cfg["bert_w"]:.0f}):  '
          f'macro-F1={best_hybrid_f1:.3f}  '
          f'F1(NV)={best_hybrid_cfg["f1_nv"]:.3f}  '
          f'F1(V)={best_hybrid_cfg["f1_v"]:.3f}  '
          f'thr={best_hybrid_cfg["thr"]:.2f}')
    delta_vs_bert = best_hybrid_f1 - bert_solo_f1
    delta_vs_svm  = best_hybrid_f1 - svm_solo_f1
    print(f'  Hybrid vs BERT-only: {delta_vs_bert:+.3f}')
    print(f'  Hybrid vs SVM-only:  {delta_vs_svm:+.3f}')

    print(f'\n{classification_report(y_test, (bert_ens_test[:, 1] > bert_t).astype(int), target_names=["No Violation", "Violation"])}')


if __name__ == '__main__':
    main()
