"""
run_lime_bert.py — LIME explainability for ChunkedBERT, with stop-word post-filtering.

Loads the saved ChunkedBERT checkpoint (lime_bert_ckpt) and runs LIME on a balanced
sample of test cases. Stop words, non-alpha tokens, and tokens shorter than 3 chars
are removed from the *reported* word list — the model still receives raw text.

Results are printed and saved to logs/lime_bert_results.txt.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/run_lime_bert.py
    CUDA_VISIBLE_DEVICES=1 python scripts/run_lime_bert.py --n_cases 50 --n_samples 300
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

from lime.lime_text import LimeTextExplainer

MANUAL_LEMMA_MAP = {
    'authorities': 'authority', 'applicants': 'applicant',
    'proceedings': 'proceeding', 'rights': 'right',
    'violations': 'violation',  'articles': 'article',
}
LEGAL_STOP = _SW | {'court', 'case', 'mr', 'mrs', 'ms'}
wnl = WordNetLemmatizer()


def lime_normalise(word: str) -> str:
    """Lemmatise a single LIME surface-form word (same logic as SVM tokenizer)."""
    w = word.lower().strip()
    toks = [t for t in w.split() if t.isalpha()]
    if not toks:
        return ''
    t = toks[0]
    return MANUAL_LEMMA_MAP.get(t, wnl.lemmatize(t, pos='n'))


def is_content_word(word: str) -> bool:
    """True if word should appear in LIME output (not a stop word / punctuation)."""
    return (word
            and word.isalpha()
            and word not in LEGAL_STOP
            and len(word) > 2)


# ---------------------------------------------------------------------------
# Dataset + model (identical to echr.ipynb)
# ---------------------------------------------------------------------------

class ECHRChunkedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, chunk_size=510, n_chunks=4):
        self.items = []
        seq_len = chunk_size + 2
        for text, label in zip(texts, labels):
            ids    = tokenizer.encode(text, add_special_tokens=False)
            raw    = [ids[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
            n_real = sum(1 for c in raw if c)
            iids, masks = [], []
            for chunk in raw:
                full = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                full = full[:seq_len]
                pad  = seq_len - len(full)
                masks.append(torch.tensor([1]*len(full) + [0]*pad, dtype=torch.long))
                full = full + [tokenizer.pad_token_id]*pad
                iids.append(torch.tensor(full, dtype=torch.long))
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  default='data/processed/processed.csv')
    parser.add_argument('--ckpt_dir',   default='results/legalbert_chunked_original_final/lime_bert_ckpt')
    parser.add_argument('--model_name', default='nlpaueb/legal-bert-base-uncased')
    parser.add_argument('--n_cases',    type=int, default=50,
                        help='Balanced test cases to explain (half viol, half no-viol)')
    parser.add_argument('--n_samples',  type=int, default=500,
                        help='LIME perturbations per case')
    parser.add_argument('--n_features', type=int, default=20,
                        help='LIME features returned per case')
    parser.add_argument('--freq_min',   type=int, default=2,
                        help='Min case frequency to include a word in aggregation')
    parser.add_argument('--top_n',      type=int, default=20,
                        help='Words per side in bar chart')
    parser.add_argument('--out_dir',    default='logs')
    parser.add_argument('--device',     default=None)
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(f'cuda:{args.device}' if args.device.isdigit()
                              else args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

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

    X_test = test_df['facts'].tolist()
    y_test = test_df['label'].tolist()
    print(f'Test set: {len(X_test)} cases  '
          f'({sum(y_test)} violation, {len(y_test)-sum(y_test)} no-violation)')

    # ── Load model from checkpoint ────────────────────────────────────────────
    if not os.path.exists(os.path.join(args.ckpt_dir, 'model.pt')):
        print(f'ERROR: checkpoint not found at {args.ckpt_dir}')
        print('Run echr.ipynb with TRAIN_FROM_SCRATCH=True first.')
        sys.exit(1)

    print(f'Loading checkpoint from {args.ckpt_dir} ...')
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model     = ChunkedBERT(args.model_name).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, 'model.pt'), map_location=device))
    model.eval()
    print('Model loaded.')

    # ── LIME predict_proba wrapper ────────────────────────────────────────────
    def bert_predict_proba(texts):
        ds = ECHRChunkedDataset(list(texts), [0]*len(texts), tokenizer)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch['input_ids'].to(device),
                               batch['attention_mask'].to(device),
                               batch['chunk_mask'].to(device))
                probs.append(torch.softmax(logits, -1).cpu().numpy())
        return np.vstack(probs)

    # ── Balanced sample ───────────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    y_arr = np.array(y_test)
    viol_idx   = np.where(y_arr == 1)[0]
    noviol_idx = np.where(y_arr == 0)[0]
    n_each = args.n_cases // 2
    sample_idx = np.concatenate([
        rng.choice(viol_idx,   min(n_each, len(viol_idx)),   replace=False),
        rng.choice(noviol_idx, min(n_each, len(noviol_idx)), replace=False),
    ])
    print(f'\nLIME sample: {len(sample_idx)} cases  '
          f'({(y_arr[sample_idx]==1).sum()} violation, '
          f'{(y_arr[sample_idx]==0).sum()} no-violation)')
    print(f'Perturbations/case: {args.n_samples}  Features/case: {args.n_features}')

    # ── Run LIME ─────────────────────────────────────────────────────────────
    explainer = LimeTextExplainer(
        class_names=['No Violation', 'Violation'], random_state=42)

    records = []
    for i, idx in enumerate(sample_idx):
        exp = explainer.explain_instance(
            X_test[idx], bert_predict_proba,
            num_features=args.n_features,
            num_samples=args.n_samples,
            labels=(1,),
        )
        true_label = 'Violation' if y_test[idx] == 1 else 'No Violation'
        for word, weight in exp.as_list(label=1):
            norm = lime_normalise(word)
            if is_content_word(norm):
                records.append({'case': int(idx), 'true_label': true_label,
                                 'word': norm, 'weight': weight})
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(sample_idx)} cases explained', flush=True)

    lime_df = pd.DataFrame(records)
    print(f'\nTotal records after stop-word filter: {len(lime_df)}  '
          f'| unique words: {lime_df["word"].nunique()}')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = (lime_df.groupby('word')['weight']
           .agg(mean_weight='mean', freq='count')
           .query(f'freq >= {args.freq_min}')
           .sort_values('mean_weight'))

    top_pos = agg[agg['mean_weight'] > 0].tail(args.top_n)
    top_neg = agg[agg['mean_weight'] < 0].head(args.top_n)
    top_all = pd.concat([top_neg, top_pos]).sort_values('mean_weight')

    print(f'\nWords after freq≥{args.freq_min} filter: {len(agg)}  '
          f'(violation: {(agg.mean_weight>0).sum()}, '
          f'no-violation: {(agg.mean_weight<0).sum()})')
    print(f'\nTop violation words:     {list(top_pos.tail(15).index[::-1])}')
    print(f'Top no-violation words:  {list(top_neg.head(15).index)}')

    # ── Diverging bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_all) * 0.38)))
    colors = ['steelblue' if w < 0 else 'salmon' for w in top_all['mean_weight']]
    ax.barh(range(len(top_all)), top_all['mean_weight'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_all)))
    ax.set_yticklabels(top_all.index, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Mean LIME weight toward Violation  (negative = toward No Violation)')
    ax.set_title('LIME Feature Attribution — LegalBERT Chunked  |  Original (stop-word filtered)',
                 fontsize=11)
    ax.legend(handles=[
        Patch(color='salmon',    label='→ Violation'),
        Patch(color='steelblue', label='→ No Violation'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'lime_bert_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f'\nPlot saved: {plot_path}')

    # ── Save text results ─────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'lime_bert_results.txt')
    with open(out_path, 'w') as f:
        f.write(f'LIME Results — LegalBERT Chunked (stop-word filtered)\n')
        f.write(f'Cases: {len(sample_idx)}  Perturbations: {args.n_samples}  '
                f'Features: {args.n_features}  freq_min: {args.freq_min}\n\n')
        f.write(f'Total records: {len(lime_df)}  Unique words: {lime_df["word"].nunique()}\n')
        f.write(f'After freq≥{args.freq_min}: {len(agg)} words  '
                f'(viol: {(agg.mean_weight>0).sum()}, '
                f'no-viol: {(agg.mean_weight<0).sum()})\n\n')
        f.write(f'Top violation words (mean LIME weight):\n')
        for w, row in top_pos.tail(15).iloc[::-1].iterrows():
            f.write(f'  {w:<20}  {row.mean_weight:+.4f}  (freq={row.freq:.0f})\n')
        f.write(f'\nTop no-violation words (mean LIME weight):\n')
        for w, row in top_neg.head(15).iterrows():
            f.write(f'  {w:<20}  {row.mean_weight:+.4f}  (freq={row.freq:.0f})\n')
    print(f'Results saved: {out_path}')


if __name__ == '__main__':
    main()
