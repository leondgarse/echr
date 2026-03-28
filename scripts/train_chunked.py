"""
train_chunked.py — Sliding-window DeBERTa for ECHR violation prediction.

Splits each document into consecutive 510-token chunks (no overlap), encodes
each chunk independently through DeBERTa, then mean-pools the CLS tokens across
all real chunks to produce a single document representation. This gives full
document coverage regardless of document length.

Architecture:
    text → chunks → DeBERTa [CLS] per chunk → mean pool → Linear(768→2)

Usage:
    python scripts/train_chunked.py \
        --data_dir data_v1 \
        --output_dir results/deberta_chunked \
        --epochs 5 --batch_size 2 --grad_accum 8 \
        --learning_rate 2e-5 --seeds 0 1 2 3 \
        --focal_gamma 2 --max_chunks 8
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Dataset — splits documents into fixed-size chunks
# ---------------------------------------------------------------------------

class ChunkedECHRDataset(Dataset):
    def __init__(self, df, tokenizer, chunk_size=510, max_chunks=8,
                 mask_shortcuts=False):
        self.df           = df.reset_index(drop=True)
        self.tokenizer    = tokenizer
        self.chunk_size   = chunk_size
        self.max_chunks   = max_chunks
        self.mask_shortcuts = mask_shortcuts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row['text'])
        label = int(row['label'])

        if self.mask_shortcuts:
            from dataset import mask_shortcuts as _mask
            text = _mask(text, self.tokenizer.mask_token or '[MASK]')

        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Split into consecutive chunks of chunk_size tokens
        raw_chunks = [tokens[i:i + self.chunk_size]
                      for i in range(0, max(len(tokens), 1), self.chunk_size)]
        n_real = min(len(raw_chunks), self.max_chunks)
        raw_chunks = raw_chunks[:self.max_chunks]

        # Pad to max_chunks with empty chunks
        while len(raw_chunks) < self.max_chunks:
            raw_chunks.append([])

        seq_len = self.chunk_size + 2  # +2 for [CLS] and [SEP]
        input_ids_list     = []
        attention_mask_list = []

        for chunk_toks in raw_chunks:
            ids  = ([self.tokenizer.cls_token_id]
                    + chunk_toks
                    + [self.tokenizer.sep_token_id])
            ids  = ids[:seq_len]
            pad_len = seq_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids  = ids + [self.tokenizer.pad_token_id] * pad_len
            input_ids_list.append(torch.tensor(ids,  dtype=torch.long))
            attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

        # chunk_mask marks real vs padding chunks
        chunk_mask = [1] * n_real + [0] * (self.max_chunks - n_real)

        return {
            'input_ids':      torch.stack(input_ids_list),       # (max_chunks, seq_len)
            'attention_mask': torch.stack(attention_mask_list),  # (max_chunks, seq_len)
            'chunk_mask':     torch.tensor(chunk_mask, dtype=torch.float),  # (max_chunks,)
            'labels':         torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model — shared encoder + mean-pool over chunk CLS tokens
# ---------------------------------------------------------------------------

class ChunkedDeberta(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1, attn_pool=False):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.attn_pool  = attn_pool
        if attn_pool:
            self.chunk_attn = nn.Linear(hidden, 1, bias=False)

    def forward(self, input_ids, attention_mask, chunk_mask):
        """
        Args:
            input_ids:      (B, C, L)
            attention_mask: (B, C, L)
            chunk_mask:     (B, C) — 1 for real chunks, 0 for padding
        Returns:
            logits: (B, 2)
        """
        B, C, L = input_ids.shape

        # Encode all chunks in one batched forward pass: (B*C, L)
        flat_ids  = input_ids.view(B * C, L)
        flat_mask = attention_mask.view(B * C, L)

        outputs  = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb  = outputs.last_hidden_state[:, 0, :]   # (B*C, H)
        cls_emb  = cls_emb.view(B, C, -1)               # (B, C, H)

        if self.attn_pool:
            # Learned attention over chunk CLS tokens; mask out padding chunks
            scores = self.chunk_attn(cls_emb).squeeze(-1)          # (B, C)
            scores = scores.masked_fill(chunk_mask == 0, float('-inf'))
            weights = torch.softmax(scores, dim=-1)                 # (B, C)
            pooled = (weights.unsqueeze(-1) * cls_emb).sum(1)      # (B, H)
        else:
            # Weighted mean pool — only real chunks contribute
            cm     = chunk_mask.unsqueeze(-1)                       # (B, C, 1)
            pooled = (cls_emb * cm).sum(1) / cm.sum(1).clamp(min=1e-9)  # (B, H)

        logits = self.classifier(self.dropout(pooled))   # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# LLRD optimizer
# ---------------------------------------------------------------------------

def build_llrd_optimizer(model, lr, wd, decay):
    encoder    = model.encoder
    num_layers = len(encoder.encoder.layer)
    groups = [{'params': list(model.classifier.parameters()), 'lr': lr, 'weight_decay': wd}]
    if hasattr(encoder, 'pooler') and encoder.pooler is not None:
        groups.append({'params': list(encoder.pooler.parameters()),
                       'lr': lr * decay, 'weight_decay': wd})
    for depth, layer in enumerate(reversed(encoder.encoder.layer)):
        groups.append({'params': list(layer.parameters()),
                       'lr': lr * (decay ** (depth + 2)), 'weight_decay': wd})
    groups.append({'params': list(encoder.embeddings.parameters()),
                   'lr': lr * (decay ** (num_layers + 2)), 'weight_decay': wd})
    lrs = [g['lr'] for g in groups]
    print(f"LLRD lr range: {min(lrs):.2e} → {max(lrs):.2e}")
    return AdamW(groups, eps=1e-8)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, weight, gamma):
    log_p = F.log_softmax(logits, dim=-1)
    ce    = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if gamma > 0:
        p_t  = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)
        ce   = ((1 - p_t) ** gamma) * ce
    return ce.mean()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        cmask = batch['chunk_mask'].to(device)
        logits = model(ids, mask, cmask)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch['labels'].numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, logits.argmax(-1), average='macro', zero_division=0
    )
    return f1, logits, labels


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one(args, train_df, val_df, test_df, model_seed):
    set_seed(model_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = ChunkedDeberta(args.model_name, attn_pool=args.attn_pool).to(device)

    weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                   y=train_df['label'].values)
    class_w = torch.tensor(weights, dtype=torch.float, device=device)

    train_ds = ChunkedECHRDataset(train_df, tokenizer,
                                  max_chunks=args.max_chunks,
                                  mask_shortcuts=args.mask_shortcuts)
    val_ds   = ChunkedECHRDataset(val_df,   tokenizer,
                                  max_chunks=args.max_chunks,
                                  mask_shortcuts=args.mask_shortcuts)
    test_ds  = ChunkedECHRDataset(test_df,  tokenizer,
                                  max_chunks=args.max_chunks,
                                  mask_shortcuts=args.mask_shortcuts)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)

    optimizer  = build_llrd_optimizer(model, args.learning_rate, 0.01,
                                      args.llrd_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.15 * total_steps)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1, best_state, patience_count = 0.0, None, 0
    run_dir = os.path.join(args.output_dir, f'seed_{model_seed}')
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss, n_steps = 0.0, 0

        for step, batch in enumerate(train_loader):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            cmask = batch['chunk_mask'].to(device)
            lbls  = batch['labels'].to(device)

            logits = model(ids, mask, cmask)
            loss   = focal_loss(logits, lbls, class_w, args.focal_gamma)
            loss   = loss / args.grad_accum
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_steps += 1

        val_f1, _, _ = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(n_steps * args.grad_accum, 1)
        print(f"  Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val macro-F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(run_dir, 'best_model.pt'))
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(
        os.path.join(run_dir, 'best_model.pt'), map_location=device
    ))
    test_f1, test_logits, test_labels = evaluate(model, test_loader, device)
    val_f1,  val_logits,  val_labels  = evaluate(model, val_loader,  device)
    print(f"  seed={model_seed} → test macro-F1: {test_f1:.4f}")
    return test_logits, val_logits, test_labels, val_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )


def load_and_split(data_path, data_seed, temporal=False):
    df = pd.read_csv(data_path)
    mask = (_contains_article(df['violation_articles'], '6') |
            _contains_article(df['nonviolation_articles'], '6'))
    df = df[mask].copy()
    df['label'] = _contains_article(df['violation_articles'], '6').astype(int)
    print(f"Art.6 subset: {len(df)} cases, {df['label'].mean():.1%} violation")

    if temporal:
        cutoff = int(np.percentile(df['year'].dropna(), 75))
        print(f"  Temporal split: train year<{cutoff}, test year>={cutoff}")
        train_val = df[df['year'] < cutoff].copy()
        test_df   = df[df['year'] >= cutoff].copy()
        val_size  = 0.15 / 0.75
        train_df, val_df = train_test_split(
            train_val, test_size=val_size, stratify=train_val['label'],
            random_state=data_seed
        )
    else:
        train_val, test_df = train_test_split(
            df, test_size=0.25, stratify=df['label'], random_state=data_seed
        )
        val_size = 0.15 / 0.75
        train_df, val_df = train_test_split(
            train_val, test_size=val_size, stratify=train_val['label'],
            random_state=data_seed
        )
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  model: {args.model_name}")
    print(f"max_chunks={args.max_chunks}  chunk_size=510  "
          f"effective_coverage={args.max_chunks * 510} tokens  "
          f"pooling={'attn' if args.attn_pool else 'mean'}  "
          f"cda={args.mask_shortcuts}")

    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.data_dir, 'processed', 'processed.csv')
    train_df, val_df, test_df = load_and_split(data_path, args.data_seed,
                                                temporal=args.temporal)

    all_test_logits = []
    all_val_logits  = []
    y_true = test_df['label'].values
    y_val  = val_df['label'].values

    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Training with model seed={seed}")
        print(f"{'='*50}")
        tl, vl, _, _ = train_one(args, train_df, val_df, test_df, seed)
        all_test_logits.append(tl)
        all_val_logits.append(vl)

    def _softmax(l):
        e = np.exp(l - l.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    avg_probs     = np.stack([_softmax(l) for l in all_test_logits]).mean(0)
    avg_val_probs = np.stack([_softmax(l) for l in all_val_logits]).mean(0)

    # Threshold tuning on val set
    best_t, best_vf1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        vf = f1_score(y_val, (avg_val_probs[:, 1] > t).astype(int),
                      average='macro', zero_division=0)
        if vf > best_vf1:
            best_vf1, best_t = vf, float(t)

    print(f"\n{'='*50}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*50}")
    print(f"  Threshold tuning: best_t={best_t:.2f}  val_f1={best_vf1:.4f}")

    y_default = avg_probs.argmax(1)
    y_tuned   = (avg_probs[:, 1] > best_t).astype(int)

    _, _, f1_def, _ = precision_recall_fscore_support(
        y_true, y_default, average='macro', zero_division=0)
    f1p_def = f1_score(y_true, y_default, average=None, labels=[0, 1])
    print(f"  [t=0.50 default]  macro_f1={f1_def:.4f}  "
          f"F1(NV)={f1p_def[0]:.4f}  F1(V)={f1p_def[1]:.4f}")

    acc = accuracy_score(y_true, y_tuned)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_tuned, average='macro', zero_division=0)
    f1p = f1_score(y_true, y_tuned, average=None, labels=[0, 1])
    print(f"  Seeds:           {args.seeds}")
    print(f"  accuracy:        {acc:.4f}")
    print(f"  macro_f1:        {mf1:.4f}  (threshold={best_t:.2f})")
    print(f"  macro_precision: {mp:.4f}")
    print(f"  macro_recall:    {mr:.4f}")
    print(f"  f1_no_violation: {f1p[0]:.4f}")
    print(f"  f1_violation:    {f1p[1]:.4f}")
    print()
    print(classification_report(y_true, y_tuned,
                                 target_names=['No Violation', 'Violation'],
                                 zero_division=0))

    np.savez(os.path.join(args.output_dir, 'test_probs.npz'),
             probs=avg_probs, labels=y_true)
    print(f"Test probs saved to {args.output_dir}/test_probs.npz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str,   required=True)
    parser.add_argument('--model_name',     type=str,   default='microsoft/deberta-v3-base')
    parser.add_argument('--output_dir',     type=str,   default='results/chunked')
    parser.add_argument('--epochs',         type=int,   default=5)
    parser.add_argument('--batch_size',     type=int,   default=2,
                        help='Samples per batch (each sample = max_chunks sequences)')
    parser.add_argument('--grad_accum',     type=int,   default=8)
    parser.add_argument('--learning_rate',  type=float, default=2e-5)
    parser.add_argument('--llrd_decay',     type=float, default=0.9)
    parser.add_argument('--patience',       type=int,   default=3)
    parser.add_argument('--focal_gamma',    type=float, default=2.0)
    parser.add_argument('--max_chunks',     type=int,   default=8,
                        help='Max 510-token chunks per document (default 8 = ~4080 tokens)')
    parser.add_argument('--seeds',          type=int,   nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--data_seed',      type=int,   default=42)
    parser.add_argument('--mask_shortcuts', action='store_true', default=False)
    parser.add_argument('--attn_pool',     action='store_true', default=False,
                        help='Attention-weighted chunk pooling instead of mean pooling')
    parser.add_argument('--temporal',      action='store_true', default=False,
                        help='Temporal split: train on year<p75, test on year>=p75')
    main(parser.parse_args())
