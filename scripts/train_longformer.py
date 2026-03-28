"""
train_longformer.py — Fine-tune Legal-Longformer for ECHR violation prediction.

Uses LongformerForSequenceClassification with global attention on [CLS] token,
natively handling up to 4096 tokens without chunking. Applies the same training
recipe as train_chunked.py: focal loss, LLRD, multi-seed ensemble, threshold tuning.

Model: lexlms/legal-longformer-base (4096 tokens, legally pretrained)
Architecture: Longformer → CLS global attention → Linear(768→2)

Usage:
    python scripts/train_longformer.py \
        --data_dir data_v1 \
        --output_dir results/legal_longformer_v1 \
        --epochs 5 --batch_size 2 --grad_accum 8 \
        --learning_rate 2e-5 --seeds 0 1 2 3

    # Temporal split:
    python scripts/train_longformer.py \
        --data_dir data_v1 \
        --output_dir results/legal_longformer_v1_temporal \
        --temporal --seeds 0 1 2 3
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
from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    LongformerConfig,
    set_seed,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LongformerECHRDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=4096, mask_shortcuts=False):
        self.df              = df.reset_index(drop=True)
        self.tokenizer       = tokenizer
        self.max_len         = max_len
        self.mask_shortcuts  = mask_shortcuts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row['text'])
        label = int(row['label'])

        if self.mask_shortcuts:
            from dataset import mask_shortcuts as _mask
            text = _mask(text, self.tokenizer.mask_token or '[MASK]')

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids      = encoding['input_ids'].squeeze(0)       # (max_len,)
        attention_mask = encoding['attention_mask'].squeeze(0)   # (max_len,)

        # Global attention on CLS token (index 0)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[0] = 1

        return {
            'input_ids':             input_ids,
            'attention_mask':        attention_mask,
            'global_attention_mask': global_attention_mask,
            'labels':                torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# LLRD optimizer — adapted for Longformer layer structure
# ---------------------------------------------------------------------------

def build_llrd_optimizer(model, lr, wd, decay):
    """
    Longformer layer structure:
      model.longformer.embeddings
      model.longformer.encoder.layers[0..11]
      model.classifier (pooler + dense)
    """
    encoder    = model.longformer
    num_layers = len(encoder.encoder.layer)
    groups = [
        {'params': list(model.classifier.parameters()), 'lr': lr, 'weight_decay': wd},
    ]
    for depth, layer in enumerate(reversed(encoder.encoder.layer)):
        groups.append({
            'params': list(layer.parameters()),
            'lr': lr * (decay ** (depth + 2)),
            'weight_decay': wd,
        })
    groups.append({
        'params': list(encoder.embeddings.parameters()),
        'lr': lr * (decay ** (num_layers + 2)),
        'weight_decay': wd,
    })
    lrs = [g['lr'] for g in groups]
    print(f"LLRD lr range: {min(lrs):.2e} → {max(lrs):.2e}  ({num_layers} layers)")
    return AdamW(groups, eps=1e-8)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits, labels, weight, gamma):
    log_p = F.log_softmax(logits, dim=-1)
    ce    = F.nll_loss(log_p, labels, weight=weight, reduction='none')
    if gamma > 0:
        p_t = log_p.exp().gather(1, labels.unsqueeze(1)).squeeze(1)
        ce  = ((1 - p_t) ** gamma) * ce
    return ce.mean()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        input_ids             = batch['input_ids'].to(device)
        attention_mask        = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        all_logits.append(outputs.logits.cpu().numpy())
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
    model     = LongformerForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        attention_window=512,   # local attention window; global on CLS
    ).to(device)

    weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                   y=train_df['label'].values)
    class_w = torch.tensor(weights, dtype=torch.float, device=device)

    train_ds = LongformerECHRDataset(train_df, tokenizer,
                                     max_len=args.max_len,
                                     mask_shortcuts=args.mask_shortcuts)
    val_ds   = LongformerECHRDataset(val_df,   tokenizer,
                                     max_len=args.max_len,
                                     mask_shortcuts=args.mask_shortcuts)
    test_ds  = LongformerECHRDataset(test_df,  tokenizer,
                                     max_len=args.max_len,
                                     mask_shortcuts=args.mask_shortcuts)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0)

    optimizer    = build_llrd_optimizer(model, args.learning_rate, 0.01,
                                        args.llrd_decay)
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.15 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1, patience_count = 0.0, 0
    run_dir = os.path.join(args.output_dir, f'seed_{model_seed}')
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss, n_steps = 0.0, 0

        for step, batch in enumerate(train_loader):
            input_ids             = batch['input_ids'].to(device)
            attention_mask        = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            lbls                  = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            loss = focal_loss(outputs.logits, lbls, class_w, args.focal_gamma)
            loss = loss / args.grad_accum
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
            best_val_f1    = val_f1
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
    # Clean up checkpoint to save disk space
    os.remove(os.path.join(run_dir, 'best_model.pt'))

    test_f1, test_logits, test_labels = evaluate(model, test_loader, device)
    val_f1,  val_logits,  val_labels  = evaluate(model, val_loader,  device)
    print(f"  seed={model_seed} → test macro-F1: {test_f1:.4f}")
    return test_logits, val_logits, test_labels, val_labels


# ---------------------------------------------------------------------------
# Data loading helpers
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
            random_state=data_seed,
        )
    else:
        train_val, test_df = train_test_split(
            df, test_size=0.25, stratify=df['label'], random_state=data_seed,
        )
        val_size = 0.15 / 0.75
        train_df, val_df = train_test_split(
            train_val, test_size=val_size, stratify=train_val['label'],
            random_state=data_seed,
        )
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  model: {args.model_name}")
    print(f"max_len={args.max_len}  temporal={args.temporal}  "
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
    parser.add_argument('--model_name',     type=str,   default='lexlms/legal-longformer-base')
    parser.add_argument('--output_dir',     type=str,   default='results/legal_longformer')
    parser.add_argument('--max_len',        type=int,   default=4096,
                        help='Max token length (Longformer supports up to 4096)')
    parser.add_argument('--epochs',         type=int,   default=5)
    parser.add_argument('--batch_size',     type=int,   default=1,
                        help='Per-step batch size (4096 tokens is ~4× larger than 512)')
    parser.add_argument('--grad_accum',     type=int,   default=16,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    parser.add_argument('--learning_rate',  type=float, default=2e-5)
    parser.add_argument('--llrd_decay',     type=float, default=0.9)
    parser.add_argument('--patience',       type=int,   default=3)
    parser.add_argument('--focal_gamma',    type=float, default=2.0)
    parser.add_argument('--seeds',          type=int,   nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--data_seed',      type=int,   default=42)
    parser.add_argument('--mask_shortcuts', action='store_true', default=False)
    parser.add_argument('--temporal',       action='store_true', default=False,
                        help='Use temporal train/test split (train year<75th-pct, test year>=75th-pct)')
    args = parser.parse_args()
    main(args)
