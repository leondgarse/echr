"""
train_multitask.py — Multi-task ECHR violation prediction.

Trains a shared DeBERTa encoder with 4 binary classification heads,
one per ECHR article (3, 5, 6, 8). Each case contributes to a head
only if that article appears in its violation_articles or
nonviolation_articles columns.

Architecture:
    text → DeBERTa [CLS] (768d) → 4 × Linear(768→2) heads

Primary evaluation metric: macro-F1 on Article 6 head (comparable to
the single-task baseline in src/train.py).

Usage example:
    python scripts/train_multitask.py \
        --data_dir data \
        --model_name microsoft/deberta-v3-base \
        --output_dir results/multitask \
        --epochs 20 \
        --batch_size 4 \
        --grad_accum 4 \
        --learning_rate 2e-5 \
        --llrd_decay 0.9 \
        --patience 5 \
        --seeds 42 0 1 2
"""

import argparse
import os
import sys
import shutil

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

# Allow importing ECHRDataset from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset import ECHRDataset  # noqa: E402  (after sys.path modification)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTICLES = [3, 5, 6, 8]          # The four article heads, in order
ART6_IDX = ARTICLES.index(6)     # Index of the Article 6 head (= 2)


# ---------------------------------------------------------------------------
# Multi-task model: shared DeBERTa encoder + 4 independent linear heads
# ---------------------------------------------------------------------------

class MultiTaskECHR(nn.Module):
    """Shared DeBERTa encoder with one binary head per ECHR article.

    The encoder produces a [CLS] representation (index 0 of
    last_hidden_state). Each head is an independent Linear(hidden→2)
    with a preceding Dropout layer.

    Args:
        model_name: HuggingFace model identifier.
        articles:   Ordered list of article numbers (determines head count).
        dropout:    Dropout probability before each classification head.
    """

    def __init__(self, model_name: str, articles=ARTICLES, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # One head per article; stored as a ModuleList so parameters are
        # registered and moved to the correct device automatically.
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 2) for _ in articles
        ])
        self.articles = articles

    def forward(self, input_ids, attention_mask):
        """Return list of logits tensors, one per head (shape: B×2 each)."""
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        # [CLS] token representation — index 0 of the sequence dimension
        cls_repr = outputs.last_hidden_state[:, 0, :]   # (B, H)
        cls_repr = self.dropout(cls_repr)
        logits_per_head = [head(cls_repr) for head in self.heads]
        return logits_per_head  # list of (B, 2) tensors


# ---------------------------------------------------------------------------
# Multi-task Dataset — wraps ECHRDataset tokenisation logic
# ---------------------------------------------------------------------------

class MultiTaskECHRDataset(Dataset):
    """Dataset that emits per-article binary labels alongside tokenised text.

    For each sample the returned dict contains:
        input_ids, attention_mask  — standard token tensors (from ECHRDataset)
        labels_<K>                 — label for article K (0/1), or -1 if the
                                     case is not relevant for that article
    The -1 sentinel is used during loss computation to mask out heads that
    should not contribute to a given sample's loss.

    Internally we reuse ECHRDataset's tokenisation (head_tail truncation,
    mask_shortcuts) by wrapping a single-row DataFrame per sample.
    To avoid per-sample overhead we pre-compute encodings via a vectorised
    ECHRDataset over the full DataFrame, then override labels.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 512,
                 truncation: str = 'head_tail', mask_shortcuts: bool = False,
                 articles=ARTICLES):
        self.articles = articles
        self.df = df.reset_index(drop=True)

        # Build an ECHRDataset for tokenisation only; we ignore its 'labels'.
        # We pass df directly; ECHRDataset will use df['label'] but we
        # override the returned label below.
        self._base = ECHRDataset(
            data=self.df,
            tokenizer=tokenizer,
            max_len=max_len,
            truncation=truncation,
            mask_shortcuts=mask_shortcuts,
        )

        # Pre-compute per-article labels: shape (n_samples, n_articles)
        # Value is 0, 1, or -1 (not applicable).
        self._labels = self._build_label_matrix(self.df, articles)

    @staticmethod
    def _contains_article(series: pd.Series, article: str) -> pd.Series:
        """Return boolean Series: True if article appears in semicolon-list."""
        return series.fillna('').str.split(';').apply(
            lambda lst: any(a.strip() == article for a in lst)
        )

    @classmethod
    def _build_label_matrix(cls, df: pd.DataFrame,
                             articles) -> np.ndarray:
        """Build (n, len(articles)) integer array with 0/1/-1 entries."""
        n = len(df)
        mat = np.full((n, len(articles)), -1, dtype=np.int64)
        for col_idx, art in enumerate(articles):
            art_str = str(art)
            in_violation    = cls._contains_article(df['violation_articles'],    art_str)
            in_nonviolation = cls._contains_article(df['nonviolation_articles'], art_str)
            relevant = in_violation | in_nonviolation
            mat[relevant, col_idx] = in_violation[relevant].astype(int)
        return mat

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Reuse ECHRDataset tokenisation (head_tail / mask_shortcuts etc.)
        base_item = self._base[idx]
        sample = {
            'input_ids':      base_item['input_ids'],
            'attention_mask': base_item['attention_mask'],
        }
        for col_idx, art in enumerate(self.articles):
            sample[f'labels_{art}'] = torch.tensor(
                self._labels[idx, col_idx], dtype=torch.long
            )
        return sample


# ---------------------------------------------------------------------------
# Helper: extract per-article label arrays from a dataset (for metrics)
# ---------------------------------------------------------------------------

def get_article_labels(dataset: MultiTaskECHRDataset,
                       art_idx: int) -> np.ndarray:
    """Return the raw label array for one article head (includes -1)."""
    return dataset._labels[:, art_idx]


# ---------------------------------------------------------------------------
# LLRD optimizer — identical logic to src/train.py, adapted for MultiTaskECHR
# ---------------------------------------------------------------------------

def build_llrd_optimizer(model: MultiTaskECHR, lr: float,
                          weight_decay: float, llrd_decay: float) -> AdamW:
    """Layer-wise learning rate decay.

    Groups:
        - classifier heads       → lr (full rate)
        - top transformer layer  → lr × d^2
        - ...
        - embeddings             → lr × d^(num_layers+2)

    Parameters:
        model:        MultiTaskECHR instance.
        lr:           Base (maximum) learning rate.
        weight_decay: AdamW weight decay.
        llrd_decay:   Per-layer decay factor d (e.g. 0.9).
    """
    d = llrd_decay
    encoder = model.encoder

    # DeBERTa uses encoder.encoder.layer for transformer blocks
    if not hasattr(encoder, 'encoder') or not hasattr(encoder.encoder, 'layer'):
        raise ValueError(
            "Cannot resolve transformer layer stack from model.encoder. "
            "Expected a DeBERTa-style architecture with .encoder.encoder.layer."
        )
    num_layers = len(encoder.encoder.layer)

    groups = []

    # All classification heads share the top learning rate
    groups.append({
        'params':       list(model.heads.parameters()),
        'lr':           lr,
        'weight_decay': weight_decay,
    })

    # Pooler (if present)
    if hasattr(encoder, 'pooler') and encoder.pooler is not None:
        groups.append({
            'params':       list(encoder.pooler.parameters()),
            'lr':           lr * (d ** 1),
            'weight_decay': weight_decay,
        })

    # Transformer layers — top layer gets d^2, bottom gets d^(num_layers+1)
    for depth, layer in enumerate(reversed(encoder.encoder.layer)):
        groups.append({
            'params':       list(layer.parameters()),
            'lr':           lr * (d ** (depth + 2)),
            'weight_decay': weight_decay,
        })

    # Embeddings
    groups.append({
        'params':       list(encoder.embeddings.parameters()),
        'lr':           lr * (d ** (num_layers + 2)),
        'weight_decay': weight_decay,
    })

    lrs = [g['lr'] for g in groups]
    print(f"LLRD lr range: {min(lrs):.2e} → {max(lrs):.2e}")
    return AdamW(groups, eps=1e-8)


# ---------------------------------------------------------------------------
# Compute per-article class weights for weighted cross-entropy
# ---------------------------------------------------------------------------

def compute_article_weights(dataset: MultiTaskECHRDataset,
                             device: torch.device) -> list:
    """Return list of (2,)-float tensors, one per article head.

    Weights are computed from the training-set labels only (labels != -1).
    Uses sklearn's 'balanced' strategy.
    """
    weights = []
    for col_idx in range(len(dataset.articles)):
        labels = dataset._labels[:, col_idx]
        relevant = labels[labels != -1]
        if len(relevant) == 0 or len(np.unique(relevant)) < 2:
            # Fallback: equal weights if only one class present
            w = torch.ones(2, dtype=torch.float, device=device)
        else:
            w_arr = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=relevant,
            )
            w = torch.tensor(w_arr, dtype=torch.float, device=device)
        weights.append(w)
        print(f"  Article {dataset.articles[col_idx]} weights: "
              f"no-violation={w[0]:.3f}, violation={w[1]:.3f} "
              f"({(relevant==1).sum()}/{len(relevant)} positive)")
    return weights


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def _contains_article(series: pd.Series, article: str) -> pd.Series:
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )


def load_and_split(data_path: str, temporal_split: bool,
                   data_seed: int, val_fraction: float = 0.15):
    """Load processed.csv and produce train/val/test DataFrames.

    Split strategy mirrors src/train.py: 75/25 train-test (or temporal),
    then val_fraction of the trainval pool becomes validation.

    Stratification is always on the Article 6 label (cases not relevant
    to Art.6 are assigned label=0 for stratification purposes only).
    """
    df = pd.read_csv(data_path)

    # Derive Article 6 label for stratification (and primary evaluation)
    art6_viol    = _contains_article(df['violation_articles'],    '6')
    art6_nonviol = _contains_article(df['nonviolation_articles'], '6')
    df['art6_label'] = art6_viol.astype(int)
    df['art6_relevant'] = (art6_viol | art6_nonviol).astype(bool)

    n_relevant = df['art6_relevant'].sum()
    print(f"Total cases: {len(df)} | Art.6 relevant: {n_relevant} "
          f"({n_relevant/len(df):.1%}) | "
          f"Art.6 violation rate: {df.loc[df['art6_relevant'], 'art6_label'].mean():.1%}")

    # Report coverage per article
    for art in ARTICLES:
        art_str = str(art)
        in_v = _contains_article(df['violation_articles'],    art_str)
        in_n = _contains_article(df['nonviolation_articles'], art_str)
        rel  = (in_v | in_n).sum()
        print(f"  Article {art}: {rel} relevant cases "
              f"({in_v.sum()} violations, {in_n.sum()} non-violations)")

    # Stratify splits on Art.6 label (matches single-task baseline)
    strat_col = 'art6_label'
    val_size  = val_fraction / (1 - 0.25)

    if temporal_split:
        cutoff    = int(df['year'].quantile(0.75))
        print(f"Temporal split: train year < {cutoff}, test year >= {cutoff}")
        train_val = df[df['year'] <  cutoff].copy()
        test_df   = df[df['year'] >= cutoff].copy()
    else:
        train_val, test_df = train_test_split(
            df, test_size=0.25,
            stratify=df[strat_col],
            random_state=data_seed,
        )

    train_df, val_df = train_test_split(
        train_val, test_size=val_size,
        stratify=train_val[strat_col],
        random_state=data_seed,
    )

    for split_name, split_df in [('Train', train_df),
                                  ('Val',   val_df),
                                  ('Test',  test_df)]:
        art6_rel = split_df['art6_relevant'].sum()
        print(f"  {split_name:5s}: {len(split_df)} cases "
              f"| Art.6 relevant: {art6_rel} "
              f"({split_df.loc[split_df['art6_relevant'], 'art6_label'].mean():.1%} viol.)")

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Collate function — pads variable-length label keys
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Default collate works fine since all tensors are fixed-length."""
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


# ---------------------------------------------------------------------------
# Compute multi-task loss for a single batch
# ---------------------------------------------------------------------------

def multitask_loss(logits_list, batch, article_weights, device,
                   articles=ARTICLES):
    """Sum of per-article weighted cross-entropy losses.

    For each article head:
        - Select samples whose label != -1 (i.e. the case is relevant).
        - Compute weighted CE on those samples only.
        - If no relevant sample in the batch, skip this head.

    Args:
        logits_list:     List of (B, 2) logit tensors, one per article.
        batch:           Dict from DataLoader (contains labels_<K> keys).
        article_weights: List of (2,)-float tensors from compute_article_weights.
        device:          Target device.
        articles:        List of article numbers (must match logits_list order).

    Returns:
        Scalar loss tensor (sum across active heads), or 0 if no head active.
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_active = 0

    for col_idx, art in enumerate(articles):
        labels_key = f'labels_{art}'
        labels = batch[labels_key].to(device)           # (B,), values 0/1/-1

        # Mask: only cases relevant to this article
        mask = (labels != -1)
        if mask.sum() == 0:
            continue

        logits_masked = logits_list[col_idx][mask]      # (n_rel, 2)
        labels_masked = labels[mask]                    # (n_rel,)

        loss = nn.CrossEntropyLoss(
            weight=article_weights[col_idx]
        )(logits_masked, labels_masked)

        total_loss = total_loss + loss
        n_active += 1

    return total_loss, n_active


# ---------------------------------------------------------------------------
# Evaluation — returns per-article metrics dict and all predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, article_weights, device, articles=ARTICLES):
    """Run inference over a DataLoader and compute per-article macro-F1.

    Returns:
        metrics: dict mapping 'art<K>_macro_f1' → float for each article.
        art6_preds: np.ndarray of predicted labels for Art.6 relevant cases.
        art6_true:  np.ndarray of true labels for Art.6 relevant cases.
        art6_logits: np.ndarray shape (n_relevant, 2) — Art.6 head logits.
    """
    model.eval()

    # Accumulators: per-article lists of (true, pred, logit)
    all_true   = [[] for _ in articles]
    all_pred   = [[] for _ in articles]
    all_logits_art6 = []

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits_list = model(input_ids, attention_mask)  # list of (B, 2)

        for col_idx, art in enumerate(articles):
            labels = batch[f'labels_{art}']             # (B,) on CPU
            mask   = (labels != -1)
            if mask.sum() == 0:
                continue

            logits_masked = logits_list[col_idx][mask].cpu()
            labels_masked = labels[mask]
            preds_masked  = logits_masked.argmax(-1)

            all_true[col_idx].extend(labels_masked.numpy().tolist())
            all_pred[col_idx].extend(preds_masked.numpy().tolist())

            if col_idx == ART6_IDX:
                all_logits_art6.append(logits_masked.numpy())

    metrics = {}
    for col_idx, art in enumerate(articles):
        y_true = np.array(all_true[col_idx])
        y_pred = np.array(all_pred[col_idx])
        if len(y_true) == 0:
            metrics[f'art{art}_macro_f1'] = float('nan')
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics[f'art{art}_macro_f1'] = float(f1)

    # Art.6 arrays for detailed reporting
    art6_true   = np.array(all_true[ART6_IDX])
    art6_pred   = np.array(all_pred[ART6_IDX])
    art6_logits = (np.concatenate(all_logits_art6, axis=0)
                   if all_logits_art6 else np.empty((0, 2), dtype=np.float32))

    return metrics, art6_pred, art6_true, art6_logits


# ---------------------------------------------------------------------------
# Single training run — returns Art.6 test logits for ensemble
# ---------------------------------------------------------------------------

def train_one(args, train_df, val_df, test_df, model_seed: int):
    """Train one multi-task model and return Art.6 test logits (n, 2)."""
    set_seed(model_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build datasets
    train_ds = MultiTaskECHRDataset(
        train_df, tokenizer, max_len=512,
        truncation='head_tail',
        mask_shortcuts=args.mask_shortcuts,
    )
    val_ds = MultiTaskECHRDataset(
        val_df, tokenizer, max_len=512,
        truncation='head_tail',
        mask_shortcuts=args.mask_shortcuts,
    )
    test_ds = MultiTaskECHRDataset(
        test_df, tokenizer, max_len=512,
        truncation='head_tail',
        mask_shortcuts=args.mask_shortcuts,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Compute class weights from training set
    print("Computing per-article class weights from training set:")
    article_weights = compute_article_weights(train_ds, device)

    # Model + optimizer
    model = MultiTaskECHR(args.model_name).to(device)
    optimizer = build_llrd_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=0.01,
        llrd_decay=args.llrd_decay,
    )

    # Linear warmup scheduler: 15% warmup then linear decay
    total_steps   = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps  = int(0.15 * total_steps)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Early stopping state
    best_art6_f1   = -1.0
    best_epoch     = -1
    patience_count = 0
    best_state     = None  # will hold deepcopy of model state_dict

    run_dir = os.path.join(args.output_dir, f'seed_{model_seed}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining seed={model_seed} | "
          f"total_steps={total_steps} | warmup={warmup_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss   = 0.0
        n_steps      = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits_list = model(input_ids, attention_mask)
            loss, n_active = multitask_loss(
                logits_list, batch, article_weights, device
            )

            if n_active == 0:
                # Batch had no labeled samples for any head — skip
                optimizer.zero_grad()
                continue

            # Normalise loss by accumulation steps
            loss = loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_steps += 1

            epoch_loss += loss.item() * args.grad_accum  # un-normalise for logging

        avg_loss = epoch_loss / max(n_steps * args.grad_accum, 1)

        # Validation
        val_metrics, _, val_true, _ = evaluate(
            model, val_loader, article_weights, device
        )
        art6_val_f1 = val_metrics.get(f'art6_macro_f1', float('nan'))

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"val Art.6 macro-F1={art6_val_f1:.4f} | "
              + " | ".join(
                  f"Art.{art}={val_metrics.get(f'art{art}_macro_f1', float('nan')):.4f}"
                  for art in ARTICLES
              ))

        # Early stopping on Art.6 val macro-F1
        if art6_val_f1 > best_art6_f1:
            best_art6_f1   = art6_val_f1
            best_epoch     = epoch
            patience_count = 0
            # Save best weights to disk (avoids keeping full model in RAM)
            best_ckpt_path = os.path.join(run_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch={best_epoch}, best val Art.6 F1={best_art6_f1:.4f})")
                break

    # Reload best checkpoint
    print(f"  Loading best checkpoint from epoch {best_epoch}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # Test evaluation
    test_metrics, test_pred, test_true, test_logits = evaluate(
        model, test_loader, article_weights, device
    )
    art6_test_f1 = test_metrics.get('art6_macro_f1', float('nan'))
    print(f"  seed={model_seed} → test Art.6 macro-F1: {art6_test_f1:.4f}")
    for art in ARTICLES:
        key = f'art{art}_macro_f1'
        print(f"    Art.{art} test macro-F1: {test_metrics.get(key, float('nan')):.4f}")

    if len(test_true) > 0:
        print(classification_report(
            test_true, test_pred,
            target_names=['No Violation', 'Violation'],
            zero_division=0,
        ))

    return test_logits, test_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model:  {args.model_name}")
    print(f"data_seed={args.data_seed} | model_seeds={args.seeds} | "
          f"temporal_split={args.temporal_split} | "
          f"mask_shortcuts={args.mask_shortcuts}")

    os.makedirs(args.output_dir, exist_ok=True)

    data_path = os.path.join(args.data_dir, 'processed', 'processed.csv')
    train_df, val_df, test_df = load_and_split(
        data_path,
        temporal_split=args.temporal_split,
        data_seed=args.data_seed,
    )

    # Collect Art.6 test logits across ensemble seeds
    all_logits = []
    shared_true = None  # same for every seed (fixed data_seed)

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training with model seed={seed}")
        print(f"{'='*60}")
        logits, y_true = train_one(args, train_df, val_df, test_df,
                                   model_seed=seed)
        all_logits.append(logits)
        if shared_true is None:
            shared_true = y_true

    # Soft ensemble: average softmax probabilities across seeds
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS (Art.6 head)")
    print(f"{'='*60}")

    def softmax(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs_stack = np.stack([softmax(l) for l in all_logits])  # (n_seeds, n, 2)
    avg_probs   = probs_stack.mean(axis=0)                    # (n, 2)
    y_pred      = avg_probs.argmax(axis=1)
    y_true      = shared_true

    if y_true is not None and len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        f1_per = f1_score(y_true, y_pred, average=None,
                          labels=[0, 1], zero_division=0)

        print(f"  Seeds:           {args.seeds}")
        print(f"  accuracy:        {acc:.4f}")
        print(f"  macro_f1:        {macro_f1:.4f}")
        print(f"  macro_precision: {macro_p:.4f}")
        print(f"  macro_recall:    {macro_r:.4f}")
        print(f"  f1_no_violation: {f1_per[0]:.4f}")
        print(f"  f1_violation:    {f1_per[1]:.4f}")
        print()
        print(classification_report(
            y_true, y_pred,
            target_names=['No Violation', 'Violation'],
            zero_division=0,
        ))

        # Save Art.6 ensemble probabilities — compatible with soft_ensemble.py
        probs_path = os.path.join(args.output_dir, 'test_probs.npz')
        np.savez(probs_path, probs=avg_probs, labels=y_true)
        print(f"Art.6 test probabilities saved to {probs_path}  "
              f"(shape: {avg_probs.shape})")
    else:
        print("  No Art.6-relevant test cases found — skipping ensemble report.")

    # Save tokenizer for downstream use
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer')
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-task ECHR violation prediction (Articles 3, 5, 6, 8).'
    )
    parser.add_argument('--data_dir',       type=str,   default='data',
                        help='Root data directory (expects processed/processed.csv)')
    parser.add_argument('--model_name',     type=str,   default='microsoft/deberta-v3-base',
                        help='HuggingFace model identifier for the shared encoder')
    parser.add_argument('--output_dir',     type=str,   default='results/multitask',
                        help='Directory for checkpoints and output files')
    parser.add_argument('--epochs',         type=int,   default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size',     type=int,   default=4,
                        help='Per-device batch size')
    parser.add_argument('--grad_accum',     type=int,   default=1,
                        help='Gradient accumulation steps (simulates larger batch)')
    parser.add_argument('--learning_rate',  type=float, default=2e-5,
                        help='Base (maximum) learning rate for LLRD')
    parser.add_argument('--llrd_decay',     type=float, default=0.9,
                        help='Per-layer LR decay factor for LLRD (e.g. 0.9)')
    parser.add_argument('--patience',       type=int,   default=5,
                        help='Early stopping patience on Art.6 val macro-F1')
    parser.add_argument('--seeds',          type=int,   nargs='+',
                        default=[42, 0, 1, 2],
                        help='Model init seeds to ensemble')
    parser.add_argument('--data_seed',      type=int,   default=42,
                        help='Seed for train/val/test split (fixed across ensemble)')
    parser.add_argument('--mask_shortcuts', action='store_true', default=False,
                        help='CDA: replace country/place/month tokens with [MASK]')
    parser.add_argument('--temporal_split', action='store_true', default=False,
                        help='Use temporal split (train on older, test on newer cases)')
    parser.add_argument('--articles',       type=int,   nargs='+',
                        default=[3, 5, 6, 8],
                        help='Article heads to train (must include 6). Default: 3 5 6 8')
    args = parser.parse_args()
    if 6 not in args.articles:
        raise ValueError("--articles must include 6 (the primary evaluation head)")
    globals()['ARTICLES'] = args.articles
    globals()['ART6_IDX'] = args.articles.index(6)
    main(args)
