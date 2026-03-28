"""
Adversarial debiasing for temporal spurious correlations.
Implements DANN (Domain-Adversarial Neural Networks) with a Gradient Reversal Layer.
The encoder is forced to produce year-invariant representations by adding a
year-bucket predictor with reversed gradients.

Usage:
  python src/adv_train.py --data_dir data --adv_lambda 0.5 --seeds 42 0 1 2
"""

import argparse
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset import ECHRDataset


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return _GRL.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Adversarial mean-pool classifier
# ---------------------------------------------------------------------------

class AdversarialClassifier(nn.Module):
    """
    BERT encoder + mean pooling + two heads:
      1. Classification head  → predicts violation/no-violation
      2. Year-bucket head     → predicts year quartile (adversarial, via GRL)
    """
    NUM_YEAR_BUCKETS = 4

    def __init__(self, model_name, num_labels=2, dropout=0.1,
                 adv_lambda=0.5, class_weights=None):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        # Adversarial year-bucket predictor
        self.year_head  = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_YEAR_BUCKETS),
        )
        self.adv_lambda    = adv_lambda
        self.class_weights = class_weights   # set by trainer after moving to device

    def mean_pool(self, last_hidden, attention_mask):
        mask  = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask,
                labels=None, year_bucket=None, **kwargs):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            w    = self.class_weights.to(logits.device) if self.class_weights is not None else None
            loss = CrossEntropyLoss(weight=w)(logits, labels)

            if year_bucket is not None and self.adv_lambda > 0:
                adv_pooled  = grad_reverse(pooled, self.adv_lambda)
                year_logits = self.year_head(adv_pooled)
                adv_loss    = CrossEntropyLoss()(year_logits, year_bucket)
                loss        = loss + adv_loss

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Dataset with year_bucket
# ---------------------------------------------------------------------------

class ECHRDatasetWithYear(ECHRDataset):
    """Extends ECHRDataset to include year_bucket in each sample."""

    def __init__(self, data, tokenizer, max_len, truncation, year_bins):
        super().__init__(data, tokenizer, max_len, truncation)
        self.year_bins = year_bins   # numpy array of bin edges from np.quantile

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        year = int(self.data.iloc[idx]['year'])
        # Assign to quartile 0-3
        bucket = int(np.searchsorted(self.year_bins, year, side='right'))
        bucket = min(bucket, AdversarialClassifier.NUM_YEAR_BUCKETS - 1)
        item['year_bucket'] = torch.tensor(bucket, dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Trainer that passes year_bucket to model
# ---------------------------------------------------------------------------

class AdvTrainer(Trainer):

    def __init__(self, llrd_decay=0.9, **kwargs):
        super().__init__(**kwargs)
        self.llrd_decay = llrd_decay

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels      = inputs.pop("labels")
        year_bucket = inputs.pop("year_bucket", None)
        outputs     = model(**inputs, labels=labels, year_bucket=year_bucket)
        loss        = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        model = self.model
        lr, wd, d = self.args.learning_rate, self.args.weight_decay, self.llrd_decay

        # AdversarialClassifier always stores backbone as .encoder
        encoder    = model.encoder
        # But .encoder itself might be a BERT/RoBERTa/DeBERTa model —
        # its transformer layers live at encoder.encoder.layer
        if not hasattr(encoder, 'encoder'):
            raise ValueError("Cannot resolve encoder.encoder.layer for LLRD")
        num_layers = len(encoder.encoder.layer)

        groups = []
        groups.append({"params": list(model.classifier.parameters()),
                        "lr": lr, "weight_decay": wd})
        groups.append({"params": list(model.year_head.parameters()),
                        "lr": lr, "weight_decay": wd})
        if hasattr(encoder, 'pooler') and encoder.pooler is not None:
            groups.append({"params": list(encoder.pooler.parameters()),
                            "lr": lr * (d ** 1), "weight_decay": wd})
        for depth, layer in enumerate(reversed(encoder.encoder.layer)):
            groups.append({"params": list(layer.parameters()),
                            "lr": lr * (d ** (depth + 2)), "weight_decay": wd})
        groups.append({"params": list(encoder.embeddings.parameters()),
                        "lr": lr * (d ** (num_layers + 2)), "weight_decay": wd})

        lrs = [g["lr"] for g in groups]
        print(f"LLRD lr range: {min(lrs):.2e} → {max(lrs):.2e}")
        self.optimizer = AdamW(groups, eps=1e-8)
        return self.optimizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    f1_per = f1_score(labels, preds, average=None, labels=[0, 1])
    return {
        'accuracy':        acc,
        'macro_f1':        macro_f1,
        'macro_precision': macro_p,
        'macro_recall':    macro_r,
        'f1_no_violation': float(f1_per[0]),
        'f1_violation':    float(f1_per[1]),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )


def load_and_split(data_path, article6_only, temporal_split, data_seed,
                   val_fraction=0.15):
    df = pd.read_csv(data_path)
    if article6_only:
        mask = (_contains_article(df['violation_articles'], '6') |
                _contains_article(df['nonviolation_articles'], '6'))
        df   = df[mask].copy()
        df['label'] = _contains_article(df['violation_articles'], '6').astype(int)
        print(f"Article 6 subset: {len(df)} cases, {df['label'].mean():.1%} violation")
    else:
        print(f"All articles: {len(df)} cases, {df['label'].mean():.1%} violation")

    val_size = val_fraction / (1 - 0.25)
    if temporal_split:
        cutoff    = int(df['year'].quantile(0.75))
        print(f"Temporal split: train year < {cutoff}, test year >= {cutoff}")
        train_val = df[df['year'] < cutoff].copy()
        test_df   = df[df['year'] >= cutoff].copy()
    else:
        train_val, test_df = train_test_split(
            df, test_size=0.25, stratify=df['label'], random_state=data_seed)

    train_df, val_df = train_test_split(
        train_val, test_size=val_size, stratify=train_val['label'],
        random_state=data_seed)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Year bins from training data quartiles (edges between buckets)
    year_bins = np.quantile(train_df['year'].values,
                             [0.25, 0.5, 0.75]).astype(int)
    print(f"  Year bins (quartile edges): {year_bins.tolist()}")

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            year_bins)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one(args, train_df, val_df, test_df, year_bins, model_seed):
    set_seed(model_seed)

    weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                    y=train_df['label'].values)
    class_weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AdversarialClassifier(
        args.model_name, adv_lambda=args.adv_lambda,
        class_weights=class_weights,
    )

    train_dataset = ECHRDatasetWithYear(train_df, tokenizer, args.max_len,
                                         args.truncation, year_bins)
    val_dataset   = ECHRDatasetWithYear(val_df,   tokenizer, args.max_len,
                                         args.truncation, year_bins)
    test_dataset  = ECHRDatasetWithYear(test_df,  tokenizer, args.max_len,
                                         args.truncation, year_bins)

    run_dir = os.path.join(args.output_dir, f"seed_{model_seed}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.15,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=model_seed,
        gradient_accumulation_steps=args.grad_accum,
        remove_unused_columns=False,   # keep year_bucket in batch
    )

    trainer = AdvTrainer(
        llrd_decay=args.llrd_decay,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    preds_out = trainer.predict(test_dataset)
    logits    = preds_out.predictions
    y_pred    = logits.argmax(-1)
    y_true    = test_df['label'].values
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"  seed={model_seed} → test macro-F1: {f1:.4f}")
    return logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {args.model_name} | adv_lambda: {args.adv_lambda} "
          f"| seeds: {args.seeds}")

    data_path = os.path.join(args.data_dir, "processed", "processed.csv")
    train_df, val_df, test_df, year_bins = load_and_split(
        data_path, args.article6_only, args.temporal_split, args.data_seed)
    y_true = test_df['label'].values

    all_logits = []
    for seed in args.seeds:
        print(f"\n{'='*50}\nTraining seed={seed}\n{'='*50}")
        logits = train_one(args, train_df, val_df, test_df, year_bins, seed)
        all_logits.append(logits)

    # Ensemble
    print(f"\n{'='*50}\nENSEMBLE RESULTS\n{'='*50}")
    softmax_stack = np.stack(
        [np.exp(l) / np.exp(l).sum(axis=1, keepdims=True) for l in all_logits])
    avg_probs = softmax_stack.mean(axis=0)
    y_pred    = avg_probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1])

    print(f"  Seeds: {args.seeds}")
    print(f"  accuracy:        {acc:.4f}")
    print(f"  macro_f1:        {macro_f1:.4f}")
    print(f"  macro_precision: {macro_p:.4f}")
    print(f"  macro_recall:    {macro_r:.4f}")
    print(f"  f1_no_violation: {f1_per[0]:.4f}")
    print(f"  f1_violation:    {f1_per[1]:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                  target_names=['No Violation', 'Violation']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str,   default="data")
    parser.add_argument("--model_name",      type=str,
                        default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--output_dir",      type=str,   default="results/adv")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--max_len",         type=int,   default=512)
    parser.add_argument("--learning_rate",   type=float, default=2e-5)
    parser.add_argument("--llrd_decay",      type=float, default=0.9)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--adv_lambda",      type=float, default=0.5,
                        help="GRL lambda: 0=off, 0.5=balanced, 1.0=strong adversary")
    parser.add_argument("--data_seed",       type=int,   default=42)
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 0, 1, 2])
    parser.add_argument("--truncation",      type=str,   default="head_tail",
                        choices=["head", "head_tail"])
    parser.add_argument("--article6_only",   action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--temporal_split",  action="store_true", default=False)
    parser.add_argument("--grad_accum",      type=int,   default=1)
    args = parser.parse_args()
    main(args)
