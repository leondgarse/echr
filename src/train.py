
import argparse
import os
import random
import shutil
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    set_seed,
)
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import wordnet, stopwords as nltk_stopwords
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
from dataset import ECHRDataset


# ---------------------------------------------------------------------------
# EDA (Easy Data Augmentation) for minority-class oversampling
# SR: synonym replacement   RD: random deletion
# Applied only to NV (label=0) training rows to improve class balance.
# ---------------------------------------------------------------------------

_EDA_STOP = (set(nltk_stopwords.words('english'))
             | {'court', 'case', 'mr', 'mrs', 'ms', 'applicant', 'government',
                'article', 'paragraph', 'convention'})


def _eda_synonym_replace(words: list, n: int, rng: random.Random) -> list:
    words = list(words)
    candidates = [i for i, w in enumerate(words)
                  if w.lower() not in _EDA_STOP and w.isalpha() and len(w) > 3]
    rng.shuffle(candidates)
    replaced = 0
    for i in candidates:
        if replaced >= n:
            break
        syns = []
        for syn in wordnet.synsets(words[i]):
            for lemma in syn.lemmas():
                s = lemma.name().replace('_', ' ')
                if s.lower() != words[i].lower() and ' ' not in s:
                    syns.append(s)
        if syns:
            words[i] = rng.choice(syns)
            replaced += 1
    return words


def _eda_random_delete(words: list, p: float, rng: random.Random) -> list:
    if len(words) <= 1:
        return words
    return [w for w in words if rng.random() > p]


def _eda_one(text: str, seed: int) -> str:
    """One EDA pass: synonym replace 3 words + random delete 5% tokens."""
    rng   = random.Random(seed)
    words = text.split()
    words = _eda_synonym_replace(words, n=3, rng=rng)
    words = _eda_random_delete(words, p=0.05, rng=rng)
    return ' '.join(words) if words else text


def apply_eda(train_df: pd.DataFrame, n_aug: int, data_seed: int) -> pd.DataFrame:
    """Add n_aug EDA copies of every NV training row.
    Each copy uses a deterministic but distinct random seed so results
    are reproducible and each copy is a different perturbation.
    """
    nv_df = train_df[train_df['label'] == 0]
    new_rows = []
    for row_idx, (_, row) in enumerate(nv_df.iterrows()):
        for k in range(n_aug):
            aug_seed = data_seed + row_idx * 1000 + k
            aug_row  = row.copy()
            aug_row['text'] = _eda_one(str(row['text']), seed=aug_seed)
            new_rows.append(aug_row)
    if not new_rows:
        return train_df
    aug_df   = pd.DataFrame(new_rows)
    combined = pd.concat([train_df, aug_df], ignore_index=True).sample(
        frac=1, random_state=data_seed
    ).reset_index(drop=True)
    print(f"  EDA augmented NV ×{n_aug}: +{len(new_rows)} rows → "
          f"{len(combined)} total, {combined['label'].mean():.1%} violation rate")
    return combined


# ---------------------------------------------------------------------------
# Mean-pooling classifier model
# ---------------------------------------------------------------------------

class MeanPoolClassifier(nn.Module):
    """Transformer encoder + mean pooling + linear head.
    Backbone stored as .encoder (single reference — avoids safetensors dup)."""
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs   = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state           # (B, T, H)
        # mean pooling — exclude padding tokens
        mask_exp  = attention_mask.unsqueeze(-1).float()
        summed    = (token_emb * mask_exp).sum(1)
        counts    = mask_exp.sum(1).clamp(min=1e-9)
        pooled    = summed / counts                     # (B, H)
        logits    = self.classifier(self.dropout(pooled))
        loss      = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits, labels)
        # Return a ModelOutput-like object
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Metrics  (macro-F1 is primary, matching SVM evaluation)
# ---------------------------------------------------------------------------

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1])
    return {
        'accuracy':        acc,
        'macro_f1':        macro_f1,
        'macro_precision': macro_p,
        'macro_recall':    macro_r,
        'f1_no_violation': float(f1_per_class[0]),
        'f1_violation':    float(f1_per_class[1]),
    }


# ---------------------------------------------------------------------------
# Trainer: weighted loss + label smoothing + LLRD optimizer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):

    def __init__(self, class_weights=None, llrd_decay=0.9,
                 label_smoothing=0.0, focal_gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.class_weights    = class_weights
        self.llrd_decay       = llrd_decay
        self.label_smoothing  = label_smoothing
        self.focal_gamma      = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weight  = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.focal_gamma > 0:
            # Focal loss: down-weights easy examples, focuses on hard ones
            # FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)
            import torch.nn.functional as F
            log_probs = F.log_softmax(logits, dim=-1)
            probs     = log_probs.exp()
            # per-sample CE with class weights
            ce = F.nll_loss(log_probs, labels,
                            weight=weight, reduction='none')
            p_t       = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_w   = (1 - p_t) ** self.focal_gamma
            loss      = (focal_w * ce).mean()
        else:
            loss = CrossEntropyLoss(
                weight=weight,
                label_smoothing=self.label_smoothing,
            )(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Layer-wise LR decay: higher layers train faster, lower layers slower.
        Works for both BERT-based (.bert) and RoBERTa-based (.roberta) models."""
        model = self.model
        lr    = self.args.learning_rate
        wd    = self.args.weight_decay
        d     = self.llrd_decay

        # Resolve encoder backbone — bert/roberta/deberta/longformer
        # (AutoModelForSeqClass) or .encoder (MeanPoolClassifier)
        encoder = (getattr(model, 'bert',       None)
                   or getattr(model, 'roberta',    None)
                   or getattr(model, 'deberta',    None)
                   or getattr(model, 'longformer', None)
                   or getattr(model, 'encoder',    None))
        if encoder is None or not hasattr(encoder, 'encoder'):
            raise ValueError("Cannot find transformer backbone for LLRD")

        num_layers = len(encoder.encoder.layer)

        groups = []
        groups.append({"params": list(model.classifier.parameters()),
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
# Data loading + splitting
# ---------------------------------------------------------------------------

def _contains_article(series, article):
    return series.fillna('').str.split(';').apply(
        lambda lst: any(a.strip() == article for a in lst)
    )


def load_and_split(data_path, article6_only, temporal_split,
                   data_seed, val_fraction=0.15):
    """
    Load processed.csv, split into train/val/test.
    data_seed controls the split — keep fixed across ensemble runs.
    """
    df = pd.read_csv(data_path)

    if article6_only:
        mask = (_contains_article(df['violation_articles'], '6') |
                _contains_article(df['nonviolation_articles'], '6'))
        df = df[mask].copy()
        df['label'] = _contains_article(df['violation_articles'], '6').astype(int)
        print(f"Article 6 subset: {len(df)} cases, "
              f"{df['label'].mean():.1%} violation rate")
    else:
        print(f"All articles: {len(df)} cases, "
              f"{df['label'].mean():.1%} violation rate")

    if temporal_split:
        cutoff    = int(df['year'].quantile(0.75))
        print(f"Temporal split: train year < {cutoff}, test year >= {cutoff}")
        train_val = df[df['year'] < cutoff].copy()
        test_df   = df[df['year'] >= cutoff].copy()
    else:
        train_val, test_df = train_test_split(
            df, test_size=0.25, stratify=df['label'], random_state=data_seed
        )

    if val_fraction > 0:
        val_size = val_fraction / (1 - 0.25)
        train_df, val_df = train_test_split(
            train_val, test_size=val_size, stratify=train_val['label'],
            random_state=data_seed
        )
        print(f"  Train : {len(train_df)}  (violation {train_df['label'].mean():.1%})")
        print(f"  Val   : {len(val_df)}    (violation {val_df['label'].mean():.1%})")
    else:
        train_df = train_val
        val_df   = train_val.sample(frac=0.10, random_state=data_seed)  # tiny proxy for callbacks
        print(f"  Train : {len(train_df)}  (violation {train_df['label'].mean():.1%})  [no held-out val]")

    print(f"  Test  : {len(test_df)}   (violation {test_df['label'].mean():.1%})")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Single training run — returns test logits
# ---------------------------------------------------------------------------

def train_one(args, train_df, val_df, test_df, model_seed, idf_dict=None):
    """Train one model with a given model_seed; return (test_logits, val_logits)."""
    set_seed(model_seed)

    # EDA augmentation: generate n_aug synonym-replaced / word-deleted NV copies
    if args.eda_nv > 0:
        train_df = apply_eda(train_df, n_aug=args.eda_nv, data_seed=model_seed)

    # NV oversampling: repeat minority-class rows in training set
    if args.oversample_nv > 1:
        nv_rows = train_df[train_df['label'] == 0]
        extra   = pd.concat([nv_rows] * (args.oversample_nv - 1), ignore_index=True)
        train_df = pd.concat([train_df, extra], ignore_index=True).sample(
            frac=1, random_state=model_seed
        ).reset_index(drop=True)
        print(f"  Oversampled NV ×{args.oversample_nv}: "
              f"{len(train_df)} train rows, "
              f"{train_df['label'].mean():.1%} violation rate")

    weights = compute_class_weight('balanced',
                                    classes=np.array([0, 1]),
                                    y=train_df['label'].values)
    class_weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.mean_pool:
        model = MeanPoolClassifier(args.model_name, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        )

    train_dataset = ECHRDataset(train_df, tokenizer, args.max_len, args.truncation,
                                mask_shortcuts=args.mask_shortcuts, idf_dict=idf_dict)
    val_dataset   = ECHRDataset(val_df,   tokenizer, args.max_len, args.truncation,
                                mask_shortcuts=args.mask_shortcuts, idf_dict=idf_dict)
    test_dataset  = ECHRDataset(test_df,  tokenizer, args.max_len, args.truncation,
                                mask_shortcuts=args.mask_shortcuts, idf_dict=idf_dict)

    run_dir = os.path.join(args.output_dir, f"seed_{model_seed}")
    # Clean up stale checkpoints — Trainer auto-resumes if dir exists
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
        save_total_limit=1,     # keep only best checkpoint — prevents disk fill
        seed=model_seed,
        gradient_accumulation_steps=args.grad_accum,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        llrd_decay=args.llrd_decay,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    preds_out  = trainer.predict(test_dataset)
    logits     = preds_out.predictions   # shape (n_test, 2)
    val_preds  = trainer.predict(val_dataset)
    val_logits = val_preds.predictions   # shape (n_val, 2)

    # Print per-seed result
    y_pred = logits.argmax(-1)
    y_true = test_df['label'].values
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"  seed={model_seed} → test macro-F1: {f1:.4f}")

    return logits, val_logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  data_seed: {args.data_seed}  "
          f"|  model_seeds: {args.seeds}  |  truncation: {args.truncation}  "
          f"|  label_smoothing: {args.label_smoothing}")

    data_path = os.path.join(args.data_dir, "processed", "processed.csv")

    # Data split is fixed by data_seed — same test set for all ensemble members
    train_df, val_df, test_df = load_and_split(
        data_path,
        article6_only=args.article6_only,
        temporal_split=args.temporal_split,
        data_seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    y_true = test_df['label'].values

    # Fit TF-IDF on training text for sentence-selection truncation.
    # Fitted on train only to avoid any val/test leakage into sentence scores.
    idf_dict = None
    if args.truncation == 'tfidf_sentence':
        tfidf_vec = TfidfVectorizer(
            min_df=3, max_df=0.9, sublinear_tf=True,
            analyzer='word', token_pattern=r'[a-zA-Z]+',
        )
        tfidf_vec.fit(train_df['text'].fillna('').tolist())
        idf_dict = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_))
        print(f"TF-IDF vocab size: {len(idf_dict):,} terms "
              f"(default OOV IDF: {max(idf_dict.values()):.2f})")

    # --- train one model per seed and collect logits ---
    all_logits     = []
    all_val_logits = []
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Training with model seed={seed}")
        print(f"{'='*50}")
        logits, val_logits = train_one(args, train_df, val_df, test_df,
                                       model_seed=seed, idf_dict=idf_dict)
        all_logits.append(logits)
        all_val_logits.append(val_logits)

    # --- ensemble: average softmax probabilities ---
    print(f"\n{'='*50}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*50}")

    def _softmax(l):
        e = np.exp(l - l.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    softmax_stack = np.stack([_softmax(l) for l in all_logits])
    avg_probs     = softmax_stack.mean(axis=0)

    # Threshold tuning on val set — find t* that maximises val macro-F1
    val_probs_stack = np.stack([_softmax(l) for l in all_val_logits])
    avg_val_probs   = val_probs_stack.mean(axis=0)
    y_val           = val_df['label'].values
    best_thresh, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        vp = (avg_val_probs[:, 1] > t).astype(int)
        vf = f1_score(y_val, vp, average='macro', zero_division=0)
        if vf > best_val_f1:
            best_val_f1, best_thresh = vf, float(t)
    print(f"  Threshold tuning: best_thresh={best_thresh:.2f}  val_macro_f1={best_val_f1:.4f}")

    y_pred_default = avg_probs.argmax(axis=1)
    y_pred         = (avg_probs[:, 1] > best_thresh).astype(int)

    # Default (t=0.5) metrics
    acc0 = accuracy_score(y_true, y_pred_default)
    _, _, f1_0, _ = precision_recall_fscore_support(y_true, y_pred_default, average='macro')
    f1_per0 = f1_score(y_true, y_pred_default, average=None, labels=[0, 1])
    print(f"  [t=0.50 default]  macro_f1={f1_0:.4f}  acc={acc0:.4f}  "
          f"F1(NV)={f1_per0[0]:.4f}  F1(V)={f1_per0[1]:.4f}")

    # Threshold-tuned metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1])

    print(f"  Seeds: {args.seeds}")
    print(f"  accuracy:        {acc:.4f}")
    print(f"  macro_f1:        {macro_f1:.4f}  (threshold={best_thresh:.2f})")
    print(f"  macro_precision: {macro_p:.4f}")
    print(f"  macro_recall:    {macro_r:.4f}")
    print(f"  f1_no_violation: {f1_per[0]:.4f}")
    print(f"  f1_violation:    {f1_per[1]:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                  target_names=['No Violation', 'Violation']))

    # Save ensemble probabilities for downstream soft-ensemble combination
    probs_path = os.path.join(args.output_dir, "test_probs.npz")
    np.savez(probs_path, probs=avg_probs, labels=y_true)
    print(f"Test probabilities saved to {probs_path}")

    # Save final ensemble model (last seed's model as representative)
    save_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        type=str,   default="data")
    parser.add_argument("--model_name",      type=str,   default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--output_dir",      type=str,   default="results")
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--max_len",         type=int,   default=512)
    parser.add_argument("--learning_rate",   type=float, default=2e-5)
    parser.add_argument("--llrd_decay",      type=float, default=0.9)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--focal_gamma",     type=float, default=0.0,
                        help="Focal loss gamma (0=off, 2=standard focal loss)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0=off, 0.1=recommended)")
    parser.add_argument("--data_seed",       type=int,   default=42,
                        help="Seed for train/val/test split — fixed across ensemble")
    parser.add_argument("--seeds",           type=int,   nargs="+",
                        default=[42, 0, 1, 2],
                        help="Model init seeds to ensemble")
    parser.add_argument("--truncation",      type=str,   default="head_tail",
                        choices=["head", "head_tail", "head_reasoning_tail", "tfidf_sentence"])
    parser.add_argument("--mask_shortcuts",  action="store_true", default=False,
                        help="CDA: replace country/place/month tokens with [MASK]")
    parser.add_argument("--mean_pool",       action="store_true", default=False,
                        help="Use mean pooling instead of CLS token")
    parser.add_argument("--article6_only",   action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--temporal_split",  action="store_true", default=False)
    parser.add_argument("--val_fraction",    type=float, default=0.15,
                        help="Fraction of total data for validation (0 = train/test only, no early stopping val)")
    parser.add_argument("--grad_accum",      type=int,   default=1,
                        help="Gradient accumulation steps (simulate larger batch)")
    parser.add_argument("--oversample_nv",  type=int,   default=1,
                        help="Repeat NV (non-violation) training rows N times (1=off, 3=triple NV)")
    parser.add_argument("--eda_nv",         type=int,   default=0,
                        help="EDA augmentation: generate N synonym-replaced copies of each NV training row (0=off)")
    args = parser.parse_args()
    main(args)
