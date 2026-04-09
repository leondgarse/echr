# The Artificial Judge: TL;DR

**"Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the ECHR"**

---

## What We Did

We trained NLP models to predict whether the European Court of Human Rights found an Article 6 violation, then asked: are these models learning legal reasoning, or exploiting spurious correlations?

**Setup:**
- Input: FACTS-only section from HUDOC API (LAW section excluded to prevent obvious leakage)
- Metric: macro-F1 (balances violation/non-violation classes)
- Primary dataset (v1): 1,205 Article 6 cases, 81.9% violation rate, 8 countries (POL/ROU/RUS/GBR/DEU/FRA/BEL/TUR)
- Original dataset: 436 cases, 73.4% violation, 3 countries (RUS/TUR/GBR)
- Evaluation: random 75/25 stratified split

---

## Models and Results

| Model | Input | v1 random | Orig random |
|---|---|---|---|
| TF-IDF + SVM | Full text | 0.725 | 0.726 |
| TF-IDF + LogReg | Full text | 0.686 | **0.735** |
| TF-IDF + SVM | 512 tok (head+tail) | 0.678 | — |
| TF-IDF + SVM | 1024 tok | 0.715 | — |
| TF-IDF + SVM | 2048 tok | 0.728 | — |
| LegalBERT fine-tuned | 512 tok (head+tail) | 0.678 | 0.680 |
| DeBERTa-v3 fine-tuned | 512 tok (head+tail) | 0.719 | — |
| Adversarial LegalBERT | 512 tok | 0.712 | — |
| Legal-Longformer | 4096 tok (native) | 0.689 | — |
| NeoBERT fine-tuned | 4096 tok (native) | 0.693 | — |
| DeBERTa chunked 4× | 2040 tok (4×510) | 0.726 | — |
| **LegalBERT chunked 4×** | **2040 tok (4×510)** | **0.760** | **0.748** |
| LegalBERT chunked + CDA | 2040 tok | 0.756 | 0.748 |

---

## Key Findings

### 1. Spurious correlations dominate all models

SVM's top violation features: `1997` (weight 0.44), `sąd` (Polish "court", 0.40), `warsaw` (0.38), `appended` (0.37), `1998` (0.34), `1994` (0.31), `bucharest` (0.28). These are year tokens and country proxies. Poland has a 93% violation rate, Romania 89%. The model predicts violation because the case is Polish from the 1990s — not because of legal merits.

### 2. SVM's apparent advantage was a truncation artefact

ECHR FACTS sections are long: median ~813 tokens, mean ~1,237 tokens; 65.5% exceed 512 tokens. SVM sees the full document; BERT-family models are hard-capped at 512. The smoking gun:

| SVM token budget | Macro-F1 (v1) | Coverage |
|---|---|---|
| 512 tok (same as BERT) | **0.678** | ~35% of docs fully covered |
| 1024 tok | 0.715 | ~59% |
| 2048 tok | 0.728 | ~82% |
| Full text | 0.725 | 100% |
| LegalBERT fine-tuned @ 512 tok | **0.678** | ~35% |

When SVM is restricted to the same 512 tokens BERT sees, it scores 0.678 — essentially identical to fine-tuned LegalBERT (0.678). The entire apparent 0.047 SVM lead over LegalBERT disappears under a token-fair comparison. SVM's advantage was seeing the full document, not being a better model.

### 3. Coverage + legal pretraining finally beats SVM

Sliding-window encoding (4×510-token chunks, mean-pool CLS) with LegalBERT achieves **0.760 on v1** and **0.748 on the original dataset**, beating full-text SVM on both. Both ingredients are necessary:
- DeBERTa chunked (coverage without legal pretraining) = 0.726
- LegalBERT at 512 tokens (legal pretraining without coverage) = 0.678
- **LegalBERT chunked (both)** = **0.760**

Legal-Longformer (`lexlms/legal-longformer-base`) with native 4096-token attention = 0.689 — despite combining both ingredients, full self-attention over 4096 tokens is noisier than chunked mean-pooling. Forced local summarisation per 510-token segment is a better inductive bias for this task.

### 4. Coverage ceiling at ~2040 tokens

LegalBERT chunked 8× (4080 tokens) = 0.709 < 4× (2040 tokens) = 0.760. Once ~75% of documents are covered, additional context adds noise through mean-pooling of more CLS vectors rather than signal.

### 5. CDA is context-dependent

Counterfactual Data Augmentation — masking country/place/month tokens during training — strongly helps on the 3-country dataset (+0.022 → 0.748) but is neutral on v1 (−0.004). On datasets with dominant country priors (RUS 80%, TUR 83%, GBR 64%), CDA forces genuine legal learning. On diverse multi-country datasets, country tokens carry useful legal-system signal that masking removes.

### 6. Non-violation features are more substantive

Violation features are event-driven and templated: `quashed`, `detention`, `delay`, `state security`. Non-violation features reflect evaluative legal reasoning: `whether`, `particular`, `legislative`, `provision`, `meaning`. Non-violation predictions are closer to genuine legal reasoning; violation predictions pattern-match on case origin and procedural failure types.

---

## What Didn't Work

- **DeBERTa-large**: collapsed to majority-class prediction — over-parameterised for ~900 training cases
- **Legal-Longformer (4096 tok native)**: 0.689 — full self-attention over 4096 tokens is noisier than 4-chunk mean-pool; confirmed across LR/epoch/data/pooling tuning
- **NeoBERT (250M, general domain, 4096 tok)**: 0.693 — matches Legal-Longformer despite no legal pretraining; 28-layer depth causes training instability; both ~0.06 below LegalBERT chunked
- **More epochs (10 vs 5)**: worse (0.730 vs 0.760) — overfitting
- **8 chunks vs 4 chunks**: worse — coverage ceiling
- **Adversarial debiasing on DeBERTa**: hurts random split (−0.043) — conflicts with DeBERTa's disentangled attention
- **Multi-task learning (Art. 3+5+6+8 shared encoder)**: hurts Art. 6 (−0.012) — failing auxiliary heads corrupt the shared encoder
- **Dataset expansion to v3 (3,212 cases, 84.7% violation)**: all models degrade — new countries have near-100% violation rates, worsening class imbalance

---

## Bottom Line

The apparent superiority of TF-IDF+SVM over fine-tuned transformers on ECHR violation prediction was an artefact of input truncation: at the same 512-token budget, both score ~0.678. Once document coverage is restored via sliding-window chunked encoding combined with legal domain pretraining, LegalBERT finally surpasses SVM (+0.035 on v1, +0.022 on original dataset).

Despite this improvement, all models exploit spurious correlations rather than learning law. SVM's top features are year tokens and country names; per-country accuracy tracks violation rates. The task is primarily lexical: models learn which countries and time periods generate violations, not why the court ruled as it did.
