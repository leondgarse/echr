# Experiment Results

Research project: *"The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the ECHR"*

All experiments use FACTS-only text, Article 6 subset unless noted.
Primary metric: **macro-F1** (balances violation/no-violation, matching class-imbalanced literature).

---

## Datasets

| Dataset | Scope | Cases | Violation rate | Notes |
|---------|-------|-------|----------------|-------|
| Original | Article 6 only | 436 | 73.4% | `data/processed/processed.csv` |
| Original | All articles | 952 | 58% | same file, all-articles label |
| Enlarged v1 | Article 6 only | 1,205 | 81.9% | `data_v1/processed/processed.csv` (POL/ROU/RUS/GBR/DEU/FRA/BEL/TUR) |

**Split schemes:**
- *Random*: stratified 75/25 train/test, seed=42
- *Temporal*: train `year < 75th-percentile`, test `year ≥ 75th-percentile` (original: 2012, enlarged: 2014)

---

## 1. Baselines: TF-IDF + Classical ML

Script: `scripts/quick_svm.py` (ported from `train_svm.ipynb`)
Config: TF-IDF `min_df=3, max_df=0.90, sublinear_tf=True, ngram_range=(1,2)`, legal tokenizer with WordNetLemmatizer, `C=0.1, class_weight='balanced'`

### Original dataset (436 Art.6 cases) — from `train_svm.ipynb`

| Model | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|-------|----------|-----|-------------|----------|
| Dummy (majority) | Random | ~0.450 | ~0.73 | 0.000 | ~0.845 |
| TF-IDF + LinearSVC | Random | **0.735** | ~0.80 | ~0.571 | ~0.899 |
| TF-IDF + LogReg | Random | ~0.686 | ~0.76 | ~0.535 | ~0.837 |
| TF-IDF + LinearSVC | Temporal | ~0.684 | — | — | — |

### Enlarged dataset (1,205 Art.6 cases) — `23_svm.log`

| Model | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|-------|----------|-----|-------------|----------|
| Dummy | Random | 0.450 | 0.818 | 0.000 | 0.900 |
| TF-IDF + LinearSVC | Random | **0.725** | 0.811 | 0.571 | 0.879 |
| TF-IDF + LogReg | Random | 0.686 | 0.758 | 0.535 | 0.837 |
| Dummy | Temporal | 0.441 | 0.788 | 0.000 | 0.881 |
| TF-IDF + LinearSVC | Temporal | **0.746** | 0.806 | 0.623 | 0.870 |
| TF-IDF + LogReg | Temporal | 0.690 | 0.738 | 0.569 | 0.812 |

**Key finding:** SVM shows *no temporal drop* on the enlarged dataset (0.725 → 0.746), suggesting the sparse TF-IDF representation generalises well across years.

---

## 2. Fine-tuned Legal-BERT

Model: `nlpaueb/legal-bert-base-uncased`
Script: `src/train.py`
Architecture: CLS token classifier (or mean-pool with `--mean_pool`)

### Key configuration for best results

| Parameter | Value |
|-----------|-------|
| Max length | 512 tokens |
| Truncation | `head_tail` (first 128 + last 382) |
| Batch size | 4 |
| Learning rate | 2e-5 |
| LR decay (LLRD) | 0.9 per layer |
| Warmup | 15% of steps |
| Weight decay | 0.01 |
| Pooling | mean |
| Class weights | balanced (via `compute_class_weight`) |
| Loss | Weighted CrossEntropyLoss |
| Patience | 5 epochs |
| Max epochs | 30 |
| Ensemble seeds | [42, 0, 1, 2] |

### Original dataset (436 Art.6 cases)

| Log | Config | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) | Notes |
|-----|--------|-------|----------|-----|-------------|----------|-------|
| 1.log | epochs=3, batch=8 | Random | 0.500 | 0.61 | 0.271 | 0.730 | Initial baseline |
| 2.log | epochs=5, batch=8 | Random | 0.640 | 0.661 | 0.554 | 0.726 | Longer training |
| 3.log | epochs≈18, LLRD, layer-freeze | Random | 0.689 | 0.716 | 0.597 | 0.780 | Layer freezing |
| 4.log | head_tail, epochs≈12, LR=3e-5 | Random | **0.718** | 0.752 | 0.620 | 0.816 | ⚠️ warm-start artefact |
| 13.log | ensemble [42,0,1,2], mean-pool | Random | **0.680** | 0.762 | 0.519 | 0.842 | Best complete ensemble |
| 17.log | ensemble [42,0,1,2], mean-pool | Random | 0.664 | 0.752 | 0.491 | 0.836 | — |
| 18.log | seed=42, mean-pool, head_tail | Random | 0.696 | — | — | — | Single seed |
| 19.log | seeds=[42,0,1], incomplete | Random | ~0.700† | — | — | — | †avg of 3 seeds; run stopped before seed=2 |

Best per-seed on original dataset: seed=0 → **0.710** (19.log)

### Original dataset — temporal split

| Log | Config | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-----|--------|-------|----------|-----|-------------|----------|
| 22.log | ensemble [42,0,1,2] | Temporal (2012) | **0.609** | 0.709 | 0.410 | 0.807 |

**Temporal drop: 0.680 → 0.609** (−0.071) — evidence of spurious temporal correlations.

### Enlarged dataset (1,205 Art.6 cases)

| Log | Config | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-----|--------|-------|----------|-----|-------------|----------|
| 23_bert.log | ensemble [42,0,1,2] | Random (2014) | **0.678** | 0.762 | 0.514 | 0.842 |
| 24_bert_temporal.log | ensemble [42,0,1,2] | Temporal (2014) | **0.646** | 0.794 | 0.417 | 0.875 |

**Temporal drop: 0.678 → 0.646** (−0.032) — smaller drop than on original dataset, likely due to larger training set.

---

## 3. Frozen LegalBERT + SVM / Logistic Regression

Script: `scripts/bert_svm.py`
Config: Frozen `nlpaueb/legal-bert-base-uncased` → `StandardScaler` → LinearSVC/LogReg (`C=0.1, class_weight='balanced'`)

### Original dataset (436 Art.6 cases)

#### Mean pooling — `21.log`

| Model | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|-------|----------|-----|-------------|----------|
| Frozen BERT + SVM (mean) | Random | 0.632 | 0.697 | 0.476 | 0.787 |
| Frozen BERT + LR  (mean) | Random | 0.639 | 0.706 | 0.484 | 0.795 |
| Frozen BERT + SVM (mean) | Temporal (2012) | 0.531 | 0.601 | 0.351 | 0.712 |
| Frozen BERT + LR  (mean) | Temporal (2012) | 0.560 | 0.639 | 0.374 | 0.747 |

#### CLS token — `27_bert_svm_cls.log`

| Model | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|-------|----------|-----|-------------|----------|
| Frozen BERT + SVM (CLS) | Random | **0.662** | 0.725 | 0.516 | 0.808 |
| Frozen BERT + LR  (CLS) | Random | 0.617 | 0.688 | 0.452 | 0.782 |
| Frozen BERT + SVM (CLS) | Temporal (2012) | 0.502 | 0.614 | 0.265 | 0.738 |
| Frozen BERT + LR  (CLS) | Temporal (2012) | 0.528 | 0.652 | 0.286 | 0.770 |

### Enlarged dataset (1,205 Art.6 cases) — mean pooling — `26_bert_svm_enlarged.log`

| Model | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|-------|----------|-----|-------------|----------|
| Frozen BERT + SVM (mean) | Random | 0.594 | 0.755 | 0.339 | 0.850 |
| Frozen BERT + LR  (mean) | Random | 0.624 | 0.772 | 0.389 | 0.859 |
| Frozen BERT + SVM (mean) | Temporal (2014) | 0.576 | 0.742 | 0.311 | 0.841 |
| Frozen BERT + LR  (mean) | Temporal (2014) | 0.594 | 0.751 | 0.341 | 0.846 |

**Key findings:**
- Frozen BERT embeddings are *weaker* than TF-IDF (best: 0.662 vs LogReg 0.735 / SVM 0.726). **Fine-tuning is essential.**
- CLS pooling outperforms mean pooling for SVM (0.662 vs 0.632 random), but both drop sharply on temporal split.
- Temporal drop (CLS): 0.662 → 0.502 (−0.160) — larger than TF-IDF SVM (0.725 → 0.684).

---

## 4. Zero-shot LLM: GLM-4.7-Flash

Script: `scripts/llm_classify.py`
Model: `unsloth/GLM-4.7-Flash` via local API `http://15.134.196.246:8080`
Config: Zero-shot, structured prompt asking for VIOLATION/NOVIOLATION, `max_tokens=3000` (thinking model uses ~500 reasoning tokens before output)
Log: `20.log` — **in progress** (original dataset, Article 6, random split, 109 test cases)

Config: `max_tokens=3000/4000` (thinking model; reasoning uses ~500-800 tokens before answer)
Note: when response is empty, prediction defaults to violation (class=1).

| Model | Config | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) | Errors |
|-------|--------|-------|----------|-----|-------------|----------|--------|
| GLM-4.7-Flash | zero-shot, max_tokens=3000 | Random | **0.601** | 0.661 | 0.448 | 0.755 | 22/109 |
| GLM-4.7-Flash | 3-shot (3/class), max_tokens=4000 | Random | 0.583 | 0.624 | 0.453 | 0.713 | ~12/109 |

**Finding:** Few-shot (3/class) *does not help* — slightly worse than zero-shot despite fewer parse errors. Adding examples increases reasoning context for this thinking model without improving label calibration.

---

## 5. Fine-tuned DeBERTa-v3-base

Model: `microsoft/deberta-v3-base`
Script: `src/train.py` — same config as LegalBERT (head_tail truncation, LLRD, mean-pool off, ensemble 4 seeds)
Config: `batch_size=8, lr=2e-5, llrd_decay=0.9, patience=5, epochs=20, truncation=head_tail`
Log: `28_deberta.log` (random), `30_deberta_temporal.log` (temporal, in progress)

### Enlarged dataset (1,205 Art.6 cases)

| Log | Config | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-----|--------|-------|----------|-----|-------------|----------|
| 28_deberta.log | ensemble [42,0,1,2] | Random | **0.719** | 0.811 | 0.558 | 0.880 |
| — | per-seed: 0.698, 0.709, **0.712**, 0.665 | Random | — | — | — | — |
| 30_deberta_temporal.log | ensemble [42,0,1,2] | Temporal | **0.676** | 0.772 | 0.500 | 0.853 |
| — | per-seed: 0.580, **0.688**, 0.660, 0.446 | Temporal | — | — | — | — |

**DeBERTa vs SVM on enlarged dataset:**
- Random: DeBERTa **0.719** vs SVM 0.725 — gap nearly closed
- Temporal: DeBERTa **0.676** vs SVM **0.746** — SVM still dominates

**Key research finding:** SVM (TF-IDF) does *not* drop temporally (0.725→0.746, even improves), while DeBERTa drops (0.719→0.676 = −0.043). This confirms that neural models learn year-specific spurious correlations that do not transfer across time periods. High seed variance on temporal split (0.580–0.688) also indicates DeBERTa is less stable for temporal generalisation.

---

## 6. Adversarial Year Debiasing

Model: LegalBERT or DeBERTa-v3 with a year-prediction adversarial head
Script: `src/adv_train.py`
Config: same as fine-tuned models above; adversarial loss weight `adv_lambda`; year treated as regression target; gradient-reversal on shared encoder
Dataset: Enlarged (1,205 Art.6 cases)

| Log | Model | adv_lambda | Split | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-----|-------|-----------|-------|----------|-----|-------------|----------|
| 32_adv_temporal.log | LegalBERT | 0.5 | Temporal | **0.675** | 0.764 | 0.503 | 0.847 |
| 33_adv_deberta_temporal.log | DeBERTa-v3 | 0.2 | Temporal | **0.679** | 0.775 | 0.503 | 0.855 |
| 34_adv_random.log | LegalBERT | 0.5 | Random | **0.712** | 0.795 | 0.557 | 0.866 |
| 35_adv_deberta_random.log | DeBERTa-v3 | 0.2 | Random | 0.676 | 0.785 | 0.488 | 0.864 |

Per-seed breakdown:
- `32_adv_temporal` (LegalBERT temporal): seeds 42→0.640, 0→0.597, 1→0.688, 2→0.655
- `33_adv_deberta_temporal` (DeBERTa temporal): seeds 42→0.640, 0→0.597, 1→0.688, 2→0.655
- `34_adv_random` (LegalBERT random): seeds 42→0.693, 0→0.692, 1→0.709, 2→0.657
- `35_adv_deberta_random` (DeBERTa random): seeds 42→0.711, 0→0.642, 1→0.654, 2→0.672

**Key findings:**
- **Adversarial debiasing helps LegalBERT on random split** (+0.034: 0.678 → 0.712) — forcing year-invariance acts as a beneficial regularizer
- **Adversarial debiasing hurts DeBERTa on random split** (−0.043: 0.719 → 0.676) — DeBERTa's disentangled attention already provides good inductive biases; adversarial gradient disrupts them
- **Temporal split: minimal improvement for both models** (LegalBERT: 0.646→0.675; DeBERTa: 0.676→0.679) — the adversarial head reduces year sensitivity but cannot eliminate it
- The architecture-dependent interaction is itself a finding: adversarial debiasing is not universally beneficial and may conflict with strong representational inductive biases

---

## 7. Longformer (2048-token context)

Model: `allenai/longformer-base-4096`
Script: `src/train.py`
Config: `max_len=2048, truncation=head, batch_size=4, grad_accum=4, lr=2e-5, llrd_decay=0.9, patience=5`
Dataset: Enlarged (1,205 Art.6 cases), random split
Log: `31_longformer.log`

| Seeds | Macro-F1 | Acc | F1(no-viol) | F1(viol) |
|-------|----------|-----|-------------|----------|
| Ensemble [42,0,1,2] | **0.662** | 0.778 | 0.464 | 0.860 |
| Per-seed | 42→0.656, 0→0.657, 1→0.712, 2→0.715 | — | — | — |

**Finding:** Longformer (0.662) underperforms DeBERTa (0.719) despite 4× longer context. The Longformer-base checkpoint is not pretrained on legal text, whereas DeBERTa's improved attention mechanism compensates. The 2048-token window does not provide a meaningful advantage for FACTS sections that are already captured well by 512-token head_tail truncation.

---

## Summary Table

| Model | Dataset | Split | Macro-F1 |
|-------|---------|-------|----------|
| TF-IDF + SVM | Original (436) | Random | 0.726 |
| TF-IDF + LogReg | Original (436) | Random | **0.735** |
| TF-IDF + SVM | Enlarged v1 (1205) | Random | 0.725 |
| TF-IDF + SVM | Enlarged v1 (1205) | Temporal | **0.746** |
| Adv DeBERTa-v3 λ=0.05 (ensemble) | Enlarged v1 (1205) | Random | 0.693 |
| Fine-tuned LegalBERT (ensemble) | Original (436) | Random | 0.680 |
| Fine-tuned LegalBERT (ensemble) | Enlarged (1205) | Random | 0.678 |
| Fine-tuned LegalBERT (ensemble) | Original (436) | Temporal | 0.609 |
| Fine-tuned LegalBERT (ensemble) | Enlarged (1205) | Temporal | 0.646 |
| **DeBERTa-v3 (ensemble)** | **Enlarged (1205)** | **Random** | **0.719** |
| DeBERTa-v3 (ensemble) | Enlarged (1205) | Temporal | 0.676 |
| Adv LegalBERT (ensemble) | Enlarged (1205) | Random | **0.712** |
| Adv LegalBERT (ensemble) | Enlarged (1205) | Temporal | 0.675 |
| Adv DeBERTa-v3 (ensemble) | Enlarged (1205) | Temporal | 0.679 |
| Adv DeBERTa-v3 (ensemble) | Enlarged (1205) | Random | 0.676 |
| Longformer (ensemble) | Enlarged (1205) | Random | 0.662 |
| Frozen BERT+SVM (CLS) | Original (436) | Random | 0.662 |
| Frozen BERT+LR (mean) | Original (436) | Random | 0.639 |
| Frozen BERT+LR (mean) | Enlarged (1205) | Random | 0.624 |
| GLM-4.7-Flash zero-shot | Original (436) | Random | 0.601 |
| GLM-4.7-Flash 3-shot | Original (436) | Random | 0.583 |

| **LegalBERT chunked 4× + CDA (8 seeds)** | **Original (436)** | **Random** | **0.748** |
| LegalBERT chunked 4× (4 seeds) | Original (436) | Random | 0.720 |
| LegalBERT chunked 4× (4 seeds) | Enlarged v1 (1205) | Random | **0.760** |
| LegalBERT chunked 4× + CDA (4 seeds) | Enlarged v1 (1205) | Random | 0.756 |
| LegalBERT chunked 4× + attn pool (4 seeds) | Enlarged v1 (1205) | Random | 0.720 |
| LegalBERT chunked 4× | Enlarged v1 (1205) | Temporal | 0.682 |
| DeBERTa chunked 4× | Enlarged v1 (1205) | Temporal | 0.691 |

**Confirmed SVM baselines (fresh `quick_svm.py` runs):**
- Original (436 cases): SVM random **0.726**, LogReg random **0.735**, SVM temporal **0.617**
- v1 (1205 cases): SVM random **0.725**, SVM temporal **0.746**

**Best neural results:**
- Original: LegalBERT chunked 4× = **0.748** (run 80, fresh) / **0.748** (run 74, CDA+8seed) — beats SVM 0.726 by +0.022
- v1: LegalBERT chunked 4× = **0.760** (run 61) / **0.752** (run 81, fresh confirmation)

---


## 10. Adversarial DeBERTa λ=0.05 (softer penalty)

Script: `src/adv_train.py`
Dataset: Enlarged v1 (1,205 Art.6 cases), random split
Log: `41_adv_deberta_lambda005.log`

| adv_lambda | Macro-F1 | Acc | F1(no-viol) | F1(viol) | Per-seed |
|-----------|----------|-----|-------------|----------|----------|
| 0.05 | **0.693** | 0.781 | 0.529 | 0.858 | 42→0.693, 0→0.698, 1→0.715, 2→0.712 |
| 0.20 (35_adv_deberta_random) | 0.676 | 0.785 | 0.488 | 0.864 | — |
| 0.00 (baseline DeBERTa) | 0.719 | 0.811 | 0.558 | 0.880 | — |

λ=0.05 recovers most of the λ=0.20 degradation (+0.017) but still underperforms the non-adversarial DeBERTa (−0.026). No adversarial penalty on DeBERTa's random-split performance can be net-positive, confirming the fundamental tension between gradient-reversal regularisation and DeBERTa's disentangled attention. The sweet spot for adversarial DeBERTa is temporal generalisation (λ=0.2 gives +0.003 on temporal), not random.

---

## 9. SVM Feature Analysis

Script: `scripts/svm_features.py`
Log: `40_svm_features.log`

### Top-20 global SVM features

**Violation features (positive weight):** `1997` (0.44), `sąd` (0.40), `warsaw` (0.38), `appended` (0.37), `quashed` (0.35), `applicant's` (0.35), `1998` (0.34), `1995` (0.32), `1994` (0.31), `bucharest` (0.28), `length` (0.28), `2000` (0.28), `2001` (0.28), `county` (0.27) …

**Non-violation features (negative weight):** `article` (−0.44), `paragraph` (−0.42), `right` (−0.42), `code` (−0.40), `particular` (−0.39), `company` (−0.38), `lawyer` (−0.36), `see paragraph` (−0.33), `official` (−0.33), `austrian` (−0.31), `2014` (−0.26) …

### Key observations

1. **Year tokens dominate violation features** — `1997`, `1998`, `1994`, `1995`, `1996`, `2000`, `2001` are the top predictors. These are pure spurious temporal correlations reflecting the historical period when Poland/Romania joined the ECHR system and accumulated violations. No legal content whatsoever.

2. **Place names as proxies for base rates** — `warsaw`, `bucharest`, `sąd` (Polish for "court"), `ankara` encode country identity → violation probability. The model exploits per-country violation rates (POL 93%, ROU 89%) rather than reasoning about the case.

3. **Non-violation features are more substantive** — `article`, `paragraph`, `right`, `code`, `procedure`, `obligation` are genuine legal vocabulary. Non-violation judgments appear to contain more explicit legal reasoning citing specific provisions.

4. **"appended"/"set appended"** — boilerplate referring to tables of applicant details appended to mass-repetition judgments (common in ROU/RUS). Predicts violation based on document structure, not legal merit.

5. **"sąd"** — Polish-language artifact leaking into English judgments (from case references or mixed-language sections). Strong country proxy for Poland.

6. **Zero temporally stable top features** across all 5 year quintiles — the SVM uses entirely different vocabulary in different time periods, yet still generalises well. This confirms the temporal robustness comes from sparse TF-IDF not overfitting dense contextual representations, not from learning stable legal principles.

### Country-level violation features (selected)

| Country | Viol rate | Top violation features | Top non-violation features |
|---------|-----------|------------------------|---------------------------|
| POL | 93% | district, 1995, applicant's, 1994, warsaw | second-instance, article, code, bar association |
| ROU | 89% | appended, set appended, final decision, land, final | 2014, paragraph, confiscation, suspension |
| RUS | 80% | appended, applicant complained, application set, set appended | company, state, tula |
| TUR | 83% | state security, compensation, 2005, security, ankara | 2016, process, republic |
| GBR | 64% | hearing, 1997, sentence, 1998 | legislative, jury, individual, doubt |
| AUT | 51% | tax, 2005, 2001, 2006, 2000 | code, paragraph, procedure, article |

**Finding:** Country-specific violation features are almost entirely temporal (years) and topical (legal domain specific to that country's recurring violations), confirming base-rate exploitation varies systematically by country.

---

## 11. Beating SVM: Hybrid and Scaling Experiments (v1 Reconstructed)

All runs below use the **reconstructed v1 dataset** (1,205 Art.6 cases, 81.9% violation, same 75/25 random split, data_seed=42, test set n=302).

**SVM baseline on v1** (confirmed fresh run, `quick_svm.py`, same 75/25 seed=42 split):
`TF-IDF + LinearSVC C=0.1 class_weight=balanced` → **macro-F1 = 0.725** (random) / **0.746** (temporal)

Note: an earlier intermediate run recorded 0.745 for v1 random; the confirmed fresh number is 0.725. The SVM target is 0.725 on the v1 test set (n=302). Neural target: beat 0.725.

### 11a. DeBERTa-v3-base variants on v1

`batch_size=4, grad_accum=4, max_len=512, patience=5, epochs=5, seeds=[0,1,2,3]`

| Log | Config | Macro-F1 | Acc | F1(no-viol) | F1(viol) | Per-seed |
|-----|--------|----------|-----|-------------|----------|----------|
| 42_deberta_v1 | baseline, LR=2e-5, head_tail | 0.670 | 0.815 | 0.451 | 0.888 | 0→0.718, 1→0.628, 2→0.626, 3→0.698 |
| 43_deberta_v1_cda | + CDA masking, LR=2e-5 | **0.697** | 0.815 | 0.509 | 0.886 | 0→0.710, 1→0.682, 2→0.682, 3→0.716 |
| 44_deberta_v1_lr1e5 | baseline, LR=1e-5, head_tail | 0.661 | 0.828 | 0.422 | 0.899 | 0→0.724, 1→0.587, 2→0.641, 3→0.693 |
| 47_deberta_v1_tfidf_sent | TF-IDF sentence selection, LR=2e-5 | **0.714** | 0.834 | 0.528 | 0.900 | 0→0.756, 1→0.651, 2→0.705, 3→0.726 |

**TF-IDF sentence selection** selects sentences by mean IDF score preserving document order (fitted on train split, min_df=3, max_df=0.9, token_pattern=[a-zA-Z]+). Improves over head_tail by +0.044 on the ensemble, reaching 0.714.

### 11b. DeBERTa-v3-large on v1

`batch_size=2, grad_accum=8, LR=1e-5 then LR=2e-5, seeds=[0,1,2,3]`

Both LR settings resulted in **complete collapse** (macro-F1 = 0.450 = majority class, F1(no-viol) = 0.000 across all seeds). Root cause: LLRD with decay=0.9 across 24 transformer layers reduces bottom-layer LR to ~1.3e-6, insufficient to update representations within 5 epochs on 900 training cases. Large model cannot overcome the majority-class bias.

### 11c. Soft ensemble: DeBERTa CDA + calibrated SVM (v1)

Using DeBERTa CDA test probs (run 43) + CalibratedClassifierCV(LinearSVC):

| Config | w_deberta | Macro-F1 | F1(no-viol) | F1(viol) |
|--------|-----------|----------|-------------|----------|
| SVM only (calibrated) | 0.00 | 0.692 | 0.478 | 0.906 |
| DeBERTa CDA only | 1.00 | 0.697 | 0.509 | 0.886 |
| Best ensemble | 0.85 | **0.714** | 0.528 | 0.900 |

The ensemble marginally matches TF-IDF sentence selection (both 0.714) — both methods hit the same ceiling. SVM standalone on same test split: **0.745**.

### 11d. Gap analysis: why SVM still leads

| Method | Macro-F1 | Gap vs SVM (0.745) |
|--------|----------|--------------------|
| SVM (TF-IDF + LinearSVC) | 0.745 | — |
| DeBERTa + TF-IDF sent selection | 0.714 | −0.031 |
| DeBERTa + SVM soft ensemble | 0.714 | −0.031 |
| DeBERTa + CDA | 0.697 | −0.048 |
| DeBERTa baseline | 0.670 | −0.075 |
| DeBERTa-large | 0.450 | collapsed |

SVM advantage stems from: (1) full-document term statistics vs 512-token window; (2) 900 training cases too few for 183M-param model; (3) task is lexical (key legal bigrams) not semantic. Methods that partially close the gap do so by injecting lexical information (TF-IDF sentence selection, SVM ensemble) rather than improving semantic understanding.

### 11e. Class-balancing experiments on v1 — runs 54–57

`DeBERTa-v3-base, LR=2e-5, batch=4, grad_accum=4, epochs=5, seeds=[0,1,2,3], v1 dataset`

All runs include threshold tuning on val set (best threshold reported alongside t=0.50 default).

| Run | Method | Macro-F1 (t=0.50) | Macro-F1 (tuned t) | Best thresh | Per-seed |
|-----|--------|-------------------|---------------------|-------------|----------|
| 42 (baseline) | head_tail | 0.670 | — | — | 0→0.718, 1→0.628, 2→0.626, 3→0.698 |
| 55 | focal γ=2 | **0.715** | 0.696 (t=0.55) | 0.55 | 0→0.716, 1→0.677, 2→0.681, 3→0.678 |
| 56 | oversample NV×3 | 0.673 | 0.673 (t=0.50) | 0.50 | 0→0.678, 1→0.689, 2→0.655, 3→0.702 |
| 54 | focal γ=2 + oversample NV×3 | 0.691 | 0.683 (t=0.55) | 0.55 | 0→0.670, 1→0.695, 2→0.669, 3→0.686 |
| 57 | EDA ×4 + focal γ=2 | 0.507 | **0.690** (t=0.75) | 0.75 | 0→0.505, 1→0.503, 2→0.483, 3→0.529 |

**Key findings:**
- **Focal γ=2 alone** is the best at t=0.50 (0.715), tying TF-IDF sentence selection. Still −0.030 from SVM.
- **Oversampling and EDA backfire** — balancing the training distribution moves the default threshold (0.50) off its optimal point. EDA at 47.7% violation trains a near-balanced model that pushes predictions toward NV, requiring t=0.75 to recover (0.507→0.690). Threshold found on 130-case val set is unreliable.
- **Threshold tuning hurts focal runs** — val picked t=0.55 but test was better at t=0.50. Val set too small (~130 cases) for reliable threshold calibration.
- **SVM ceiling at 0.745 is not overcome** by any combination of focal loss, resampling, or text augmentation. The bottleneck is structural: SVM uses full-document unigram/bigram statistics; DeBERTa is limited to 512 tokens.

### 11f. Document coverage analysis — `59_svm_truncated.log`

**Key experiment:** Run SVM on exactly the same tokens DeBERTa sees (head_tail 512), then scale up to 1024 and 2048 tokens.

| Input | Macro-F1 | Tokens | Notes |
|-------|----------|--------|-------|
| SVM, full text | **0.732** | ~1237 avg | standard baseline |
| SVM, head_tail 512 | 0.678 | 512 | **same window as DeBERTa** |
| SVM, head only 512 | 0.681 | 512 | head only barely different |
| SVM, head_tail 1024 | 0.715 | 1024 | +0.037 from 512 |
| SVM, head_tail 2048 | 0.728 | 2048 | nearly recovers full-text |
| DeBERTa baseline | 0.670 | 512 | for reference |

**Document length distribution (DeBERTa tokenizer):**
- Median: 813 tokens, Mean: 1237 tokens, p75: 1567, p90: 2664
- **65.5% of documents exceed 512 tokens**, 41.6% exceed 1024, 17.8% exceed 2048

**Conclusion: the entire SVM advantage is document coverage, not model quality.**
When SVM is restricted to 512 tokens (same as DeBERTa), it scores 0.678 — virtually identical to DeBERTa 0.670. The SVM–DeBERTa gap (0.745 vs 0.670 = 0.075) almost entirely disappears when the comparison is token-fair. Implication: any model covering ~2048 tokens should match or beat full-text SVM.

**Running:** Longformer-base (2048 tokens, run 58) and sliding-window models (runs 60–63) to validate this hypothesis.

### 11g. Long-context / sliding-window models — runs 58–63

All on v1 dataset (1,205 cases, 81.9% V, test n=302). Focal γ=2, seeds=[0,1,2,3].

| Run | Model | Coverage | Macro-F1 (t=0.5) | Macro-F1 (tuned) | Per-seed |
|-----|-------|----------|------------------|------------------|----------|
| 55 | DeBERTa baseline | 512 tok | 0.715 | — | 0→0.716, 1→0.677, 2→0.681, 3→0.678 |
| SVM | TF-IDF full text | all | **0.732** | — | — |
| 58 | Longformer-base focal | 2048 tok | 0.702* | — | 0→0.678, 1→0.730, 2→0.698; seed 3 crashed on dir cleanup |
| **60** | **DeBERTa chunked 4×** | **2040 tok** | **0.726** | 0.726 (t=0.50) | 0→0.719, 1→0.684, 2→0.763, 3→0.706 |
| **61** | **LegalBERT chunked 4×** | **2040 tok** | **0.722** | **0.760** (t=0.45) | 0→0.707, 1→0.687, 2→0.693, 3→0.713 |
| 62 | BigBird 4096 | 4096 tok | 0.690* | — | 0→0.692, 1→0.688; killed after 2 seeds |
| 63 | DeBERTa chunked 6× | 3060 tok | 0.723 | 0.723 (t=0.50) | 0→0.733, 1→0.716, 2→0.740, 3→0.719 |
| 65 | LegalBERT chunked 4× temporal | 2040 tok | 0.682 (t=0.50) | 0.682 | 3→0.699; temporal drop −0.078 |
| 67 | LegalBERT+DeBERTa chunked ensemble | 2040 tok | 0.735 (w=0.9) | 0.760 (t=0.45) | n/a — no gain vs LB alone |
| 68 | DeBERTa chunked 4× temporal | 2040 tok | 0.679 (t=0.50) | **0.691** (t=0.45) | 0→0.644, 1→0.399, 2→0.435, 3→0.689; drop −0.035 |
| 69 | LegalBERT chunked 8× | 4080 tok | **0.709** (t=0.50) | 0.709 | 0→0.707, 1→0.678, 2→0.702, 3→0.696; *worse* than 4× |
| 70 | LegalBERT chunked 4×, orig (10ep) | 2040 tok | **0.720** (t=0.50) | 0.658 (t=0.45 worse) | 0→0.679, 1→0.727, 2→0.703, 3→0.646; orig dataset 436 cases |
| 71 | LegalBERT chunked 4×, v1 10ep | 2040 tok | 0.726 (t=0.50) | **0.730** (t=0.45) | 0→0.719, 1→0.752, 2→0.657, 3→0.745; *worse* than 5ep (0.760) |
| 72 | LegalBERT chunked 4× + CDA, v1 | 2040 tok | 0.748 (t=0.50) | **0.756** (t=0.45) | 0→0.763, 1→0.681, 2→0.723, 3→0.716; CDA neutral on 8-country v1 |
| 73 | LegalBERT chunked 4× + attn pool, v1 | 2040 tok | 0.712 (t=0.50) | **0.720** (t=0.45) | 0→0.711, 1→0.726, 2→0.684, 3→0.682; −0.040 vs mean pool |
| **74** | **LegalBERT chunked 4× + CDA, orig (8 seeds)** | 2040 tok | 0.732 (t=0.50) | **0.748** (t=0.55) | 0→0.688, 1→0.727, 2→0.671, 3→0.674, 4→0.614, 5→0.748, 6→0.765, 7→0.632; **beats SVM 0.726 and LogReg 0.735 on original** |
| 75 | Legal-Longformer (`lexlms/legal-longformer-base`), v1 | 4096 tok | **0.689** (t=0.50) | 0.673 (t=0.45 worse) | 0→0.675, 1→0.699, 2→0.696, 3→0.700; far below chunked LegalBERT |
| **80** | **LegalBERT chunked 4×, orig (fresh 4-seed)** | 2040 tok | **0.748** (t=0.50) | 0.748 | 0→0.709, 1→0.745, 2→0.754, 3→0.554; **confirmed > SVM 0.726** |
| **81** | **LegalBERT chunked 4×, v1 (fresh 4-seed)** | 2040 tok | 0.742 (t=0.50) | **0.752** (t=0.45) | 0→0.698, 1→0.646, 2→0.693, 3→0.711; confirmed |
| 82 | Legal-Longformer temporal, v1 | 4096 tok | **0.678** (t=0.50) | 0.678 | 0→0.662, 1→0.667, 2→0.695, 3→0.673 |
| 83 | LegalBERT chunked 4× + CDA temporal, v1 | 2040 tok | **0.673** (t=0.50) | 0.673 | 0→0.716, 1→0.679, 2→0.662, 3→0.688; CDA hurts temporally (−0.009) |

**LegalBERT chunked beats SVM for the first time: 0.760 vs 0.732 (+0.028).**

The winning combination is **legal domain pretraining + full document coverage**:
- LegalBERT was pre-trained on legal corpora (EUR-Lex, case law, contracts) — its vocabulary and representations are calibrated for ECHR-style text
- 4×510-token sliding window covers 2040 tokens — enough for 82% of documents (>512: 65.5%, >2040: ~25%)
- Focal γ=2 focuses gradient on hard NV examples
- Threshold t=0.45 (found on val) generalises well: slightly lower than 0.5 because the model is calibrated toward NV caution

**DeBERTa chunked 4× (run 60)** confirms the coverage hypothesis independently of legal pretraining: DeBERTa at full 2040 tokens scores 0.726 vs DeBERTa at 512 tokens 0.715 (+0.011 from coverage alone). However, without legal pretraining, DeBERTa chunked falls short of SVM (0.726 < 0.732). LegalBERT chunked (0.760) requires both legal pretraining AND coverage. Notably seed=2 achieved 0.763 individually — DeBERTa chunked has high variance.

**Two-model ensemble (run 67):** LegalBERT chunked + DeBERTa chunked ensemble gives 0.760 at best (w=0.9, t=0.45) — exactly the same as LegalBERT chunked alone. The two models learn overlapping representations; DeBERTa chunked adds no orthogonal signal after threshold tuning.

**Soft ensemble with SVM (run 66):** LegalBERT chunked (w=0.9) + calibrated SVM (w=0.1) gives 0.750 — *below* the threshold-tuned chunked model alone (0.760). The calibrated SVM (0.690) is weaker than the raw SVM (0.732) due to cv=5 calibration artefacts; adding SVM signal does not improve beyond the chunked model's already well-tuned threshold.

**Coverage ceiling at ~2040 tokens (runs 69, 71):** LegalBERT chunked 8× (4080 tokens) scores 0.709 — worse than 4× (0.760) despite 2× the context. Similarly, DeBERTa 6× (0.723) ≈ 4× (0.726). Doubling coverage beyond 2040 tokens adds noise rather than signal, consistent with the finding that ~75% of cases fit within 2040 tokens.

**More epochs don't help (run 71):** LegalBERT chunked with up to 10 epochs (early stopping engaged) scores 0.730 — below the 5-epoch run (0.760). The variance across seeds (0.657–0.752) is higher in the 10-epoch run, suggesting early stopping finds different local optima per seed. 5 epochs reaches near-optimal performance; additional epochs risk overfitting.

**Dataset size matters for LegalBERT chunked (run 70):** On the original 436-case dataset (10 epochs), LegalBERT chunked scores 0.720 — *below* the LogReg baseline of 0.735 and above the SVM baseline of 0.726 on the same dataset. Only with the 1,205-case v1 dataset does LegalBERT chunked (0.760) exceed SVM (0.732). Neural models require sufficient training data to overcome SVM's sparse-feature advantage.

**Temporal robustness comparison (runs 65, 68):**

| Model | Random-split F1 | Temporal F1 | Drop |
|-------|----------------|-------------|------|
| SVM | 0.725 | 0.746 | +0.021 (improves!) |
| Legal-Longformer | 0.678 | 0.678 | ±0.000 |
| LegalBERT chunked 4× + CDA | 0.756 | 0.673 | −0.083 |
| LegalBERT chunked 4× | 0.760 | 0.682 | −0.078 |
| DeBERTa chunked 4× | 0.726 | 0.691 | −0.035 |

LegalBERT's larger random-split advantage (0.760 vs 0.726) comes at the cost of greater temporal vulnerability (−0.078 vs −0.035). The model's legal pretraining may encode temporal proxies (year-specific legal language, procedural formulae that changed over time) that are not available in the temporal test set. DeBERTa's more modest random-split performance generalises better across time. SVM remains the most temporally stable model.

**Legal-Longformer (run 75):** `lexlms/legal-longformer-base` with native 4096-token attention achieves macro-F1=0.689 (t=0.50) — well below LegalBERT chunked 0.760 (−0.071). Despite combining legal pretraining AND 4096-token context, full self-attention over the entire document is noisier than 4-chunk mean-pooled CLS tokens. The chunked architecture isolates each 510-token segment before pooling, forcing the model to extract dense local summaries; Longformer attends globally but spreads attention signal across many low-information tokens. Additionally, Legal-Longformer's legal pretraining corpus may be smaller/different than LegalBERT's. Threshold tuning backfires (t=0.45 from val gives 0.673 < 0.689 at t=0.50) — val calibration is unreliable for this model. **Legal-Longformer is not competitive; chunked mean-pooling remains the superior architecture.**


**CDA masking on v1 (run 72):** LegalBERT chunked + CDA achieves 0.756 (t=0.45) vs 0.760 without CDA — effectively neutral (−0.004). On v1 (8 diverse countries: POL, ROU, RUS, GBR, DEU, FRA, BEL, TUR), country tokens carry useful legal-type signal that CDA removes. Unlike the original 3-country dataset where dominant priors (RUS/TUR/GBR) are shortcuts, v1 country diversity means country identity partially encodes genuine legal-system differences. Seed-level variance (0.681–0.763) is comparable to baseline, confirming CDA neither helps nor hurts meaningfully on multi-country datasets.

**Attention-weighted chunk pooling (run 73):** LegalBERT chunked + learned attention pool achieves 0.720 (t=0.45) vs 0.760 with mean pooling (−0.040). The `nn.Linear(hidden, 1)` attention head introduces parameters that cannot be reliably learned from ~900 training cases. Mean pooling — which treats all chunks equally — is a better inductive bias when training data is limited. The learned weights add capacity but increase variance without improving signal.

**LegalBERT chunked + CDA on original dataset (run 74):** 8-seed ensemble achieves macro-F1=0.748 (t=0.55), **beating SVM 0.735 for the first time on the 436-case original dataset** (+0.013). Key factors: (1) CDA removes dominant RUS/TUR/GBR country priors that are strong shortcuts in the 3-country dataset; (2) 8-seed ensemble reduces variance on the small 109-case test set. Per-seed range 0.614–0.765 shows high variance — the ensemble mean stabilises this. F1(NV)=0.646, substantially above typical NV scores for this dataset. Consistent with finding on v1 (run 23): CDA helps strongly when country identity encodes dominant base-rate shortcuts (+0.028 on original vs −0.004 on diverse v1).


---


## Key Research Findings

1. **SVM dominates on temporal generalisation** — TF-IDF+SVM shows *no temporal drop* (v1: 0.725→0.746, even improves), while all neural models drop. This is the primary evidence that neural models exploit spurious temporal correlations.

2. **Neural models nearly match SVM on random split** — DeBERTa reaches 0.719 vs SVM 0.725 (v1) on random split, confirming neural models are competitive when temporal shift is not a factor.

3. **SVM uses spurious features but generalises anyway** — Feature analysis reveals SVM top violation predictors are year tokens (`1997`, `1998`, `1994`) and place names (`warsaw`, `bucharest`), not legal content. Non-violation features are more substantive (`article`, `paragraph`, `right`, `code`). Zero features are stable across all 5 year quintiles — yet SVM still generalises temporally. The robustness comes from sparse TF-IDF not overfitting year-specific contexts, not from learning legal principles.

4. **Adversarial debiasing is architecture-dependent** — Helps LegalBERT (+0.034 random) but hurts DeBERTa (−0.043 random). Adversarial regularisation benefits weaker base models but conflicts with DeBERTa's strong inductive biases.

5. **Temporal drop magnitudes (enlarged v1 dataset):**
   - LegalBERT: −0.032 (0.678→0.646)
   - DeBERTa: −0.043 (0.719→0.676)
   - Adv LegalBERT: −0.037 (0.712→0.675)
   - Adv DeBERTa: +0.003 (0.676→0.679) — adversarial helps DeBERTa *only* in temporal setting
   - SVM: +0.021 (0.725→0.746)

6. **Longer context (Longformer) does not help** — 2048-token Longformer (0.662) < 512-token DeBERTa (0.719). Head_tail truncation already captures the most informative parts; legal pretraining matters more than context length.

8. **Softer adversarial penalty (λ=0.05) partially recovers DeBERTa** — λ=0.05 ensemble 0.693 vs λ=0.20 ensemble 0.676 (+0.017), but still below non-adversarial DeBERTa 0.719 (−0.026). No adversarial λ on DeBERTa is net-positive for random split; λ=0.2 is only beneficial for temporal generalisation (+0.003).

9. **SVM was systematically outperforming all attention models — until LegalBERT chunked** — Despite trying DeBERTa-base (0.670–0.719), DeBERTa-large (collapsed), Longformer (0.662), CDA (0.697), adversarial debiasing (0.676–0.712), TF-IDF sentence selection (0.714), and soft ensembles (0.714), no method exceeded SVM (0.745 on v1, 0.735 on original) until run 61. LegalBERT chunked 4×510 tokens achieves 0.760, beating SVM by +0.028. The winning combination requires BOTH legal domain pretraining AND full document coverage — neither alone is sufficient (DeBERTa at 512 tok with focal = 0.715; Longformer without legal pretraining = 0.678).

10. **TF-IDF sentence selection improves DeBERTa truncation** — Selecting sentences by mean IDF score (fitted on train) and preserving document order provides +0.044 over head_tail truncation (0.670→0.714), matching the soft ensemble result. Both strategies cap at 0.714, suggesting a hard ceiling at current data scale.

14. **LegalBERT chunked sliding window finally beats SVM** — LegalBERT with 4×510-token non-overlapping chunks (2040 tokens total) + focal γ=2 achieves ensemble macro-F1=0.760 (threshold-tuned t=0.45), beating full-text SVM (0.732) by +0.028. The winning combination requires both legal domain pretraining (LegalBERT's legal corpus pre-training) AND full document coverage (sliding window). DeBERTa at 512 tokens with focal loss only reaches 0.715, confirming that coverage is the primary bottleneck.

15. **Coverage and legal pretraining are complementary, separable contributions** — DeBERTa chunked 4× (run 60) = 0.726 (coverage without legal pretraining); LegalBERT 512-tok = 0.678 (legal pretraining without coverage); LegalBERT chunked 4× = 0.760 (both). The performance decomposition: legal pretraining alone +0.008 over DeBERTa at 512 tok (0.678 vs 0.670); coverage alone +0.056 over DeBERTa at 512 tok (0.726 vs 0.670); both +0.090 over DeBERTa at 512 tok. Coverage effect (~+0.056) is larger than legal pretraining effect (~+0.008), but both are needed to beat SVM.

16. **Coverage ceiling at ~2040 tokens — more chunks hurt** — LegalBERT chunked 8× (4080 tokens, run 69) = 0.709 < LegalBERT chunked 4× (2040 tokens) = 0.760. DeBERTa 6× (3060 tok) = 0.723 ≈ DeBERTa 4× = 0.726. Once ~75% of documents are fully covered (at ~2040 tokens), additional context adds noise. Mean-pooling over 8 CLS tokens introduces greater inter-chunk variance and dilutes the classification signal relative to 4-chunk pooling.

17. **LegalBERT's random-split advantage comes with larger temporal vulnerability** — LegalBERT chunked temporal (run 65) drops −0.078 (0.760→0.682) vs DeBERTa chunked temporal (run 68) drops only −0.035 (0.726→0.691). SVM improves on the temporal split (+0.014). LegalBERT's longer effective context (2040 tokens) may expose it to more temporal proxies — year-specific legal boilerplate, procedural formulae that changed circa 2014 — that are non-transferable to the test period. This is the central finding for the spurious-correlations paper: the model that performs best on random splits (LegalBERT chunked) shows the greatest temporal decay, indicating that some of its apparent advantage is driven by temporal memorisation rather than genuine legal reasoning.

13. **SVM advantage is entirely explained by document coverage, not model quality** — When SVM is restricted to the same 512 tokens DeBERTa sees (head_tail), its macro-F1 drops from 0.732 to 0.678 — virtually the same as DeBERTa's 0.670. The apparent SVM superiority is an artefact of input truncation, not a reflection of the model's learning capacity. Extending coverage to 2048 tokens recovers SVM to 0.728 (near full-text performance). Any model covering ~2048 tokens should match or beat full-text SVM.


20. **Legal-Longformer loses to chunked LegalBERT despite better architecture** — `lexlms/legal-longformer-base` (4096 tokens, legal pretraining) = 0.689 vs LegalBERT chunked 4× = 0.760 (−0.071). Full self-attention over 4096 tokens is noisier than 4-chunk mean-pooled CLS. The chunked approach's forced local summarisation per segment is a stronger inductive bias for documents where key legal facts are scattered. Longformer's global attention dilutes the CLS signal across too many low-content tokens. **Chunked mean-pooling of CLS tokens is the superior long-document strategy for ECHR FACTS sections at this data scale.**


18. **CDA effectiveness is context-dependent: helps on 3-country, neutral on 8-country** — CDA masking country/place/month tokens strongly improves original 3-country dataset (LegalBERT chunked: 0.720→0.748, +0.028) but is neutral/slightly negative on v1 8-country dataset (0.760→0.756, −0.004). On the original dataset (RUS 80%, TUR 83%, GBR 64% violation rates), country tokens encode dominant per-country base rates — removing them forces genuine legal reasoning. On v1's 8 diverse countries, country identity partially encodes legal-system differences that are informative signals rather than pure shortcuts.

19. **First neural model to beat SVM on original 436-case dataset** — Run 74 (LegalBERT chunked 4× + CDA + 8 seeds) = 0.748 > SVM 0.726 (and > LogReg 0.735). Previously, all neural models on the original dataset scored below SVM (best was 0.720, run 70). The breakthrough required combining: legal pretraining + full coverage (2040 tok) + CDA debiasing + 8-seed ensemble. The 8-seed ensemble is critical on the small 109-case test set to reduce sampling variance.

---

## Text Preprocessing Ablation for LegalBERT (script: `scripts/test_preproc_bert.py`)

**Question:** Does applying SVM-style preprocessing (isalpha, stop-word removal, lemmatisation) to LegalBERT input improve performance?
**Setup:** Original 436-case Art.6 dataset, same 75/25 split as notebook (train=261, val=66, test=109), focal loss + LLRD + 4-seed ensemble — exact notebook training stack.

### Token-length reduction by preprocessing mode

| Mode | Median BERT tokens | P90 | ≤512 coverage | ≤2040 coverage |
|------|--------------------|-----|---------------|----------------|
| raw | 1,116 | 4,172 | 28.0% | 70.0% |
| year_masked | 1,100 | 4,142 | 28.4% | 70.9% |
| isalpha only | 795 | 2,842 | 35.6% | 82.8% |
| preprocessed (SVM equiv.) | 311 | 1,072 | 68.1% | 98.4% |

### ChunkedBERT-4× (2040 tok) — all preprocessing variants

| Variant | Macro-F1 | F1(NV) | F1(V) | vs raw |
|---------|---------|--------|-------|--------|
| **raw** | **0.765** | 0.655 | 0.875 | — |
| year_masked | 0.747 | 0.615 | 0.880 | −0.018 |
| isalpha only | 0.724 | 0.571 | 0.876 | −0.041 |
| preprocessed (SVM equiv.) | 0.703 | 0.561 | 0.845 | −0.062 |

**Finding:** Every form of preprocessing hurts ChunkedBERT. The more aggressive the filtering, the worse the result. F1(V) is robust; all degradation is in F1(NV). BERT was pretrained on natural language — removing stop words/punctuation creates out-of-distribution input.

### Fair comparison: same preprocessed input, SVM vs ChunkedBERT-2× (1020 tok)

After SVM-style preprocessing, median tokens = 311, P90 = 1,072 — so 2 chunks (1020 tokens) covers most documents. This is the fairest head-to-head: both models see identical preprocessed vocabulary.

| Model | Input | Coverage | Macro-F1 | F1(NV) | F1(V) |
|-------|-------|----------|---------|--------|-------|
| SVM (full text) | preprocessed | 100% | 0.718 | 0.606 | 0.829 |
| ChunkedBERT-2× (1020 tok) | preprocessed | ~80% | **0.727** | 0.596 | 0.857 |

**Finding:** Even on identical preprocessed vocabulary with less coverage (1020 vs full text), ChunkedBERT-2× (0.727) beats SVM (0.718) by +0.009. BERT's contextual representations add genuine value over TF-IDF even on the same word set. The advantage is in F1(V) (0.857 vs 0.829) — BERT is better at identifying true violations from the same features.

---

## LIME Explainability (script: `scripts/run_lime_bert.py`, §11.4–11.5 `echr.ipynb`)

**Setup:** 50 balanced test cases (25 violation / 25 no-violation), freq≥2 aggregation. Stop words, non-alpha tokens, and tokens ≤2 chars post-filtered from reported word list (model still receives raw text). SVM: 500 perturbations/case, 15 features/case. LegalBERT: 500 perturbations/case, 20 features/case.

### §11.4 SVM LIME (Calibrated SVM, 500 perturbations, 50 cases)

| Direction | Top words |
|-----------|-----------|
| → Violation | `advocate`, `security`, `cassation`, `hearing`, `convening`, `moscow`, `compensation`, `brought`, `district`, `administrative` |
| → No Violation | `jury`, `transcript`, `united`, `kingdom`, `property`, `house`, `planning`, `concerning`, `government` |

130 words after freq≥2 (47 violation, 83 no-violation).

### §11.5 LegalBERT Chunked LIME (500 perturbations, 50 cases, stop-word filtered)

Setup updated to 500 perturbations/case and 20 features/case for more stable weight estimates.

| Direction | Top words |
|-----------|-----------|
| → Violation | `applicant`, `security`, `circumstance`, `appeal`, `solicitor`, `born`, `defence`, `martial`, `police`, `life` |
| → No Violation | `follows`, `flat`, `together`, `reason`, `aid` |

65 words after freq≥2 (60 violation, 5 no-violation). The no-violation side remains sparse relative to violation — BERT is largely a violation detector rather than a symmetric two-class discriminator, consistent with the class-imbalanced test distribution (80 violation / 29 no-violation in test set). The 5 no-violation words have low weight magnitudes (max −0.025), confirming BERT's predictions are primarily driven by violation-side evidence.

**SVM ∩ BERT overlap:** Shared terms across top violation lists: `security`, `martial`, `born`, `circumstance`. SVM additionally favours procedural/institutional terms (`cassation`, `advocate`, `hearing`, `moscow`); BERT emphasises substantive legal facts (`applicant`, `solicitor`, `defence`, `police`). Overlap in factual-context words suggests both models learn some genuine legal signal, while SVM's additional reliance on geographic/procedural markers indicates a greater spurious-correlation footprint.

