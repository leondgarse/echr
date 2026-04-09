# The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the European Court of Human Rights

**Binary classification of ECHR Article 6 violation outcomes from FACTS-only text.**

---

## Overview

This project investigates whether NLP models predicting European Court of Human Rights (ECHR) violation outcomes learn genuine legal principles or exploit spurious correlations — country identities, year tokens, and procedural boilerplate. Using FACTS-only input (the LAW and Operative Provisions sections are excluded to prevent outcome leakage), we compare TF-IDF baselines against transformer models across random and temporal evaluation splits.

**Key finding:** All models exploit spurious correlations. TF-IDF+SVM's top violation features are year tokens (`1997`, `1998`) and place names (`warsaw`, `bucharest`), not legal content. Neural models that outperform SVM on random splits show larger temporal drops, confirming they memorise year-specific patterns rather than learning transferable legal reasoning.

---

## Results Summary

Primary metric: **macro-F1** (balances violation/non-violation under 82% class imbalance).
Dataset: Enlarged v1 (1,205 Art.6 cases, 8 countries), random 75/25 stratified split.

| Model | Tokens | Macro-F1 |
|-------|--------|----------|
| Dummy (majority class) | — | 0.450 |
| TF-IDF + SVM | 512 (same as BERT) | 0.678 |
| LegalBERT fine-tuned | 512 | 0.678 |
| TF-IDF + SVM | full text | 0.725 |
| Legal-Longformer | 4096 (native) | 0.690 |
| NeoBERT | 4096 (native) | 0.693 |
| DeBERTa-v3-base chunked 4× | 2040 | 0.726 |
| **LegalBERT chunked 4×** | **2040** | **0.752–0.760** |

**The apparent SVM lead over fine-tuned BERT was entirely an artefact of input truncation.** At the same 512-token budget, both score 0.678. LegalBERT only exceeds SVM when given comparable document coverage via 4×510-token sliding-window chunking (macro-F1 0.760 vs SVM 0.725, +0.035). Both legal pretraining and full coverage are necessary — neither alone is sufficient.

---

## Repository Structure

```
echr/
├── echr.ipynb               # Primary analysis notebook (all Phase 2 + Phase 3 work)
├── EDA.ipynb                # Phase 1: EDA, Fighting Words, Scattertext, concordance
├── EDA.pdf                  # Exported EDA output
│
├── data/                    # Original dataset (436 Art.6 cases, RUS/TUR/GBR)
├── data_v1/                 # Enlarged dataset (1,205 Art.6 cases, 8 countries)
├── data_large/              # Extended dataset (3,212 Art.6 cases — higher imbalance)
│
├── results/                 # Saved model probability outputs (.npz) and checkpoints
│   ├── legalbert_chunked_v1_final/   # Best model: LegalBERT chunked 4× on v1
│   ├── legalbert_chunked_original_final/
│   ├── legal_longformer_v2/          # Legal-Longformer (lr=1e-5, 8ep)
│   └── neobert_v2/                   # NeoBERT (LLRD fixed, 4096 tok)
│
├── logs/                    # Training logs for all experiments
│
├── scripts/
│   ├── download_data.py     # HUDOC API data acquisition
│   ├── preprocess_data.py   # FACTS extraction, label assignment, splits
│   ├── train_longformer.py  # Legal-Longformer fine-tuning (4096 tok, CLS or mean pool)
│   ├── train_neobert.py     # NeoBERT fine-tuning (4096 tok, LLRD)
│   ├── train_chunked.py     # DeBERTa chunked fine-tuning
│   ├── analyze_bias.py      # Post-hoc bias analysis (country, year, length)
│   └── svm_features.py      # SVM TF-IDF coefficient extraction
│
├── src/
│   ├── train.py             # LegalBERT fine-tuning (legacy single-model)
│   └── dataset.py           # Dataset utilities, mask_shortcuts (CDA)
│
├── RESULTS.md               # Full experiment log with all runs and findings
├── ANALYSES.md              # Interpretive analysis of spurious correlations
└── REPORT_TLDR.md           # Executive summary of all key findings
```

---

## Datasets

| Dataset | Cases | Art.6 cases | Violation rate | Countries | Path |
|---------|-------|-------------|----------------|-----------|------|
| Original | 952 | 436 | 73.4% | RUS, TUR, GBR | `data/` |
| Enlarged v1 | 2,251 | 1,205 | 81.9% | + POL, ROU, DEU, FRA, BEL | `data_v1/` |
| Extended | 4,258 | 3,212 | 84.7% | + more countries | `data_large/` |

Data sourced from [HUDOC](https://hudoc.echr.coe.int/) via `echr-extractor`. Only the **FACTS section** is used — LAW and Operative Provisions are excluded to prevent outcome leakage (they are written after the decision and directly reveal the outcome).

---

## Quickstart

### Dependencies
```bash
pip install echr-extractor torch transformers pandas scikit-learn nltk
pip install scattertext shifterator lime xformers
```

### Data acquisition
```bash
# Original 3-country dataset
python scripts/download_data.py --countries RUS,TUR,GBR --per_country_count 200 --articles 3,5,6,8

# Enlarged v1 (8 countries)
python scripts/download_data.py \
    --countries POL,ROU,RUS,GBR,DEU,FRA,BEL,TUR \
    --per_country_count 200 --articles 3,5,6,8
```

### Preprocessing
```bash
python scripts/preprocess_data.py --data_dir data/raw
```

### Primary analysis
Open and run **`echr.ipynb`**. Set `DATASET = 'v1'` and `TRAIN_FROM_SCRATCH = False` to load pre-saved results. Set `TRAIN_FROM_SCRATCH = True` with a GPU to retrain from scratch.

### Fine-tune individual models
```bash
# LegalBERT chunked (best model)
# See echr.ipynb §7–9 for the full training stack

# Legal-Longformer
python scripts/train_longformer.py \
    --data_dir data_v1 --output_dir results/legal_longformer_v2 \
    --epochs 8 --learning_rate 1e-5 --seeds 0 1 2 3

# NeoBERT
python scripts/train_neobert.py \
    --data_dir data_v1 --output_dir results/neobert_v2 \
    --epochs 5 --learning_rate 2e-5 --seeds 0 1 2 3
```

---

## Key Findings

1. **Spurious correlations dominate all models.** SVM top violation features: `1997` (0.44), `sąd` (Polish "court", 0.40), `warsaw` (0.38), `appended` (0.37). These are year tokens and country proxies, not legal content.

2. **SVM's apparent advantage was a truncation artefact.** At 512 tokens, SVM and fine-tuned LegalBERT both score 0.678. The entire SVM lead disappears under a token-fair comparison.

3. **Coverage + legal pretraining together beat SVM.** LegalBERT chunked 4×510 tokens (2040 total) achieves 0.752–0.760, beating full-text SVM (0.725). DeBERTa chunked without legal pretraining scores 0.726 — legal pretraining alone adds ~0.03 on top of coverage.

4. **Long-context single-vector models do not close the gap.** Legal-Longformer (4096 tok, legal pretraining) = 0.690; NeoBERT (4096 tok, general) = 0.693 — both ~0.06 below LegalBERT chunked. The bottleneck is single-CLS classification vs chunked mean-pooling of independently-encoded segments.

5. **Neural models show larger temporal drops.** LegalBERT chunked: −0.078 (0.760 → 0.682); DeBERTa chunked: −0.035; SVM: +0.021 (improves). The model with the highest random-split performance memorises the most temporal patterns.

6. **CDA is context-dependent.** Masking country/place/month tokens strongly helps on the 3-country dataset (+0.028) where country identity encodes dominant base rates, but is neutral on the 8-country v1 dataset (−0.004) where country identity carries genuine legal-system signal.

7. **Non-violation features are more legally substantive.** Violation features are event-driven and templated (`quashed`, `detention`, `delay`). Non-violation features reflect evaluative legal reasoning (`whether`, `particular`, `legislative`, `provision`).

---

## Evaluation Protocol

- **Metric:** macro-F1 (equally weights violation and non-violation F1; robust to class imbalance)
- **Random split:** stratified 75/25, `data_seed=42`
- **Temporal split:** train on `year < 75th percentile`, test on `year ≥ 75th percentile` (cutoff ≈ 2014 on v1). Temporal drop = evidence of spurious temporal memorisation.
- **Ensemble:** 4 seeds `[0, 1, 2, 3]`, threshold-tuned on validation set

---

## References

- **Aletras et al. (2016).** Predicting judicial decisions of the European Court of Human Rights: a Natural Language Processing perspective. *PeerJ Computer Science.*
- **Chalkidis et al. (2022).** LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. *ACL.*
- **Medvedeva & McBride (2023).** Legal Judgment Prediction: If You Are Going to Do It, Do It Right. *NLLP @ ACL.*
- **Santosh et al. (2022).** Deconfounding Legal Judgment Prediction for European Court of Human Rights Cases. *EMNLP.*
- **Chalkidis et al. (2021).** Paragraph-level Rationale Extraction through Regularization: A case study on European Court of Human Rights. *NAACL.*
- **Bommasani et al. (2024).** NeoBERT: A Next-Generation BERT. *Transactions on Machine Learning Research.*
