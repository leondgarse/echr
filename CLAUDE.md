# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: *"The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the European Court of Human Rights"*

Investigates whether NLP models predicting ECHR violation outcomes learn genuine legal principles or exploit spurious correlations (procedural boilerplate, country names, text length).

## Key Commands

```bash
# Data acquisition (~1,200 cases from HUDOC API)
python scripts/download_data.py --countries RUS,TUR,GBR --per_country_count 200 --articles 3,5,6,8

# Preprocessing: extract FACTS sections, assign labels, create splits
python scripts/preprocess_data.py --data_dir data/raw

# Fine-tune Legal-BERT
python src/train.py --epochs 3 --batch_size 8 --output_dir results

# CPU-only training
CUDA_VISIBLE_DEVICES="" python src/train.py --epochs 1 --batch_size 2

# Bias analysis (run after training)
python scripts/analyze_bias.py
```

No `requirements.txt` exists. Key packages: `echr-extractor`, `torch`, `transformers`, `pandas`, `scikit-learn`, `lime`, `scattertext`, `shifterator`, `nltk`.

## Architecture & Data Flow

```
HUDOC API → download_data.py → data/raw/{metadata.csv, full_text.json}
                                    ↓
                          preprocess_data.py → data/processed/processed.csv
                          (1,590 raw cases → ~952 valid cases)
                                    ↓
                              echr.ipynb  ←── primary analysis notebook
                    (SVM · LegalBERT-512 · LegalBERT-Chunked · Bias · LIME)
                                    ↓
                         results/{legalbert_*/,  *.npz}   logs/*.log
```

**`echr.ipynb`** — **Primary analysis notebook** (all Phase 2 + Phase 3 work). Self-contained. Sections:
1. Configuration (`DATASET`, `TRAIN_FROM_SCRATCH`, `HP`)
2. Log discovery (parses training logs for pre-run results)
3. Data loading (Art.6 filter, label assignment, train/val/test split)
4. TF-IDF baselines — SVM + LogReg (`svm_pipe`, `lr_pipe`)
5. Document coverage analysis (word-count vs BERT 512-token limit)
6. SVM at different token budgets (512 → full text)
7–9. LegalBERT 512-token and Chunked 4×510 fine-tuning (`ChunkedBERT`)
10. Full model comparison table
11. Bias analysis: SVM feature weights, per-country F1, country-token masking probe, **LIME for SVM and LegalBERT**
12. Summary

**`EDA.ipynb`** — Phase 1 deliverable. TF-IDF analysis, Fighting Words (shifterator), scattertext corpus comparison, concordance analysis. `EDA.pdf` is the exported output.

**`train_svm.ipynb`** — Earlier standalone TF-IDF+SVM notebook (superseded by `echr.ipynb`; kept for reference).

**`scripts/preprocess_data.py`** — Extracts *only* the FACTS section (strips LAW and Operative Provisions) to prevent outcome leakage. Labels: violation=1, non-violation=0.

**`scripts/analyze_bias.py`** — Standalone post-hoc bias script (respondent state, year, text length). Superseded by §11 of `echr.ipynb`.

**`scripts/svm_features.py`** — Extracts Linear SVM TF-IDF coefficients; standalone version of §11.1.

**`scripts/topic_analysis.py`** — LDA / BERTopic topic modeling for corpus interpretation.

**`src/train.py`** + **`src/dataset.py`** — Legacy Legal-BERT fine-tuning scripts (single-model, no ensemble). Superseded by `echr.ipynb`.

**`labs_train/`** — In-class lab tutorials (not project deliverables).

## Research Context

- **FACTS-only** — all models use only the FACTS section. Using Procedure or LAW sections inflates accuracy via spurious correlations (procedural boilerplate, outcome-revealing language).
- **Article 6 scope** — Phase 2 primary analysis is Article 6 only; all-articles (952 cases) is an addon comparison.
- **Article-6-specific label** — derived from `violation_articles`/`nonviolation_articles` columns, not the original `label` column (overall case outcome).
- **`has_representation` excluded as a feature** — EDA found it is not a strong correlate of outcome in FACTS-only text.
- **Bias axes:** respondent country, judgment year (temporal shift), text length.
- **Explainability:** LIME implemented in `echr.ipynb` §11.4–11.5 for both SVM and LegalBERT Chunked. Integrated Gradients (captum) still pending.

The FACTS section extraction is the central methodological safeguard — ECHR judgments are written after the decision, so using full text risks leakage from outcome-revealing language in the LAW section.

## Primary Analysis Notebook (`echr.ipynb`)

Single self-contained notebook covering all modeling and explainability. No external `.py` dependencies for the core analysis.

### Preprocessing (`legal_tokenizer`)
Ported from `EDA.ipynb` for consistency:
- `WordNetLemmatizer` with noun POS (`wnl.lemmatize(tok, pos="n")`)
- `MANUAL_LEMMA_MAP` for irregular legal plurals (*authorities→authority*, *applicants→applicant*, etc.)
- Stop words: NLTK English + `{court, case, mr, mrs, ms}`
- Passed as `tokenizer=` to `TfidfVectorizer` (`token_pattern=None`, `lowercase=False`)

**TF-IDF parameters:** `min_df=3`, `max_df=0.90`, `sublinear_tf=True`, `ngram_range=(1,2)`
- `min_df=3` (corpus is ~436 Art.6 cases — too small for `min_df=50`)

### Models
| Variable | Architecture | Notes |
|---|---|---|
| `svm_pipe` | TF-IDF + LinearSVC | `C=0.1`, `class_weight='balanced'` |
| `lr_pipe` | TF-IDF + LogisticRegression | `C=1.0`, `class_weight='balanced'` |
| `calib_svm_pipe` | TF-IDF + CalibratedClassifierCV(LinearSVC) | Platt scaling, 5-fold; used for LIME |
| LegalBERT-512 | `nlpaueb/legal-bert-base-uncased` | Head+tail 512-token truncation |
| LegalBERT-Chunked | `ChunkedBERT` (4×510 mean-pool CLS) | Effective 2040-token coverage |

### Key design choices
- `has_representation` excluded — not a strong correlate in FACTS-only text
- `class_weight='balanced'` addresses 73%/27% Art.6 class imbalance
- Focal loss + LLRD optimizer for LegalBERT fine-tuning
- 4-seed ensemble with threshold tuning on validation set

### LIME explainability (§11.4–11.5)
- **SVM**: Platt-calibrated SVM (`calib_svm_pipe`) feeds `predict_proba` to `LimeTextExplainer`. Run on 20 balanced test cases, 500 perturbations each. Results aggregated by word frequency (≥3 appearances).
- **LegalBERT**: `bert_predict_proba` wraps `ChunkedBERT.forward`; 200 perturbations per case (slower). Requires `TRAIN_FROM_SCRATCH=True` — model must be in memory (`lime_bert_model`, `lime_bert_tok`). Skipped gracefully when loading from cached `.npz`.
- SVM∩BERT overlap printed to separate genuine legal signal from model-specific artifacts.

### Key findings
- SVM @ 512 tok ≈ LegalBERT-512 (same input budget → same performance; BERT advantage comes from coverage, not pretraining alone)
- SVM macro-F1 scales with token budget (512→3800)
- LegalBERT Chunked (2040 tok) beats full-text SVM
- Per-country F1 tracks violation rate in some countries → base-rate exploitation
- Country/place/month token masking has limited impact → bias is distributionally encoded, not concentrated in named entities

## Project Status

- Phase 1 (Data + EDA): complete — `EDA.ipynb` / `EDA.pdf`
- Phase 2 (Modeling): complete — `echr.ipynb` (SVM, LegalBERT-512, LegalBERT-Chunked)
- Phase 3 (Bias/Explainability): SVM feature weights, per-country F1, token masking, and **LIME** done (`echr.ipynb` §11); Integrated Gradients (captum) pending
- Phase 4 (Deliverables): pending
