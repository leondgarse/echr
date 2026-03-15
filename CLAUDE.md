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

No `requirements.txt` exists. Key packages: `echr-extractor`, `torch`, `transformers`, `pandas`, `scikit-learn`, `scattertext`, `shifterator`, `nltk`.

## Architecture & Data Flow

```
HUDOC API → download_data.py → data/raw/{metadata.csv, full_text.json}
                                    ↓
                          preprocess_data.py → data/processed/processed.csv
                          (1,590 raw cases → ~952 valid cases)
                                    ↓
                    src/train.py + src/dataset.py → results/final_model/
                    (Legal-BERT: nlpaueb/legal-bert-base-uncased)
                                    ↓
                          analyze_bias.py + EDA.ipynb → insights
```

**`src/dataset.py`** — PyTorch Dataset wrapping processed.csv; handles tokenization and padding for Legal-BERT.

**`src/train.py`** — Loads processed.csv, splits into train/val/test, fine-tunes Legal-BERT, reports accuracy/precision/recall/F1.

**`scripts/preprocess_data.py`** — Critical design choice: extracts *only* the FACTS section (strips LAW and Operative Provisions) to prevent outcome leakage. Labels: violation=1, non-violation=0.

**`scripts/analyze_bias.py`** — Post-hoc analysis of predictions sliced by respondent state, judgment year, and text length.

**`EDA.ipynb`** — Phase 1 group project deliverable. TF-IDF analysis, Fighting Words (shifterator), scattertext corpus comparison, concordance analysis. `EDA.pdf` is the exported output of this notebook.

**`labs_train/`** — In-class lab tutorials for Phase 2 training (not project deliverables).

## Research Context

- **FACTS-only** — all models use only the FACTS section. This is deliberate: using Procedure or LAW sections inflates accuracy via spurious correlations (procedural boilerplate, outcome-revealing language).
- **Article 6 scope** — Phase 2 analysis is scoped to Article 6 cases only for a focused comparison.
- **Article-6-specific label** — for Article 6 cases, the label is derived from `violation_articles`/`nonviolation_articles` columns, not the original `label` column (which reflects the overall case outcome).
- **Bias axes under investigation:** respondent country, judgment year (temporal shift), text length, presence of "represented" keyword
- **Explainability methods planned:** LIME, Integrated Gradients

The FACTS section extraction is the central methodological safeguard — ECHR judgments are written after the decision, so using full text risks leakage from outcome-revealing language in the LAW section.

## Phase 2 Traditional ML Notebook (`train_svm.ipynb`)

Self-contained notebook (no external `.py` dependencies). Implements TF-IDF + Linear SVM.

**Preprocessing (`legal_tokenizer`)** — ported directly from `EDA.ipynb` for consistency:
- `WordNetLemmatizer` with noun POS (`wnl.lemmatize(tok, pos="n")`)
- `MANUAL_LEMMA_MAP` for irregular legal plurals (*authorities→authority*, *applicants→applicant*, etc.)
- `LEGAL_EXTRA_STOPWORDS`: NLTK English + `{court, case, mr, mrs, ms}`
- Passed as `tokenizer=` to `TfidfVectorizer` (`token_pattern=None`, `lowercase=False`)

**TF-IDF parameters:** `min_df=3`, `max_df=0.90`, `sublinear_tf=True`, `ngram_range=(1,2)`
- `min_df=3` (not 50 as in the reference repo — corpus is ~430 Art.6 cases, too small for `min_df=50`)

**Results:** ~74% random split, ~68% temporal split (Article 6, FACTS-only).
Consistent with Aletras et al. (2016) ~79% on FACTS-only. The ~6pp temporal drop is the key finding.

**Two evaluation settings:**
1. Random stratified 75/25 split — standard baseline
2. Temporal split — train on cases before the 75th-percentile year, test on newer cases

**Outputs:** classification report (Precision/Recall/F1), confusion matrix, per-country accuracy bar chart, top discriminating TF-IDF features.

## Project Status

- Phase 1 (Data + EDA): complete — `EDA.ipynb` / `EDA.pdf`
- Phase 2 (Modeling): `train_svm.ipynb` (TF-IDF+SVM) done; Legal-BERT (`src/train.py`) in progress
- Phase 3 (Bias/Explainability): basic metrics done; keyword bias tests and LIME/IG pending
- Phase 4 (Deliverables): pending
