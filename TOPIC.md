# Topic Model Experiments — ECHR Art. 6, Random Split (75/25)

**Scripts:** `scripts/topic_hybrid.py`, `scripts/ensemble_topic.py`, `scripts/topic_analysis.py`
**Data:** Article 6 subset — 436 cases (327 train / 109 test), 73.4% violation rate
**Split:** Random stratified 75/25, `RANDOM_STATE=42`
**Classifiers:** LinearSVC + LogisticRegression, grid-searched over `C ∈ [0.01…10] × class_weight ∈ [None, balanced]`, 5-fold macro-F1

---

## Overall Best Results (across all runs)

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| TF-IDF only / LR *(baseline)* | 0.789 | 0.743 | 0.852 | 0.635 |
| Fine-tuned chunked LegalBERT end-to-end | 0.800 | **0.750** | 0.861 | 0.633 |
| **TF-IDF + BERTopic (MiniLM) / SVM** | **0.817** | **0.775** | **0.872** | **0.677** |
| TF-IDF + LDA (k=30) / SVM | 0.780 | 0.735 | 0.844 | 0.625 |
| TF-IDF + BERTopic-LegalBERT finetuned / LR | 0.789 | 0.733 | 0.855 | 0.610 |
| TF-IDF + BERT + BERTopic (ensemble) | 0.743 | 0.664 | 0.827 | 0.500 |

**BERTopic backbone:** `sentence-transformers/all-MiniLM-L6-v2` (contrastively trained for semantic similarity)

---

## Run A — LDA + BERTopic (no BERT encoding)
*`logs/topic_lda_bertopic.log`*

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| TF-IDF only / SVM | 0.761 | 0.734 | 0.819 | 0.649 |
| TF-IDF only / Logistic Reg | 0.789 | 0.743 | 0.852 | 0.635 |
| LDA only (k=10) / SVM | 0.615 | 0.589 | 0.691 | 0.488 |
| LDA only (k=20) / SVM | 0.716 | 0.685 | 0.783 | 0.587 |
| LDA only (k=30) / SVM | 0.670 | 0.641 | 0.743 | 0.538 |
| TF-IDF + LDA (k=10) / LR | 0.780 | 0.724 | 0.848 | 0.600 |
| TF-IDF + LDA (k=20) / SVM | 0.771 | 0.721 | 0.839 | 0.603 |
| TF-IDF + LDA (k=30) / SVM | 0.780 | 0.735 | 0.844 | 0.625 |
| BERTopic only (k=10–30) / SVM | 0.679 | 0.616 | 0.771 | 0.462 |
| **TF-IDF + BERTopic (k=10) / SVM** | **0.817** | **0.775** | **0.872** | **0.677** |
| TF-IDF + BERTopic (k=20,30) / SVM | 0.817 | 0.775 | 0.872 | 0.677 |

## Run B — LDA + BERTopic + frozen chunked LegalBERT
*`logs/topic_bert_hybrid.log`*

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| BERT only (frozen) / SVM | 0.688 | 0.601 | 0.787 | 0.414 |
| BERT only (frozen) / LR | 0.661 | 0.579 | 0.764 | 0.393 |
| TF-IDF + BERTopic (k=10) / SVM | 0.817 | 0.765 | 0.875 | 0.655 |
| TF-IDF + BERTopic (k=10) / LR | 0.807 | 0.770 | 0.863 | 0.677 |
| BERT + LDA (k=10–30) / SVM | 0.661 | 0.579 | 0.764 | 0.393 |
| BERT + BERTopic (k=10) / SVM | 0.688 | 0.617 | 0.782 | 0.452 |

## Run C — BERTopic with frozen LegalBERT as backbone
*`logs/topic_bert_backbone.log`*

BERTopic's internal MiniLM replaced with chunked LegalBERT (frozen) doc vectors fed directly into UMAP+HDBSCAN.

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| BERTopic-LegalBERT only (k=10) / SVM | 0.606 | 0.589 | 0.672 | 0.506 |
| TF-IDF + BERTopic-LegalBERT (k=10) / SVM | 0.780 | 0.712 | 0.852 | 0.571 |
| BERT + BERTopic-LegalBERT (k=10) / SVM | 0.679 | 0.602 | 0.777 | 0.426 |

**Finding:** Frozen LegalBERT is a worse BERTopic backbone than MiniLM (0.712 vs 0.775). LegalBERT's CLS token was not contrastively trained for sentence similarity, so UMAP clusters it poorly.

## Run D — BERTopic with fine-tuned LegalBERT as backbone
*`logs/86_topic_finetuned_backbone.log`*

Fine-tuned encoder (from `results/legalbert_chunked_encoder/hf_encoder`, macro-F1=0.750 end-to-end) used as BERTopic backbone.

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| BERT only (fine-tuned, frozen) / SVM | 0.633 | 0.585 | 0.726 | 0.444 |
| BERT only (fine-tuned, frozen) / LR | 0.706 | 0.639 | 0.795 | 0.484 |
| BERTopic-LegalBERT-ft only (k=10) / SVM | 0.505 | 0.489 | 0.578 | 0.400 |
| BERTopic-LegalBERT-ft only (k=10) / LR | 0.624 | 0.578 | 0.717 | 0.438 |
| TF-IDF + BERTopic-LegalBERT-ft (k=10) / SVM | 0.798 | 0.722 | 0.867 | 0.577 |
| **TF-IDF + BERTopic-LegalBERT-ft (k=30) / LR** | **0.789** | **0.733** | **0.855** | **0.610** |
| BERT + BERTopic-LegalBERT-ft (k=10) / SVM | 0.716 | 0.647 | 0.803 | 0.492 |
| TF-IDF + BERTopic (MiniLM) / SVM *(for reference)* | 0.817 | 0.765 | 0.875 | 0.655 |

**Finding:** Fine-tuning improves the LegalBERT backbone vs frozen (0.733 vs 0.712), but MiniLM still wins (0.775). Fine-tuning with cross-entropy optimises the classifier head, not semantic clustering — the resulting embeddings are task-adapted but not similarity-adapted.

## Run E — Stacking Ensemble: TF-IDF + fine-tuned BERT + BERTopic (MiniLM)
*`logs/87_ensemble_topic.log` — `scripts/ensemble_topic.py`*

Meta-learner: LogisticRegression stacking all three signals. BERTopic uses MiniLM backbone.

| Model | Accuracy | Macro-F1 | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| TF-IDF only | 0.789 | 0.743 | 0.852 | 0.635 |
| BERT only (fine-tuned, frozen) | 0.706 | 0.639 | 0.795 | 0.484 |
| BERTopic only (k=10) | 0.633 | 0.609 | 0.706 | 0.512 |
| TF-IDF + BERT | 0.706 | 0.639 | 0.795 | 0.484 |
| **TF-IDF + BERTopic (k=10)** | **0.807** | **0.770** | **0.863** | **0.677** |
| BERT + BERTopic (k=10) | 0.697 | 0.632 | 0.787 | 0.476 |
| TF-IDF + BERT + BERTopic (k=10) | 0.743 | 0.664 | 0.827 | 0.500 |

**Finding:** Full ensemble (0.664) is *worse* than TF-IDF alone (0.743). Frozen BERT embeddings introduce noise that degrades the meta-learner. The best pairing remains TF-IDF + BERTopic (0.770).

---

## Key Findings

### 1. Best result: TF-IDF + BERTopic (MiniLM) / SVM — Macro-F1 = 0.775
Beats TF-IDF baseline (+0.041). BERTopic's semantic topics are complementary to TF-IDF n-grams.

### 2. LDA adds almost nothing over TF-IDF
TF-IDF + LDA (k=30) = 0.735 ≈ TF-IDF alone (0.734). LDA is a lossy BoW compression of the same signal TF-IDF already has.

### 3. BERTopic > LDA as augmentation
MiniLM embeddings capture semantic similarity that BoW misses, making BERTopic topics genuinely complementary to TF-IDF. LDA topics are not.

### 4. Frozen BERT embeddings (even fine-tuned) hurt in ensembles
Fine-tuning optimises the end-to-end classifier, not the embedding space. Frozen fine-tuned BERT in a linear probe (0.639) is still worse than TF-IDF (0.743), and adding it to ensembles introduces noise.

### 5. MiniLM beats LegalBERT as a BERTopic backbone
MiniLM was trained contrastively for sentence similarity → better UMAP/HDBSCAN clusters. LegalBERT (even fine-tuned) was never trained for sentence similarity, so its embeddings cluster poorly.

### 6. BERTopic saturates at k=10 on this corpus
HDBSCAN discovers ~13 stable clusters regardless of `nr_topics` target (327 training docs is too small for finer granularity).

---

## Topic Analysis (EDA)
*`logs/85_topic_analysis.log` — `scripts/topic_analysis.py`*
*Full report: `results/topic_analysis/topic_report.md`*

Fitted LDA (k=15) and BERTopic (k=15, MiniLM) on full Art.6 corpus (436 cases).
BERTopic discovered 13 real topics with 34 outlier docs.

---

### LDA Topics — Violation vs Non-Violation signal (Δ = V − NV mean weight)

| Topic | Top words | Δ (V−NV) | Direction |
|---|---|---|---|
| 6 | proceeding, administrative, compensation, civil | +0.110 | → Violation |
| 0 | appeal, hearing, judgment, proceeding | +0.070 | → Violation |
| 4 | state, security, criminal, organisation | +0.042 | → Violation |
| 9 | criminal, proceeding, regional, hearing | +0.041 | → Violation |
| 1 | martial, officer, army, sentence | +0.041 | → Violation |
| 7 | evidence, trial, judge, police | −0.107 | → Non-Violation |
| 12 | child, authority, social, second | −0.055 | → Non-Violation |
| 2 | state, act, would, article | −0.055 | → Non-Violation |
| 14 | site, expert, report, would | −0.024 | → Non-Violation |
| 13 | police, offence, criminal, article | −0.027 | → Non-Violation |

**Interpretation:**
- Violation cases concentrate in procedural/administrative topics (6, 0, 9) — cases about unfair proceedings, delays, lack of access to court
- Military-court topics (1: martial, army) strongly predict violation — Turkey/Russia military tribunal cases
- Non-violation cases concentrate in evidential/trial topics (7: evidence, trial, judge) — cases where courts conducted proper proceedings
- Topic 12 (child, authority, social) being non-violation dominant is expected — UK child custody/care cases often find no Art.6 violation

### LDA Topics — by Respondent Country

| Topic | Top words | GBR | RUS | TUR |
|---|---|---|---|---|
| 7 | evidence, trial, judge | **0.310** | 0.031 | 0.013 |
| 2 | state, act, would | **0.235** | 0.019 | 0.051 |
| 12 | child, authority, social | **0.092** | 0.020 | 0.012 |
| 0 | appeal, hearing, judgment | 0.094 | **0.320** | 0.033 |
| 9 | criminal, proceeding, regional | 0.007 | **0.164** | 0.016 |
| 11 | cell, medical, detention | 0.011 | **0.133** | 0.017 |
| 6 | proceeding, administrative, compensation | 0.011 | 0.049 | **0.392** |
| 4 | state, security, criminal, organisation | 0.005 | 0.009 | **0.200** |
| 3 | police, prosecutor, officer | 0.017 | 0.024 | **0.120** |

**This is the spurious correlation signal:** topics cluster almost entirely by country. A classifier seeing topics 0/9/11 (Russia) vs 6/4/3 (Turkey) vs 7/2/12 (UK) is essentially learning country identity, not legal substance. Since each country has a different violation base rate, topic features can "predict" outcomes without understanding any law.

---

### BERTopic Topics

BERTopic top words are dominated by stopwords (the, of, and, to...) because BERTopic's c-TF-IDF runs on raw text, not through the legal tokenizer. The topics are semantically meaningful via embeddings but the word labels are not interpretable here. LDA (which uses the legal tokenizer via CountVectorizer) produces far more readable topic descriptions for this corpus.

Notable exception — topic 2 (`execution, karacabey, civil, bursa, land, compensation`) is clearly Turkey land/property expropriation cases, and topic 11 (`martial, army`) is military court cases. These match LDA topics 6 and 1 respectively.

---

## Stability Evaluation
*`logs/88_topic_stable_eval.log` — `scripts/topic_stable_eval.py`*
*10 data splits × 3 BERTopic seeds — `results/topic_stable_eval/`*

| Model | Macro-F1 | Accuracy | F1 (Violation) | F1 (No Violation) |
|---|---|---|---|---|
| TF-IDF + LDA (k=30) / LR | 0.711 ± 0.050 | 0.780 ± 0.046 | 0.851 ± 0.037 | 0.572 ± 0.074 |
| TF-IDF + BERTopic (k=10) / LR | 0.703 ± 0.042 | 0.781 ± 0.024 | 0.855 ± 0.015 | 0.551 ± 0.072 |
| **TF-IDF only / LR** | **0.700 ± 0.031** | 0.758 ± 0.033 | 0.830 ± 0.029 | 0.569 ± 0.045 |
| TF-IDF + BERTopic (k=10) / SVM | 0.699 ± 0.031 | 0.779 ± 0.021 | 0.853 ± 0.015 | 0.544 ± 0.054 |
| TF-IDF only / SVM | 0.697 ± 0.032 | 0.755 ± 0.033 | 0.828 ± 0.028 | 0.566 ± 0.046 |
| TF-IDF + LDA (k=30) / SVM | 0.679 ± 0.050 | 0.761 ± 0.052 | 0.837 ± 0.046 | 0.521 ± 0.081 |
| Dummy | 0.423 ± 0.000 | 0.734 ± 0.000 | 0.847 ± 0.000 | 0.000 ± 0.000 |

**Key finding:** All differences are within one standard deviation. The single-run best (TF-IDF + BERTopic / SVM = 0.775) was at the top of the natural variance range (mean 0.699, std 0.031). Topic features **do not reliably improve** over plain TF-IDF on this corpus — the dataset is too small (109 test cases per split) for the differences to be statistically meaningful.
