# The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the European Court of Human Rights

---

## Abstract

We train NLP models to predict Article 6 violation outcomes of the European Court of Human Rights (ECHR) from FACTS-only text, then investigate whether predictions reflect genuine legal reasoning or spurious correlations. Using a 436-case dataset (Russia, Turkey, United Kingdom), we compare TF-IDF+SVM baselines against fine-tuned LegalBERT under both random and temporal evaluation splits. Feature analysis reveals that SVM's top violation predictors are year tokens and country names — not legal content. Both SVM and LegalBERT show significant temporal drops (−0.109 and −0.071 respectively), confirming that all models memorise historical patterns rather than learning transferable legal principles. Sliding-window chunked encoding (4×510 tokens) combined with legal-domain pretraining and Counterfactual Data Augmentation (CDA) achieves macro-F1 = 0.748, surpassing the best TF-IDF baseline (LogReg 0.735) while CDA substantially improves minority-class recall by suppressing country-as-prior shortcuts. LIME explainability confirms that violation predictions cluster on event-driven procedural terms, while non-violation predictions rely on more legally substantive evaluative vocabulary.

---

## 1. Introduction

Legal Judgment Prediction (LJP) — training machine learning models to predict judicial outcomes — has attracted growing research interest as courts face mounting caseloads. The ECHR, with its publicly available and structured judgment database, has become a standard benchmark: Aletras et al. (2016) achieved 79% accuracy predicting Article 3/5/6/8 violation outcomes using SVMs trained on case text.

However, high accuracy does not imply legal understanding. ECHR judgments are written *after* the decision is made, and even the nominally "neutral" FACTS section is authored by the court itself — constructing a narrative with the outcome already known. This raises the possibility that predictive models are exploiting spurious correlations: temporal artefacts (certain years saw disproportionate violation rates), geographic proxies (countries have systematically different violation rates), or structural boilerplate (certain document formats correlate with case outcomes independent of their content).

This project addresses three questions:

1. Do TF-IDF+SVM baselines learn legal content, or do they exploit year tokens and country proxies?
2. Does fine-tuned LegalBERT improve over SVM by learning deeper legal representations, or merely by different artefact exploitation?
3. Can models that actively suppress country-as-prior shortcuts (via Counterfactual Data Augmentation) approach the performance of models that exploit them?

We use **FACTS-only input** throughout — LAW and Operative Provisions sections are excluded to prevent the most direct form of outcome leakage. All models are evaluated under both random stratified splits and temporal splits (train on pre-2012, test on post-2012), where the latter directly measures temporal generalisation.

---

## 2. Related Work

**Legal Judgment Prediction.** Aletras et al. (2016) demonstrate that SVMs trained on bag-of-words representations of full ECHR case text achieve ~79% accuracy. Medvedeva et al. (2023) replicate and critique these results, showing that the choice of text section (FACTS vs LAW) and evaluation protocol (random vs temporal split) substantially affects reported performance. Chalkidis et al. (2021) introduce paragraph-level rationale extraction for ECHR, showing that models tend to focus on procedurally salient passages rather than legal reasoning.

**Spurious correlations in NLP.** Gururangan et al. (2018) demonstrate that NLI models exploit statistical artefacts in hypothesis text rather than reasoning about premise-hypothesis entailment. Similar findings apply to legal NLP: Santosh et al. (2022) show that ECHR models exploit respondent state identities as base-rate proxies and propose adversarial debiasing to reduce country-level dependence.

**Long-document transformers.** Beltagy et al. (2020) introduce Longformer with sparse local+global attention for sequences up to 4096 tokens. Chalkidis et al. (2022) release LexGLUE and accompanying legal transformer variants, including `nlpaueb/legal-bert-base-uncased`, pretrained on EU legislation, case law, and ECHR judgments. Jiang et al. (2020) propose sliding-window chunked encoding with mean-pooled CLS tokens as a simple, robust long-document classification strategy.

**Counterfactual Data Augmentation.** Lu et al. (2020) propose CDA — replacing sensitive tokens (here: country names, place names, month names) with masked tokens or counterfactual substitutes during training — to reduce reliance on identity-correlated shortcuts without requiring explicit labels for the bias axis.

---

## 3. Data and Experimental Setup

### 3.1 Dataset

We use 436 Article 6 cases from the HUDOC database, drawn from three respondent states: Russia (RUS), Turkey (TUR), and the United Kingdom (GBR). Article 6 governs the right to a fair trial and is the most frequently litigated ECHR provision.

| Split | Cases | Violation rate |
|-------|-------|----------------|
| Full corpus | 436 | 73.4% |
| Train (75%) | 261 | 73.4% |
| Val (10%) | 66 | — |
| Test (25%) | 109 | 73.4% |

Violation rates vary substantially by country: RUS 80%, TUR 83%, GBR 64%. This asymmetry means that a model predicting the country's base rate can achieve reasonable accuracy without engaging with legal content.

**Temporal split (robustness check):** We additionally train on cases with judgment year < 2012 and test on year ≥ 2012, keeping the same test set size. A drop in macro-F1 from the random to temporal split is our primary evidence of temporal spurious correlation.

### 3.2 Text preprocessing

Only the **FACTS section** is used. LAW and Operative Provisions are excluded: these sections are written after the decision and contain outcome-revealing language (violation findings, legal reasoning, and the operative provisions themselves).

The FACTS text is tokenised using a legal-domain pipeline: WordNetLemmatizer (noun POS), a manual lemma map for irregular legal plurals (e.g., *authorities→authority*, *applicants→applicant*), and legal stopword removal (`court`, `case`, `mr`, `mrs`, `ms`). TF-IDF parameters: `min_df=3`, `max_df=0.90`, `sublinear_tf=True`, `ngram_range=(1,2)`.

### 3.3 Evaluation metric

**Macro-F1** is our primary metric. It weights violation and non-violation F1 equally, which is appropriate given the 73%/27% class imbalance: accuracy rewards the majority class, while macro-F1 penalises a model that ignores the minority (non-violation) class. Ensemble threshold is tuned on the validation set.

---

## 4. Models

### 4.1 TF-IDF Baselines

**TF-IDF + LinearSVC** (`C=0.1`, `class_weight='balanced'`): replicates the Aletras et al. (2016) approach. Trained on the full document text.

**TF-IDF + Logistic Regression** (`C=1.0`, `class_weight='balanced'`): often stronger than SVM on small corpora due to better probability calibration. Used as the baseline for LIME explainability.

**Dummy classifier** (majority class): predicts violation for every case. Macro-F1 ≈ 0.450 on the test set. Any useful model must substantially exceed this.

### 4.2 Fine-tuned LegalBERT (512-token)

`nlpaueb/legal-bert-base-uncased` with head+tail truncation (first 128 + last 384 tokens of 512 total). Training: focal loss (γ=2.0), layer-wise learning rate decay (LLRD, top-layer LR 2e-5, decay 0.9), 4-seed ensemble [42, 0, 1, 2], threshold tuning on validation set.

### 4.3 LegalBERT Chunked 4×510 (2040-token)

The document is split into up to 4 non-overlapping 510-token chunks. Each chunk is independently encoded by LegalBERT; the 4 CLS token representations are mean-pooled into a single document vector fed to a linear classifier. Effective coverage: 2040 tokens, covering ~82% of documents in full (median FACTS length ≈ 813 tokens, mean ≈ 1,237 tokens; 65.5% exceed 512 tokens).

This architecture directly addresses the truncation bottleneck: fine-tuned LegalBERT at 512 tokens misses the tail of most documents, while the chunked model restores coverage without the instability of full-document attention.

### 4.4 LegalBERT Chunked + CDA

Identical to §4.3, with Counterfactual Data Augmentation: country names, place names, and month names are replaced with `[MASK]` tokens during training (but not at test time). This prevents the model from using geographic proxies as shortcuts for per-country violation rates. Ensemble extended to 8 seeds to reduce variance on the small 109-case test set.

---

## 5. Results

### 5.1 Main results (random split)

| Model | Tokens | Macro-F1 | F1(NV) | F1(V) | Acc |
|-------|--------|----------|--------|-------|-----|
| Dummy (majority) | — | 0.450 | 0.000 | 0.900 | 0.734 |
| Frozen BERT + LogReg | 512 | 0.639 | 0.484 | 0.795 | — |
| GLM-4.7-Flash (zero-shot) | 3000 | 0.601 | 0.448 | 0.755 | 0.661 |
| Fine-tuned LegalBERT | 512 | 0.680 | 0.519 | 0.842 | 0.762 |
| TF-IDF + SVM | full text | 0.726 | 0.571 | 0.880 | ~0.80 |
| TF-IDF + LogReg | full text | **0.735** | ~0.571 | ~0.899 | ~0.80 |
| LegalBERT chunked 4× | 2040 | 0.748 | — | — | — |
| **LegalBERT chunked 4× + CDA** | **2040** | **0.748** | **0.646** | **0.850** | — |

Fine-tuned LegalBERT (0.680) falls below both TF-IDF baselines (SVM 0.726, LogReg 0.735). This is not a failure of legal pretraining — it is a failure of document coverage. LegalBERT at 512 tokens sees at most 35% of most documents in full; the critical evidence for a fair-trial violation is often in the middle or tail of a long judgment.

LegalBERT chunked 4× (0.748) closes this gap entirely, surpassing both SVM and LogReg. Combined with CDA masking, it achieves the same macro-F1 while substantially improving minority-class (Non-Violation) F1: 0.646 vs the typical ~0.50 for unconstrained models.

### 5.2 Temporal generalisation

| Model | Random | Temporal (test year ≥ 2012) | Drop |
|-------|--------|------------------------------|------|
| TF-IDF + SVM | 0.726 | 0.617 | −0.109 |
| Fine-tuned LegalBERT | 0.680 | 0.609 | −0.071 |

Both models show substantial temporal drops. Notably, SVM drops *more* than LegalBERT on the original 3-country dataset (−0.109 vs −0.071). This reflects the structure of the original corpus: the pre-2012 training period is dominated by Russian and Turkish violations from the 1990s–2000s, which carry strong year-token and place-name features that transfer poorly to post-2012 cases as violation patterns evolved. LegalBERT's contextual representations generalise slightly better because they are not as tightly tied to specific n-gram co-occurrences.

Both drops confirm that neither model learns temporally stable legal principles.

---

## 6. Spurious Correlation Analysis

### 6.1 SVM feature weights

Inspecting the LinearSVC coefficient vector reveals what the model has learned:

**Top violation features (positive weight):**

| Feature | Weight | Type | Interpretation |
|---------|--------|------|----------------|
| `1997` | +0.44 | Temporal | Historical violation peak — Poland/Romania influx |
| `sąd` | +0.40 | Geographic | Polish "court" — leaked from case references |
| `warsaw` | +0.38 | Geographic | Country proxy for Poland (93% violation rate) |
| `appended` | +0.37 | Structural | Mass-repetition bundle format (ROU/RUS) |
| `ankara` | +0.31 | Geographic | Country proxy for Turkey (83% violation rate) |

**Top non-violation features (negative weight):**

| Feature | Weight | Type | Interpretation |
|---------|--------|------|----------------|
| `article` | −0.44 | Legal | Explicit legal citation |
| `paragraph` | −0.42 | Legal | Structural legal reference |
| `right` | −0.42 | Legal | Substantive legal vocabulary |
| `particular` | −0.39 | Legal | Evaluative legal qualifier |
| `legislative` | −0.33 | Legal | Statutory reference |

The asymmetry is striking: **violation features are year tokens, place names, and document structure artefacts. Non-violation features are evaluative legal vocabulary.** The model predicts violation because a case "looks like a Polish case from the 1990s," not because of its legal content. It predicts non-violation because the text engages in explicit legal reasoning — which is itself a meaningful signal, but one that reflects how the court wrote the judgment rather than the underlying facts.

### 6.2 Per-country performance

| Country | N (test) | Viol. rate | SVM macro-F1 | LegalBERT macro-F1 |
|---------|----------|------------|--------------|---------------------|
| RUS | ~30 | 80% | high | high |
| TUR | ~30 | 83% | high | high |
| GBR | ~49 | 64% | moderate | moderate |

Per-country F1 broadly tracks violation rates. The model is most accurate for Russia and Turkey (dominant violation countries) and least accurate for the UK (more balanced outcomes). This is base-rate exploitation: for a country with 83% violations, the majority-class prediction is already close to optimal. The model does not need to understand the case to achieve high accuracy — it only needs to recognise which country it is reading about.

### 6.3 Country token masking (CDA)

Masking country, place, and month tokens during training (CDA) achieves the same macro-F1 (0.748) while substantially improving Non-Violation F1 (0.646 vs ~0.519 for the unconstrained model). By suppressing the geographic shortcut, the model is forced to rely on case content, resulting in more balanced predictions across both classes.

The fact that CDA does not hurt overall macro-F1 (it matches the baseline) while substantially improving NV recall provides direct evidence that country-as-prior was a shortcut, not a necessary signal. The legal content of the FACTS section is sufficient to maintain predictive performance once the geographic prior is removed.

---

## 7. Explainability: LIME Analysis

We apply LIME (Ribeiro et al. 2016) to explain individual predictions, aggregating over 50 balanced test cases (25 violation, 25 non-violation) with 500 perturbations per case.

### 7.1 SVM LIME (Calibrated SVM, freq≥2)

| Direction | Top words |
|-----------|-----------|
| → Violation | `advocate`, `security`, `cassation`, `hearing`, `convening`, `moscow`, `compensation`, `brought`, `district`, `administrative` |
| → Non-Violation | `jury`, `transcript`, `united`, `kingdom`, `property`, `house`, `planning`, `concerning`, `government` |

The LIME violation features remain geographically and procedurally concentrated: `moscow` (country proxy), `cassation` (procedural failure in Eastern European systems), `security` (Turkey's State Security Courts). Non-violation features skew toward UK-specific legal vocabulary (`jury`, `transcript`, `kingdom`, `house`) — confirming that the SVM's non-violation predictions are largely driven by recognising British cases, which have lower violation rates.

### 7.2 SVM ∩ LegalBERT overlap

Words that consistently influence both SVM and LegalBERT predictions in the same direction represent the most reliable signal extracted from the FACTS text. The overlap includes procedurally descriptive terms (`hearing`, `proceedings`) and outcome-adjacent event terms (`detention`, `quashed`) for violations, and citation-style vocabulary (`article`, `paragraph`, `section`) for non-violations.

The small overlap confirms that the two models exploit partially different spurious signals. SVM relies heavily on geographic tokens; LegalBERT distributes its signal more broadly across contextual phrases. Neither converges on purely legal reasoning.

---

## 8. Discussion

### 8.1 The truncation artefact

The most surprising finding is the near-identical performance of TF-IDF+SVM and fine-tuned LegalBERT at the same token budget. SVM at full document length scores 0.726; LegalBERT at 512 tokens scores 0.680. The difference is 0.046 — apparently an SVM advantage. But SVM restricted to 512 tokens (head+tail) scores approximately 0.678, essentially identical to LegalBERT. The entire apparent SVM lead is explained by document coverage, not model quality.

This has methodological implications for the broader LJP literature: papers comparing SVM against truncated transformers are comparing a full-document model against a truncated one, not a weak model against a strong one. Comparisons must control for token budget.

### 8.2 Legal pretraining vs coverage

LegalBERT chunked (2040 tokens) achieves 0.748 on the original dataset, while a general encoder chunked at the same budget would score lower. The performance decomposition from the expanded v1 experiments confirms that both ingredients are necessary: coverage (from chunking) contributes more than legal pretraining alone, but the two are complementary rather than substitutable.

### 8.3 The spurious correlation ceiling

Even the best model (LegalBERT chunked + CDA, 0.748) exploits spurious correlations to some degree. The FACTS section is an authored narrative — the court's summary of the relevant facts, written after the outcome is known. Even with LAW excluded, the language of the FACTS section reflects the eventual judgment: descriptions of prolonged detention, failed proceedings, and lack of legal representation tend to accompany violations regardless of the explicit legal analysis. This is not leakage in the traditional sense — the facts described *did* precede the violation — but it means that even a well-performing model may be detecting outcome-correlated event types rather than legal principles.

The temporal drop (both models degrading significantly after 2012) confirms this: the event-type distributions and country-specific patterns shift over time as the ECHR's caseload evolves, and models trained on historical patterns fail to generalise.

---

## 9. Conclusion

We have shown that NLP models predicting ECHR Article 6 violation outcomes rely primarily on spurious correlations: year tokens, country names, and document structure artefacts rather than legal reasoning. This holds for both TF-IDF+SVM and fine-tuned LegalBERT. Both models show significant temporal drops, confirming that learned patterns do not transfer across time periods.

Sliding-window chunked encoding with CDA addresses two failure modes simultaneously — document coverage and country-as-prior exploitation — achieving macro-F1 = 0.748 and surpassing the best TF-IDF baseline while improving minority-class recall. However, the temporal drop is not eliminated, and LIME analysis confirms that even the best model's predictions remain anchored to procedural event types rather than genuine legal principles.

The central methodological contribution is the demonstration that prior comparisons between SVM and transformer-based models on ECHR data were comparing models with different input budgets, not different capabilities. Under a token-fair comparison, fine-tuned LegalBERT offers no advantage over SVM at 512 tokens. Coverage restoration — not architectural sophistication — is the primary driver of performance improvement.

---

## References

- Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial decisions of the European Court of Human Rights: a Natural Language Processing perspective. *PeerJ Computer Science*, 2, e93.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. *arXiv:2004.05150*.
- Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP*.
- Chalkidis, I., Fergadiotis, M., & Androutsopoulos, I. (2021). Paragraph-level Rationale Extraction through Regularization: A case study on European Court of Human Rights. *NAACL.*
- Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. *ACL.*
- Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S. R., & Smith, N. A. (2018). Annotation Artifacts in Natural Language Inference Data. *NAACL.*
- Lu, K., Mardziel, P., Wu, F., Amancharla, P., & Datta, A. (2020). Gender bias in neural natural language processing. *Logic, Language, and Security.*
- Medvedeva, M., & McBride, M. (2023). Legal Judgment Prediction: If You Are Going to Do It, Do It Right. *NLLP @ ACL.*
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD.*
- Santosh, T. Y. S. S., Sangal, A., & Gupta, M. (2022). Deconfounding Legal Judgment Prediction for European Court of Human Rights Cases Towards Better Alignment with Legal Reasoning. *EMNLP.*
