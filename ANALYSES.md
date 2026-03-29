# Analyses

*"The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the ECHR"*

This document synthesises the analytical findings across all experiments. Raw numbers are in `RESULTS.md`; this file focuses on interpretation and research implications.

---

## 1. The Central Question: Do Models Learn Law or Shortcuts?

The short answer: **primarily shortcuts**, across all model families.

The FACTS-only setup was designed to prevent the most obvious leakage (LAW and Operative Provisions sections are written *after* the decision and directly reveal outcomes). Even so, every model examined relies heavily on spurious correlations rather than genuine legal reasoning. The three dominant shortcuts are:

### 1.1 Temporal Shortcuts (Year Tokens)

The top TF-IDF violation features are bare year tokens: `1997` (weight 0.44), `1998` (0.34), `1994` (0.31), `1995`, `1996`, `2000`, `2001`. These reflect historical accident: Poland and Romania joined the ECHR system in the mid-1990s and immediately accumulated large violation backlogs. Cases from those years are violations not because of what happened legally, but because of *when* they were judged and *which country* was respondent.

This is confirmed by the temporal split experiment: all neural models drop in macro-F1 when trained on pre-2014 cases and tested on post-2014 cases (see §2). The year tokens that worked as proxies in training do not transfer.

### 1.2 Geographic Shortcuts (Country-as-Prior)

Country names and place-specific vocabulary function as strong base-rate proxies:

| Country | Violation rate | Key violation features |
|---------|---------------|------------------------|
| POL | 93% | `warsaw`, `sąd` (Polish "court"), `district`, `1994/1995` |
| ROU | 89% | `bucharest`, `appended`, `set appended`, `final decision` |
| RUS | 80% | `appended`, `applicant complained`, `application set` |
| TUR | 83% | `ankara`, `state security`, `security` |
| AUT | 51% | `tax`, year tokens 2000–2006 |
| GBR | 64% | `hearing`, `sentence`, year tokens 1997–1998 |

The model essentially learns: "if this reads like a Polish case from the 1990s, predict violation." This is accurate in the training set but not a legal argument. `sąd` — a Polish-language artifact leaking into English judgments via case references — is the second-strongest global feature (weight 0.40).

### 1.3 Structural/Boilerplate Shortcuts

`appended` and `set appended` (weight 0.37) refer to tables of applicant details appended to mass-repetition judgments, common in ROU and RUS bundles. The model predicts violation from document structure (a bundle format) rather than legal content.

---

## 1a. SVM Feature Analysis — Full Results

Script: `scripts/svm_features.py` | Log: `40_svm_features.log` | Dataset: v1 (1,205 Art.6 cases)

### Important: two tokenisers — two different pictures

The `svm_features.py` script used a lenient tokeniser (strips punctuation, no `isalpha()` check), so year tokens like `1997` pass through and dominate the weights. The benchmark SVM (`quick_svm.py`) uses `t.isalpha()`, which already strips years before fitting. **The macro-F1 scores reported throughout were computed without year tokens.**

Running feature analysis with the correct benchmark tokeniser reveals the actual predictive features:

### Full-corpus top-30 features (benchmark tokeniser — no years, isalpha)

**VIOLATION (positive weight):**

| Feature | Weight | Type | Interpretation |
|---------|--------|------|----------------|
| `upheld judgment` | +0.222 | Legal-procedural | Domestic court upheld a flawed decision; applicant still brought to ECtHR |
| `state security` | +0.189 | Institution | Turkish State Security Courts — institutionally unfair by design (abolished 2004) |
| `cassation` | +0.186 | Procedural | Cassation proceedings improperly conducted or resulting in cyclical quashing |
| `applicant brought` | +0.187 | Procedural | Common phrasing when applicant initiates ECtHR proceedings after domestic failure |
| `hearing` | +0.187 | Legal | Access to / fairness of hearing — core Art.6 issue |
| `quashed judgment` | +0.175 | Procedural | Domestic judgment quashed and re-adjudicated in a loop → procedural unfairness |
| `quashed` | +0.156 | Procedural | Same; standalone token |
| `february` | +0.154 | Month | Temporal proxy; months correlated with certain filing patterns (residual shortcut) |
| `detention` | +0.154 | Legal | Pre-trial detention length — Art.6 violation type |
| `criminal proceeding` | +0.148 | Legal | Criminal trials attract heightened Art.6 scrutiny |
| `compensation` | +0.147 | Legal | Applicant seeking compensation — typical violation case posture |
| `civil` | +0.143 | Legal | Civil proceedings as Art.6 subject matter |
| `second applicant` | +0.133 | Boilerplate | Multi-applicant mass-case template language |
| `delay` | +0.132 | Legal | Length of proceedings — most common Art.6 violation type |
| `written` | +0.130 | Procedural | Written procedure without oral hearing — Art.6 issue |
| `judgment` | +0.128 | Generic | High frequency in violation fact patterns |
| `army` | +0.126 | Institution | Turkish military courts / UK army discipline cases |
| `treasury` | +0.126 | Institution | State Treasury (RUS enforcement cases) / Treasury Solicitor (UK) |
| `born life` | +0.123 | Boilerplate | "born in [year], currently serving life sentence" — template phrase in criminal cases |
| `pecuniary` | +0.123 | Legal | "pecuniary damage" — Art.41 just satisfaction claim; signals violation posture |
| `security` | +0.123 | Institution | Security court / state security apparatus |
| `applicant requested` | +0.123 | Procedural | Applicant's request denied or unaddressed — violation context |
| `representative` | +0.117 | Procedural | Legal representative; procedurally significant in certain violation patterns |
| `brought` | +0.118 | Procedural | "brought proceedings" — applicant-initiated |
| `charge` | +0.116 | Legal | Criminal charge — Art.6§2 presumption of innocence cases |
| `appeal` | +0.116 | Procedural | Appeal proceedings; fairness of appellate review |
| `illegal` | +0.116 | Legal | Illegal detention / illegal proceedings |
| `partially` | +0.115 | Outcome | Partial domestic success → still brought to ECtHR |
| `administrative dismissed` | +0.117 | Procedural | Administrative proceedings dismissed — access to court issue |
| `initiated` | +0.116 | Procedural | Initiated proceedings |

**NON-VIOLATION (negative weight):**

| Feature | Weight | Type | Interpretation |
|---------|--------|------|----------------|
| `applicant together` | −0.293 | Narrative | UK-style phrasing; multi-party narrative cases tend non-violation |
| `legislative` | −0.252 | Legal | Legislative framework cited — indicates structured legal reasoning |
| `say` | −0.229 | Narrative | Narrative past-tense style; more common in detailed GBR acquittal reasoning |
| `government` | −0.220 | Procedural | Government submissions; more prominent in reasoned non-violation |
| `said` | −0.211 | Narrative | Same narrative style |
| `official` | −0.197 | Legal | Official body reference; Western EU procedural language |
| `carried` | −0.197 | Procedural | "carried out" investigations / procedures — due process observed |
| `concerning` | −0.195 | Reasoning | Evaluative framing: "concerning the applicant's situation" |
| `might` | −0.191 | Reasoning | Conditional / evaluative language in legal analysis |
| `whether` | −0.182 | Reasoning | "whether the applicant received…" — evaluative framing, mark of legal scrutiny |
| `particular` | −0.181 | Reasoning | "in the particular circumstances" — individualised evaluation |
| `flat` | −0.178 | Factual | "flat" (apartment); UK/RUS housing cases — systematically non-violation |
| `individual` | −0.177 | Reasoning | Individualised assessment; UK legal style |
| `entered` | −0.175 | Procedural | "entered into force" / "entered a plea" — process-respecting language |
| `came` | −0.173 | Narrative | Narrative past tense; detailed case description style |
| `act` | −0.171 | Legal | Statutory act cited — legislative grounding of non-violation |
| `jury` | −0.170 | Legal | UK jury trial cases — GBR-specific non-violation pattern under Art.6 |
| `right` | −0.164 | Legal | Explicit right invoked — present more in reasoned acquittals |
| `would` | −0.161 | Reasoning | Conditional reasoning; evaluative analysis of what would constitute a violation |
| `provision` | −0.161 | Legal | Legal provision cited — structured non-violation reasoning |
| `conversation` | −0.160 | Factual | Surveillance/interception of conversations — UK non-violation pattern |
| `local` | −0.160 | Institutional | Local authority / local court — non-violation context |
| `discovered` | −0.158 | Factual | Factual detail; narrative style |
| `provide` | −0.155 | Legal | "provide reasons" / "provide access" — due process observed |
| `terrorist` | −0.154 | Factual | UK terrorism suspects — GBR fair-trial non-violation pattern |
| `surveillance` | −0.153 | Legal | Surveillance cases — UK Art.6 non-violation pattern |
| `meaning` | −0.153 | Reasoning | "within the meaning of Article 6" — definitional reasoning |
| `one` | −0.152 | Generic | High frequency in GBR/AUT narrative style |
| `assessment` | −0.151 | Reasoning | "assessment of evidence" — evaluative language |
| `part` | −0.150 | Procedural | "on the part of" — procedural language |

### Per-country top-5 features (benchmark tokeniser)

| Country | Cases | Viol% | Top violation features | Top non-violation features |
|---------|-------|-------|------------------------|---------------------------|
| GBR | 148 | 65% | `hearing`, `sentence`, `army`, `delay`, `legal` | `legislative`, `jury`, `individual`, `official`, `say` |
| RUS | 167 | 77% | `authority`, `republic`, `pension`, `favour`, `town` | `together`, `life applicant`, `flat`, `property`, `state property` |
| TUR | 121 | 79% | `state security`, `civil`, `judgment`, `upheld judgment`, `circumstance applicant` | `fact`, `act`, `carried`, `defendant`, `lawyer` |

**Country-level observations:**

- **GBR violations**: `hearing`, `delay`, `sentence`, `legal` — genuine Art.6 content. UK violations predominantly concern length of proceedings, fair hearing, and army discipline cases. The model is partially learning real legal patterns here.
- **GBR non-violations**: `legislative`, `jury`, `individual`, `surveillance`, `terrorist` — UK cases involving legislative frameworks, jury trials, and terrorism suspects are systematically non-violation. The model learns the UK legal landscape, not universal Art.6 principles.
- **RUS violations**: `authority`, `pension`, `favour`, `quashed` — enforcement failures (pension non-payment, authorities ignoring judgments) and quashed-judgment cycles. Still partly topical (RUS-specific violation types) rather than universal.
- **RUS non-violations**: `flat`, `property`, `state property` — housing and property cases in Russia's favour. Country-specific topic.
- **TUR violations**: `state security`, `army`, `civil` — Turkish State Security Courts and military tribunals are institutionally unfair under Art.6. The model correctly identifies this, but it is learning an institution that no longer exists (abolished 2004) rather than a transferable legal principle.
- **TUR non-violations**: `fact`, `act`, `carried`, `defendant`, `lawyer` — more substantive legal language indicating reasoned acquittals.

### Tokeniser discrepancy note

The `svm_features.py` tokeniser strips punctuation but does **not** require `t.isalpha()`, so year tokens (`1997`, `1998`, …) pass through and appear at the top of the weight list. The benchmark SVM (`quick_svm.py`) uses `t.isalpha()` and already excludes years before fitting. The macro-F1 scores reported throughout (SVM 0.725–0.726, LogReg 0.735, etc.) were therefore computed **without year tokens**. Note: on the original 436-case dataset, LogReg (0.735) slightly outperforms LinearSVC (0.726); the previously cited "SVM 0.735 on original" was the LogReg result. Years are available shortcuts that the SVM exploits heavily *when permitted*, but they are not strictly needed.

### Debiasing probe: removing place names and years

Script: `scripts/svm_debiased.py` | Dataset: original (436 Art.6 cases, RUS/TUR/GBR)

Three conditions tested on identical splits:

| Condition | Random macro-F1 | Temporal macro-F1 |
|-----------|-----------------|-------------------|
| Baseline (years already excl. via `isalpha`) | 0.726 | 0.617 |
| + remove place names (country/city stopwords) | **0.739** | **0.622** |
| + remove places + explicitly strip year tokens | 0.739 | 0.622 |

**Removing place names marginally *improves* performance (+0.013 random, +0.005 temporal).** Explicitly stripping year tokens on top adds nothing further. Two conclusions:

1. **Place names are redundant shortcuts.** The same signal is encoded elsewhere in the text — legal domain vocabulary, procedural language patterns specific to each country's violation type. The model adapts to alternative features when place names are removed.
2. **Removing spurious features does not close the temporal gap.** The random→temporal drop persists (0.739→0.622 = −0.117) with or without place names and years. The temporal gap is not solely caused by geographic/year shortcuts; it reflects deeper distributional shifts in the case mix across time periods.

The practical implication: place-name and year masking alone is not a sufficient debiasing strategy for either SVM or (by extension) neural models.

### Temporal stability

**Zero features** are stable in the top-20 across all five year quintiles. The SVM uses completely different predictive vocabulary in each era:
- Early era (pre-1997): older ECHR member states, different vocabulary
- Accession era (1997–2004): Polish/Romanian year tokens, `warsaw`, `bucharest` dominate
- Post-accession (2004–2010): `2005`, `2004`, different country mix
- Modern era (2010–2016): newer patterns emerge
- Recent (2016+): `2014` shifts to non-violation marker (temporal reversal)

The implication: **the SVM has no stable legal knowledge**. Its temporal robustness comes entirely from the fact that sparse TF-IDF does not compound these shifting vocabularies into dense contextual representations that would overfit the training distribution.

---

## 2. Temporal Generalisation: SVM vs Neural Models

The temporal split (train on pre-2014, test on post-2014) is the primary diagnostic for spurious correlation. A model exploiting temporal shortcuts should drop; a model using stable legal features should not.

### Temporal drop by model (Enlarged v1, 1,205 cases)

| Model | Random | Temporal | Drop |
|-------|--------|----------|------|
| TF-IDF + SVM | 0.725 | **+0.746** | **+0.021** |
| DeBERTa-v3 | 0.719 | 0.676 | −0.043 |
| Adv LegalBERT (λ=0.5) | 0.712 | 0.675 | −0.037 |
| Adv DeBERTa (λ=0.2) | 0.676 | 0.679 | −0.003 |
| LegalBERT | 0.678 | 0.646 | −0.032 |
| Frozen BERT+LR | 0.639 | 0.560 | −0.079 |
| Frozen BERT+SVM (CLS) | 0.662 | 0.502 | −0.160 |

**SVM not only avoids temporal drop — it improves (+0.021).** All neural models degrade. Frozen BERT representations suffer the worst (−0.160), suggesting that contextual embeddings encode temporal information more densely than sparse bag-of-words.

### The paradox: SVM uses spurious features but generalises

Feature analysis (§1.1) shows the SVM's top features are year tokens and country names — themselves spurious. Yet it generalises temporally. The explanation: sparse TF-IDF representations do not *overfit* to year-specific vocabulary clusters the way dense contextual representations do. A neural model learns that the combination of contextual patterns characteristic of the 1990s–2000s Polish legal discourse predicts violation; when the distribution shifts post-2014, this dense representation fails. TF-IDF just counts tokens, so a year token `1997` contributes a fixed additive signal that does not interact with surrounding context — the representation is less sensitive to distributional shift.

In other words: **SVM generalises despite using spurious features; neural models fail because of how they encode those features.**

Zero features are stable across all five year quintiles in the temporal stability analysis, yet SVM still generalises. This rules out the hypothesis that SVM succeeds by learning stable legal vocabulary — it succeeds by not overfitting unstable vocabulary.


### Temporal robustness of long-context models (runs 65, 68)

The results are now in — and the pessimistic hypothesis was confirmed.

| Model | Random | Temporal | Drop |
|-------|--------|----------|------|
| TF-IDF + SVM | 0.725 | **0.746** | **+0.021** |
| Legal-Longformer (4096 tok) | 0.689 | 0.678 | −0.011 |
| LegalBERT chunked 4× + CDA | 0.756 | 0.673 | −0.083 |
| LegalBERT chunked 4× | **0.760** | 0.682 | **−0.078** |
| DeBERTa chunked 4× | 0.726 | 0.691 | −0.035 |

**LegalBERT chunked + CDA suffers the largest temporal drop** (−0.083) — worse than without CDA (−0.078). CDA removes country tokens that help the model orient toward relevant time periods, making it more temporally fragile rather than less. DeBERTa chunked shows a more moderate drop (−0.035), and SVM continues to improve temporally (+0.021). Legal-Longformer has the smallest drop (−0.011) but is also the weakest random-split model (0.689) — its stability comes from mediocrity, not robustness.

**Why does LegalBERT chunked degrade more?**
1. *More context → more temporal proxies.* By reading 2040 tokens of each case, LegalBERT encounters year tokens, decade-specific boilerplate, and period-specific institutional names (e.g. `state security court`, abolished 2004) distributed throughout the full FACTS section. A 512-token model misses much of this; paradoxically, reading less exposes it to fewer temporal shortcuts.
2. *Legal pretraining amplifies temporal representations.* LegalBERT was pre-trained on EUR-Lex (up to a certain year) and contains implicit temporal biases in its embedding space. Fine-tuning on ECHR data reinforces these biases, and they transfer poorly to post-2014 test cases.
3. *DeBERTa's disentangled attention partially separates temporal from content signal*, offering modest protection that LegalBERT's coupled attention does not.

**The core finding for the paper:** The highest-performing model on random split (LegalBERT chunked, 0.760) shows the greatest temporal vulnerability (−0.078). This is the clearest possible demonstration that random-split performance overstates generalisability. Models with better test-set performance are *more* reliant on spurious temporal correlations, not less. SVM — which appears to use spurious features (year tokens) — actually generalises best across time because sparse features do not interact contextually and do not overfit to distributional shift.

**Note on DeBERTa chunked temporal stability:** Per-seed variance for run 68 is very high (seeds 0, 3: 0.644/0.689 vs seeds 1, 2: 0.399/0.435). Two seeds essentially collapsed. The ensemble (0.691) is inflated by the two good seeds masking the two failed ones. This variance is a robustness concern: chunked DeBERTa under temporal distribution shift is sensitive to initialisation.

---

## 3. Model Hierarchy and What It Reveals

### On random split (in-distribution)

```
LegalBERT chunked 4× (0.760) > LegalBERT chunked+CDA (0.756) > DeBERTa chunked 4× (0.726) > SVM (0.725)
≈ DeBERTa (0.719) > Adv LegalBERT (0.712) > Legal-Longformer (0.689) > Adv DeBERTa λ=0.05 (0.693)
> LegalBERT (0.678) > Frozen BERT+LR (0.639) > GLM zero-shot (0.601)
```

LegalBERT chunked sliding window is now the clear leader on random split, having overcome SVM by resolving the truncation bottleneck. Neural models with full document coverage (2040 tokens) outperform both SVM and truncated neural models.

### On temporal split (out-of-distribution)

```
SVM (0.746) > DeBERTa chunked 4× (0.691) > Adv DeBERTa λ=0.2 (0.679)
≈ Adv LegalBERT (0.675) ≈ DeBERTa (0.676) > LegalBERT chunked 4× (0.682) > Legal-Longformer (0.678)
> LegalBERT chunked+CDA (0.673) > LegalBERT (0.646) >> Frozen BERT (0.502–0.560)
```

The ordering inverts relative to random split. SVM dominates by a large margin. LegalBERT chunked (0.682) drops below SVM-level models despite having the best random-split F1. Coverage and temporal robustness are **orthogonal** — the coverage fix that beats SVM on random splits creates no temporal benefit. DeBERTa chunked (0.691) is slightly better temporally than LegalBERT chunked (0.682), and adversarial training provides marginal recovery for non-chunked models.

### LLM zero-shot (GLM-4.7-Flash): 0.601

The zero-shot LLM underperforms all trained models, including the frozen BERT baseline (0.639). Few-shot examples (3/class) make it slightly worse (0.583). This suggests that the task requires familiarity with the specific ECHR article 6 violation criteria and case distribution, which few-shot examples cannot convey to a general-purpose reasoning model.

---

## 4. Adversarial Year Debiasing

### What it does

A gradient-reversal layer is added to the shared encoder with a year-regression head. The encoder is trained to predict violations while being *prevented* from encoding year information. This is a direct structural intervention against temporal spurious correlations.

### Architecture-dependent effects

| Model | Baseline (random) | + Adversarial (random) | Effect |
|-------|-------------------|------------------------|--------|
| LegalBERT | 0.678 | 0.712 (λ=0.5) | **+0.034** |
| DeBERTa-v3 | 0.719 | 0.676 (λ=0.2) | −0.043 |
| DeBERTa-v3 | 0.719 | 0.693 (λ=0.05) | −0.026 |

Adversarial debiasing **helps LegalBERT but hurts DeBERTa**. The divergence is interpretable:

- LegalBERT has weaker inductive biases and relies more on raw statistical patterns, including year-correlated vocabulary. Removing year information forces it to find more label-relevant features, acting as a beneficial regulariser.
- DeBERTa's disentangled attention mechanism (separate positional and content embeddings) already provides strong inductive biases that resist spurious correlations. The adversarial gradient-reversal signal competes with these learned representations and degrades them.

This is an **architecture-dependent finding**: adversarial debiasing is not universally beneficial and should be applied selectively based on the base model's capacity for structured representation.

### Adversarial effect on temporal split

On temporal split, adversarial training provides marginal improvement for both models:
- LegalBERT: 0.646 → 0.675 (+0.029)
- DeBERTa: 0.676 → 0.679 (+0.003)

The adversarial head reduces year-sensitivity but cannot close the SVM gap. This suggests the temporal spurious correlations in neural models are not fully captured by a single linear year-regression adversary — they are distributed across many correlated linguistic features that the adversary cannot fully suppress.

---

## 5. Context Length and Legal Pretraining: Both Matter, Coverage More

**Revised findings after sliding-window experiments:**

The earlier finding that Longformer (2,048 tokens, 0.662) underperforms DeBERTa (0.719) was correctly attributed to lack of legal pretraining — but this masked a larger story about document coverage.

**Key experiment (run 59):** When SVM is restricted to the *same* 512 tokens DeBERTa sees, it scores 0.678 — virtually identical to DeBERTa 0.670. The full SVM advantage (0.745 vs 0.670 = 0.075) almost entirely disappears when the comparison is token-fair. This definitively shows the SVM advantage was never about model quality but purely about seeing the full document.

**Sliding-window results confirm this:**

| Model | Coverage | Macro-F1 | Interpretation |
|-------|----------|----------|----------------|
| SVM, 512 tok | 512 | 0.678 | ~same as DeBERTa |
| DeBERTa 512 tok (focal) | 512 | 0.715 | slightly better than SVM at same coverage |
| SVM, 2048 tok | 2048 | 0.728 | near full-text SVM |
| SVM, full text | all | 0.732 | coverage-based advantage |
| DeBERTa chunked 4× | 2040 | 0.726 | +0.011 over DeBERTa 512-tok |
| LegalBERT chunked 4× | 2040 | **0.760** | **beats SVM: +0.028** |

**The decomposition:**
- Coverage effect (DeBERTa 512→2040): +0.011 (0.715→0.726)
- Legal pretraining effect (DeBERTa→LegalBERT at 512): ~+0.008 (0.678→0.686 approx)
- Both together (DeBERTa 512→LegalBERT chunked): +0.045 (0.715→0.760)
- These effects are **multiplicative** — legal pretraining helps more when the model sees the full document

**Conclusion:** Both pretraining domain alignment and document coverage are necessary to beat SVM. Coverage is the larger effect. Longformer (0.662) failed because it lacked legal pretraining; LegalBERT at 512 tokens lacked coverage. The combination — legal pretraining + sliding-window coverage — is what finally beats SVM.

### Coverage ceiling: ~2040 tokens (runs 69, 63)

| Model | Chunks | Coverage | Macro-F1 |
|-------|--------|----------|----------|
| LegalBERT chunked | 4× | 2040 tok | **0.760** |
| LegalBERT chunked | 8× | 4080 tok | 0.709 ← worse |
| DeBERTa chunked | 4× | 2040 tok | **0.726** |
| DeBERTa chunked | 6× | 3060 tok | 0.723 ≈ same |

Doubling coverage beyond 2040 tokens hurts LegalBERT (−0.051) and provides no benefit for DeBERTa. Once ~75% of documents are fully covered (at ~2040 tokens), additional context introduces noise through the mean-pooling of CLS tokens: 8 chunks amplify inter-chunk variance while adding mostly repetitive or low-information tail content. This is the coverage ceiling.

### Dataset size requirement (run 70)

LegalBERT chunked on the original 436-case dataset (10 epochs, early stopping) scores **0.720** — below SVM's 0.735 on the same dataset. Only with the 1,205-case v1 dataset does LegalBERT chunked (0.760) exceed SVM (0.732). This confirms that neural models require sufficient training data to overcome SVM's sparse-feature robustness advantage: at 436 cases, the model underfits; at 1,205 cases, it surpasses SVM.

### More epochs offer no improvement (run 71)

LegalBERT chunked trained for up to 10 epochs (early stopping with patience=3) scores **0.730** — below the 5-epoch result of 0.760. Early stopping engages at epochs 4–8, similar to the 5-epoch run. The inter-seed variance is higher (0.657–0.752) suggesting that additional budget reaches different local optima per seed. **5 epochs is sufficient; the 5-epoch result (0.760) is the near-optimal configuration** for this dataset size and model.

### Legal-Longformer: native long-context loses to chunked mean-pool (run 75)

`lexlms/legal-longformer-base` with native 4096-token attention achieves **0.689** (random) and **0.678** (temporal) — both well below LegalBERT chunked 4×. This is counterintuitive: Legal-Longformer combines the two ingredients that made LegalBERT chunked succeed (legal pretraining + full document coverage), yet performs far worse.

The explanation lies in the pooling mechanism. LongformerForSequenceClassification uses global attention on the [CLS] token, which attends to all 4096 tokens. Across a noisy 4096-token FACTS section containing procedural boilerplate, repetitive applicant lists, and low-information preambles, the global [CLS] attention signal is diluted. By contrast, the chunked approach forces the model to produce a dense local summary per 510-token segment before mean-pooling — an implicit denoising step. **Chunked CLS mean-pooling is a stronger inductive bias for this task than global Longformer attention.**

Temporal stability of Legal-Longformer (−0.011) is the smallest drop observed, but this reflects the model being weak in both settings rather than genuine temporal robustness. A model scoring 0.689/0.678 has limited room to degrade.

---

## 6. Class Imbalance and F1(no-violation)

F1 on the minority class (no-violation) is consistently the weakest metric:

| Model | Dataset | F1(no-viol) | F1(viol) |
|-------|---------|-------------|----------|
| SVM (best) | v1 temporal | 0.623 | 0.870 |
| DeBERTa (best) | v1 random | 0.558 | 0.880 |
| Adv LegalBERT | v1 random | 0.557 | 0.866 |
| LegalBERT | v1 temporal | 0.417 | 0.875 |

The high violation rate (75–82%) across all dataset versions makes non-violation recall structurally difficult.

---

## 7. The Violation/Non-Violation Asymmetry

The benchmark SVM (no years, `isalpha` tokeniser) reveals a structural asymmetry between the two classes that goes deeper than just spurious features.

### Violation features: events and institutional failures

Top violation tokens describe *things that happened*: `quashed`, `cassation`, `detention`, `delay`, `hearing`, `criminal proceeding`, `charge`, `illegal`, `upheld judgment`. These are procedural acts and institutional failures — the narrative of what went wrong domestically before the case reached Strasbourg. Even where genuine legal content is present, it encodes specific violation *types* (length of proceedings, quashed-judgment cycles, unfair hearing) rather than the legal reasoning behind the finding.

Residual shortcuts co-exist: `state security` / `army` (Turkish institutional proxies), `february` (month as subtle temporal surrogate), `born life` / `second applicant` (mass-case boilerplate), `treasury` / `pecuniary` (country-correlated procedural language).

### Non-violation features: reasoning and evaluation

Top non-violation tokens describe *legal analysis being conducted*: `whether`, `particular`, `might`, `would` (conditional/evaluative framing), `legislative`, `provision`, `right`, `act`, `meaning` (statutory citation), `jury`, `surveillance`, `terrorist`, `conversation` (UK fact patterns that systematically produce acquittals).

Non-violation judgments contain more explicit citations to legal provisions and reasoned analysis — the court explains *why* no violation occurred, using evaluative language that the model picks up as signal. Violation judgments, especially mass-repetition cases, follow templated language referencing applicant lists and established violation patterns, which is more formulaic.

### The core asymmetry

| Dimension | Violation cases | Non-violation cases |
|-----------|----------------|---------------------|
| Language style | Event-driven ("was detained", "was quashed") | Evaluative ("whether", "in the particular circumstances") |
| Legal citation | Sparse; violation type established by precedent | Explicit; provisions, articles, legislative framework cited |
| Document structure | Templated / mass-case boilerplate common | More individualised reasoning |
| Shortcut density | High (institutional names, months, boilerplate) | Lower; residual shortcuts are country-specific (jury, terrorist) |

**Implication**: models predicting non-violation are partly learning to recognise *the presence of legal reasoning*, while models predicting violation are partly learning to recognise *case origin and procedural failure patterns*. Neither is fully learning universal legal principles — but non-violation predictions are closer to the right kind of signal.

---

## 8. Implications for Attention Models

The SVM feature analysis and the violation/non-violation asymmetry provide concrete, actionable guidance for neural model design and training.

### 8.1 What attention models are likely doing wrong

The temporal drop (DeBERTa: −0.043 on v1) combined with the feature analysis suggests neural models are attending to:

1. **Contextual clusters around institutional names** — `state security`, `army`, `cassation`, `treasury` are embedded in dense contextual neighbourhoods that encode country identity. An attention model learns that the co-occurrence pattern around `state security` predicts violation, but this pattern changes post-2004 (Turkish State Security Courts abolished).
2. **Temporal vocabulary distributions** — not just year tokens themselves (which are filtered) but the vocabulary *associated* with different eras: case law language, institutional names, procedural terminology that evolved over time.
3. **Template detection** — mass-case boilerplate (`born life`, `second applicant`, `pecuniary`) occurs in dense contextual patterns that a neural model memorises as violation indicators. These patterns don't transfer when case volumes and filing practices change.

### 8.2 Targeted training interventions

**Counterfactual data augmentation (CDA)**

The most direct intervention: systematically replace country-identifying tokens during training to force the model to learn country-invariant features.
- Mask or replace institutional names (`state security`, `army`, `cassation`) with generic placeholders during training
- Replace country-specific currency/place tokens (`rub`, `favour`, `treasury`) with `[INSTITUTION]`, `[CURRENCY]`, `[OFFICIAL]` tokens
- The model must then predict from procedural content alone

This is stronger than the adversarial year-debiasing approach (§4) because it intervenes at the token level rather than the representation level.

**Month masking**

`february` appears as a top violation feature — a subtle temporal shortcut invisible to year-stripping. Masking all month names during training would close this residual temporal leak.

**Targeted attention supervision**

Given the asymmetry (violation = event tokens, non-violation = reasoning tokens), attention supervision can be applied:
- For training examples with known violation type (e.g. length-of-proceedings cases), supervise attention to focus on `delay`, `hearing`, `criminal proceeding` rather than co-occurring institutional names
- Penalise high attention weights on a predefined shortcut token list (years, months, place names, institutional names) during training via an auxiliary attention regularisation loss:
  `L_attn = λ · Σ attention_weight(token) for token in SHORTCUT_SET`

**Token-level bias probes (actionable next step)**

Before any training modification, run targeted masking probes:
1. Replace all country/institutional tokens → measure F1 drop → quantifies geographic shortcut contribution
2. Replace all month tokens → quantifies residual temporal shortcut
3. Replace `quashed`, `cassation`, `state security` one at a time → identifies which institutional features are load-bearing vs redundant

### 8.3 Exploiting the asymmetry: attend to reasoning language

Non-violation features are more substantive: `whether`, `particular`, `legislative`, `provision`, `right`, `meaning`. These mark *evaluative reasoning* in the text. Two approaches to amplify this signal:

**Reasoning-span highlighting**
Sentences containing evaluative markers (`whether the applicant`, `in the particular circumstances`, `within the meaning of`) are more likely to contain the court's legal reasoning. A simple pre-processing step: up-weight these sentences in the head_tail truncation scheme (currently first 128 + last 382 tokens). Alternatively, add a third segment: the highest-density reasoning span (identified by keyword density) as a middle chunk.

**Contrastive pre-fine-tuning**
Create pairs of (violation, non-violation) cases with similar factual content but different outcomes — e.g. two length-of-proceedings cases, one violation and one not. Fine-tune with a contrastive objective so the model must learn what distinguishes the legal analysis, not just the fact pattern. The SVM feature asymmetry provides a principled basis for selecting these pairs: cases that share violation-type vocabulary (`delay`, `hearing`) but differ in evaluative vocabulary (`whether`, `legislative`).

### 8.4 Architecture-level takeaways

| Finding | Implication |
|---------|-------------|
| Adversarial debiasing hurts DeBERTa but helps LegalBERT | Do not apply gradient-reversal to models with strong positional inductive biases; use token-level interventions (CDA, masking) instead |
| Longformer underperforms DeBERTa despite 4× context | Legal reasoning tokens are short and dense; long context adds noise from procedural boilerplate more than it adds signal. Smarter truncation > longer window |
| Non-violation reasoning language is short and evaluative | CLS token or first-sentence pooling may actually capture more reasoning signal than mean-pooling over the full FACTS section, which averages in event-driven boilerplate |


### 8.5 Summary: recommended training pipeline for a debiased model

1. **Preprocessing**: mask month tokens and institutional shortcut list (`[MASK]` or generic replacement) — free, ~0 compute cost
2. **Truncation**: replace head_tail with head + reasoning-span (keyword-density-detected middle section) + tail
3. **Training objective**: standard cross-entropy + attention regularisation penalty on shortcut tokens (λ=0.01–0.05)
4. **Validation metric**: prioritise temporal macro-F1, not random-split macro-F1 — a model that improves on random but degrades on temporal is learning shortcuts, not law
5. **Probing suite**: run token-masking probes after each major training change to verify which shortcuts have been reduced

1. **LIME / Integrated Gradients**: Do neural models attend to the same temporal/geographic tokens identified in SVM feature analysis, or do they exploit different spurious patterns? Confirming this cross-model would strengthen the finding.

2. **Country masking probe**: What is the macro-F1 drop when all country names and place names are masked before inference? This would directly quantify the contribution of geographic shortcuts.

3. **Year masking probe**: Similarly, masking all 4-digit year tokens would isolate the temporal shortcut contribution.


4. **Article-level generalisation (deferred)**: Do the same spurious correlation patterns appear for Articles 3, 5, and 8, or is Article 6 anomalous due to its procedural nature?

---

## 9. From SVM Dominance to LegalBERT Chunked: A Resolution

After running DeBERTa-base (multiple configs), DeBERTa-large, Longformer, adversarial debiasing, CDA, TF-IDF sentence selection, and soft ensembles — none beat SVM. The SVM was finally beaten by **LegalBERT with sliding-window chunked encoding** (run 61, 0.760 vs SVM 0.732).

### 9.1 Why SVM dominated earlier attempts

**1. Truncation was the primary bottleneck — but not recognised as such.**
65.5% of FACTS sections exceed 512 tokens (median 813 tokens, mean 1237). SVM sees *every word's* TF-IDF contribution. DeBERTa sees 512 tokens. The smoking gun: when SVM is restricted to the *same* 512 tokens, it scores 0.678 ≈ DeBERTa's 0.670. The apparent 0.075 SVM gap was almost entirely a truncation artefact.

Longformer (2048-token window, 0.662) failed to fix this because it lacks legal pretraining. Without legal vocabulary alignment, more context is noise, not signal.

**2. 900 training cases are insufficient for 183M-parameter models.**
DeBERTa-base has 183M parameters vs SVM's ~12K effective parameters (TF-IDF features × 2 classes). With ~900 training examples, SVM's simpler hypothesis class generalises better. DeBERTa-large (400M params) collapsed completely — consistent with severe over-parameterisation.

**3. The task is primarily lexical.**
Top SVM features are specific legal bigrams: *"reasonable time"*, *"legal assistance"*, *"access court"*, *"adequately reasoned"*, *"fair trial"*. Predicting Art.6 violation reduces to counting these phrases across the full document. LegalBERT's pretraining helps it recognise these legal constructs; DeBERTa's general-English pretraining is less aligned.

**4. 81.9% class imbalance amplifies causes 2 and 3.**
Effective non-violation training examples: ~130. SVM handles this via a globally-optimal convex objective with `class_weight='balanced'`. DeBERTa with 183M parameters has far more capacity to overfit the majority class.

### 9.2 What finally worked: sliding-window chunked encoding

| Method | Macro-F1 | Key change |
|--------|----------|-----------|
| DeBERTa baseline, 512 tok | 0.670 | — |
| DeBERTa + focal, 512 tok | 0.715 | focal loss (γ=2) |
| SVM, full text | 0.725 | full document |
| DeBERTa chunked 4×, 2040 tok | 0.726 | full coverage |
| Legal-Longformer, 4096 tok | 0.689 | coverage + legal pretraining, but global attn diluted |
| **LegalBERT chunked 4×, 2040 tok** | **0.760** | **coverage + legal pretraining + chunked pooling** |

The sliding-window approach (split document into consecutive 510-token chunks, encode each independently, mean-pool CLS embeddings) solves the truncation bottleneck without requiring a new model architecture. The key insight: LegalBERT's legal domain pretraining makes it robust to the fragmented context each chunk sees — it can interpret partial FACTS sections because its pretraining exposed it to similar legal language patterns.

### 9.3 What the earlier gap-closing attempts revealed

Methods that partially closed the DeBERTa–SVM gap all injected *lexical* information rather than improving *semantic* representations:

| Method | Gain | Mechanism |
|--------|------|-----------|
| TF-IDF sentence selection | +0.044 | More discriminative lexical content in 512-token window |
| CDA masking | +0.027 | Removed shortcut tokens → model uses remaining legal content |
| Soft ensemble (DeBERTa + SVM) | +0.017 | Directly blends SVM's full-document statistics |

LegalBERT chunked breaks this pattern — it improves via *coverage* (sees the same tokens SVM sees) and *legal representations* (understands what those tokens mean in legal context), not by injecting SVM's signal.

### 9.4 Research framing (updated)

The sequence of SVM dominance followed by LegalBERT chunked resolution is itself a publishable contribution:

> *"TF-IDF+SVM appears to outperform modern attention models on ECHR violation prediction due to: (1) document truncation discarding discriminative content; (2) insufficient training data (900 cases) for over-parameterised models; (3) the task's primarily lexical nature. Once document coverage is restored via sliding-window chunked encoding, and combined with legal domain pretraining (LegalBERT), neural models finally surpass SVM (+0.028). The gap was not a fundamental limitation of attention models, but an artefact of the 512-token context window. This finding also confirms that both full document coverage AND legal pretraining are necessary — neither alone is sufficient."*

The spurious correlations story remains valid — and deepens: LegalBERT chunked (0.760 random) drops to 0.682 temporally (−0.078), its highest random-split score correlating with its largest temporal fragility. **The model that appears most capable on standard evaluation is most reliant on temporal shortcuts.** SVM, despite being a bag-of-words model, remains the most temporally robust (0.746). For the paper, this is the key thesis: random-split performance is a misleading metric for legal AI; temporal generalisation exposes whether a model has learned legal principles or memorised historical accident patterns.

### 9.5 Original dataset: LegalBERT chunked finally beats SVM (confirmed)

On the original 436-case dataset (RUS/TUR/GBR only), LegalBERT chunked 4× (run 80, fresh 4-seed) achieves **0.748** — confirmed above both SVM (0.726) and LogReg (0.735). The breakthrough requires:
1. **4×510 sliding window** — covers 2040 tokens vs the original 512-token truncation
2. **Legal pretraining** — LegalBERT understands fragmented FACTS sections
3. **Focal loss γ=2** — corrects for the 73.4% violation imbalance

With CDA masking (run 74, 8 seeds): also 0.748 — CDA adds nothing on the 3-country dataset once coverage is already 2040 tokens (the country-prior shortcuts are overridden by the richer document context). The improvement over SVM (+0.022) requires only the 4-seed ensemble; the earlier finding that 8 seeds were needed for the original dataset was not confirmed in run 80 (4 seeds gives the same 0.748).

**The original dataset result is now: LegalBERT chunked 0.748 > SVM 0.726 > LogReg 0.735? No — LogReg 0.735 < 0.748.** Full ordering on original random split: LegalBERT chunked (0.748) > LogReg (0.735) > SVM (0.726).

---
