# ECHR Project Roadmap

Based on the [Group Project Proposal](https://notebooklm.google.com/notebook/98c3051b-1b21-4204-b7cd-7892f9350f20) and the current codebase status.

## Phase 1: Foundation (Completed)
**Goal:** Setup environment, acquire data, and prepare for modeling.
- [x] **Data Acquisition**: `scripts/download_data.py` implements robust downloading of Violation/Non-Violation cases from HUDOC.
  - Supports `--countries RUS,TUR,GBR` and `--per_country_count 200` for balanced 1,200-case dataset.
- [x] **Data Preprocessing**: `scripts/preprocess_data.py` extracts the "FACTS" section (addressing data provenance) and splits data into Train/Val/Test.
- [x] **Environment**: Dependencies (`echr-extractor`, `transformers`, `scattertext`, `shifterator`, etc.) are established.
- [x] **EDA Notebook**: `EDA.ipynb` includes:
  - Per-country summary table, correlation matrix, "represented" keyword analysis
  - TF-IDF, N-grams, Fighting Words (shifterator), Scattertext, Concordance
  - Text length & respondent state visualizations

## Phase 2: Modeling (In Progress)
**Goal:** Establish predictive baselines and state-of-the-art models.
- [x] **Legal-BERT Model**: `src/train.py` implements fine-tuning of `nlpaueb/legal-bert-base-uncased`.
- [ ] **SVM Baseline**: Implement a TF-IDF + Linear SVM baseline (N-gram=5) as per the proposal (referencing Aletras et al. 2016).
    *   *Action*: Create `src/train_svm.py`.
- [ ] **Model Comparison**: Compare Accuracy, Precision, Recall, and F1 between BERT and SVM.

## Phase 3: Deep Dive & Bias Analysis (Next Steps)
**Goal:** "Open the black box" and detect spurious correlations.
- [ ] **Basic Bias Metrics**: `scripts/analyze_bias.py` currently covers:
    - [x] Respondent State Bias (Country)
    - [x] Temporal Bias (Year)
    - [x] Text Length Bias
- [ ] **Specific Keyword Testing**: Test for the "represented" keyword artifact.
    *   *Action*: Update `scripts/analyze_bias.py` to calculate correlation between "represented" counts and Violation labels.
- [ ] **Explainability**: Implement looking inside the model.
    *   *Action*: Add LIME or Integrated Gradients to `scripts/analyze_bias.py` (or a new script) to visualize feature importance for specific cases.

## Phase 4: Deliverables & Synthesis
**Goal:** Final reporting.
- [ ] **Results Compilation**: Aggregate all metrics into a comparative report.
- [ ] **Critical Analysis**: Conclude whether the model learns "law" or "bias".

## Suggested Next Tasks
1.  **Implement SVM Baseline**: Essential for the research question "Deep Learning vs Traditional ML".
2.  **Enhance Bias Analysis**: Add the specific "represented" keyword check.
3.  **Run Full Experiment**: Train both models on the full 500-2000 case dataset and generate the comparison.
