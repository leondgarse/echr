### **Project Title:**
**The Artificial Judge: Evaluating Legal Reasoning and Spurious Correlations in NLP Models of the European Court of Human Rights**

### **1. Project Overview & Motivation**
The European Court of Human Rights (ECHR) faces a significant backlog of cases, prompting interest in AI-driven tools to triage or predict judicial outcomes. While recent studies demonstrate that Natural Language Processing (NLP) models can predict outcomes with high accuracy (79%–82%), significant questions remain regarding *how* these models reach their conclusions.

This project investigates the "black box" of Legal Judgment Prediction (LJP). Specifically, we aim to determine if predictive models are learning genuine legal principles or merely exploiting linguistic artifacts and spurious correlations (e.g., the presence of specific procedural keywords or metadata). Drawing on the course’s focus on **data provenance** and **data generation**, we critique the reliance on post-hoc judicial texts as input data, acknowledging that the "facts" in these documents are often constructed narratives rather than objective reality.

### **2. Key Research Questions**
1.  **Predictive Accuracy:** How do traditional machine learning models (Linear SVM with N-grams) compare against modern Transformer-based models (Legal-BERT) in predicting ECHR violation outcomes?
2.  **Explainability & Legal Reasoning:** Do these models rely on legally substantive text (e.g., specific fact patterns) or spurious correlations (e.g., procedural terms like "represented by," or country names)?
3.  **Generalization:** Does the model's performance degrade when tested on cases from a future time period, indicating a failure to learn evolving legal standards (the "temporal shift" problem)?

### **3. Data Source & Provenance**
We will utilize the **ECHR-OD (Open Data)** or **LexGLUE** datasets, which provide structured access to ECHR judgments scraped from the HUDOC database.
*   **Dataset Size:** We target approximately 500–2,000 cases to meet the course requirement for a non-textual or textual dataset.
*   **Data Provenance Issue:** We acknowledge a critical limitation identified in the literature: our input data (the "Facts" section of judgments) is generated *after* the decision is made, potentially leaking the outcome. We will address this by strictly removing "Law" and "Operative Provisions" sections during preprocessing.

### **4. Methodology**
Framed as a **binary classification task** (Violation vs. No Violation), scoped to **Article 6** cases.
All models use **FACTS section only** — stripping LAW and Operative Provisions prevents outcome leakage.
Labels are Article-6-specific (derived from `violation_articles`/`nonviolation_articles` columns).

**A. Exploratory Data Analysis (EDA)** — `EDA.ipynb` / `EDA.pdf`
*   TF-IDF and Fighting Words (Monroe et al. 2008) analysis of violation vs. non-violation vocabulary.
*   Scattertext corpus comparison and concordance analysis.
*   Spurious keyword investigation: `represented` rate by class and country.
*   Shared lexical pipeline: lemmatization (`WordNetLemmatizer`, noun POS), legal stopwords (`court`, `case`, `mr`, `mrs`, `ms`), `MANUAL_LEMMA_MAP` for irregular legal plurals.

**B. Modeling:**
1.  **Baseline Model** (`train_svm.ipynb`) — TF-IDF + Linear SVM replicating Aletras et al. (2016). Uses the same `legal_tokenizer` as EDA for consistency. Results: ~74% random split, ~68% temporal split.
2.  **Advanced Model** (`src/train.py`) — Fine-tune **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`).

**C. Evaluation & Critique:**
*   **Two split strategies:** random stratified 75/25 (standard) and temporal split (train on older cases, test on future cases). The ~6pp drop on temporal split is the primary evidence of temporal spurious correlation.
*   **Per-country accuracy:** accuracy closely tracking violation rate per country indicates the model exploits class base rates rather than case facts.
*   **Top SVM features:** presence of country names or procedural boilerplate in the highest-weight features is direct evidence of spurious correlation.
*   **Explainability (planned):** Integrated Gradients or LIME to verify whether highlighted regions align with legally substantive text.

### **6. References**
*   **Aletras et al. (2016).** Predicting judicial decisions of the European Court of Human Rights: a Natural Language Processing perspective. *PeerJ Computer Science*.
*   **Chalkidis et al. (2022).** LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. *ACL*.
*   **Medvedeva & McBride (2023).** Legal Judgment Prediction: If You Are Going to Do It, Do It Right. *NLLP @ ACL*.
*   **Santosh et al. (2022).** Deconfounding Legal Judgment Prediction for European Court of Human Rights Cases. *EMNLP*.
*   **Quemy & Wrembel (2021).** ECHR-DB: On building an integrated open repository of legal documents. *Information Systems*.

### **6. Execution Steps**

#### **1. Environment Setup**
```bash
pip install echr-extractor torch transformers pandas scikit-learn nltk scattertext shifterator
```

#### **2. Data Acquisition**
```bash
python scripts/download_data.py --countries RUS,TUR,GBR --per_country_count 200 --articles 3,5,6,8
```
Saves to `data/raw/metadata.csv` and `data/raw/full_text.json`.

#### **3. Data Preprocessing**
```bash
python scripts/preprocess_data.py --data_dir data/raw
```
Extracts FACTS sections only; saves to `data/processed/processed.csv` (~952 cases).

#### **4. Phase 1 — EDA**
Open and run `EDA.ipynb`. Exported output: `EDA.pdf`.

#### **5. Phase 2 — Traditional ML (SVM)**
Open and run `train_svm.ipynb`. Self-contained; no external `.py` dependencies.

#### **6. Phase 2 — Legal-BERT**
```bash
python src/train.py --epochs 3 --batch_size 8 --output_dir results
# CPU only:
CUDA_VISIBLE_DEVICES="" python src/train.py --epochs 1 --batch_size 2
```
