# Applied AI & ML Essentials — Capstone Project

**Total marks:** 100 · **Parts:** 4 (independently deliverable)

This repository implements the full data-to-AI lifecycle across four parts: data cleaning
and EDA, supervised machine learning, intelligent FAQ matching (TF-IDF + hybrid search),
and an LLM-powered structured extraction feature with safety guardrails.

## Repository Structure

```
IITP_AIML_Capstone/
├── part1/   → Data Acquisition, Cleaning, and Exploratory Analysis   (25 marks)
├── part2/   → Supervised Machine Learning Model                     (30 marks)
├── part3/   → Intelligent FAQ Matching                               (25 marks)
├── part 4/  → LLM-Powered Feature: Structured JSON Extraction        (20 marks)
└── README.md → this file
```

Each part is self-contained: its own notebook, its own README with dataset details,
findings, and run instructions, and its own output files (plots / cleaned CSV / etc).

## Part Summaries

### [Part 1 — Data Cleaning & EDA](./part1/README.md)
- Dataset: e-commerce order data (800 rows × 11 columns after cleaning).
- Null analysis, duplicate removal, dtype correction, skewness, IQR outlier detection.
- 5 required visualizations + correlation heat map, all saved as PNGs.
- Spearman vs Pearson comparison and grouped aggregation with written interpretation.
- Output: `cleaned_data.csv`, used as input to Part 2.

### [Part 2 — Supervised ML: Regression + Classification](./part2/README.md)
- Regression: Linear Regression vs Ridge on a continuous target, with coefficient
  interpretation.
- Classification: Logistic Regression with class-imbalance handling, confusion matrix,
  ROC/AUC, decision-threshold sensitivity, and an L2 regularization (C) comparison.
- Bootstrap confidence interval (500 resamples) on the AUC difference between models.
- Leak-free pipeline: scaler fit only on training data.

### [Part 3 — Intelligent FAQ Matching](./part3/README.md)
- `FAQMatcher` class: TF-IDF vectorization + cosine similarity, with `match()`,
  `best_match()`, and `explain_match()`.
- `hybrid_search()`: merges keyword search (Part 1-style) with TF-IDF, keeping the
  highest score per FAQ.
- Comparison demo across 3 test queries showing why TF-IDF/hybrid search beats naive
  keyword matching on paraphrased questions.

### [Part 4 — LLM-Powered Feature: Structured JSON Extraction](./part%204/README.md)
- Track A: extracts customer name, product, issue type, sentiment, and refund flag from
  free-text support messages via an LLM API (OpenRouter), with a few-shot prompt at
  `temperature=0`.
- Output is validated against a JSON schema (`jsonschema`) with a safe fallback on
  failure.
- PII guardrail: regex-blocks any input containing an email or phone number before the
  LLM is ever called.
- Includes a temperature 0 vs 0.7 A/B comparison and a 3-input end-to-end demonstration
  table.

## How to Run

Each part folder contains its own README with exact setup and run steps. In general:

```bash
# Part 1 and Part 2 (local, no API key needed)
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook part1/"capstone Q1.ipynb"
jupyter notebook part2/Capstonre_part2.ipynb

# Part 3 (local, no API key needed)
pip install scikit-learn
jupyter notebook part3/Capstone_part3.ipynb

# Part 4 (requires a free OpenRouter API key: https://openrouter.ai/keys)
pip install requests jsonschema
jupyter notebook "part 4"/capstone_part4.ipynb
```

No API keys or secrets are committed anywhere in this repository. Part 4 requests the
`LLM_API_KEY` interactively at runtime.

## Notes on Data

Parts 1 and 2 use a structured dataset with 800+ rows, 11 columns, a numeric target
(`purchase_amount`), and multiple categorical columns, satisfying the size and shape
requirements for the assignment. Parts 3 and 4 use small hand-crafted datasets (FAQ set
and sample support messages) appropriate to their respective tasks, as specified in the
brief.
