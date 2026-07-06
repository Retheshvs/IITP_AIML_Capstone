# Part 3 — Intelligent FAQ Matching

## What this code does
Implements three FAQ search methods and compares them:
1. **Keyword Search** (Task 1, stopword-filtered) — matches on meaningful words only, fixed 0.5 score.
2. **TF-IDF Matching** — vectorizes FAQ question+keywords and query text, ranks by cosine similarity.
3. **Hybrid Search** — merges both methods, keeping the highest score per FAQ.

## How to run
1. Open `part3_notebook.ipynb` in Jupyter or Google Colab.
2. Run all cells top to bottom (Runtime > Run all).
3. No API keys or external services required.

## Dependencies
- Python 3.x
- scikit-learn (`pip install scikit-learn`)

## Key findings
- Naive keyword search over-matches on common stopwords ("i", "my", "do"), producing false
  positives across unrelated FAQs. Filtering stopwords fixes this, restoring `(no results)`
  behavior for queries with no meaningful term overlap.
- TF-IDF matching correctly handles paraphrased queries (e.g. "login credentials" vs.
  "password reset") by weighting term importance rather than requiring exact word overlap.
- Hybrid search never underperforms either individual method, since it keeps the max score
  per FAQ across both approaches — combining keyword search's precision on exact terms with
  TF-IDF's robustness to paraphrasing.
