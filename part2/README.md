# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

**Marks:** 30 · **Input:** `cleaned_data.csv` (produced in Part 1)

## 1. Labels

- **Regression label (`y_reg`):** `purchase_amount` — the continuous order value.
- **Classification label (`y_clf`):** `is_returned` — a natural binary column already
  present in the dataset (1 = order was returned, 0 = not returned). This was used
  directly instead of binarizing `purchase_amount` at its median, since a real
  business-relevant binary outcome was already available.
- Both `order_id` and `is_returned` are dropped from the feature matrix `X` when
  predicting `purchase_amount`, and `purchase_amount` is dropped when predicting
  `is_returned`, so neither target leaks into its own feature set.
- Remaining missing values in `customer_rating` (216 nulls) are filled with the column
  median before encoding.

## 2. Categorical Encoding

- `product_category` and `payment_method` have no natural order (nominal), so both are
  **one-hot encoded** with `pd.get_dummies(..., drop_first=True)`. Dropping the first
  dummy avoids multicollinearity (the dropped category is implied when all other dummies
  are 0) and avoids the false-ordinal-relationship problem that label/integer encoding
  would introduce — e.g. encoding `Clothing=0, Electronics=1, Home=2, Sports=3` would
  incorrectly imply `Sports > Home` in magnitude, which has no real meaning for an
  unordered category.
- No ordinal columns were present in this dataset, so no label encoding was needed.

## 3. Leak-Free Train-Test Split and Scaling

`X` and both labels are split with `train_test_split(X, y, test_size=0.2,
random_state=42)`. `StandardScaler` is **fit only on `X_train`**, then used to transform
both `X_train` and `X_test`. Fitting the scaler on the full dataset (train + test) would
leak test-set mean/variance into the training process — the model would implicitly "see"
statistics from data it's supposed to be evaluated on, producing an overly optimistic
performance estimate that would not hold on genuinely unseen data.

## 4. Regression — Linear Regression vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 289.33 | 0.612 |
| Ridge (alpha=1.0) | 288.72 | 0.613 |

**Top 3 features by \|coefficient\|:**

| Feature | Coefficient |
|---|---|
| `delivery_days` | +31.12 |
| `product_category_Sports` | +2.31 |
| `satisfaction_score` | +1.15 |

`delivery_days` dominates: a one-standard-deviation increase in delivery days is
associated with a ~31-unit increase in predicted `purchase_amount`, holding other
scaled features constant — far larger than any other feature, suggesting higher-value
orders in this dataset tend to also have longer delivery windows (e.g. bulkier or
higher-value shipments). The `Sports` category coefficient means orders in that category
are associated with a modest increase in predicted purchase amount relative to the
baseline (dropped) category.

Ridge produces a nearly identical MSE/R² to plain Linear Regression here (288.72 vs
289.33), with only a marginal improvement. `alpha` controls the strength of the L2
penalty added to the loss function, which shrinks coefficients toward zero to reduce
variance at the cost of a small amount of bias. Because this dataset's features are not
strongly collinear and the plain OLS coefficients are already fairly small and stable,
Ridge has little room to improve on OLS — the two models end up producing a very similar
coefficient profile. Ridge would diverge more from OLS on a dataset with strong
multicollinearity, where OLS coefficients tend to be large and unstable.

## 5. Classification — Logistic Regression

**Class imbalance:** the training set is 85.3% "not returned" vs 14.7% "returned" — well
below the 35% threshold, so imbalance handling was required. `class_weight='balanced'`
was used (rather than SMOTE) to reweight the loss function inversely to class frequency,
avoiding the need to synthesize new minority-class rows on a fairly small dataset.

**Confusion matrix (test set, threshold = 0.5):**

| | Predicted: Not Returned | Predicted: Returned |
|---|---|---|
| **Actual: Not Returned** | 68 | 64 |
| **Actual: Returned** | 19 | 9 |

**Classification report:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (Not Returned) | 0.78 | 0.52 | 0.62 | 132 |
| 1 (Returned) | 0.12 | 0.32 | 0.18 | 28 |

Accuracy: 0.481 · **AUC: 0.383** (ROC curve saved as `roc_curve.png`)

**Formulas:**
- Precision = TP / (TP + FP) — of all orders predicted as "returned," what fraction
  actually were returned.
- Recall = TP / (TP + FN) — of all orders that were actually returned, what fraction the
  model caught.

For this task, **recall on the "returned" class matters more than precision**: missing a
return (a false negative) means the business fails to flag a customer likely to send an
item back, losing the chance to intervene (e.g. proactive support, retention offer),
whereas a false positive (flagging an order that turns out fine) only costs a low-value
follow-up check.

**Honest finding:** the AUC of 0.383 is *below* 0.5, meaning this logistic regression
model separates the two classes worse than random guessing on this test set. This
indicates `is_returned` has little to no learnable linear relationship with the features
available in this dataset (order value, delivery days, ratings, category, payment
method) — the returns in this data appear close to random noise with respect to these
features, rather than the model being miscalibrated. A non-linear model or additional
features (e.g. product defect rate, prior return history) would likely be needed to
predict returns meaningfully on data like this.

## 6. Decision-Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.176 | 0.929 | 0.295 |
| 0.40 | 0.163 | 0.750 | 0.268 |
| 0.50 | 0.123 | 0.321 | 0.178 |
| 0.60 | 0.059 | 0.036 | 0.044 |
| 0.70 | 0.000 | 0.000 | 0.000 |

The threshold that maximizes F1 on this dataset is **0.30**. Since recall matters more
than precision for this task (missing a return is costlier than a false alarm), the
threshold should be **lowered** from the default 0.5 toward 0.30 — this raises recall
from 0.32 to 0.93, at the cost of precision dropping from 0.12 to 0.18 (still low, but
precision was already weak at every threshold given the model's poor overall separation
noted above). The trade-off is worth it here: catching 93% of actual returns, even with
many false alarms, is more useful operationally than catching only 32% of them.

## 7. Regularization Experiment (C=1.0 vs C=0.01)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.123 | 0.321 | 0.383 |
| C=0.01 (strong L2) | 0.156 | 0.429 | 0.407 |

`C` is the inverse of the L2 regularization strength in scikit-learn's
`LogisticRegression` (smaller `C` = stronger penalty on large coefficients). Reducing `C`
from 1.0 to 0.01 **improved** precision, recall, and AUC slightly on this dataset —
suggesting the baseline model was mildly overfitting the small amount of real signal
(or noise) in the training data, and shrinking the coefficients more aggressively
generalized marginally better to the test set. That said, both models still perform
close to or below chance level (AUC ≈ 0.4), so this improvement is small in absolute
terms.

## 8. Bootstrap Confidence Interval for AUC Difference

500 bootstrap resamples were drawn from the test set (`np.random.choice` with
replacement); for each resample, the AUC of the C=1.0 model minus the AUC of the C=0.01
model was computed.

- **Mean AUC difference:** −0.0238
- **95% CI:** [−0.0475, −0.0037]

The interval **excludes zero**, meaning the C=0.01 (more regularized) model's AUC
advantage over the C=1.0 baseline is consistent across resampled subsets of the test
data — not just a fluke of one particular train/test split. Note the difference is
negative (C=1.0 minus C=0.01), confirming the more regularized model is reliably better,
consistent with the regularization comparison above.

## Files

- `Capstonre_part2.ipynb` — full pipeline (Tasks 1–7)
- `cleaned_data.csv` — input, produced in Part 1
- `roc_curve.png` — ROC curve for the baseline (C=1.0) logistic regression model

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook Capstonre_part2.ipynb
```

Run all cells top to bottom. No API keys or external services are required for this part.
