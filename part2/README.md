# Part 2 — Supervised ML Model

## Label Definitions
- y_reg: [column name]
- y_clf: 1 if [target] > median ([value]), else 0

## Encoding
[which columns were ordinal vs one-hot, and why one-hot avoids false ordinal relationships]

## Data Leakage Prevention
[explain scaler fit only on training data]

## Regression Results
[paste MSE/R2 table, Ridge vs Linear, top 3 coefficients, alpha explanation]

## Classification Results
[confusion matrix, classification report, AUC value, Precision/Recall formulas]

## Threshold Sensitivity
[paste threshold table, state which maximizes F1, cost tradeoff]

## Regularization Experiment
[C=1.0 vs C=0.01 table, explain what C controls, better or worse]

## Bootstrap AUC Confidence Interval
[mean difference, CI bounds, whether it excludes zero, interpretation]
