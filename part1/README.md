# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## 1. Dataset

This project uses a **synthetic e-commerce order dataset** (`ecommerce_raw.csv`), generated
by the script itself with a fixed random seed (`np.random.seed(42)`) rather than downloaded
from an external source. It was built this way so it would satisfy — and let me demonstrate —
every technique required by this task: missing values at different rates, duplicate rows, a
numeric column stored as text, positive skew, negative skew, and IQR-detectable outliers.

**How to get the dataset:** run `part1_eda.py` (or the notebook). Step 0 at the top of the
script generates `ecommerce_raw.csv` automatically — no external download needed.

**Raw shape:** 815 rows × 11 columns (before cleaning)
**Final shape:** 800 rows × 11 columns (after cleaning)

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_age` | Customer age in years |
| `purchase_amount` | Order value (target numeric column) |
| `quantity` | Units purchased |
| `discount_percent` | Discount applied (stored as text, e.g. `"10%"`) |
| `customer_rating` | Post-purchase rating, 1–5 |
| `delivery_days` | Days taken to deliver |
| `product_category` | Electronics / Clothing / Home / Books / Sports |
| `payment_method` | Credit Card / Debit Card / UPI / Cash on Delivery |
| `satisfaction_score` | Post-purchase satisfaction, 0–100 |
| `is_returned` | 1 if the order was returned, else 0 |

## 2. Code

All cleaning, EDA, and visualization code is in `part1_eda.py` — runs top-to-bottom, no
manual steps between tasks. Can be split into notebook cells at each `# TASK n` marker.

## 3. Step-by-step findings

### Task 1 — Load and inspect
Loaded with `pd.read_csv()`. Initial shape: **815 rows × 11 columns**.

### Task 2 — Null value analysis
| Column | Null count | Null % |
|---|---|---|
| `discount_percent` | 12 | 1.47% |
| `customer_rating` | 222 | **27.24%** |
| `satisfaction_score` | 40 | 4.91% |

`customer_rating` exceeds the 20% threshold and was deliberately not filled here —
median-filling a column that's over a quarter missing would overwrite genuine signal with
a constant value. It's handled explicitly in Task 9a instead. `satisfaction_score`
(4.91% missing) was filled with its median.

### Task 3 — Duplicates
`df.duplicated().sum()` found **15 duplicate rows**. Removed via `drop_duplicates()`
(815 → 800 rows). Null percentages barely shifted (e.g. `customer_rating` 27.24% → 27.0%),
meaning duplicated rows weren't disproportionately null.

### Task 4 — Data type correction
`discount_percent` was stored as text (`"10%"`, plus 12 rows of `"N/A"` noise). Converted
by stripping `%` and applying `pd.to_numeric(errors='coerce')` — correctly produced 12 NaNs
matching the 12 original "N/A" entries, then filled with the median.

`product_category` and `payment_method` converted from object to `category` dtype:
- Memory before: 149,112 bytes
- Memory after: 59,847 bytes
- Saved: 89,265 bytes (~60% reduction)

### Task 5 — Skewness
Highest |skew|: **`purchase_amount` (+4.284)**. Strong positive skew — a long right tail
from a handful of large/bulk orders pulls the mean well above the median, making the mean
an unreliable "typical value" and median the safer choice for imputation.

### Task 6 — Outliers (IQR)
| Column | Bounds | Outliers |
|---|---|---|
| `purchase_amount` | (−19.69, 95.14) | **47** |
| `quantity` | (0.00, 8.00) | **16** |

Documented, not dropped — likely genuine bulk/high-value orders rather than data errors.

### Task 7 — Visualizations
See full plot-by-plot descriptions in Section 4 below.

### Task 8 — Correlation
Highest |correlation|: `purchase_amount` ↔ `delivery_days` (r ≈ 0.815). Possibly partially
causal (larger orders take longer to pick/pack/ship), but could be confounded by a third
variable — e.g. order complexity or multi-warehouse fulfillment routing.

### Task 9a — Imputation comparison
| Column | Mean | Median | Skew |
|---|---|---|---|
| `purchase_amount` | 43.920 | 34.830 | +4.284 |
| `quantity` | 4.101 | 4.000 | +3.865 |

Both positively skewed → median used for both, since the mean in each case is inflated by
the high-value tail. Confirmed zero remaining nulls in both columns after imputation.

### Task 9b — Spearman vs Pearson
| Pair | Spearman | Pearson | \|Diff\| |
|---|---|---|---|
| `quantity` vs `satisfaction_score` | −0.009 | −0.052 | 0.042 |
| `discount_percent` vs `quantity` | 0.002 | 0.044 | 0.041 |
| `is_returned` vs `purchase_amount` | 0.038 | −0.002 | 0.040 |

All three differences are small — these pairs show weak, largely noise-level relationships
rather than meaningful monotonic-nonlinear patterns. The Task 8 pair (`purchase_amount`–
`delivery_days`, Pearson ≈ 0.815) remains the dataset's one genuinely strong relationship
and the one I'd prioritize for feature selection in Part 2.

### Task 9c — Grouped aggregation
| Category | Mean | Std | Count |
|---|---|---|---|
| Books | 43.18 | 28.50 | 131 |
| Clothing | 40.00 | 24.49 | 211 |
| Electronics | 46.11 | 38.31 | 231 |
| Home | 43.96 | 32.84 | 150 |
| **Sports** | **49.27** | **62.83** | 77 |

Sports has both the highest mean and, by a wide margin, the highest std (62.83 vs
Electronics' next-highest 38.31) — high within-group variance means category alone can't
reliably predict a Sports order's value. Ratio of highest-to-lowest group mean ≈ **1.23**
— a modest ~23% gap, suggesting `product_category` carries some but not strong predictive
signal on its own.

## 4. Plot descriptions

All plots are saved as PNG files in `plots/`, produced via `plt.savefig()` directly from
the code (no external editing).

**1. `line_purchase_amount.png` — Line plot of purchase amount by row index**
Plots `purchase_amount` against the DataFrame's row index using `plt.plot()`. There's no
meaningful trend across the index (row order isn't chronological in this dataset), but the
plot makes the presence of extreme values immediately visible as sharp upward spikes
scattered through an otherwise low, flat baseline — a first visual hint of the outliers
confirmed later in Task 6.

**2. `bar_mean_purchase_by_category.png` — Bar chart of mean purchase amount by category**
Shows `df.groupby('product_category')['purchase_amount'].mean()` as a bar chart. Sports has
the tallest bar (highest average order value, ≈49.27), Clothing the shortest (≈40.00). This
visual matches and supports the Task 9c grouped-aggregation table.

**3. `histogram_top_skew.png` — Histogram of `purchase_amount` (the most skewed column)**
20-bin histogram (`sns.histplot`, `bins=20`) of `purchase_amount`, the column identified in
Task 5 as having the highest absolute skewness. The shape is clearly right-skewed: a tall
concentration of bars at low-to-moderate values, tapering into a long, thin, sparse tail
stretching toward high values — the visual signature of the +4.284 skew value reported
numerically in Task 5.

**4. `scatter_purchase_vs_delivery.png` — Scatter plot of purchase amount vs delivery days**
Plots `purchase_amount` (x) against `delivery_days` (y) via `sns.scatterplot`. Shows a
clear positive relationship, but the shape of the point cloud curves and flattens rather
than forming a straight line — consistent with the Task 8/9b finding that these two
variables are strongly correlated but in a mildly non-linear, monotonic way (delivery time
rises with order value but at a decreasing rate for larger orders).

**5. `box_purchase_by_category.png` — Box plot of purchase amount split by category**
`sns.boxplot` of `purchase_amount` grouped by `product_category`. Sports shows the widest
interquartile range and the most/highest-value outlier points above its whisker, visually
confirming the high standard deviation (62.83) found in Task 9c. Clothing's box is the
narrowest and sits lowest, matching its status as the lowest-mean, most tightly clustered
category.

**6. `correlation_heatmap.png` — Correlation heat map of all numeric columns**
`sns.heatmap(annot=True)` of the full Pearson correlation matrix from Task 8. Each cell is
color-coded and numerically annotated. The `purchase_amount`/`delivery_days` cell stands out
as the darkest off-diagonal cell (r ≈ 0.815) — the strongest relationship in the dataset by
a wide margin, with every other pair of variables shown as pale, near-zero-correlation cells.

## 5. Output files
- `cleaned_data.csv` — final cleaned dataset (800 rows × 11 columns)
- `plots/` — all 6 saved figures (5 required visualizations + correlation heat map)

## 6. How to run
\`\`\`bash
pip install pandas numpy matplotlib seaborn scipy
python part1_eda.py
\`\`\`
This single command regenerates the raw dataset, runs every task in order, prints all
reported statistics to the console, saves all plots to `./plots/`, and writes
`cleaned_data.csv`. No manual dataset download is required.
