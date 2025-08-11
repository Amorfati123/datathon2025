# ICU Sepsis Prediction with Temporal Fusion Transformer (TFT)

This repository contains a **teaching notebook** that walks through an end-to-end, reproducible pipeline for **early sepsis prediction from ICU vital signs**. We start with messy, irregular hourly measurements, build several **classical forecasting baselines** (ARIMA, Holt-Winters) to reason about missingness and short sequences, and then construct a **multivariate sequence classification** model using the **Temporal Fusion Transformer (TFT)**. Throughout, the notebook emphasizes **transparent preprocessing**, **sound evaluation**, and **in-notebook diagnostics** you can reuse in other clinical time-series projects.

> This work was produced as part of a grant-funded education effort aligned with the CIMDAR-HIVE vision: building practical capacity for multimodal health AI in resource-limited institutions through hands-on, datathon-style learning.

---

## Key highlights

- **Data shaping for sequences:**  
  - Harmonizes each ICU stay to a **fixed 24-hour window** (`hour = 0..23`) using robust trailing forecasting.  
  - Handles missingness via **Forward Fill**, **Linear Interpolation**, and **Mean Imputation**, then **ARIMA-based extension** for late-hour gaps.  
  - Guarantees **no NaNs** in the vital features before modeling.

- **Classical forecasting sanity checks:**  
  - Compares **ARIMA** and **Holt-Winters** on a per-stay, per-vital basis for the final 6 hours (train: hours 0–17; test: 18–23).  
  - Reports **MAE, RMSE, MAPE, R²** and produces interpretable plots that make missing-data decisions explicit.

- **Baseline classifiers:**  
  - **Logistic Regression** and **Random Forest** trained on engineered features with **Robust scaling** (median/IQR) to dampen outliers.  
  - Includes **5-fold stratified CV** with `AUC`, `Accuracy`, and `F1`.

- **Deep learning (TFT):**  
  - Builds `TimeSeriesDataSet` objects (encoder: 22 hours; decoder: 2 hours) with **stay-level groupings** to prevent leakage.  
  - Trains **Temporal Fusion Transformer** with PyTorch Forecasting + Lightning, **early stopping** on validation loss, and thorough evaluation: **AUROC, AUPRC, Precision/Recall, Specificity, F1, Brier score**, and diagnostic curves (ROC/PR/Calibration/Confusion Matrix).

---

## Data (expected schema)

The notebook expects a de-identified ICU cohort organized as hourly rows per stay:

- **Columns (time-varying):**  
  `stay_id, hour, heart_rate, sbp, dbp, mbp, resp_rate, spo2, temperature, charttime (optional)`
- **Label:**  
  `sepsis` (binary; assigned per stay).  
- The teaching notebook balances the cohort for pedagogy (≈ equal sepsis / non-sepsis examples).

> You don’t need raw EHR access to run the notebook; it operates on pre-aggregated hourly tables. Replace the sample DataFrames with your own as long as you keep the column names.

---

## Pipeline overview

1. **Profiling missingness and coverage**
   - Counts stays with `< 24` hours and identifies starts with missing vitals.

2. **Imputation + trailing forecasting to 24h**
   - For each stay & vital:
     - Fill intra-series gaps (FFill / Linear / Mean).
     - If late hours are missing, **fit ARIMA** on observed prefix and **forecast to hour 23**.
     - Final **safety net:** forward/back fill then global median to remove any residual NaNs.

3. **Classical forecasting baselines**
   - ARIMA(1,1,1) and Holt-Winters (additive trend) on the imputed series (train 0–17, forecast 18–23).
   - Visualizes **Observed (train/test)** vs **Forecasts** per imputation method to explain behavior.

4. **Baseline classification**
   - Train/test split at **stay level** (no leakage).
   - **RobustScaler** (median/IQR) **fit on train only**, applied to train & test.
   - Evaluate **LogReg** and **Random Forest**; then **5-fold stratified CV** for stability.

5. **TFT modeling**
   - `TimeSeriesDataSet` with:
     - `group_ids = ["stay_id_int"]`
     - `time_idx = "time_idx"` (0–23)
     - `target = "sepsis"`
     - `time_varying_unknown_reals = [vitals...]`
     - `time_varying_known_reals = ["time_idx"]`
   - TFT hyperparams (teaching defaults):  
     `learning_rate=1e-3, hidden_size=16, attention_head_size=2, dropout=0.1, hidden_continuous_size=8, loss=CrossEntropy()`
   - **EarlyStopping:** monitor `val_loss`, `patience=5`.

6. **Evaluation & diagnostics**
   - **AUROC, AUPRC, Accuracy, Precision, Recall, Specificity, F1, Brier**.
   - **ROC, PR, Confusion Matrix, Calibration** plots.

---

## Results (representative)

### Baseline classifiers (test set, scaled features)

| Model                | Accuracy | AUC   | F1   |
|---------------------|---------:|------:|-----:|
| Logistic Regression | 0.653    | 0.716 | 0.650 |
| Random Forest       | 0.678    | 0.738 | 0.682 |

**5-fold CV (mean ± std AUC):**
- Logistic Regression: **0.721 ± 0.007**  
- Random Forest: **0.741 ± 0.007**

### TFT (validation set, last step)

- **AUROC:** 0.734  
- **AUPRC:** 0.714  
- **Accuracy:** 0.671  
- **Precision:** 0.731  
- **Recall (Sensitivity):** 0.538  
- **Specificity:** 0.803  
- **F1:** 0.620  
- **Brier score:** 0.206

> Takeaway: TFT shows **strong discrimination (AUROC/AUPRC)** and **good calibration** (Brier), with a **precision-leaning operating point** (higher precision, moderate recall). Threshold tuning or class-aware losses can shift this balance if higher sensitivity is required clinically.

### Forecasting sanity checks (per-vital)
- **Heart Rate:** ARIMA generally outperforms Holt-Winters across imputations.  
- **SBP:** Best MAE sometimes with **Mean Imputation + ARIMA**, but **R² can be negative** on short/noisy series—evidence that classical univariate models struggle for some vitals.

---

## Reproducibility
- Stay-level split with random_state=42 ensures the same admissions stay in the same fold across runs.

- TFT uses Lightning’s Trainer; we enable early stopping for stable comparisons.

- If you want fully deterministic training, set deterministic=True and seed builders; note this can reduce speed and may change results slightly.

## Design choices & lessons
- Why ARIMA pre-forecast? Medical time series often end early (transfers, discharge). TFT expects contiguous sequences; trailing ARIMA fills give a principled way to reach 24h while minimizing leakage.

- Why RobustScaler, not StandardScaler? Vital signs can be heavy-tailed. Median/IQR scaling reduces the effect of outliers on linear models.

- Batch size matters: Very small batches (e.g., 1) degrade batch-norm/optimizer statistics and can reduce TFT performance; moderate batch sizes (e.g., 128–256) performed better in our tests.

## Limitations & next steps
- Sensitivity vs Precision: Current threshold favors precision. Consider threshold tuning, class weights, or focal loss to increase recall if clinically required.

- Feature set: We model vitals only. Adding labs, demographics, interventions, or textual notes (multimodal) may improve performance.

- Temporal labeling: We predict sepsis status using the first 24h. Alternative labeling strategies (e.g., horizon-based, event-time alignment) can be explored.

## Acknowledgments
- Educational framing inspired by the CIMDAR-HIVE initiative to democratize multimodal health AI through short courses and datathons.

- Built with PyTorch Forecasting, PyTorch Lightning, statsmodels, and scikit-learn.

