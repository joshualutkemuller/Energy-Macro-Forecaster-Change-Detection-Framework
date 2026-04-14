# Change Detection & EDA Integration

**CSE 6242 Data & Visual Analytics · Spring 2026 · Team 132**
**File:** `dash_app_jl_with_cd.py` · **Branch:** `change-detection-implementation`

> This document covers what was built, the statistical methods and mathematics behind each feature, and a prioritized list of recommended additions for future sprints.

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [EDA Integration — Pre-Computed Pipeline Outputs](#2-eda-integration--pre-computed-pipeline-outputs)
   - 2.1 [Cross-Correlation Function (CCF)](#21-cross-correlation-function-ccf)
   - 2.2 [Lagged OLS Regression](#22-lagged-ols-regression)
   - 2.3 [Granger Causality](#23-granger-causality)
   - 2.4 [Pearson Correlation Heatmap](#24-pearson-correlation-heatmap)
3. [Change Detection Methods](#3-change-detection-methods)
   - 3.1 [CUSUM (Cumulative Sum Control Chart)](#31-cusum-cumulative-sum-control-chart)
   - 3.2 [PELT (Pruned Exact Linear Time)](#32-pelt-pruned-exact-linear-time)
4. [Key Empirical Findings from the Data](#4-key-empirical-findings-from-the-data)
5. [Recommended Additions — EDA](#5-recommended-additions--eda)
6. [Recommended Additions — Change Detection](#6-recommended-additions--change-detection)
7. [Implementation Notes](#7-implementation-notes)

---

## 1. What Was Built

`dash_app_jl_with_cd.py` is a feature-extended copy of `dash_app.py`. It adds two new full-width dashboard sections without modifying any existing behavior.

### Statistical Analysis Section

Loads the pre-computed CSV outputs from the `grp132_datawrangler` EDA pipeline and renders them as interactive Plotly charts:

| Component | Chart/Element | Data Source |
|-----------|---------------|-------------|
| Cross-correlation chart | Bar chart, lags −18 to +18, 95% CI bands | `outputs/ccf_wti_vs_*.csv` |
| Lagged regression chart | R² bars + beta line (dual-axis), lags 0–18 | `outputs/lagged_reg_wti_vs_*.csv` |
| Granger causality table | HTML table, F-stat and p-value per lag | `outputs/granger_wti_vs_*.csv` |
| Correlation heatmap | Interactive heatmap, 10 key variables | `outputs/corr_pearson_monthly.csv` |

All four update when the **Analysis Target** radio selector changes (rig count, production, fed funds, unemployment, or industrial production).

### Change Detection Section

Applies statistical change-point detection to the user's currently selected date window and chosen series (WTI price, rig count, or U.S. production):

| Component | Method | Controls | Output |
|-----------|--------|----------|--------|
| Main series chart | CUSUM or PELT | Series, method, sensitivity | Time series with regime-break vertical lines |
| Secondary statistics chart | CUSUM: S+/S− statistic | Threshold, drift | Cumulative sum plot with threshold line |
| | PELT: segmented means | Penalty | Colour-coded regime segments with mean lines |
| Summary text | Both | — | Detected break dates listed |

---

## 2. EDA Integration — Pre-Computed Pipeline Outputs

### 2.1 Cross-Correlation Function (CCF)

#### What it shows

The CCF answers: *At which lag does WTI price best predict (or follow) a target series?* Positive lags mean WTI leads the target; negative lags mean the target leads WTI.

#### Mathematics

For two mean-centred, stationary time series $\{x_t\}$ (WTI) and $\{y_t\}$ (target), the cross-correlation at lag $k$ is:

$$
\hat{\rho}_{xy}(k) = \frac{\hat{\gamma}_{xy}(k)}{\hat{\sigma}_x \, \hat{\sigma}_y}
$$

where the sample cross-covariance is:

$$
\hat{\gamma}_{xy}(k) =
\begin{cases}
\dfrac{1}{T} \displaystyle\sum_{t=1}^{T-k} (x_t - \bar{x})(y_{t+k} - \bar{y}) & k \geq 0 \\[10pt]
\hat{\gamma}_{yx}(-k) & k < 0
\end{cases}
$$

The **95% confidence interval** for a white-noise null hypothesis is:

$$
\text{CI}_{95} = \pm \frac{1.96}{\sqrt{T}}
$$

Bars exceeding this band are statistically significant at the 5% level. The pipeline computes this for lags $k \in [-18, +18]$ months using the full dataset ($T = 155$ monthly observations), giving $\text{CI}_{95} \approx \pm 0.157$.

#### Dashboard rendering

- Bars are coloured by significance: orange (positive, significant), red (negative, significant), grey (not significant)
- The darkest bar marks the lag with the highest absolute CCF
- Positive/negative CI lines are drawn as dotted horizontals
- A vertical line at lag 0 separates "WTI leads" from "target leads"

---

### 2.2 Lagged OLS Regression

#### What it shows

For each lag $k$, this fits a simple OLS regression of the target at time $t$ on WTI at time $t - k$. The R² curve shows how much variance in future target activity is explained by current WTI, as a function of the lead time.

#### Mathematics

At each lag $k \in \{0, 1, \ldots, 18\}$, estimate:

$$
y_{t+k} = \alpha_k + \beta_k \, x_t + \varepsilon_t
$$

by ordinary least squares, where:
- $y_{t+k}$ is the target series (e.g., rig count) at time $t + k$
- $x_t$ is WTI price at time $t$
- $\hat{\beta}_k$ is the estimated slope coefficient (e.g., rigs added per $1/bbl increase in WTI)
- $R^2_k$ is the coefficient of determination — the fraction of variance in $y_{t+k}$ explained by $x_t$

The OLS estimator is:

$$
\hat{\beta}_k = \frac{\sum_{t}(x_t - \bar{x})(y_{t+k} - \bar{y}_{t+k})}{\sum_{t}(x_t - \bar{x})^2}
$$

Statistical significance of $\hat{\beta}_k$ is tested with a $t$-statistic:

$$
t_k = \frac{\hat{\beta}_k}{\text{SE}(\hat{\beta}_k)}, \quad \text{SE}(\hat{\beta}_k) = \sqrt{\frac{\hat{\sigma}^2_\varepsilon}{\sum_t (x_t - \bar{x})^2}}
$$

under the null $H_0: \beta_k = 0$, using a $t_{n-2}$ reference distribution.

#### Dashboard rendering

- Left axis: R² bars, peak lag highlighted in dark orange
- Right axis: beta coefficient line, with filled circles for p < 0.05 and ✕ markers for non-significant lags

---

### 2.3 Granger Causality

#### What it shows

Granger (1969) causality tests whether past values of WTI improve forecasts of the target *beyond what the target's own history already explains*. A significant result does not prove economic causality — it means WTI has incremental predictive content.

#### Mathematics

At lag order $p$, estimate two models:

**Restricted model** (target's own history only):
$$
y_t = \alpha + \sum_{j=1}^{p} \phi_j \, y_{t-j} + \varepsilon_t
$$

**Unrestricted model** (target history + WTI history):
$$
y_t = \alpha + \sum_{j=1}^{p} \phi_j \, y_{t-j} + \sum_{j=1}^{p} \delta_j \, x_{t-j} + \varepsilon_t
$$

The $F$-test for the joint significance of the WTI lags is:

$$
F = \frac{(RSS_R - RSS_U) / p}{RSS_U / (T - 2p - 1)} \sim F_{p,\, T-2p-1}
$$

where $RSS_R$ and $RSS_U$ are the residual sums of squares from the restricted and unrestricted models, respectively.

**H₀:** $\delta_1 = \delta_2 = \cdots = \delta_p = 0$ (WTI does not Granger-cause the target).

Rejecting H₀ at $p < 0.05$ means past WTI values contain statistically significant predictive information about the target.

#### Dashboard rendering

Each row of the table shows the lag order, F-statistic, p-value, and a green "Reject H₀" badge or a grey "Fail to reject" badge. Rows update when the target selector changes.

---

### 2.4 Pearson Correlation Heatmap

#### What it shows

The symmetric Pearson correlation matrix for 10 key project variables computed on the full monthly dataset. Warm colours indicate positive correlation; cool colours indicate negative.

#### Mathematics

The Pearson correlation between two variables $X$ and $Y$ over $T$ observations:

$$
r_{XY} = \frac{\sum_{t=1}^{T}(X_t - \bar{X})(Y_t - \bar{Y})}{\sqrt{\sum_{t=1}^{T}(X_t - \bar{X})^2} \cdot \sqrt{\sum_{t=1}^{T}(Y_t - \bar{Y})^2}}
$$

This measures *linear* association, ranging from −1 (perfect negative) to +1 (perfect positive). Values between −0.157 and +0.157 are not statistically significant at 5% in this dataset ($\text{CI}_{95} = \pm 1.96/\sqrt{155}$).

---

## 3. Change Detection Methods

### 3.1 CUSUM (Cumulative Sum Control Chart)

#### Origin and purpose

Introduced by Page (1954) as an industrial quality-control procedure, CUSUM was adapted for economic time series by Brown, Durbin, and Evans (1975). It detects *persistent* mean shifts — not momentary spikes — making it well-suited to identifying when the oil price–drilling relationship changes regime.

#### Mathematics

Let $\{x_t\}_{t=1}^{T}$ be the raw time series. First standardise:

$$
z_t = \frac{x_t - \hat{\mu}}{\hat{\sigma}}
$$

where $\hat{\mu}$ and $\hat{\sigma}$ are the sample mean and standard deviation of the full window. Define two recursive statistics:

$$
S^+_t = \max\!\left(0,\; S^+_{t-1} + z_t - k\right)
$$

$$
S^-_t = \max\!\left(0,\; S^-_{t-1} - z_t - k\right)
$$

with $S^+_0 = S^-_0 = 0$.

- $S^+_t$ accumulates evidence of an **upward** shift
- $S^-_t$ accumulates evidence of a **downward** shift
- $k$ is the **drift (allowance) parameter** — the size of shift (in $\hat{\sigma}$ units) that the test should not react to. Typical values: $k \in [0.25, 1.0]$.

A **change point** is declared when either statistic exceeds the decision threshold $h$:

$$
S^+_t > h \quad \text{or} \quad S^-_t > h
$$

Upon detection, both statistics are reset to zero (the "resetting CUSUM"), and the search for the next change point continues.

#### Parameter guide

| Parameter | Dashboard control | Effect | Typical range |
|-----------|------------------|--------|---------------|
| $h$ (threshold) | "CUSUM Threshold" slider | Higher = fewer, larger breaks required | 3–8 for monthly economic series |
| $k$ (drift) | "CUSUM Drift" slider | Higher = ignores small persistent shifts | 0.25–1.0 |

**Interpreting the statistic chart:** When S+ or S− rises steeply and then resets, the series has undergone an upward or downward mean shift at the reset point. A flat statistic near zero indicates a stable regime.

#### Why CUSUM for this project

CUSUM is interpretable, requires no distributional assumptions beyond stationarity, and is computationally $O(T)$. The resetting form used here naturally handles multiple regime changes over a long history — exactly the pattern expected in oil markets across the 2014 price collapse, the 2020 COVID shock, and the 2021–2022 energy crisis.

---

### 3.2 PELT (Pruned Exact Linear Time)

#### Origin and purpose

Proposed by Killick, Fearnhead, and Eckley (2012), PELT solves the multiple change-point problem *exactly* (unlike greedy or window-based heuristics) with a computational cost that is linear in $T$ under a pruning condition. It finds the set of change points that minimises a penalised cost function, treating each segment between change points as a separate regime.

#### Mathematics

Let $\tau = \{\tau_1, \tau_2, \ldots, \tau_m\}$ be a set of $m$ change points (indices into the series). PELT seeks the partition that minimises:

$$
\min_{\tau} \left[ \sum_{i=0}^{m} \mathcal{C}(x_{(\tau_i+1):\tau_{i+1}}) + \beta \cdot m \right]
$$

where:
- $\mathcal{C}(\cdot)$ is a **cost function** measuring how poorly a single model fits a segment (the implementation uses an RBF/Gaussian kernel cost, which is more robust to non-normal distributions than a simple variance cost)
- $\beta$ is the **penalty constant** — it controls the trade-off between fit quality and model complexity (number of segments). Higher $\beta$ means fewer change points.

The RBF cost for a segment $y_{a:b}$ is defined via the kernel:

$$
\mathcal{C}(y_{a:b}) = -\sum_{i=a}^{b} \sum_{j=a}^{b} k(y_i, y_j)
$$

where $k(y_i, y_j) = \exp\!\left(-\tfrac{\|y_i - y_j\|^2}{2\sigma^2}\right)$.

PELT's efficiency comes from *pruning*: at each step, candidate change points that cannot be optimal for any future extension are permanently removed from consideration. Under mild conditions, the expected number of active candidates stays bounded even as $T \to \infty$, giving linear expected run time.

#### Parameter guide

| Parameter | Dashboard control | Effect |
|-----------|------------------|--------|
| $\beta$ (penalty) | "PELT Penalty" slider | Higher = fewer breaks, larger regimes. Start at 10–20 for monthly data and adjust until the number of regimes is meaningful |

#### Why PELT for this project

PELT is optimal (not approximate) and handles non-Gaussian series well with the RBF kernel — important for oil price data, which has fat tails and the famous April 2020 negative-price extreme. Unlike CUSUM, PELT simultaneously optimises the *number* and *locations* of all change points rather than processing them sequentially.

#### Requirement

PELT requires the `ruptures` Python package:

```bash
pip install ruptures
```

If `ruptures` is not installed, the dashboard displays an informational message and the CUSUM method remains fully functional.

---

## 4. Key Empirical Findings from the Data

These results come directly from the pre-computed `grp132_datawrangler` pipeline outputs (full dataset, 2013–present, 155 monthly observations).

### WTI → Rig Count

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Peak CCF lag | +4 months | WTI price leads rig count by approximately 4 months |
| Peak CCF value | r = 0.602 | Moderate-to-strong positive correlation at 4-month lead |
| Peak R² (lagged regression) | 0.418 at lag 4 | WTI alone explains ~42% of rig count variance 4 months out |
| Beta at peak lag | ~15.9 rigs per $/bbl | A $10/bbl WTI increase predicts ~159 additional rigs 4 months later |
| Granger causality | Significant at all lags 1–12 (all p < 0.01) | Strong predictive content: past WTI reliably improves rig count forecasts |

This directly supports the proposal's hypothesis of a 2–4 month lag between price changes and drilling activity changes.

### WTI → U.S. Production

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Peak CCF lag | +10–12 months | WTI leads production by ~10–12 months (much longer lag than rigs) |
| Peak CCF value | r ≈ 0.121 | Weak correlation — production is a slow-moving, capital-committed variable |
| Peak R² | < 0.02 at any lag | WTI alone explains very little production variance |
| Granger causality | Significant at all lags 1–12 | Incremental predictive content exists, but effect size is small |

The weak WTI–production correlation makes physical sense: once wells are drilled they produce regardless of short-term price moves. Production reflects decisions made 12–24 months earlier.

---

## 5. Recommended Additions — EDA

The following features are not yet in `dash_app_jl_with_cd.py` and are recommended for future sprints, ordered by estimated analytical value.

### 5.1 Asymmetric Response Analysis (High Priority)

The proposal explicitly asks whether companies *contract drilling faster than they expand*. This requires splitting WTI returns into positive and negative shocks and running the CCF and lagged regression separately on each subset.

**How to implement:**

```python
monthly_df["wti_shock_pos"] = monthly_df["wti_mom_pct"].clip(lower=0)
monthly_df["wti_shock_neg"] = monthly_df["wti_mom_pct"].clip(upper=0)
```

Then compute CCF and lagged OLS separately for positive vs. negative shocks against rig count. The Apergis, Ewing & Payne (2021) reference in the proposal found that negative price shocks cause faster rig reductions than positive shocks of the same magnitude cause rig additions. Surfacing this asymmetry directly would be a key contribution of the dashboard.

**Add to dashboard as:** A toggle "Shock type: All / Positive / Negative" in the Statistical Analysis section, switching the CCF and regression charts between the three subsets.

---

### 5.2 Rolling Correlation Over Time (High Priority)

The static CCF covers the full dataset but cannot show whether the WTI–rig relationship has *changed* over time. A rolling 36-month or 48-month Pearson correlation between WTI(t) and rig_count(t+4) plotted as a time series would reveal structural changes in the relationship strength.

**How to implement:**

```python
monthly_df["wti_lead4"] = monthly_df["wti_price_weekly"].shift(4)
monthly_df["rolling_corr"] = (
    monthly_df["wti_lead4"]
    .rolling(36)
    .corr(monthly_df["rig_count"])
)
```

**Add to dashboard as:** A new chart card showing the rolling correlation over time, with recession bands overlaid. Periods where the correlation drops near zero would indicate decoupling between price and drilling — a key finding for the proposal's threshold/asymmetry questions.

---

### 5.3 Regime-Conditioned Summary Statistics (Medium Priority)

Once CUSUM or PELT identifies regime breaks, compute and display summary statistics *within each regime* (mean WTI, mean rig count, mean production, mean macro variables). This transforms the change-point detection from a visual feature into an analytical one.

**What to show:** A table below the change-detection charts with one row per detected regime, showing the regime date range, observations count, mean and standard deviation of each key variable, and the percentage change from the previous regime.

---

### 5.4 Threshold Detection / Price Level Analysis (Medium Priority)

The proposal asks whether drilling expansion/contraction is triggered at a *specific price threshold*. This is distinct from detecting a change *in the time domain* — it asks whether there is a WTI price level above/below which the rig-count response changes.

**Approach:** Fit a piecewise linear regression (segmented regression) of rig count on WTI price:

$$
\text{rig}_t = \alpha + \beta_1 \cdot \text{WTI}_t + \beta_2 \cdot \max(0, \text{WTI}_t - \theta) + \varepsilon_t
$$

where $\theta$ is the unknown breakpoint (estimated via a grid search over the WTI price range). A significant $\hat{\beta}_2$ confirms a nonlinear response above threshold $\hat{\theta}$.

Alternatively, plot the scatter of rig count vs. WTI (already in the dashboard) with a LOESS smooth overlaid — visual non-linearity in the smooth would suggest a threshold effect without formal model fitting.

---

### 5.5 Vector Autoregression (VAR) Impulse Response (Medium Priority)

The proposal mentions VAR as the current industry standard. Adding a VAR impulse response function (IRF) to the dashboard would directly benchmark the project's approach against the literature.

A bivariate VAR(p) in WTI and rig count estimates how a one-standard-deviation shock to WTI propagates through the system over future periods. The IRF shows the expected path of rig count following such a shock, with confidence bands.

**Library:** `statsmodels.tsa.api.VAR` — no additional dependencies beyond what is already installed.

---

### 5.6 Volatility Regime Chart (Lower Priority)

Plot a 12-month rolling standard deviation of WTI returns (i.e., a simple historical volatility measure). High-volatility periods correspond to price uncertainty, which the proposal suggests may suppress drilling activity even when prices are elevated (companies need *stable* high prices, not volatile ones).

Overlaying this volatility series on the rig count chart would visually test the "price stabilisation" hypothesis in the proposal.

---

### 5.7 State-Level Lag Heterogeneity (Lower Priority)

Different states/basins have different cost structures and therefore different price responsiveness. The existing regional map shows end-of-window rig counts, but does not show *lag heterogeneity*.

**Extension:** For each selected state/basin, compute the lag at which WTI-to-state-rig-count CCF peaks, and colour the geographic bubbles by that optimal lag rather than by current rig count. Texas (low-cost Permian) would be expected to respond faster than higher-cost basins.

---

## 6. Recommended Additions — Change Detection

### 6.1 PELT on Returns / First Differences (High Priority)

The current implementation runs PELT on the raw level series (e.g., WTI price in $/bbl). Change-point detection on a non-stationary level series can produce spurious results because the cost function is sensitive to the overall trend. A more statistically principled approach is to run PELT on the **first-differenced or log-return series**:

```python
wti_returns = monthly_df["wti_price_weekly"].pct_change().dropna()
```

This makes the series approximately stationary before detecting changes in its variance or mean. The current approach works acceptably for visual purposes but should be refined for any quantitative results reported in the final paper.

---

### 6.2 Bayesian Online Change Point Detection (BOCPD) (High Priority)

BOCPD (Adams & MacKay, 2007) is a probabilistic alternative to CUSUM and PELT that outputs a *posterior probability of a change point at each time step* rather than a binary decision. This is more informative for a dashboard because it shows the *confidence* the model has in each detected break.

The `bayesian_changepoint_detection` Python package implements this. The output is a heatmap of run-length probabilities: each row is a time step and each column is a hypothesised run length (time since last change point). Bright spots on the diagonal indicate high-probability change points.

**Why it's valuable for this project:** The proposal's risk section notes the subjectivity in defining a "price spike." BOCPD replaces the subjective threshold with a calibrated posterior probability, providing a more defensible answer to the question "was this a structural break?"

---

### 6.3 CUSUM on Regression Residuals (Medium Priority)

The Brown, Durbin, and Evans (1975) formulation of CUSUM — cited in the proposal — applies the method not to the raw price series but to the **recursive residuals** of a regression model. This tests whether the *relationship* between WTI and rig count is stable over time, rather than whether WTI itself is stable.

**How to implement:** At each time $t$, fit an OLS regression of rig count on WTI using data up to $t - 1$, generate a one-step-ahead forecast residual, and cumulate these standardised residuals. A structural break in the regression relationship (e.g., the industry becoming less price-sensitive after the 2014 shale revolution) shows up as a drift in this CUSUM statistic.

This is a more direct test of the proposal's core hypothesis than applying CUSUM to prices alone.

---

### 6.4 Binary Segmentation as a Fallback (Lower Priority)

PELT requires `ruptures`. Binary segmentation (BS) is a faster, simpler alternative that is available in `ruptures` but can also be implemented in pure `numpy`. BS recursively finds the single best split point in a segment and repeats. It is not optimal like PELT but is computationally cheaper and produces visually similar results for the lag ranges relevant to this dataset.

Providing a BS fallback when `ruptures` is unavailable would make the PELT-equivalent feature accessible to users who have not installed the package.

---

### 6.5 Change-Point Annotation Against Historical Events (Lower Priority)

The proposal mentions "validating breakpoints against known historical events." A lookup table mapping approximate date ranges to annotated events would add context to any detected change points:

| Date range | Event |
|------------|-------|
| 2014-06 to 2016-02 | OPEC November 2014 no-cut decision; shale price war |
| 2020-02 to 2020-05 | COVID-19 demand collapse; April 2020 negative WTI |
| 2021-11 to 2022-06 | Post-COVID supply squeeze; Russia–Ukraine War |

When a detected change point falls within 2 months of a known event, annotate the vertical line with the event name. This would make the change-detection section directly useful for the "who cares?" narrative in the proposal.

---

## 7. Implementation Notes

### Running the dashboard

```bash
# From the project root
pip install ruptures   # optional — enables PELT
python dash_app_jl_with_cd.py
# Open http://127.0.0.1:8050
```

### File relationship

```
dash_app.py                    ← original, untouched
dash_app_jl_with_cd.py         ← this branch's working file
  ├── All original code
  ├── EDA_OUTPUTS dict (loads grp132_datawrangler/outputs/ CSVs at startup)
  ├── Statistical Analysis section  (new layout + callback)
  └── Change Detection section      (new layout + callback)
```

### Adding new EDA pipeline outputs to the dashboard

1. Add the CSV filename to the `EDA` dict in `dash_app_jl_with_cd.py`:
   ```python
   EDA["my_new_output"] = _load_eda_csv("my_new_output.csv")
   ```
2. Write a chart builder function following the pattern of `build_ccf_fig` or `build_lag_reg_fig`.
3. Add a `dcc.Graph` component to the layout and wire it to a callback.

### Dependencies added by this file

| Package | Use | Required? |
|---------|-----|-----------|
| `ruptures` | PELT change-point detection | Optional — dashboard degrades gracefully if missing |

No other new dependencies. All EDA is loaded from pre-computed CSVs rather than re-computed at runtime, so the dashboard starts quickly even on large datasets.

---

## References

- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2), 100–115.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *Journal of the Royal Statistical Society: Series B*, 37(2), 149–163.
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590–1598.
- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
- Apergis, N., Ewing, B. T., & Payne, J. E. (2021). The asymmetric relationship of oil prices and production on drilling rig trajectory. *Resources Policy*, 71, 101990.
- Khalifa, A., Caporin, M., & Hammoudeh, S. (2017). The relationship between oil prices and rig counts: The importance of lags. *Energy Economics*, 63, 213–226.
