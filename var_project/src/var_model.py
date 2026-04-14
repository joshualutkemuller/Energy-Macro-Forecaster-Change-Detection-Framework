import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
file_path = Path("data/master_monthly.csv")
if not file_path.exists():
   raise FileNotFoundError(
       f"Could not find {file_path.resolve()}. "
       "Make sure master_monthly.csv is inside the data folder."
   )
df = pd.read_csv(file_path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")
df = df[~df.index.isna()].sort_index()
print("Dataset shape:", df.shape)
required_cols = ["wti_price_weekly", "rig_count", "indpro"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
   raise KeyError(f"Missing required columns: {missing_cols}")
data = df[required_cols].copy()
for col in required_cols:
   data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.dropna()
def adf_test(series, name):
   series = series.dropna()
   result = adfuller(series, autolag="AIC")
   print(f"\nADF Test: {name}")
   print(f"Test Statistic: {result[0]:.4f}")
   print(f"p-value: {result[1]:.4f}")
   if result[1] < 0.05:
       print("Result: Stationary")
   else:
       print("Result: Not stationary")
adf_test(data["wti_price_weekly"], "WTI Price (level)")
adf_test(data["rig_count"], "Rig Count (level)")
adf_test(data["indpro"], "INDPRO (level)")
transformed = pd.DataFrame(index=data.index)
transformed["wti_mom_pct"] = data["wti_price_weekly"].pct_change() * 100
transformed["rig_mom_pct"] = data["rig_count"].pct_change() * 100
transformed["indpro_mom_pct"] = data["indpro"].pct_change() * 100
transformed = transformed.replace([float("inf"), float("-inf")], pd.NA).dropna()
adf_test(transformed["wti_mom_pct"], "WTI % Change")
adf_test(transformed["rig_mom_pct"], "Rig % Change")
adf_test(transformed["indpro_mom_pct"], "INDPRO % Change")
var_data = transformed[["wti_mom_pct", "rig_mom_pct", "indpro_mom_pct"]]
model = VAR(var_data)
maxlags = min(6, max(1, len(var_data) // 5))
lag_selection = model.select_order(maxlags=maxlags)
selected_lag = lag_selection.aic
if selected_lag is None or not isinstance(selected_lag, int) or selected_lag < 1:
   selected_lag = 1
print(f"\nSelected lag: {selected_lag}")
results = model.fit(selected_lag)
print("\nVAR Results Summary:")
print(results.summary())
irf = results.irf(12)
irf.plot(response="rig_mom_pct", impulse="wti_mom_pct", orth=False)
plt.tight_layout()
plt.show()
forecast_steps = 12
lag_order = results.k_ar
forecast_input = var_data.values[-lag_order:]
forecast_values = results.forecast(forecast_input, steps=forecast_steps)
last_date = var_data.index[-1]
future_index = pd.date_range(start=last_date, periods=forecast_steps + 1, freq="MS")[1:]
forecast_df = pd.DataFrame(
   forecast_values,
   index=future_index,
   columns=var_data.columns
)
print("\n12-Month Forecast:")
print(forecast_df)
plt.figure(figsize=(10, 5))
plt.plot(var_data.index, var_data["rig_mom_pct"], label="Historical")
plt.plot(forecast_df.index, forecast_df["rig_mom_pct"], label="Forecast")
plt.title("Rig Count Forecast (% Change)")
plt.xlabel("Date")
plt.ylabel("Percent Change")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
transformed.to_csv("outputs/var_transformed_data.csv")
forecast_df.to_csv("outputs/var_forecast_12m.csv")
with open("outputs/var_summary.txt", "w", encoding="utf-8") as f:
   f.write("LAG ORDER SELECTION\n")
   f.write(str(lag_selection.summary()))
   f.write("\n\nVAR MODEL SUMMARY\n")
   f.write(str(results.summary()))
print("\nSaved files to outputs/")
