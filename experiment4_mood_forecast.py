# experiment4_mood_forecast.py
# AI for Mental Health – Mood Forecast (Prophet + Plotly)
# Works in GitHub Codespaces / local VS Code / Colab (no Streamlit needed)

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Settings
# -----------------------------
USE_SAMPLE = True        # set to False to read mood.csv (columns: date,mood_score)
YEARS_TO_PREDICT = 2     # 1..4 typically
FORECAST_OUT = "forecast_results.csv"

# -----------------------------
# Load data
# -----------------------------
if USE_SAMPLE:
    # 3 years of synthetic mood data (0–100), with weekly & yearly cycles + noise
    rng = pd.date_range("2022-01-01", periods=3 * 365, freq="D")
    weekly = 5 * np.sin(2 * np.pi * (rng.dayofweek) / 7)
    yearly = 6 * np.sin(2 * np.pi * (rng.dayofyear) / 365.25)
    noise = np.random.normal(0, 3, len(rng))
    mood = np.clip(70 + weekly + yearly + noise, 40, 95).round(1)
    df = pd.DataFrame({"date": rng, "mood_score": mood})
else:
    # Expect a file called mood.csv in the same folder
    # with columns: date,mood_score
    df = pd.read_csv("mood.csv")
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df or "mood_score" not in df:
        raise ValueError("CSV must have columns: date,mood_score")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ignore_index=True)

print(f"Rows: {len(df)}")
print("Raw data (last 10):")
print(df.tail(10))

# -----------------------------
# Prepare Prophet dataframe
# -----------------------------
df_p = df.rename(columns={"date": "ds", "mood_score": "y"})

# Build & fit Prophet model
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.5
)
m.fit(df_p)

# Create future dataframe
future = m.make_future_dataframe(periods=YEARS_TO_PREDICT * 365, freq="D")
forecast = m.predict(future)

# -----------------------------
# Interactive time-series with rangeslider
# -----------------------------
print("\nOpening interactive plots...")

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df["date"], y=df["mood_score"],
    mode="lines", name="mood_score", line=dict(color="#ff6b6b")
))

# 14-day moving average for clarity
df_ma = df.copy()
df_ma["ma14"] = df_ma["mood_score"].rolling(14).mean()
fig_ts.add_trace(go.Scatter(
    x=df_ma["date"], y=df_ma["ma14"],
    mode="lines", name="14d MA", line=dict(color="#4dabf7")
))

fig_ts.update_layout(
    title="Mood time series (with rangeslider)",
    xaxis=dict(rangeslider=dict(visible=True)),
    xaxis_title="Date",
    yaxis_title="Mood (0–100)",
    height=450,
)
fig_ts.show()

# -----------------------------
# Interactive forecast plot (Prophet)
# -----------------------------
fig_forecast = plot_plotly(m, forecast)  # yhat + confidence intervals
fig_forecast.update_layout(
    title=f"Mood Forecast — {YEARS_TO_PREDICT} year(s) ahead",
    xaxis_title="Date",
    yaxis_title="Mood (forecast)",
    height=500
)
fig_forecast.show()

# -----------------------------
# Forecast table preview + save
# -----------------------------
cols = ["ds", "yhat_lower", "yhat", "yhat_upper", "trend"]
print("\nForecast table (last 10 rows):")
print(forecast[cols].tail(10))

forecast[cols].to_csv(FORECAST_OUT, index=False)
print(f"\n✅ Forecast saved to: {FORECAST_OUT}")

# -----------------------------
# Quick KPIs
# -----------------------------
last_actual = float(df["mood_score"].iloc[-1])
last_pred = float(forecast["yhat"].iloc[-1])
delta = last_pred - last_actual
print(f"\nLast actual mood: {last_actual:.1f}")
print(f"Forecast at horizon end: {last_pred:.1f}")
print(f"Δ (forecast - actual): {delta:+.1f}")
