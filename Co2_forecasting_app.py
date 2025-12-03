import io
import math
import requests
import joblib
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

sns.set(style="whitegrid")

OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

# -------------------------
# CACHING & LOADERS
# -------------------------
@st.cache_data(show_spinner=True)
def load_owid() -> pd.DataFrame:
    r = requests.get(OWID_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

@st.cache_resource
def load_rf_model(path: str = "Tuned_rf_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load RF model from {path}: {e}")
        return None

@st.cache_resource
def load_encoder(path: str = "encoder.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None  # encoder is optional for the current classifier inputs

# -------------------------
# HELPERS
# -------------------------
def choose_scale(actual_values, forecast_values):
    options = {"tonnes": 1.0, "kilotons": 1000.0}
    best_scale, best_diff = 1.0, float("inf")
    for scale in options.values():
        diff = abs(np.median(actual_values) - np.median(forecast_values * scale))
        if diff < best_diff:
            best_scale, best_diff = scale, diff
    return best_scale

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100.0
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def plot_timeseries(actual_df, forecast_df, country, unit_label):
    plt.figure(figsize=(10, 4.5))
    plt.plot(actual_df["year"], actual_df["actual"], marker="o", label="Actual")
    plt.plot(forecast_df["year"], forecast_df["forecast"], marker="X", linestyle="--", label="Forecast")
    plt.xlabel("Year")
    plt.ylabel(f"CO₂ ({unit_label})")
    plt.title(f"Actual vs Forecast — {country}")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

def plot_residuals(val_df):
    plt.figure(figsize=(10, 3))
    sns.barplot(x="year", y="residual", data=val_df)
    plt.axhline(0, linestyle="--", color="k")
    plt.xticks(rotation=45)
    plt.title("Residuals (Actual − Predicted)")
    st.pyplot(plt.gcf())
    plt.close()

def plot_parity(val_df):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x="actual", y="forecast", data=val_df)
    mn = min(val_df["actual"].min(), val_df["forecast"].min())
    mx = max(val_df["actual"].max(), val_df["forecast"].max())
    plt.plot([mn, mx], [mn, mx], "--r")
    plt.title("Actual vs Predicted (parity)")
    st.pyplot(plt.gcf())
    plt.close()

# -------------------------
# UI Layout
# -------------------------
st.set_page_config(page_title="CO₂ Dashboard", layout="wide")
st.title("🌍 CO₂ Forecast Validator & Dashboard")

owid = load_owid()

# --- Sidebar: global options ---
st.sidebar.header("Global Settings")
countries = sorted(owid["country"].dropna().unique())
default_index = countries.index("India") if "India" in countries else 0
country = st.sidebar.selectbox("Country (for validation)", countries, index=default_index)

unit_choice = st.sidebar.radio("Reporting unit", ["Metric Tons (t)", "Kilotons (kt)"])
use_prophet = st.sidebar.checkbox("Generate forecast with Prophet", value=True)
horizon = st.sidebar.number_input("Prophet forecast horizon (years)", min_value=1, max_value=50, value=10)
uploaded_forecast = st.sidebar.file_uploader("Upload forecast CSV (optional)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Classifier (your saved models)")
rf_model = load_rf_model("Tuned_rf_model.pkl")
encoder = load_encoder("encoder.pkl")
if rf_model is None:
    st.sidebar.error("RF model not loaded — classifier will be disabled.")
else:
    st.sidebar.success("RandomForest model loaded.")

# -------------------------
# Section 1 — Forecast validation
# -------------------------
st.header("1) Forecast Validator")

# Prepare actuals for selected country
df_country = owid[owid["country"] == country][["year", "co2"]].dropna()
if df_country.empty:
    st.error(f"No OWID data for {country}. Choose another country.")
else:
    df_country = df_country[df_country["year"] >= 1970].copy()
    df_country["year"] = df_country["year"].astype(int)
    # unit handling
    if unit_choice.endswith("(kt)"):
        df_country["actual"] = df_country["co2"] / 1000.0
        unit_label = "kt"
    else:
        df_country["actual"] = df_country["co2"]
        unit_label = "t"

    st.sidebar.markdown(f"Available actual years for {country}: {df_country['year'].min()} — {df_country['year'].max()}")

    # Build or load forecast
    if use_prophet:
        st.info("Fitting Prophet on OWID 'co2' series (may take a few seconds).")
        df_ts = df_country[["year", "actual"]].rename(columns={"year": "ds", "actual": "y"})
        df_ts["ds"] = pd.to_datetime(df_ts["ds"].astype(str) + "-01-01")
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        try:
            model.fit(df_ts[["ds", "y"]])
            future = model.make_future_dataframe(periods=horizon, freq="Y")
            forecast = model.predict(future)
            df_fc = forecast[["ds", "yhat"]].copy()
            df_fc["year"] = df_fc["ds"].dt.year
            my_forecast = df_fc[["ds", "year", "yhat"]].copy()
            st.success("Prophet forecast generated.")
        except Exception as e:
            st.error(f"Prophet fitting failed: {e}")
            st.stop()
    else:
        if uploaded_forecast is None:
            st.warning("Upload a forecast CSV with columns 'ds' and 'yhat' or enable Prophet.")
            st.stop()
        my_forecast = pd.read_csv(uploaded_forecast, parse_dates=["ds"])
        if "yhat" not in my_forecast.columns:
            st.error("Uploaded CSV must contain 'ds' and 'yhat' columns.")
            st.stop()
        my_forecast["year"] = pd.to_datetime(my_forecast["ds"]).dt.year

    st.subheader(f"Forecast preview for {country}")
    st.dataframe(my_forecast.tail(10))

    # Merge on year and validate
    merged = pd.merge(my_forecast, df_country[["year", "actual"]], on="year", how="inner").sort_values("year")
    if merged.empty:
        st.warning("No overlapping years between forecast and actuals — cannot compute validation metrics.")
    else:
        # choose scale to convert forecast to OWID units (tonnes)
        scale = choose_scale(merged["actual"].values, merged["yhat"].values)
        merged["forecast"] = merged["yhat"] * scale  # forecast in metric tonnes
        # if user requested reporting in kt, convert later for display
        metrics = compute_metrics(merged["actual"].values, merged["forecast"].values)

        st.subheader("Validation Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics['MAE']:.2f} t")
        col2.metric("RMSE", f"{metrics['RMSE']:.2f} t")
        col3.metric("MAPE", f"{metrics['MAPE']:.2f} %")
        col4.metric("R²", f"{metrics['R2']:.4f}")

        # Prepare display table in chosen unit
        display_df = merged[["year", "actual", "forecast"]].copy()
        if unit_label == "kt":
            display_df["actual_report"] = display_df["actual"] / 1000.0
            display_df["forecast_report"] = display_df["forecast"] / 1000.0
        else:
            display_df["actual_report"] = display_df["actual"]
            display_df["forecast_report"] = display_df["forecast"]

        st.subheader("Forecast vs Actual (overlapping years)")
        st.dataframe(display_df[["year", "actual_report", "forecast_report"]].set_index("year").rename(columns={
            "actual_report": f"Actual ({unit_label})",
            "forecast_report": f"Forecast ({unit_label})"
        }))

        # Plots
        plot_timeseries(df_country.rename(columns={"co2": "co2"}), merged.assign(forecast=merged["forecast"]), country, unit_label)
        merged["residual"] = merged["actual"] - merged["forecast"]
        merged["abs_pct_err"] = (merged["residual"].abs() / (merged["actual"] + 1e-9)) * 100.0
        plot_residuals(merged)
        plot_parity(merged)

        st.download_button("Download merged forecast vs actual", merged.to_csv(index=False), file_name=f"merged_{country}.csv", mime="text/csv")

# -------------------------
# Section 2 — Global Map + Top Emitters + Region filter
# -------------------------
st.header("2) Global Map & Exploratory Charts")

# Filter OWID to country-level (iso_code length == 3) safely:
owid_copy = owid.copy()
owid_copy["iso_code"] = owid_copy["iso_code"].astype(str).fillna("")
owid_countries = owid_copy[owid_copy["iso_code"].str.len() == 3].copy()

# Year slider for the map
years_available = sorted(owid_countries["year"].dropna().unique().tolist())
if years_available:
    default_map_year = max(years_available)
else:
    default_map_year = 2020
map_year = st.slider("Map Year", min_value=int(min(years_available)), max_value=int(max(years_available)), value=int(default_map_year), step=1) if years_available else st.slider("Map Year", 1970, 2024, 2020)

df_map = owid_countries[owid_countries["year"] == map_year]

st.subheader(f"Global CO₂ Emissions — {map_year}")
fig = px.choropleth(
    df_map,
    locations="iso_code",
    color="co2",
    hover_name="country",
    color_continuous_scale="Reds",
    labels={"co2": "CO₂ (metric tonnes)"},
)
fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# Top-10 emitters (bar chart)
st.subheader(f"Top 10 Emitters — {map_year}")
df_top10 = df_map.sort_values("co2", ascending=False).head(10)
if not df_top10.empty:
    fig2 = px.bar(df_top10, x="country", y="co2", text="co2", title=f"Top 10 CO₂ Emitters — {map_year}", color="co2", color_continuous_scale="Reds")
    fig2.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig2.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No data for top-10 emitters for this year.")

# Region-wise filtering (if available)
st.subheader("Region-wise View")
if "region" in owid.columns:
    regions = sorted(owid["region"].dropna().unique().tolist())
    region_sel = st.selectbox("Select region", ["All"] + regions)
    if region_sel == "All":
        df_region = owid_countries[owid_countries["year"] == map_year]
    else:
        df_region = owid_countries[(owid_countries["region"] == region_sel) & (owid_countries["year"] == map_year)]
    if df_region.empty:
        st.warning("No data for this region/year.")
    else:
        fig3 = px.bar(df_region.sort_values("co2", ascending=False).head(50), x="country", y="co2", title=f"CO₂ in {region_sel} — {map_year}", color="co2", color_continuous_scale="Reds")
        fig3.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Region column not present in OWID dataset version; skipping region charts.")

# Time-lapse animated map
st.subheader("CO₂ Emissions Over Time (Animated)")
try:
    anim = px.choropleth(
        owid_countries,
        locations="iso_code",
        color="co2",
        hover_name="country",
        animation_frame="year",
        color_continuous_scale="Reds",
        labels={"co2": "CO₂ (t)"}
    )
    anim.update_layout(height=650)
    st.plotly_chart(anim, use_container_width=True)
except Exception as e:
    st.warning(f"Animated map could not be created: {e}")

# # -------------------------
# # Section 3 — Simple Classifier UI (your saved RF)
# # -------------------------
# st.header("3) CO₂ Emission Classifier (Saved RandomForest)")

# if rf_model is None:
#     st.warning("RandomForest model not available — skip classifier section.")
# else:
#     st.write("Enter sample features (model was trained on 1-person rows: Kilotons of Co2, Metric Tons Per Capita).")
#     sample_co2_kt = st.number_input("Kilotons of CO₂ (kt)", min_value=0.0, value=1000.0, format="%.2f")
#     sample_percap = st.number_input("Metric Tons Per Capita", min_value=0.0, value=1.0, format="%.3f")

#     if st.button("Predict (RF)"):
#         # Prepare input exactly like training: if your model expects 'Kilotons of Co2' in kt and 'Metric Tons Per Capita'
#         input_df = pd.DataFrame([[sample_co2_kt, sample_percap]], columns=["Kilotons of Co2", "Metric Tons Per Capita"])
#         try:
#             pred = rf_model.predict(input_df)[0]
#             prob = rf_model.predict_proba(input_df)[0][1] if hasattr(rf_model, "predict_proba") else None
#             if pred == 1:
#                 st.success(f"🌡 HIGH CO₂ Emissions (prob: {prob:.2f})" if prob is not None else "🌡 HIGH CO₂ Emissions")
#             else:
#                 st.info(f"💧 LOW CO₂ Emissions (prob: {prob:.2f})" if prob is not None else "💧 LOW CO₂ Emissions")
#         except Exception as e:
#             st.error(f"Prediction failed — check model input format. Error: {e}")

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.caption("Data source: Our World in Data (OWID). Prophet used for quick demo forecasting. For robust production forecasts use proper preprocessing, per-country model tuning, train/validation splits, and uncertainty estimation.")
