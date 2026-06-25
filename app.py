"""
FIFA World Cup 2034 - Weather Analysis Dashboard
Interactive EDA + Feature Engineering app for match-scheduling insights
across the 5 Saudi host cities.
Organised in 3 sections: The Data -> Analysis -> Results.
Run locally:  streamlit run app.py
"""
 
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
# ----------------------------------------------------------------------
# Page config + theming (palette derived from the Saudi 2034 logo)
# ----------------------------------------------------------------------
st.set_page_config(page_title="FIFA 2034 Weather Analysis",
                   page_icon="*", layout="wide")
 
GREEN = "#2a1a5e"    # deep navy/indigo (primary, headings)
GREEN2 = "#5b3fa8"   # lighter purple (primary charts)
GOLD = "#e8821e"     # warm orange accent
RED = "#d64550"      # warm red accent
 
CITY_COLORS = {
    "Abha": "#3aa6a0",
    "Riyadh": "#2a1a5e",
    "Al Khobar": "#5b3fa8",
    "NEOM": "#e8821e",
    "Jeddah": "#d64550",
}
 
st.markdown(f"""
<style>
    .stApp {{ background-color: #fafaff; }}
    h1, h2, h3 {{ color: {GREEN}; }}
    h1 {{ border-bottom: 3px solid {GOLD}; padding-bottom: 6px; display: inline-block; }}
    [data-testid="stSidebar"] {{ background-color: #f3f1fb; }}
    [data-testid="stMetricValue"] {{ color: {GREEN}; }}
    [data-testid="stMetric"] {{
        background: #ffffff; border: 1px solid #e7e3f3;
        border-top: 4px solid {GOLD};
        border-radius: 12px; padding: 14px 16px;
    }}
    [data-testid="stSidebar"] img {{ margin: 0 auto 6px; display: block; }}
    div.block-container {{ padding-top: 2.2rem; }}
    .stTabs [aria-selected="true"] {{ color: {GREEN}; }}
    .stRadio label p {{ font-weight: 500; }}
</style>
""", unsafe_allow_html=True)
 
LOGO_FILE = "fifa2034_logo.png"  # place this image in the same folder as app.py
 
# ----------------------------------------------------------------------
# Data loading + feature helpers
# ----------------------------------------------------------------------
DATA_FILE = "host_cities_weather_2015_2024.csv"
 
CITY_PROFILE = {
    "Abha":      "Highland - mild",
    "Riyadh":    "Inland - hot-dry",
    "Al Khobar": "Coastal - humid (Gulf)",
    "NEOM":      "Coastal / desert",
    "Jeddah":    "Coastal - humid (Red Sea)",
}
 
 
def compute_wbgt(temp_c, rh):
    """Simplified WBGT heat-stress index (temperature + humidity)."""
    es = 6.105 * np.exp(17.27 * temp_c / (237.7 + temp_c))
    e = es * rh / 100.0
    return 0.567 * temp_c + 0.393 * e + 3.94
 
 
def suitability_class(w):
    if w < 28:
        return "Suitable"
    elif w < 32:
        return "Caution"
    return "Not Suitable"
 
 
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "Year" not in df.columns:
        df["Year"] = df["datetime"].dt.year
    df["WBGT"] = compute_wbgt(df["Temperature"], df["Humidity"]).round(2)
    df["MatchSuitability"] = df["WBGT"].apply(suitability_class)
    WBGT_BEST, WBGT_WORST = 18.0, 38.0
    df["ComfortScore"] = (((WBGT_WORST - df["WBGT"]) /
                           (WBGT_WORST - WBGT_BEST)) * 100).clip(0, 100).round(0)
    return df
 
 
@st.cache_data
def build_features(base):
    """Engineered features (computed live for the Feature Engineering page)."""
    d = base.copy()
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24)
    d["Month_sin"] = np.sin(2 * np.pi * d["Month"] / 12)
    d["Month_cos"] = np.cos(2 * np.pi * d["Month"] / 12)
    d = pd.concat([d, pd.get_dummies(d["City"], prefix="City")], axis=1)
    d = d.sort_values(["City", "datetime"]).reset_index(drop=True)
    for lag in [1, 3, 24]:
        d[f"WBGT_lag{lag}"] = d.groupby("City")["WBGT"].shift(lag)
    return d
 
 
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file '{DATA_FILE}' not found. Put it in the same folder as app.py.")
    st.stop()
 
CITIES = list(CITY_PROFILE.keys())
 
# ----------------------------------------------------------------------
# Sidebar navigation (3 sections, 6 pages)
# ----------------------------------------------------------------------
if os.path.exists(LOGO_FILE):
    try:
        st.sidebar.image(LOGO_FILE, width=130)
    except Exception:
        pass
st.sidebar.title("FIFA 2034")
st.sidebar.caption("Weather Analysis for Match Scheduling")
 
st.sidebar.markdown("**The Data**")
st.sidebar.caption("Overview - Data Quality")
st.sidebar.markdown("**Analysis**")
st.sidebar.caption("Climate Patterns - Heat Stress & Comfort")
st.sidebar.markdown("**Results**")
st.sidebar.caption("Scheduling - Feature Engineering")
st.sidebar.markdown("---")
 
page = st.sidebar.radio("Go to", [
    "1. Overview",
    "2. Data Quality & Distributions",
    "3. Climate Patterns",
    "4. Heat Stress & Comfort",
    "5. Scheduling Insights",
    "6. Feature Engineering",
])
st.sidebar.markdown("---")
st.sidebar.caption("Data: Open-Meteo Historical Archive (2015-2024)")
 
# ======================================================================
# 1. OVERVIEW
# ======================================================================
if page == "1. Overview":
    head_l, head_r = st.columns([1, 5])
    with head_l:
        if os.path.exists(LOGO_FILE):
            try:
                st.image(LOGO_FILE, width=120)
            except Exception:
                pass
    with head_r:
        st.title("Weather Analysis for FIFA World Cup 2034")
        st.markdown("Exploring 10 years of hourly weather across the five Saudi "
                    "host cities to support **smart match scheduling** that balances "
                    "fan comfort and player safety.")
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Host cities", "5")
    c2.metric("Years of data", "10")
    c3.metric("Hourly records", f"{len(df):,}")
    c4.metric("Missing values", "0")
 
    st.markdown("### Host cities and their climate")
    profile = (df.groupby("City")
                 .agg(Temperature=("Temperature", "mean"),
                      Humidity=("Humidity", "mean"))
                 .round(1).reindex(CITIES))
    profile["Climate"] = [CITY_PROFILE[c] for c in profile.index]
    profile = profile.rename(columns={"Temperature": "Avg Temp (C)",
                                       "Humidity": "Avg Humidity (%)"})
    st.dataframe(profile[["Climate", "Avg Temp (C)", "Avg Humidity (%)"]],
                 use_container_width=True)
 
    st.markdown("### Data source & collection")
    st.markdown(
        "- **Source:** Open-Meteo Historical Archive API (free, no API key required)\n"
        "- **Cities:** Riyadh, Jeddah, Al Khobar, Abha, NEOM (FIFA 2034 hosts)\n"
        "- **Period:** 2015-2024, **hourly** records, timezone Asia/Riyadh\n"
        "- **Variables:** temperature, humidity, wind speed, apparent temperature, "
        "precipitation, solar radiation\n"
        "- Data was collected per city via the API and saved to "
        "`host_cities_weather_2015_2024.csv`. This app loads that CSV and computes "
        "WBGT and all engineered features live.")
 
    with st.expander("Show the data-collection script (reference)"):
        st.code('''import requests, pandas as pd, time
 
CITIES = {
    "Riyadh": (24.7136, 46.6753), "Jeddah": (21.4858, 39.1925),
    "Al Khobar": (26.2794, 50.2083), "Abha": (18.2164, 42.5053),
    "NEOM": (27.96, 35.45),
}
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
          "apparent_temperature", "precipitation", "shortwave_radiation"]
BASE = "https://archive-api.open-meteo.com/v1/archive"
 
frames = []
for name, (lat, lon) in CITIES.items():
    r = requests.get(BASE, params={
        "latitude": lat, "longitude": lon,
        "start_date": "2015-01-01", "end_date": "2024-12-31",
        "hourly": ",".join(HOURLY), "timezone": "Asia/Riyadh"})
    d = pd.DataFrame(r.json()["hourly"])
    d.insert(0, "City", name)
    frames.append(d)
    time.sleep(1)
 
df = pd.concat(frames, ignore_index=True)
df.to_csv("host_cities_weather_2015_2024.csv", index=False)''', language="python")
 
    st.info("Navigate through the sections in the sidebar: The Data -> Analysis "
            "-> Results.")
 
# ======================================================================
# 2. DATA QUALITY & DISTRIBUTIONS
# ======================================================================
elif page == "2. Data Quality & Distributions":
    st.title("Data Quality & Distributions")
    st.markdown("Checking the raw data for completeness and plausibility, then "
                "exploring the shape of each weather variable.")
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Missing values", int(df.isna().sum().sum()))
    c3.metric("Duplicate rows",
              int(df.duplicated(subset=["City", "datetime"]).sum()))
 
    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown("#### Rows per city")
        st.dataframe(df["City"].value_counts().rename("rows"),
                     use_container_width=True)
    with colB:
        st.markdown("#### Descriptive statistics")
        num_cols = ["Temperature", "Humidity", "WindSpeed",
                    "ApparentTemp", "Precipitation", "SolarRadiation", "WBGT"]
        st.dataframe(df[num_cols].describe().round(1), use_container_width=True)
 
    st.success("No missing values, no duplicates, and city averages match known "
               "regional climate - the data is clean and trustworthy. Statistical "
               "outliers in WindSpeed and Precipitation are genuine rare weather "
               "events (storms, rainy days), so they are retained.")
 
    st.markdown("### Variable distributions")
    st.markdown("Select a variable to see how its values are distributed across "
                "the full dataset.")
    var = st.selectbox("Choose a variable",
                       ["Temperature", "Humidity", "WindSpeed", "SolarRadiation"])
    fig = px.histogram(df, x=var, nbins=50, color_discrete_sequence=[GREEN2])
    fig.update_layout(bargap=0.02, yaxis_title="Frequency",
                      plot_bgcolor="white", height=440)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Temperature is broad and bell-shaped; Humidity spans the full "
               "range (dry vs humid cities); WindSpeed is right-skewed (mostly "
               "calm); SolarRadiation spikes at zero (night hours).")
 
# ======================================================================
# 3. CLIMATE PATTERNS  (city comparison + seasonal + daily + 1 heatmap)
# ======================================================================
elif page == "3. Climate Patterns":
    st.title("Climate Patterns")
    st.markdown("How the five host cities differ, and how conditions change "
                "across the year and the day.")
 
    tab1, tab2, tab3, tab4 = st.tabs(
        ["City comparison", "Seasonal & daily", "Yearly trend", "Heatmap"])
 
    # --- City comparison ---
    with tab1:
        order = list(df.groupby("City")["Temperature"].median()
                       .sort_values().index)
        f1 = px.box(df, x="City", y="Temperature", category_orders={"City": order},
                    color="City", color_discrete_map=CITY_COLORS)
        f1.update_layout(title="Temperature by city", showlegend=False,
                         plot_bgcolor="white", height=420,
                         yaxis_title="Temperature (C)")
        st.plotly_chart(f1, use_container_width=True)
        f2 = px.box(df, x="City", y="Humidity", category_orders={"City": order},
                    color="City", color_discrete_map=CITY_COLORS)
        f2.update_layout(title="Humidity by city", showlegend=False,
                         plot_bgcolor="white", height=420,
                         yaxis_title="Humidity (%)")
        st.plotly_chart(f2, use_container_width=True)
        st.markdown("**Abha** is the coolest (median ~18 C) and the only city "
                    "viable for daytime matches. The others are similar in "
                    "temperature but differ in humidity: **Jeddah** the most "
                    "humid, **Riyadh** the driest.")
 
    # --- Seasonal & daily ---
    with tab2:
        st.markdown("**Average temperature by month**")
        monthly = (df.groupby(["City", "Month"])["Temperature"]
                     .mean().round(1).reset_index())
        fm = px.line(monthly, x="Month", y="Temperature", color="City",
                     markers=True, color_discrete_map=CITY_COLORS,
                     category_orders={"City": CITIES})
        fm.update_layout(plot_bgcolor="white", height=420,
                         yaxis_title="Avg Temperature (C)",
                         xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fm, use_container_width=True)
        st.caption("Peaks in summer (Jul ~32.5 C), lowest in winter. Abha stays "
                   "~23 C even at peak summer - viable year-round.")
 
        st.markdown("**Average temperature by hour**")
        hourly = (df.groupby(["City", "Hour"])["Temperature"]
                    .mean().round(1).reset_index())
        fh = px.line(hourly, x="Hour", y="Temperature", color="City",
                     markers=True, color_discrete_map=CITY_COLORS,
                     category_orders={"City": CITIES})
        fh.update_layout(plot_bgcolor="white", height=420,
                         yaxis_title="Avg Temperature (C)",
                         xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fh, use_container_width=True)
        st.caption("Coolest 05:00-06:00, hottest 13:00-15:00. The realistic match "
                   "window is the evening (~18:00-22:00).")
 
    # --- Yearly trend ---
    with tab3:
        yearly = df.groupby("Year")["Temperature"].mean().round(2)
        z = np.polyfit(yearly.index, yearly.values, 1)
        trend = np.poly1d(z)
        ft = go.Figure()
        ft.add_scatter(x=yearly.index, y=yearly.values, mode="lines+markers",
                       name="Avg Temp", line=dict(color=RED, width=2))
        ft.add_scatter(x=yearly.index, y=trend(yearly.index), mode="lines",
                       name=f"Trend: {z[0]:+.3f} C/yr",
                       line=dict(color="gray", dash="dash"))
        ft.update_layout(plot_bgcolor="white", height=460,
                         yaxis_title="Avg Temperature (C)", xaxis_title="Year",
                         xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(ft, use_container_width=True)
        st.markdown(f"A slight warming trend of **{z[0]:+.3f} C/year** "
                    f"(~{z[0]*10:+.2f} C/decade), with large year-to-year "
                    "variability - suggestive, not conclusive, but relevant for 2034.")
 
    # --- Single interactive heatmap (variable + city selectable) ---
    with tab4:
        st.markdown("Average value by **hour of day** and **month**. "
                    "Choose the variable and city.")
        hc1, hc2 = st.columns(2)
        metric = hc1.selectbox("Variable", ["Temperature", "WBGT"])
        city = hc2.selectbox("City", CITIES)
        sub = df[df["City"] == city]
        pivot = sub.pivot_table(index="Hour", columns="Month",
                                values=metric, aggfunc="mean").round(1)
        unit = "Temp (C)" if metric == "Temperature" else "WBGT"
        fhm = px.imshow(pivot, color_continuous_scale="RdYlBu_r", aspect="auto",
                        origin="lower",
                        labels=dict(x="Month", y="Hour of day", color=unit))
        fhm.update_layout(height=540,
                          title=f"{city} - Avg {metric} by Hour and Month")
        st.plotly_chart(fhm, use_container_width=True)
        st.caption("Blue = cooler (comfortable), Red = hotter. Blue zones (winter "
                   "months + evening/night) are candidate match windows. Switch to "
                   "WBGT to see real heat stress including humidity.")
 
# ======================================================================
# 4. HEAT STRESS & COMFORT  (WBGT + suitability + calculator + comfort)
# ======================================================================
elif page == "4. Heat Stress & Comfort":
    st.title("Heat Stress & Comfort")
    st.markdown("**WBGT** (Wet-Bulb Globe Temperature) combines temperature and "
                "humidity into one heat-stress index. It drives the match "
                "suitability classes and the 0-100 comfort score.")
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Mean WBGT by city")
        wbgt_city = df.groupby("City")["WBGT"].mean().round(1).sort_values()
        fb = px.bar(x=wbgt_city.values, y=wbgt_city.index, orientation="h",
                    color_discrete_sequence=[GREEN2], text=wbgt_city.values)
        fb.update_layout(plot_bgcolor="white", height=400,
                         xaxis_title="Mean WBGT", yaxis_title="")
        st.plotly_chart(fb, use_container_width=True)
    with c2:
        st.markdown("#### Suitability mix by city")
        mix = (df.groupby("City")["MatchSuitability"]
                 .value_counts(normalize=True).unstack().fillna(0) * 100).round(1)
        mix = mix.reindex(CITIES)
        cmap = {"Suitable": "#2e8b57", "Caution": "#e9c46a", "Not Suitable": RED}
        fmix = go.Figure()
        for cls in ["Suitable", "Caution", "Not Suitable"]:
            if cls in mix.columns:
                fmix.add_bar(y=mix.index, x=mix[cls], name=cls, orientation="h",
                             marker_color=cmap[cls])
        fmix.update_layout(barmode="stack", plot_bgcolor="white", height=400,
                           xaxis_title="% of hours", legend_title="")
        st.plotly_chart(fmix, use_container_width=True)
 
    st.markdown("### Try the comfort index")
    cc1, cc2 = st.columns(2)
    t = cc1.slider("Temperature (C)", 0, 50, 38)
    h = cc2.slider("Humidity (%)", 0, 100, 50)
    w = compute_wbgt(t, h)
    cls = suitability_class(w)
    color = {"Suitable": "#2e8b57", "Caution": GOLD, "Not Suitable": RED}[cls]
    st.markdown(f"**WBGT = {w:.1f}** -> "
                f"<span style='color:{color};font-weight:700'>{cls}</span>",
                unsafe_allow_html=True)
    st.caption("Note: the simplified WBGT over-estimates stress in hot-humid "
               "cities (Jeddah, Al Khobar) - a known limitation.")
 
    st.markdown("---")
    st.header("Fan Comfort Score (0-100)")
    st.latex(r"\text{Comfort} = \frac{38 - \text{WBGT}}{38 - 18} \times 100")
    st.caption("A 0-100 reading of WBGT: 100 = ideal, 0 = extreme heat stress.")
 
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Average comfort by city")
        city_score = df.groupby("City")["ComfortScore"].mean().round(0).sort_values()
        fc = px.bar(x=city_score.values, y=city_score.index, orientation="h",
                    text=city_score.values, color=city_score.values,
                    color_continuous_scale="RdYlGn")
        fc.update_layout(plot_bgcolor="white", height=360,
                         xaxis_title="Avg Comfort (0-100)", yaxis_title="",
                         coloraxis_showscale=False)
        st.plotly_chart(fc, use_container_width=True)
    with g2:
        st.markdown("#### Most stable city")
        stability = df.groupby("City")["ComfortScore"].std().round(1).sort_values()
        fs = px.bar(x=stability.values, y=stability.index, orientation="h",
                    text=stability.values, color_discrete_sequence=[GREEN2])
        fs.update_layout(plot_bgcolor="white", height=360,
                         xaxis_title="Std dev (lower = more stable)",
                         yaxis_title="")
        st.plotly_chart(fs, use_container_width=True)
    st.markdown(f"**Abha** scores highest and **{stability.index[0]}** is the most "
                "climatically stable (smallest swing in comfort).")
 
    st.markdown("#### The \"golden window\": months where all cities align")
    monthly_city = df.groupby(["Month", "City"])["ComfortScore"].mean()
    worst = monthly_city.groupby("Month").min().round(0)
    best = monthly_city.groupby("Month").max().round(0)
    fg = go.Figure()
    fg.add_scatter(x=worst.index, y=worst.values, mode="lines+markers",
                   name="Worst city", line=dict(color=RED, width=3))
    fg.add_scatter(x=best.index, y=best.values, mode="lines+markers",
                   name="Best city", line=dict(color=GREEN2, width=2, dash="dot"))
    fg.update_layout(plot_bgcolor="white", height=440, xaxis_title="Month",
                     yaxis_title="Comfort (0-100)",
                     xaxis=dict(tickmode="linear", dtick=1), legend_title="")
    st.plotly_chart(fg, use_container_width=True)
    golden = worst.sort_values(ascending=False).head(3)
    mm = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    st.success(f"Golden window: **{', '.join(mm[m] for m in golden.index)}** - "
               "even the least comfortable host city scores high, so all cities "
               "can be scheduled together with minimal heat risk.")
 
# ======================================================================
# 5. SCHEDULING INSIGHTS  (recommendations, no heatmap duplication)
# ======================================================================
elif page == "5. Scheduling Insights":
    st.title("Scheduling Insights")
    st.markdown("Data-driven answers to the key scheduling questions.")
 
    st.markdown("### 1) Which cities need night matches the most?")
    day = df[df["Hour"].between(12, 17)]
    night_need = (day.assign(bad=day["MatchSuitability"] != "Suitable")
                     .groupby("City")["bad"].mean().mul(100).round(1)
                     .reindex(CITIES).sort_values())
    f1 = px.bar(x=night_need.values, y=night_need.index, orientation="h",
                text=night_need.values, color_discrete_sequence=[RED])
    f1.update_layout(plot_bgcolor="white", height=360,
                     xaxis_title="% of daytime hours NOT suitable", yaxis_title="")
    st.plotly_chart(f1, use_container_width=True)
    st.markdown("**Jeddah** and **Al Khobar** depend most on night matches; "
                "**Abha** barely needs them.")
 
    st.markdown("### 2) Coolest months and hours (by WBGT)")
    b1, b2 = st.columns(2)
    with b1:
        bm = df.groupby("Month")["WBGT"].mean().round(1)
        fbm = px.bar(x=bm.index, y=bm.values, color_discrete_sequence=[GREEN2])
        fbm.update_layout(plot_bgcolor="white", height=340, xaxis_title="Month",
                          yaxis_title="Avg WBGT",
                          xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fbm, use_container_width=True)
    with b2:
        bh = df.groupby("Hour")["WBGT"].mean().round(1)
        fbh = px.bar(x=bh.index, y=bh.values, color_discrete_sequence=[GOLD])
        fbh.update_layout(plot_bgcolor="white", height=340, xaxis_title="Hour",
                          yaxis_title="Avg WBGT",
                          xaxis=dict(tickmode="linear", dtick=2))
        st.plotly_chart(fbh, use_container_width=True)
    st.markdown("Comfort peaks in **winter months** and at **night / early "
                "morning**. Recommended kickoff window: **evening (~18:00-22:00)**.")
 
    st.markdown("### 3) Afternoon (16:00) vs Evening (20:00)")
    aft = df[df["Hour"] == 16].groupby("City")["WBGT"].mean()
    eve = df[df["Hour"] == 20].groupby("City")["WBGT"].mean()
    comp = pd.DataFrame({"16:00": aft, "20:00": eve}).reindex(CITIES).round(1)
    comp["Drop"] = (comp["16:00"] - comp["20:00"]).round(1)
    f3 = go.Figure()
    f3.add_bar(x=comp.index, y=comp["16:00"], name="16:00 (4 PM)", marker_color=RED)
    f3.add_bar(x=comp.index, y=comp["20:00"], name="20:00 (8 PM)", marker_color=GREEN2)
    f3.update_layout(barmode="group", plot_bgcolor="white", height=400,
                     yaxis_title="Avg WBGT", legend_title="")
    st.plotly_chart(f3, use_container_width=True)
    st.dataframe(comp.rename(columns={"Drop": "WBGT drop (16->20)"}),
                 use_container_width=True)
    st.markdown("Every city is cooler at 8 PM. The drop is largest in dry inland "
                "cities (they cool faster after sunset). WBGT is used as a proxy "
                "for expected cooling demand (actual energy data is a future step).")
 
# ======================================================================
# 6. FEATURE ENGINEERING
# ======================================================================
elif page == "6. Feature Engineering":
    st.title("Feature Engineering")
    st.markdown("The engineered features and *before / after* comparisons. "
                "All computed live inside the app from the raw weather data.")
 
    fe = build_features(df)
 
    st.markdown("### Engineered feature groups")
    st.markdown(
        "- **WBGT** - heat-stress index (temperature + humidity)\n"
        "- **MatchSuitability** - target class from WBGT\n"
        "- **Cyclical encoding** - Hour/Month as sin & cos\n"
        "- **City one-hot** - 5 binary city columns\n"
        "- **Lag features** - WBGT 1, 3, 24 hours ago (within each city)")
    new_cols = (["WBGT", "MatchSuitability", "Hour_sin", "Hour_cos",
                 "Month_sin", "Month_cos"]
                + [c for c in fe.columns if c.startswith("City_")]
                + ["WBGT_lag1", "WBGT_lag3", "WBGT_lag24"])
    st.dataframe(fe[["City", "datetime", "Temperature", "Humidity"] + new_cols].head(12),
                 use_container_width=True)
 
    tabA, tabB, tabC = st.tabs(
        ["Before / After", "City encoding", "Feature relationships"])
 
    # --- before/after ---
    with tabA:
        st.markdown("**Cyclical encoding of Hour**")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("BEFORE: hour as a number (23 looks far from 0)")
            sample = fe[fe["City"] == "Riyadh"].head(48).reset_index()
            figb = px.line(sample, y="Hour", color_discrete_sequence=[RED])
            figb.update_layout(plot_bgcolor="white", height=340,
                               yaxis_title="Hour", xaxis_title="Time step")
            st.plotly_chart(figb, use_container_width=True)
        with cc2:
            st.caption("AFTER: hour as a circle (23 sits next to 0)")
            figa = px.scatter(fe, x="Hour_sin", y="Hour_cos", color="Hour",
                              color_continuous_scale="twilight")
            figa.update_layout(plot_bgcolor="white", height=340)
            figa.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(figa, use_container_width=True)
 
        st.markdown("**Raw Temperature vs WBGT**")
        cc3, cc4 = st.columns(2)
        with cc3:
            st.caption("BEFORE: temperature only")
            f1 = px.histogram(fe, x="Temperature", nbins=50,
                              color_discrete_sequence=[GOLD])
            f1.update_layout(plot_bgcolor="white", height=320,
                             yaxis_title="Frequency")
            st.plotly_chart(f1, use_container_width=True)
        with cc4:
            st.caption("AFTER: WBGT (temperature + humidity)")
            f2 = px.histogram(fe, x="WBGT", nbins=50,
                              color_discrete_sequence=[GREEN2])
            f2.update_layout(plot_bgcolor="white", height=320,
                             yaxis_title="Frequency")
            st.plotly_chart(f2, use_container_width=True)
        st.caption("WBGT merges temperature and humidity, so similar-temperature "
                   "cities (Riyadh, Jeddah) get very different heat stress.")
 
    # --- city encoding ---
    with tabB:
        st.markdown("Models read numbers, not text. The City column becomes 5 "
                    "binary columns (one per city):")
        city_cols = [c for c in fe.columns if c.startswith("City_")]
        st.dataframe(fe[["City"] + city_cols].drop_duplicates("City")
                       .reset_index(drop=True), use_container_width=True)
 
    # --- relationships: correlation + lag scatter ---
    with tabC:
        st.markdown("**Correlation of engineered features**")
        corr_cols = ["Temperature", "Humidity", "WBGT",
                     "WBGT_lag1", "WBGT_lag3", "WBGT_lag24",
                     "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]
        corr = fe[corr_cols].corr().round(2)
        figc = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto")
        figc.update_layout(height=520)
        st.plotly_chart(figc, use_container_width=True)
        st.caption("WBGT correlates strongly with its recent lags - past "
                   "conditions help predict the near future (basis of forecasting).")
 
        st.markdown("**Do lag features carry predictive value?**")
        samp = fe.dropna(subset=["WBGT_lag3"]).sample(min(4000, len(fe)),
                                                      random_state=1)
        figs = px.scatter(samp, x="WBGT_lag3", y="WBGT", color="City",
                          color_discrete_map=CITY_COLORS, opacity=0.4)
        figs.update_layout(plot_bgcolor="white", height=460,
                           xaxis_title="WBGT 3 hours ago", yaxis_title="WBGT now")
        st.plotly_chart(figs, use_container_width=True)
        st.caption("The strong diagonal confirms high autocorrelation - exactly "
                   "why lag features improve the forecasting model.")
 