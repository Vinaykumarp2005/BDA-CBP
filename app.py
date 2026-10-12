import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Air Quality Big Data Analytics Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Big Data Analytics Dashboard")
st.markdown(
    "**End-to-End Analytics Pipeline:** HDFS Storage • MapReduce Summarization • "
    "Hive Analytical Queries • Pig Latin Temporal Analysis • Streamlit UI"
)

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_aqi_data_expanded.csv")
    df['data_date'] = pd.to_datetime(df['data_date'])
    numeric_cols = ['hour_of_day', 'aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'temperature_c', 'humidity_pct', 'wind_speed_kmh']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `final_aqi_data_expanded.csv` not found. Please ensure the file is saved in the same folder.")
    st.stop()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("🔍 Global Filters")

all_states = sorted(df["state"].unique())
selected_states = st.sidebar.multiselect("Select State / City:", all_states, default=all_states)

all_statuses = sorted(df["status"].dropna().unique())
selected_status = st.sidebar.multiselect("AQI Severity Status:", all_statuses, default=all_statuses)

hour_range = st.sidebar.slider("Hour of Day Range (24h):", 0, 23, (0, 23))

filtered_df = df[
    (df["state"].isin(selected_states)) &
    (df["status"].isin(selected_status)) &
    (df["hour_of_day"].between(hour_range[0], hour_range[1]))
]

if filtered_df.empty:
    st.warning("No records match the selected filter criteria. Please broaden your selection.")
    st.stop()

# ----------------- TOP KPI METRICS -----------------
kpi2, kpi3, kpi4, kpi5 = st.columns(4)
# kpi1.metric("Total Records Processed", f"{len(filtered_df):,}")
kpi2.metric("Mean AQI", f"{filtered_df['aqi'].mean():.1f}")
kpi3.metric("Peak PM2.5 (µg/m³)", f"{filtered_df['pm2_5'].max():.1f}")
kpi4.metric("Mean Temperature", f"{filtered_df['temperature_c'].mean():.1f} °C")
kpi5.metric("Avg Wind Speed", f"{filtered_df['wind_speed_kmh'].mean():.1f} km/h")

st.divider()

# ----------------- TABBED ANALYTICS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & Status (MR / Hive)",
    "🧪 Pollutants In-Depth",
    "⏰ Temporal & Weather Trends (Pig)",
    "🔗 Correlation Analysis",
    "📋 Data Explorer"
])

# TAB 1: OVERVIEW & MAPREDUCE / HIVE
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hive Aggregation: Average AQI by City")
        avg_aqi_city = filtered_df.groupby("state")["aqi"].mean().reset_index().sort_values(by="aqi", ascending=False)
        fig_bar = px.bar(
            avg_aqi_city,
            x="state",
            y="aqi",
            color="aqi",
            color_continuous_scale="Reds",
            text_auto='.1f',
            labels={"state": "City / State", "aqi": "Average AQI"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # with col2:
    #     st.subheader("MapReduce Frequency Count: AQI Status Distribution")
    #     status_counts = filtered_df["status"].value_counts().reset_index()
    #     status_counts.columns = ["status", "count"]
    #     fig_donut = px.pie(
    #         status_counts,
    #         names="status",
    #         values="count",
    #         hole=0.45,
    #         color_discrete_sequence=px.colors.qualitative.Bold
    #     )
    #     st.plotly_chart(fig_donut, use_container_width=True)

# TAB 2: POLLUTANTS BREAKDOWN
with tab2:
    st.subheader("Multi-Pollutant Average Comparison Across Regions")
    pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co']
    pollutant_summary = filtered_df.groupby("state")[pollutants].mean().reset_index()
    
    fig_pollutants = px.bar(
        pollutant_summary.melt(id_vars='state', var_name='Pollutant', value_name='Concentration'),
        x="state",
        y="Concentration",
        color="Pollutant",
        barmode="group",
        labels={"state": "City", "Concentration": "Average Concentration"},
        title="Particulate & Gaseous Pollutant Levels"
    )
    st.plotly_chart(fig_pollutants, use_container_width=True)

    st.subheader("PM2.5 vs PM10 Distribution by State")
    fig_scatter = px.scatter(
        filtered_df,
        x="pm10",
        y="pm2_5",
        color="state",
        size="aqi",
        hover_data=["sensor_id", "status"],
        labels={"pm10": "PM10 (µg/m³)", "pm2_5": "PM2.5 (µg/m³)"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 3: TEMPORAL & METEOROLOGICAL
with tab3:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.subheader("Pig Latin Output: Diurnal Hourly AQI Profile")
        hourly_trend = filtered_df.groupby("hour_of_day")[["aqi", "pm2_5"]].mean().reset_index()
        fig_line = px.line(
            hourly_trend,
            x="hour_of_day",
            y=["aqi", "pm2_5"],
            markers=True,
            labels={"hour_of_day": "Hour of Day (24h)", "value": "Metric Level", "variable": "Metric"},
            title="Hourly Variations (Rush Hour Peaks: 8-10 AM & 6-9 PM)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_t2:
        st.subheader("Impact of Meteorological Factors on AQI")
        weather_avg = filtered_df.groupby("hour_of_day")[["temperature_c", "wind_speed_kmh", "humidity_pct"]].mean().reset_index()
        fig_weather = px.line(
            weather_avg,
            x="hour_of_day",
            y=["temperature_c", "wind_speed_kmh", "humidity_pct"],
            markers=True,
            labels={"hour_of_day": "Hour of Day (24h)", "value": "Values", "variable": "Meteorology"}
        )
        st.plotly_chart(fig_weather, use_container_width=True)

# TAB 4: CORRELATION HEATMAP
with tab4:
    st.subheader("Pearson Correlation Heatmap (Pollutants vs Weather)")
    num_df = filtered_df[['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'temperature_c', 'humidity_pct', 'wind_speed_kmh']].dropna()
    corr = num_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# TAB 5: DATA EXPLORER
with tab5:
    st.subheader("Cleaned Dataset View (Hive / Sqoop Export)")
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_aqi_data.csv",
        mime="text/csv"
    )