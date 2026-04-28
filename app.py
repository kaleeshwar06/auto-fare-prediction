import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="AutoIntelligence", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 35px; color: #00FFC2; }
    .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 20px; border-radius: 15px; }
    div[data-testid="stExpander"] { border: none !important; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'base_fare': [30]*100,
        'charge_per_km': [12]*100,
        'charge_per_min': [2]*100,
        'booking_fee': [10]*100,
        'distance': np.random.uniform(1, 15, 100),
        'travel_time': np.random.uniform(5, 45, 100),
        'hour': np.random.randint(0, 24, 100)
    })
    noise = np.random.normal(0, 2, 100)
    data['fare'] = (data['base_fare'] + data['distance']*12 + data['travel_time']*2 + data['booking_fee']) + noise
    data['demand'] = data['hour'].apply(lambda x: np.random.randint(22, 30) if 8<=x<=11 or 17<=x<=21 else np.random.randint(5, 12))
    return data

df = get_data()

X_fare = df[['distance', 'travel_time']]
y_fare = df['fare']
X_dem = df[['hour']]
y_dem = df['demand']

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50),
    "XGBoost": XGBRegressor()
}

for m in models.values():
    m.fit(X_fare, y_fare)

demand_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50),
    "XGBoost": XGBRegressor()
}

for m in demand_models.values():
    m.fit(X_dem, y_dem)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80)
    st.title("Settings")
    st.markdown("---")
    dist = st.slider("Trip Distance (km)", 1.0, 30.0, 8.5)
    dur = st.slider("Trip Duration (min)", 5, 120, 25)
    hr = st.select_slider("Time of Day", options=list(range(24)), value=18)
    st.markdown("---")
    model_type = st.selectbox("Intelligence Engine", list(models.keys()))
    st.info("The model updates in real-time as you adjust sliders.")

st.title("🚖 Prediction Dashboard")
st.caption("Auto-Rickshaw Fare & Demand Analytics Engine")

c1, c2, c3 = st.columns(3)

raw_fare = models[model_type].predict([[dist, dur]])[0]
if model_type == "Linear Regression":
    fare_output = raw_fare - 0.002
else:
    fare_output = raw_fare

dem_output = int(demand_models[model_type].predict([[hr]])[0])

with c1:
    st.metric("Estimated Fare", f"₹{fare_output:.2f}", delta="Real-time")
with c2:
    st.metric("Predicted Demand", f"{dem_output} rides/hr", delta="Active")
with c3:
    status = "Peak Hour" if 8<=hr<=11 or 17<=hr<=21 else "Normal"
    st.metric("Zone Status", status)

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Spatial Demand Forecast")
    hr_range = np.arange(0, 24).reshape(-1, 1)
    preds = demand_models[model_type].predict(hr_range)
    
    fig = px.line(x=np.arange(0, 24), y=preds, labels={'x':'Hour of Day', 'y':'Demand Count'},
                  template="plotly_dark", color_discrete_sequence=['#00FFC2'])
    fig.update_traces(mode='lines+markers', fill='tozeroy')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("💡 Fare Breakdown")
    labels = ['Base', 'Distance', 'Time']
    values = [30, dist*12, dur*2]
    fig_pie = px.pie(names=labels, values=values, hole=0.5, template="plotly_dark",
                     color_discrete_sequence=['#00C9A7', '#0081CF', '#4B4453'])
    fig_pie.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.write("✅ **System Log:** Model Inference complete. Linear Regression offset applied successfully.")
