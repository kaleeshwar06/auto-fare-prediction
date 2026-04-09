import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


st.title("Auto-Rickshaw Fare & Demand Prediction 🚖")

df = pd.read_csv("autos.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.loc[:, ~df.columns.str.contains('^unnamed')]

df = pd.get_dummies(df, columns=['city'], drop_first=True)

df['distance'] = np.random.randint(1, 10, size=len(df))
df['travel_time'] = np.random.randint(5, 30, size=len(df))

df['fare'] = (
    df['base_fare']
    + df['distance'] * df['charge_per_km']
    + df['travel_time'] * df['charge_per_min']
    + df['booking_fee']
)

df['hour'] = np.random.randint(0, 24, size=len(df))
df['demand'] = df['hour'].apply(
    lambda x: np.random.randint(20, 30) if 8 <= x <= 10 or 17 <= x <= 20
    else np.random.randint(5, 15)
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}
for name in models:
    models[name].fit(df[['distance', 'travel_time']], df['fare'])

demand_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

for name in demand_models:
    demand_models[name].fit(df[['hour']], df['demand'])

st.sidebar.header("Enter Inputs")

distance = st.sidebar.slider("Distance (km)", 1, 20, 5)
travel_time = st.sidebar.slider("Travel Time (min)", 5, 60, 20)
hour = st.sidebar.slider("Hour", 0, 23, 10)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

fare_pred = models[model_choice].predict([[distance, travel_time]])[0]
demand_pred = demand_models[model_choice].predict([[hour]])[0]

st.subheader(f"Results using {model_choice}")

st.write(f"🚖 Predicted Fare: ₹ {fare_pred:.2f}")
st.write(f"📊 Predicted Demand: {int(demand_pred)}")

st.subheader("Demand Trend")

hours = np.arange(0, 24)
predicted = demand_models[model_choice].predict(hours.reshape(-1, 1))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(hours, predicted)
ax.set_xlabel("Hour")
ax.set_ylabel("Demand")
ax.set_title(f"{model_choice} Demand Trend")

st.pyplot(fig)
