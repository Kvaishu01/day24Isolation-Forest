import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Isolation Forest - Anomaly Detection", layout="centered")
st.title("🚨 Day 24 — Isolation Forest: Anomaly Detection in Network Traffic")

# Generate synthetic network traffic dataset
@st.cache_data
def generate_data(n=500, random_state=42):
    rng = np.random.RandomState(random_state)
    normal_data = rng.normal(loc=50, scale=10, size=(n, 2))  # Normal traffic
    anomalies = rng.uniform(low=20, high=100, size=(int(0.05*n), 2))  # Anomalies
    data = np.vstack([normal_data, anomalies])
    labels = np.hstack([np.ones(n), -1*np.ones(int(0.05*n))])
    return pd.DataFrame(data, columns=["Packet_Size", "Duration"]), labels

df, true_labels = generate_data()

st.subheader("📂 Sample Network Traffic Data")
st.dataframe(df.head())

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df)
predictions = model.predict(df)

# Append results
df["Anomaly"] = predictions

# Visualization
st.subheader("📊 Anomaly Detection Results")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df, x="Packet_Size", y="Duration",
    hue="Anomaly", palette={1: "blue", -1: "red"}, ax=ax
)
ax.set_title("Network Traffic Anomalies (Red = Anomalies)")
st.pyplot(fig)

# Show anomaly counts
st.subheader("🔎 Anomaly Statistics")
anomaly_counts = df["Anomaly"].value_counts()
st.write(anomaly_counts)

st.success("✅ Isolation Forest successfully detected anomalies in network traffic!")
