import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import pickle

from src.pipelines.predict_pipeline import PredictPipeline, CustomData
df = pd.read_csv("notebook/data/water_potability.csv")

st.set_page_config(
    page_title="Water Quality Prediction",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Water Quality Monitoring System")

st.sidebar.info(
    """
    This AI-powered application predicts whether water is safe for drinking
    using Machine Learning.
    """
)

st.sidebar.success("Model: Random Forest Classifier")

# ---------------- TITLE ---------------- #

st.title("AI-Based Water Quality Prediction System")

st.markdown(
    """
    Enter water quality parameters below to check whether the water is:
    
    - Safe for Drinking
    - Unsafe for Drinking
    """
)
st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(df.head())
st.markdown("---")

st.subheader("Water Potability Distribution")

potability_counts = df["Potability"].value_counts()

fig1, ax1 = plt.subplots()

ax1.bar(
    ["Unsafe", "Safe"],
    potability_counts.values
)

ax1.set_ylabel("Count")

st.pyplot(fig1)
st.markdown("---")

st.subheader("pH Value Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.hist(df["ph"].dropna(), bins=20)

ax2.set_xlabel("pH Values")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)

# ---------------- INPUTS ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH Value", value=7.0)

    Hardness = st.number_input("Hardness", value=200.0)

    Solids = st.number_input("Solids", value=10000.0)

with col2:
    Chloramines = st.number_input("Chloramines", value=7.0)

    Sulfate = st.number_input("Sulfate", value=300.0)

    Conductivity = st.number_input("Conductivity", value=400.0)

with col3:
    Organic_carbon = st.number_input("Organic Carbon", value=10.0)

    Trihalomethanes = st.number_input("Trihalomethanes", value=70.0)

    Turbidity = st.number_input("Turbidity", value=4.0)

# ---------------- PREDICTION ---------------- #

if st.button("Predict Water Quality"):

    data = CustomData(
        ph,
        Hardness,
        Solids,
        Chloramines,
        Sulfate,
        Conductivity,
        Organic_carbon,
        Trihalomethanes,
        Turbidity
    )

    pred_df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()

    result = predict_pipeline.predict(pred_df)

    st.subheader("Prediction Result")

    if result[0] == 1:
        st.success("Water is SAFE for Drinking")
    else:
        st.error("Water is NOT SAFE for Drinking")

    st.subheader("Input Parameters")

    st.dataframe(pred_df)

# ---------------- FEATURE INFO ---------------- #

st.markdown("---")

st.subheader("Important Water Quality Parameters")

feature_data = pd.DataFrame({
    "Feature": [
        "pH",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic Carbon",
        "Trihalomethanes",
        "Turbidity"
    ],
    "Importance": [8, 7, 9, 6, 5, 7, 4, 6, 8]
})

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    feature_data["Feature"],
    feature_data["Importance"]
)

plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption("Built using Python, Scikit-learn, Streamlit")