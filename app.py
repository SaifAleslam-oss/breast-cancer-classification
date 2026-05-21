import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

with open("breast_cancer_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.markdown(
    """
    <h1 style='text-align:center; color:#ff4b4b;'>
    🩺 Breast Cancer Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; font-size:18px; color:#555;'>
    Machine Learning web app using Logistic Regression to predict breast cancer diagnosis.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.subheader("Enter Tumor Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Mean Features")
    radius_mean = st.number_input("Radius Mean", value=12.5)
    texture_mean = st.number_input("Texture Mean", value=17.3)
    perimeter_mean = st.number_input("Perimeter Mean", value=80.2)
    area_mean = st.number_input("Area Mean", value=480.5)
    smoothness_mean = st.number_input("Smoothness Mean", value=0.09)
    compactness_mean = st.number_input("Compactness Mean", value=0.08)
    concavity_mean = st.number_input("Concavity Mean", value=0.04)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.02)
    symmetry_mean = st.number_input("Symmetry Mean", value=0.18)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)

with col2:
    st.markdown("### SE Features")
    radius_se = st.number_input("Radius SE", value=0.30)
    texture_se = st.number_input("Texture SE", value=1.10)
    perimeter_se = st.number_input("Perimeter SE", value=2.10)
    area_se = st.number_input("Area SE", value=25.0)
    smoothness_se = st.number_input("Smoothness SE", value=0.005)
    compactness_se = st.number_input("Compactness SE", value=0.015)
    concavity_se = st.number_input("Concavity SE", value=0.020)
    concave_points_se = st.number_input("Concave Points SE", value=0.010)
    symmetry_se = st.number_input("Symmetry SE", value=0.018)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)

with col3:
    st.markdown("### Worst Features")
    radius_worst = st.number_input("Radius Worst", value=14.1)
    texture_worst = st.number_input("Texture Worst", value=22.5)
    perimeter_worst = st.number_input("Perimeter Worst", value=90.3)
    area_worst = st.number_input("Area Worst", value=600.2)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.12)
    compactness_worst = st.number_input("Compactness Worst", value=0.18)
    concavity_worst = st.number_input("Concavity Worst", value=0.12)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.07)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.28)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)

input_data = np.array([[
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,
    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,
    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,
    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
]])

st.divider()

center_col1, center_col2, center_col3 = st.columns([1, 1, 1])

with center_col2:
    predict_button = st.button("Predict Diagnosis", use_container_width=True)

if predict_button:
    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)

    benign_prob = probability[0][0] * 100
    malignant_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.markdown(
            f"""
            <div style='background-color:#ffe5e5; padding:25px; border-radius:15px;
            text-align:center; border:2px solid #ff4b4b;'>
                <h2 style='color:#b30000;'>❌ Malignant Tumor Detected</h2>
                <p style='font-size:18px;'>Malignant Probability: {malignant_prob:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:#e6ffed; padding:25px; border-radius:15px;
            text-align:center; border:2px solid #2ecc71;'>
                <h2 style='color:#1e8449;'>✅ Benign Tumor Detected</h2>
                <p style='font-size:18px;'>Benign Probability: {benign_prob:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("This app is built for educational purposes using Logistic Regression and Streamlit.")