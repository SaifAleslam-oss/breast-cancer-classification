import streamlit as st
import numpy as np
import pickle

# Load trained pipeline
with open("breast_cancer_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# App Title
st.title("Breast Cancer Prediction App")

st.write("Enter the tumor measurements below:")

# =========================
# Mean Features
# =========================

radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")
smoothness_mean = st.number_input("Smoothness Mean")
compactness_mean = st.number_input("Compactness Mean")
concavity_mean = st.number_input("Concavity Mean")
concave_points_mean = st.number_input("Concave Points Mean")
symmetry_mean = st.number_input("Symmetry Mean")
fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

# =========================
# SE Features
# =========================

radius_se = st.number_input("Radius SE")
texture_se = st.number_input("Texture SE")
perimeter_se = st.number_input("Perimeter SE")
area_se = st.number_input("Area SE")
smoothness_se = st.number_input("Smoothness SE")
compactness_se = st.number_input("Compactness SE")
concavity_se = st.number_input("Concavity SE")
concave_points_se = st.number_input("Concave Points SE")
symmetry_se = st.number_input("Symmetry SE")
fractal_dimension_se = st.number_input("Fractal Dimension SE")

# =========================
# Worst Features
# =========================

radius_worst = st.number_input("Radius Worst")
texture_worst = st.number_input("Texture Worst")
perimeter_worst = st.number_input("Perimeter Worst")
area_worst = st.number_input("Area Worst")
smoothness_worst = st.number_input("Smoothness Worst")
compactness_worst = st.number_input("Compactness Worst")
concavity_worst = st.number_input("Concavity Worst")
concave_points_worst = st.number_input("Concave Points Worst")
symmetry_worst = st.number_input("Symmetry Worst")
fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

# =========================
# Create Input Data
# =========================

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

# =========================
# Prediction
# =========================

if st.button("Predict"):

    prediction = pipeline.predict(input_data)

    if prediction[0] == 1:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor Detected")