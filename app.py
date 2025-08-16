# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model and dataset
model = joblib.load("iris_model.pkl")
iris = load_iris()

st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("A simple ML app to classify Iris flowers using **Random Forest**.")

# Sidebar input
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.sidebar.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    pred_class = iris.target_names[prediction]

    st.success(f"🌼 The model predicts this is an **{pred_class}**")
