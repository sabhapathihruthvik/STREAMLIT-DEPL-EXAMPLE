import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------
# Train model inside the app (no pickle file needed)
# -------------------
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier (No Pickle)")
st.write("This model predicts the type of Iris flower based on input features.")

# Sidebar input sliders
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.sidebar.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    pred_class = iris.target_names[prediction]

    st.success(f"🌼 Predicted flower: **{pred_class}**")

# Show model accuracy
st.write(f"✅ Model Accuracy on test set: {accuracy:.2f}")
