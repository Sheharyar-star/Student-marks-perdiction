
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Student Marks Prediction")
st.write(
    "Predict student marks using a trained Decision Tree model."
)

MODEL_PATH = Path(__file__).parent / "decision_tree_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()

    hours = st.number_input(
        "Enter hours studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("Predict Marks"):
        input_data = pd.DataFrame({
            "Hours_Studied": [hours]
        })

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Marks: {prediction:.2f}")

except Exception as e:
    st.error(f"Could not load the model: {e}")
