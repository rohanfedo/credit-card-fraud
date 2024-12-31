!pip install scikit-learn
import sklearn
import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('optimized_random_forest_model.pkl')

st.title("Fraud Detection Model")

# Input features
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
feature3 = st.number_input('Feature 3')
feature4 = st.number_input('Feature 4')

if st.button("Predict"):
    data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                        columns=['feature1', 'feature2', 'feature3', 'feature4'])
    prediction = model.predict(data)
    st.write("Prediction:", prediction[0])
