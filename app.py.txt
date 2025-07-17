import streamlit as st
import pandas as pd
import joblib

#  Load your trained model
model = joblib.load('stroke_model.pkl')

#  Define the feature order based on how the model was trained
model_feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi',
    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children',
    'Residence_type_Rural', 'Residence_type_Urban',
    'smoking_status_Unknown', 'smoking_status_formerly smoked',
    'smoking_status_never smoked', 'smoking_status_smokes'
]

#  Streamlit UI
st.title(" Stroke Risk Prediction")
st.write("Enter patient details to predict the risk of stroke.")

#  Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 120)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
glucose = st.number_input("Average Glucose Level")
bmi = st.number_input("Body Mass Index")

work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

#  Manual one-hot encoding
work_map = {
    "Govt_job": [1, 0, 0, 0, 0],
    "Never_worked": [0, 1, 0, 0, 0],
    "Private": [0, 0, 1, 0, 0],
    "Self-employed": [0, 0, 0, 1, 0],
    "children": [0, 0, 0, 0, 1]
}
residence_map = {
    "Rural": [1, 0],
    "Urban": [0, 1]
}
smoking_map = {
    "Unknown": [1, 0, 0, 0],
    "formerly smoked": [0, 1, 0, 0],
    "never smoked": [0, 0, 1, 0],
    "smokes": [0, 0, 0, 1]
}
gender_val = 1 if gender == "Male" else 0
married_val = 1 if ever_married == "Yes" else 0

# 🧾 Assemble full input list
input_data = [gender_val, age, hypertension, heart_disease, married_val, glucose, bmi] \
             + work_map[work_type] + residence_map[residence_type] + smoking_map[smoking_status]

#  Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=model_feature_names)

#  Predict button
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)[0]
    st.subheader(" Prediction Result")
    st.write("High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke")
