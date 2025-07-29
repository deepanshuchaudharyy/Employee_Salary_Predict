import streamlit as st
import joblib
import pandas as pd
import os

# Check file sizes
def check_file_size(file_path, max_size_mb=25):
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        if size_mb > max_size_mb:
            st.error(f"Error: {file_path} is {size_mb:.2f} MB, exceeding {max_size_mb} MB limit.")
            return False
        return True
    else:
        st.error(f"Error: {file_path} not found.")
        return False

# Load the scaler and model
if check_file_size("scaler.pkl") and check_file_size("model.pkl"):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
else:
    st.stop()

st.title("Employee Salary Prediction")

# Collect inputs
department = st.selectbox('Department', ['IT', 'Finance', 'Engineering', 'HR', 'Operation'])
age = st.number_input("Employee Age", min_value=20, max_value=80, value=25)
gender = st.selectbox('Gender', ['F', 'M'])
education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
job_role = st.selectbox('Job Role', ['Specialist', 'Developer', 'Analyst', 'Technician', 'Consultant'])
years = st.number_input('Past Experience (in years)', min_value=0, step=1)

# Create a DataFrame with exact column names and order
input_data = pd.DataFrame({
    'Department': [department],
    'Age': [age],
    'Gender': [gender],
    'Education_Level': [education_level],
    'Job_Title': [job_role],
    'Years_At_Company': [years]
})

# Button for prediction
prediction_button = st.button("Predict Salary!")

st.divider()

if prediction_button:
    try:
        # Predict using the pipeline directly
        prediction = model.predict(input_data)[0]
        st.write(f"Predicted Salary: ${prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
else:
    st.write("Please use the button for the prediction")