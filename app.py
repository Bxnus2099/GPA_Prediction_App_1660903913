import streamlit as st
import pickle
import numpy as np

# โหลดโมเดล
model = pickle.load(open('model.pkl', 'rb'))

st.title("GPA Prediction App")

study = st.number_input("Study Hours per Day", 0.0, 12.0)
sleep = st.number_input("Sleep Hours per Day", 0.0, 12.0)
subjects = st.number_input("Number of Subjects", 1, 10)
phone = st.number_input("Phone Hours per Day", 0.0, 12.0)

if st.button("Predict GPA"):
    input_data = np.array([[study, sleep, subjects, phone]])
    result = model.predict(input_data)
    st.success(f"Predicted GPA: {result[0]:.2f}")
