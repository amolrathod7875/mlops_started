import streamlit as st
import joblib

model = joblib.load('LinearRegression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Test Score Predictor ")
st.write("Enter the number of hours studied")

hours = st.number_input("Hours studied :",min_value=0.0, steps=1.0)

if st.button("Predict"):
    try:
        data = [[hours]]
        scaler_data = scaler.trandform(data)
        prediction = model.predict(scaler_data)
        st.write(f"Predicted Test Score : {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error : {e}")
