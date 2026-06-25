# app.py
import streamlit as st
import pandas as pd
import joblib


model = joblib.load('house_price_model.pkl')

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("House Price Prediction App 🏠")
st.write("Enter the house details below to get an estimated price.")

st.markdown("---")


sqft = st.number_input("Area_SqFt", min_value=300, max_value=25000, value=1500, step=50)
bedrooms = st.number_input("Rooms", min_value=1, max_value=10, value=3, step=1)


st.markdown("---")

if st.button("Predict House Price", type="primary"):
   
    input_data = pd.DataFrame([[sqft, bedrooms]], 
                              columns=['Area_SqFt', 'Rooms'])
    
    
    prediction = model.predict(input_data)
    
    
    st.success(f"### 💰 Estimated Price: {prediction[0]:.2f} ")