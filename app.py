import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('credit_card_fraud_detection_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent based on uploaded CSV data.")


st.header("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type=["csv"])

def predict_fraud(input_data):
    predictions = model.predict(input_data)
    return ["Fraud" if pred[0] > 0.5 else "Not Fraud" for pred in predictions]

if uploaded_file:
    try:
        
        df = pd.read_csv(uploaded_file)
        
        required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain columns: Time, Amount, and V1 to V28.")
        else:
      
            input_data = df[required_columns].values
            
          
            df["Prediction"] = predict_fraud(input_data)
            
          
            st.subheader("Prediction Results")
            st.dataframe(df[["Time", "Amount", "Prediction"]])
            
      
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

