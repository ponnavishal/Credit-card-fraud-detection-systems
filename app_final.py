import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the model
def load_model():
    model = Sequential([
        Dense(26, input_shape=(30,), activation='relu'),  # Input layer and first hidden layer
        Dense(15, activation='relu'),                    # Second hidden layer
        Dense(1, activation='sigmoid')                   # Output layer
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# App Title
st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent based on user-provided data.")

# Input Section
st.header("Input Transaction Data")

# Input for time
time = st.number_input("Time Elapsed (seconds):", min_value=0.0, step=10.0)

# Input for amount
amount = st.number_input("Transaction Amount ($):", min_value=0.0, step=10.0)

# Input for V1 to V28 as a single comma-separated string
st.write("Enter features V1 to V28 as a comma-separated list (e.g., `0.1, -0.2, ..., 0.3`)")
v_features_text = st.text_area("Enter V1 to V28 Features:", "")

# Prediction Function
def predict_fraud(input_data):
    prediction = model.predict(input_data)
    return "Fraud" if prediction[0][0] > 0.5 else "Not Fraud"

# Prediction Button
if st.button("Predict"):
    try:
        # Parse and validate V1 to V28 inputs
        v_features = [float(x) for x in v_features_text.split(",")]
        if len(v_features) != 28:
            st.error("Please enter exactly 28 features for V1 to V28.")
        else:
            # Combine time, amount, and V1 to V28 into a single input array
            input_data = np.array([[time, amount] + v_features])
            
            # Make prediction
            result = predict_fraud(input_data)
            st.subheader("Prediction Result")
            st.write(f"The transaction is predicted to be: **{result}**")
    except ValueError:
        st.error("Invalid input. Please ensure all features are numbers, separated by commas.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
