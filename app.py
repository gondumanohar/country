# app.py

import streamlit as st
import joblib
import numpy as np

# Step 1: Load the saved model
logistic_model = joblib.load('logistic_model.pkl')

# Step 2: Create the Streamlit app
st.title("Logistic Regression Model Deployment")

# Step 3: Define input features
st.header("Input Features")

# Create input fields for your model's features (replace 'feature1', 'feature2', etc., with your actual feature names)
feature1 = st.number_input('Enter value for feature1', min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input('Enter value for feature2', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input('Enter value for feature3', min_value=0.0, max_value=100.0, value=50.0)

# Step 4: Make predictions
# Create a button to make predictions
if st.button('Predict'):
    # Collect the input features into a numpy array (ensure the input shape matches the model's expected shape)
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Make prediction
    prediction = logistic_model.predict(input_data)[0]  # Get the predicted class (0 or 1)
    
    # Display the prediction result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("The prediction is: Positive Class (1)")
    else:
        st.error("The prediction is: Negative Class (0)")
