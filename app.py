import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset (replace with your actual dataset path)
data = pd.read_csv('country_comparison_large_dataset.csv')

# Selecting features and target variable
# Drop 'Unemployment Rate (%)' (target) and unnecessary columns like 'Country' or 'Id'
X = data[['Year', 'Population (in Millions)', 'GDP (in Trillions USD)', 
          'Inflation Rate (%)', 'Life Expectancy (Years)', 
          'Healthcare Expenditure per Capita (USD)', 'Internet Penetration (%)']] 
y = data['Unemployment Rate (%)']  # Target variable

# Convert continuous target to binary categories
threshold = y.median()  # Use median to classify
y_binary = (y >= threshold).astype(int)  # Convert to binary target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit App Title
st.title("Unemployment Rate Prediction")

# Sidebar inputs for features with default values
st.sidebar.header("Input Features")

year = st.sidebar.number_input('Year', value=2020)
population_in_millions = st.sidebar.number_input('Population (in Millions)', value=50.0)
gdp_in_trillions = st.sidebar.number_input('GDP (in Trillions USD)', value=1.0)
inflation_rate = st.sidebar.number_input('Inflation Rate (%)', value=3.0)
life_expectancy = st.sidebar.number_input('Life Expectancy (Years)', value=70.0)
healthcare_expenditure = st.sidebar.number_input('Healthcare Expenditure per Capita (USD)', value=1000.0)
internet_penetration = st.sidebar.number_input('Internet Penetration (%)', value=50.0)

# Button to predict unemployment rate
if st.sidebar.button('Predict Unemployment Rate'):
    # Convert inputs to a numpy array for prediction
    input_data = np.array([[year, population_in_millions, gdp_in_trillions, 
                            inflation_rate, life_expectancy, healthcare_expenditure, 
                            internet_penetration]])
    
    # Make prediction (binary output: 0 or 1)
    predicted_unemployment = rf_model.predict(input_data)[0]
    
    # Display the predicted unemployment rate category (0: Low, 1: High)
    if predicted_unemployment == 1:
        st.subheader(f"Predicted Unemployment Rate: High (Above Median)")
    else:
        st.subheader(f"Predicted Unemployment Rate: Low (Below Median)")
