# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to train models
def train_models(data):
    # Select important features
    X = data[['Year', 'Population (in Millions)', 'GDP (in Trillions USD)', 
              'Inflation Rate (%)', 'Life Expectancy (Years)', 
              'Healthcare Expenditure per Capita (USD)', 'Internet Penetration (%)']] 
    y = data['Unemployment Rate (%)']

    # Convert continuous target to binary categories
    threshold = y.median()
    y_binary = (y >= threshold).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Train Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    
    # Train Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)
    y_pred_decision_tree = decision_tree_model.predict(X_test)

    # Train Random Forest Classifier
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)
    y_pred_random_forest = random_forest_model.predict(X_test)

    # Return the trained models and predictions
    return (logistic_model, decision_tree_model, random_forest_model), (y_test, y_pred_logistic, y_pred_decision_tree, y_pred_random_forest)

# Streamlit app
st.title("Country Comparison: Predicting Unemployment Rate")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(data.head())

    # Display selected columns for understanding
    st.subheader("Selected Features")
    st.write(data[['Year', 'Population (in Millions)', 'GDP (in Trillions USD)', 
                   'Inflation Rate (%)', 'Life Expectancy (Years)', 
                   'Healthcare Expenditure per Capita (USD)', 'Internet Penetration (%)']].head())
    
    # Train the models
    models, predictions = train_models(data)
    logistic_model, decision_tree_model, random_forest_model = models
    y_test, y_pred_logistic, y_pred_decision_tree, y_pred_random_forest = predictions

    # Evaluation metrics
    st.subheader("Model Evaluation")

    # Logistic Regression Evaluation
    st.write("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.2f}")
    st.text(classification_report(y_test, y_pred_logistic))

    # Decision Tree Evaluation
    st.write("Decision Tree Classifier")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_decision_tree):.2f}")
    st.text(classification_report(y_test, y_pred_decision_tree))

    # Random Forest Evaluation
    st.write("Random Forest Classifier")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_random_forest):.2f}")
    st.text(classification_report(y_test, y_pred_random_forest))

else:
    st.write("Please upload a CSV file containing the dataset.")
