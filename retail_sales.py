import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Read the dataset
data = pd.read_csv("retail_sales_dataset.csv")

# Check for missing values in relevant columns
print("Missing values in relevant columns:")
print(data[['Age', 'Gender', 'Product Category']].isnull().sum())

# Handle missing values (e.g., dropping or imputing)
data.dropna(subset=['Age', 'Gender', 'Product Category'], inplace=True)

# Encode categorical variables
le_gender = LabelEncoder()
data['Gender_Encoded'] = le_gender.fit_transform(data['Gender'])

# Select only age and gender as features
X = data[['Age', 'Gender_Encoded']]  # Features
y = data['Product Category']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate and print metrics
print("\nModel Performance Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Print feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
print(feature_importance)

# Save the model and the label encoder
joblib.dump(rf_model, 'product_category_model.pkl')
joblib.dump(le_gender, 'gender_encoder.pkl')

# Function to predict product category
def predict_product_category(age, gender):
    """
    Predict product category based on customer age and gender

    Parameters:
    age (int): Customer's age
    gender (str): Customer's gender ('Male' or 'Female')

    Returns:
    str: Predicted product category
    """
    # Load the saved model and encoder
    loaded_model = joblib.load('product_category_model.pkl')
    loaded_gender_encoder = joblib.load('gender_encoder.pkl')

    # Encode gender
    try:
        gender_encoded = loaded_gender_encoder.transform([gender])[0]
    except ValueError:
        raise ValueError(f"Invalid gender: {gender}. Please use 'Male' or 'Female'.")

    # Create feature array
    features = np.array([[age, gender_encoded]])

    # Make prediction
    prediction = loaded_model.predict(features)
    return prediction[0]
