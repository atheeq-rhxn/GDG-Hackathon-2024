from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Load the trained model and gender encoder when the application starts
model_path = 'product_category_model.pkl'
encoder_path = 'gender_encoder.pkl'

try:
    model = joblib.load(model_path)
    gender_encoder = joblib.load(encoder_path)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    gender_encoder = None

def validate_input(age, gender):
    """Validate input data ranges."""
    if not isinstance(age, int):
        raise ValueError("Age must be a whole number")

    if not (0 <= age <= 120):
        raise ValueError("Age must be between 0 and 120 years")

    if gender.lower() not in ['male', 'female']:
        raise ValueError("Gender must be either 'Male' or 'Female'")

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle both GET and POST requests for prediction."""
    if request.method == 'GET':
        return render_template('predict.html')

    if request.method == 'POST':
        if model is None or gender_encoder is None:
            return render_template('error.html', 
                                 error="Model or encoder not loaded. Please contact administrator.")

        try:
            # Get form data
            age = request.form.get('age')
            gender = request.form.get('gender', '').capitalize()

            if not age or not gender:
                raise ValueError("Age and gender must be provided")

            try:
                age = int(age)
            except ValueError:
                raise ValueError("Age must be a whole number")

            # Validate inputs
            validate_input(age, gender)

            # Encode gender
            gender_encoded = gender_encoder.transform([gender])[0]

            # Create feature array
            features = np.array([[age, gender_encoded]])

            # Make prediction
            prediction = model.predict(features)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(features)[0]
            class_labels = model.classes_

            # Create sorted list of (category, probability) pairs
            prob_list = sorted(zip(class_labels, probabilities), 
                               key=lambda x: x[1], 
                               reverse=True)

            return render_template('result.html', 
                                 prediction=prediction,
                                 probabilities=prob_list,
                                 age=age,
                                 gender=gender)

        except ValueError as ve:
            return render_template('error.html', 
                                 error=f"Invalid input: {str(ve)}")
        except Exception as e:
            return render_template('error.html', 
                                 error=f"An error occurred: {str(e)}")

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error="Page not found. Please return to home page."), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)
