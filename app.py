from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import logging
from logging.handlers import RotatingFileHandler
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/seedysense.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('SeedySense startup')

# Load the model
try:
    
    scaler = joblib.load('./model/scaler.pkl')
    seedy_model = joblib.load('./model/SeedySense.pkl')
    app.logger.info('Model loaded successfully')
except Exception as e:
    app.logger.error(f'Error loading model: {str(e)}')
    model = None

# Input ranges for validation
INPUT_RANGES = {
    'N': (0, 140),
    'P': (5, 145),
    'K': (5, 205),
    'temperature': (8.83, 43.68),
    'humidity': (14.26, 99.98),
    'ph': (3.50, 9.94),
    'rainfall': (20.21, 298.56)
}

def validate_input(data):
    """Validate input data against defined ranges."""
    errors = []
    for key, (min_val, max_val) in INPUT_RANGES.items():
        value = float(data.get(key, 0))
        if not min_val <= value <= max_val:
            errors.append(f"{key} must be between {min_val} and {max_val}")
    return errors

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall'])
            }

            # Validate input
            errors = validate_input(input_data)
            if errors:
                return render_template('predict.html', errors=errors)

            # Make prediction
            features = np.array([[
                input_data['N'], input_data['P'], input_data['K'],
                input_data['temperature'], input_data['humidity'],
                input_data['ph'], input_data['rainfall']
            ]])
            
            # Scale the input data using the loaded scaler
            input_data_scaled = scaler.transform(features)

            # Make prediction
            prediction = seedy_model.predict(input_data_scaled)
            final_pred = prediction[0]

            # Log prediction
            app.logger.info(f'Prediction made: {prediction} for input: {input_data}')
            
            return render_template('predict.html', 
                                 prediction=final_pred,
                                 input_data=input_data)

        except Exception as e:
            app.logger.error(f'Error making prediction: {str(e)}')
            return render_template('predict.html', 
                                 error="An error occurred while making the prediction.")
    
    return render_template('predict.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/buy-seeds')
def buy_seeds():
    return render_template('buy-seeds.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    try:
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        
        # Log the contact form submission
        app.logger.info(f'Contact form submitted by {name} ({email})')
        
        # Here you would typically send an email or store in database
        # For now, we'll just return a success message
        return render_template('contact.html', 
                             success="Thank you for your message! We'll get back to you soon.")
    
    except Exception as e:
        app.logger.error(f'Error submitting contact form: {str(e)}')
        return render_template('contact.html', 
                             error="An error occurred while submitting your message.")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
