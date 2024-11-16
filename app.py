import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

scaler = joblib.load('./model/scaler.pkl')
seedy_model = joblib.load('./model/SeedySense.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/buy-seeds')
def buy_seeds():
    return render_template('buy-seeds.html')

@app.route('/dont-contact-us')
def dont_contact_us():
    return render_template('dont-contact-us.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = seedy_model.predict(input_data_scaled)
    final_pred = prediction[0]
    
    return jsonify({'prediction': final_pred})

if __name__ == '__main__':
    app.run(debug=True)
