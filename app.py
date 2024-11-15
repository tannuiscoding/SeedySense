from flask import Flask, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

scaler = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/understand_data')
def understand_data():
    return render_template('understand_data.html')

@app.route('/predict', methods=['POST'])
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Load the SeedySense model
    seedy_model = joblib.load("../model/SeedySense.pkl")
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
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    prediction = seedy_model.predict(input_data_scaled)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)