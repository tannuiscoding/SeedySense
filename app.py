import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

from flask import Flask, render_template, request, jsonify  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

app = Flask(__name__)

# Load the fitted scaler and model
scaler = joblib.load('./model/scaler.pkl')
seedy_model = joblib.load('./model/SeedySense.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_crop():
    # Get form data
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