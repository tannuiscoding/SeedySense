from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/understand_data')
def understand_data():
    return render_template('understand_data.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    hours_studied = float(data['hours_studied'])
    previous_scores = float(data['previous_scores'])
    extracurricular_activities = 1.0 if data['extracurricular_activities'] == 'yes' else 0.0
    sleep_hours = float(data['sleep_hours'])
    sample_question_papers_practiced = float(data['sample_question_papers_practiced'])

    # Prediction formula
    y = (2.85 * hours_studied + 
         1.02 * previous_scores + 
         0.61 * extracurricular_activities + 
         0.48 * sleep_hours + 
         0.19 * sample_question_papers_practiced + 
         (-33.92))

    return jsonify({'prediction': y})

if __name__ == '__main__':
    app.run(debug=True)