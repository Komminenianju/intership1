from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'diabetes_prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)