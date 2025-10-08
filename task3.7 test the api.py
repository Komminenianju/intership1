curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":85,"BMI":28.5,"DiabetesPedigreeFunction":0.5,"Age":30}'