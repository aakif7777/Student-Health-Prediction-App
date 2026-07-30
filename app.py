from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import joblib
import pandas as pd
import numpy as np
import json


app = Flask(__name__)

# MYSQL CONFIG

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'infected_123'      # <
app.config['MYSQL_DB'] = 'student_health_db'

mysql = MySQL(app)


print("="*60)
print("Student Health Risk Prediction System")
print("Loading model files...")
print("="*60)

# LOAD MODEL FILES

model = joblib.load(
    "model/student_health_catboost_model.pkl"
)

feature_encoders = joblib.load(
    "model/feature_encoders.pkl"
)

target_encoder = joblib.load(
    "model/target_encoder.pkl"
)

median_values = joblib.load(
    "model/median_values.pkl"
)

feature_columns = joblib.load(
    "model/feature_columns.pkl"
)


print("All model files loaded successfully")

# RECOMMENDATIONS ENGINE

def generate_recommendations(data, prediction):
    tips = []
    
    sleep = float(data.get('sleep_duration', 0))
    if sleep < 5:
        tips.append("Your sleep is critically low. Aim for 7-9 hours. Try a consistent bedtime and no screens 1 hour before bed.")
    elif sleep < 6:
        tips.append("You're slightly sleep-deprived. Prioritize 7-9 hours nightly. A short 20-min nap can help recovery.")
    
    quality = data.get('sleep_quality', '')
    if quality == 'poor':
        tips.append("Poor sleep quality detected. Limit caffeine after 2pm and establish a relaxing wind-down routine.")
    
    bmi = float(data.get('bmi', 0))
    if bmi < 18.5:
        tips.append("Your BMI indicates you're underweight. Focus on nutrient-dense foods and strength training.")
    elif bmi > 30:
        tips.append("Your BMI indicates obesity. Start with low-impact exercise like walking, and consult a nutritionist.")
    elif bmi > 25:
        tips.append("Your BMI is slightly elevated. Small changes like portion control and 30 mins daily walking help.")
    
    hr = int(float(data.get('heart_rate', 0)))
    if hr > 100:
        tips.append("Elevated resting heart rate. Stress management, regular cardio, and hydration can help.")
    elif hr < 50:
        tips.append("Low heart rate detected. If you're not an athlete, consider checking with a healthcare provider.")
    
    steps = int(float(data.get('step_count', 0)))
    if steps < 3000:
        tips.append("Very low step count. Try a 20-minute walk after lunch or take stairs instead of elevators.")
    elif steps < 7000:
        tips.append("You're close to the 10,000 step goal! Add a 15-minute evening walk.")
    
    exercise = int(float(data.get('exercise_duration', 0)))
    if exercise < 15:
        tips.append("Minimal exercise. Start with 15 minutes of brisk walking or bodyweight exercises 3x a week.")
    elif exercise < 30:
        tips.append("Good start! Aim for 150 minutes/week of moderate activity for optimal health.")
    
    water = float(data.get('water_intake', 0))
    if water < 1.5:
        tips.append("Low water intake. Carry a bottle and aim for 2-3 litres daily.")
    
    cal = int(float(data.get('calorie_expenditure', 0)))
    if cal < 1200:
        tips.append("Very low calorie expenditure. Even light activity boosts metabolism and mental clarity.")
    
    stress = data.get('stress_level', '')
    if stress == 'high':
        tips.append("High stress detected. Try daily meditation, breathing exercises, or speak to a counsellor.")
    elif stress == 'medium':
        tips.append("Moderate stress. Take 10-minute breaks every hour and practice the 4-7-8 breathing technique.")
    
    activity = data.get('physical_activity_level', '')
    if activity == 'sedentary':
        tips.append("Sedentary lifestyle is a major risk. Set hourly movement reminders and stand during calls.")
    elif activity == 'moderate':
        tips.append("You're moderately active — push toward vigorous activity 2x a week for cardiovascular benefits.")
    
    smoke = data.get('smoking_alcohol', '')
    if smoke == 'yes':
        tips.append("Smoking/alcohol significantly increase health risks. Consider gradual reduction programs.")
    elif smoke == 'occasional':
        tips.append("Even occasional smoking/alcohol adds up. Try replacing the habit with tea or fruit.")
    
    if prediction == 'fit':
        tips.append("Excellent work! You're in the healthy zone. Keep maintaining these habits!")
    elif prediction == 'at-risk':
        tips.append("You're on the edge. Small consistent changes in the flagged areas above will push you into the Fit zone.")
    else:
        tips.append("Your lifestyle needs attention. Pick ONE area from above to improve this week.")
    
    return tips

# HOME PAGE

@app.route("/")
def home():
    return render_template("index.html")

# PREDICTION

@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    data = {
        "sleep_duration": float(request.form["sleep_duration"]),
        "heart_rate": float(request.form["heart_rate"]),
        "bmi": float(request.form["bmi"]),
        "calorie_expenditure": float(request.form["calorie_expenditure"]),
        "step_count": float(request.form["step_count"]),
        "exercise_duration": float(request.form["exercise_duration"]),
        "water_intake": float(request.form["water_intake"]),
        "diet_type": request.form["diet_type"],
        "stress_level": request.form["stress_level"],
        "sleep_quality": request.form["sleep_quality"],
        "physical_activity_level": request.form["physical_activity_level"],
        "smoking_alcohol": request.form["smoking_alcohol"],
        "gender": request.form["gender"]
    }

    input_df = pd.DataFrame([data])

    # Handle missing values
    for col in input_df.columns:
        if col in median_values:
            input_df[col] = input_df[col].fillna(median_values[col])

    # Encode categorical features
    for col, encoder in feature_encoders.items():
        input_df[col] = encoder.transform(input_df[col].astype(str))

    # Arrange columns same as training
    input_df = input_df[feature_columns]

    # Prediction
    prediction = model.predict(input_df)
    result = target_encoder.inverse_transform(prediction.astype(int))[0]

    # Generate recommendations
    recommendations = generate_recommendations(data, result)

    # Save to database
    cursor = mysql.connection.cursor()
    cursor.execute("""
        INSERT INTO predictions 
        (sleep_duration, heart_rate, bmi, calorie_expenditure, step_count,
         exercise_duration, water_intake, diet_type, stress_level, sleep_quality,
         physical_activity_level, smoking_alcohol, gender, prediction, recommendations)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data['sleep_duration'], data['heart_rate'], data['bmi'],
        data['calorie_expenditure'], data['step_count'], data['exercise_duration'],
        data['water_intake'], data['diet_type'], data['stress_level'],
        data['sleep_quality'], data['physical_activity_level'],
        data['smoking_alcohol'], data['gender'], result, json.dumps(recommendations)
    ))
    mysql.connection.commit()
    cursor.close()

    return render_template(
        "index.html",
        prediction=result,
        recommendations=recommendations
    )

# DASHBOARD

@app.route("/dashboard")
def dashboard():
    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT id, prediction, recommendations, created_at 
        FROM predictions ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()

    # Prepare chart data (last 10, oldest first)
    labels = []
    scores = []
    for row in reversed(rows[-10:]):
        labels.append(row[3].strftime('%d %b'))
        if row[1] == 'fit':
            scores.append(85)
        elif row[1] == 'at-risk':
            scores.append(50)
        else:
            scores.append(20)

    return render_template(
        "dashboard.html",
        predictions=rows,
        labels=labels,
        scores=scores
    )


if __name__ == "__main__":
    app.run(debug=True)