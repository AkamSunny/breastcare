import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../hello_flask/api
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))# .../hello_flask
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")    # .../hello_flask/templates

# Create app pointing to project templates
app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Diagnostic logging
print("BASE_DIR:", BASE_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("TEMPLATES_DIR:", TEMPLATES_DIR)
print("home.html exists?:", os.path.exists(os.path.join(TEMPLATES_DIR, "home.html")))
print("Jinja search path before:", getattr(app.jinja_loader, "searchpath", None))

# Load model + scaler (assumes files are in api/)
model_path = os.path.join(BASE_DIR, "xgb_breastcare.json")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

print("model_path:", model_path, "exists?", os.path.exists(model_path))
print("scaler_path:", scaler_path, "exists?", os.path.exists(scaler_path))

model = XGBClassifier()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

print("Model and scaler loaded successfully.")

@app.route("/ping")
def ping():
    return {"status": "ok", "message": "Flask app is alive!"}

@app.route("/")
def home():
    # check templates path at runtime (debug)
    print("Jinja search path now:", app.jinja_loader.searchpath)
    return render_template("home.html", query="")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def cancerPrediction():
    try:
        # Get inputs from form
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])

        # Prepare input
        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
        new_df = pd.DataFrame(data, columns=[
            'perimeter_worst', 'concave points_worst', 'concave points_mean',
            'area_mean', 'area_worst'
        ])

        # Scale & predict
        new_df_scaled = scaler.transform(new_df)
        real_pred = model.predict(new_df_scaled)
        real_proba = model.predict_proba(new_df_scaled)

        if real_pred[0] == 1:
            output1 = "The patient is diagnosed with Breast Cancer"
            confidence = float(real_proba[0][1] * 100)
        else:
            output1 = "The patient is not diagnosed with Breast Cancer"
            confidence = float(real_proba[0][0] * 100)

        output2 = f"Confidence: {confidence:.2f}%"
        return render_template("home.html", output1=output1, output2=output2,
                               query1=request.form['query1'], query2=request.form['query2'],
                               query3=request.form['query3'], query4=request.form['query4'],
                               query5=request.form['query5'])

    except Exception as e:
        error_message = f"Error: {e}"
        return render_template("home.html", output1=error_message, output2="")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
